// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2018 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/pipelines/sequential/sequential_SfM2.hpp"
#include "openMVG/sfm/pipelines/localization/SfM_Localizer.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_matches_provider.hpp"
#include "openMVG/sfm/pipelines/sequential/SfmSceneInitializer.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_BA.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres.hpp"
#include "openMVG/sfm/sfm_data_filters.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_data_triangulation.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/system/logger.hpp"

#include "third_party/histogram/histogram.hpp"
#include "third_party/htmlDoc/htmlDoc.hpp"

#include <array>
#include <ceres/types.h>
#include <functional>
#include <iostream>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

namespace openMVG {
namespace sfm {

using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::matching;

SequentialSfMReconstructionEngine2::SequentialSfMReconstructionEngine2(
  SfMSceneInitializer * scene_initializer,
  const SfM_Data & sfm_data,
  const std::string & soutDirectory,
  const std::string & sloggingFile)
  : ReconstructionEngine(sfm_data, soutDirectory),
    scene_initializer_(scene_initializer),
    sLogging_file_(sloggingFile),
    cam_type_(EINTRINSIC(PINHOLE_CAMERA_RADIAL3))
{
  if (!sLogging_file_.empty())
  {
    // setup HTML logger
    html_doc_stream_ = std::make_shared<htmlDocument::htmlDocumentStream>("SequentialReconstructionEngine SFM report.");
    html_doc_stream_->pushInfo(
      htmlDocument::htmlMarkup("h1", std::string("SequentialSfMReconstructionEngine")));
    html_doc_stream_->pushInfo("<hr>");

    html_doc_stream_->pushInfo( "Dataset info:");
    html_doc_stream_->pushInfo( "Views count: " +
      htmlDocument::toString( sfm_data.GetViews().size()) + "<br>");
    html_doc_stream_->pushInfo( "Poses count: " +
      htmlDocument::toString( sfm_data.GetPoses().size()) + "<br>");
  }
}

SequentialSfMReconstructionEngine2::~SequentialSfMReconstructionEngine2()
{
  if (!sLogging_file_.empty())
  {
    // Save the reconstruction Log
    std::ofstream htmlFileStream(sLogging_file_.c_str());
    htmlFileStream << html_doc_stream_->getDoc();
  }
}

void SequentialSfMReconstructionEngine2::SetFeaturesProvider(Features_Provider * provider)
{
  features_provider_ = provider;
}

void SequentialSfMReconstructionEngine2::SetMatchesProvider(Matches_Provider * provider)
{
  matches_provider_ = provider;
}

bool SequentialSfMReconstructionEngine2::Process() {

  //-------------------
  //-- Incremental reconstruction
  //-------------------
  //- 1. Init the reconstruction with a seed
  //- 2. While we can localize some cameras in the reconstruction
  //     a. Triangulation
  //     b. Bundle Adjustment and cleaning
  //- 3. Final bundle Adjustment
  //-------------------

  //--
  //- 1. Init the reconstruction with a Seed
  //--
  {
    if (!scene_initializer_ || !scene_initializer_->Process())
    {
      OPENMVG_LOG_ERROR << "Initialization status: Failed";
      return false;
    }
    else
    {
      OPENMVG_LOG_INFO << "Initialization status : Success";
      sfm_data_.poses = scene_initializer_->Get_sfm_data().GetPoses();
    }

    if (!InitTracksAndLandmarks())
      return false;

    if (!sfm_data_.GetPoses().empty())
    {
      const bool bTriangulation = Triangulation();
      Save(sfm_data_, stlplus::create_filespec(sOut_directory_, "Initialization", ".ply"), ESfM_Data(ALL));
      const size_t init_tracks_before = sfm_data_.GetLandmarks().size();
      const size_t init_angle_removed = RemoveOutliers_AngleError(sfm_data_, Square(2.0));
      const size_t init_pixel_removed = RemoveOutliers_PixelResidualError(sfm_data_, Square(4.0));
      OPENMVG_LOG_INFO
        << "-- Init filter stats:"
        << " #tracks: " << init_tracks_before
        << " -angle(" << init_angle_removed << ", thresh=" << Square(2.0) << "deg)"
        << " -pixel(" << init_pixel_removed << ", thresh=" << Square(4.0) << "px)"
        << " -> " << sfm_data_.GetLandmarks().size();

      //-- Display some statistics
      OPENMVG_LOG_INFO
        << "\n\n-------------------------------" << "\n"
        << "-- Starting Structure from Motion (statistics) with:\n"
        << "-- #Camera calibrated: " << sfm_data_.GetPoses().size()
        << " from " << sfm_data_.GetViews().size() << " input images.\n"
        << "-- #Tracks, #3D points: " << sfm_data_.GetLandmarks().size() << "\n"
        << "-------------------------------";

      if (!bTriangulation)
      {
        // We are not able to triangulate enough 3D points for the given poses.
        return false;
      }
    }
  }

  if (sfm_data_.GetPoses().empty())
  {
    return false;
  }

  //--
  //- 2. While we can localize some cameras in the reconstruction
  //     a. Triangulate the landmarks
  //     b. Perform Bundle Adjustment and cleaning (batched)
  //--
  IndexT resection_round = 0;

  // Count total views that need poses (for early termination)
  const IndexT total_views = static_cast<IndexT>(sfm_data_.GetViews().size());

  // BA batching: only run the expensive global BA + triangulation + filtering
  // cycle when enough new cameras have been added since the last BA.
  // Each new camera gets a good initial pose from RefinePose (single-camera
  // BA against fixed 3D points), so deferring global BA is safe.
  // We track poses_at_last_ba to decide when to trigger.
  IndexT poses_at_last_ba = sfm_data_.GetPoses().size();

  // Minimum number of new cameras before we run global BA.
  // This avoids running a full BA cycle (which costs ~13s on 1M tracks)
  // after adding just 2-4 cameras.  The threshold adapts: early in
  // reconstruction when there are few cameras, we BA more often to
  // maintain accuracy; later when the scene is stable, we batch more.
  auto ba_batch_threshold = [](IndexT current_poses) -> IndexT {
    if (current_poses < 50)
      return 5;   // BA every 5 new cameras (scene still fragile)
    if (current_poses < 150)
      return 15;  // BA every 15 new cameras (scene stabilizing)
    return 30;    // BA every 30 new cameras (scene stable, BA expensive)
    };

  // Incrementally estimate the pose of the cameras based on a confidence score.
  const std::array<float, 2> track_inlier_ratios = { 0.2, 0.0 };
  for (auto track_inlier_ratio = track_inlier_ratios.cbegin();
    track_inlier_ratio < track_inlier_ratios.cend(); ++track_inlier_ratio)
  {
    // Early exit: all views have been reconstructed
    if (sfm_data_.GetPoses().size() >= total_views)
      break;

    IndexT pose_before = sfm_data_.GetPoses().size();
    while (AddingMissingView(*track_inlier_ratio))
    {
      const IndexT poses_added_since_ba =
        sfm_data_.GetPoses().size() - poses_at_last_ba;
      const IndexT threshold = ba_batch_threshold(sfm_data_.GetPoses().size());
      const bool all_views_done = sfm_data_.GetPoses().size() >= total_views;

      // Run BA cycle only when enough cameras accumulated, all views
      // are done, or no new cameras were added (stall — let BA clean up).
      if (poses_added_since_ba >= threshold || all_views_done)
      {
        // Create new 3D points
        Triangulation();
        // Adjust the scene
        {
          Bundle_Adjustment_Ceres::BA_Ceres_options options;
          if (sfm_data_.GetPoses().size() > 100 &&
            (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE) ||
              ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
            )
          {
            options.preconditioner_type_ = ceres::JACOBI;
            options.linear_solver_type_ = ceres::SPARSE_SCHUR;
          }
          else
          {
            options.linear_solver_type_ = ceres::DENSE_SCHUR;
          }
          options.max_num_iterations_ = 3;
          Bundle_Adjustment_Ceres bundle_adjustment_obj(options);
          const Optimize_Options ba_refine_options
          (ReconstructionEngine::intrinsic_refinement_options_,
            ReconstructionEngine::extrinsic_refinement_options_,
            Structure_Parameter_Type::ADJUST_ALL,
            Control_Point_Parameter(),
            this->b_use_motion_prior_
          );
          bundle_adjustment_obj.Adjust(sfm_data_, ba_refine_options);
        }
        // Remove unstable triangulations and camera poses
        const size_t tracks_before_filter = sfm_data_.GetLandmarks().size();
        const size_t poses_before_filter = sfm_data_.GetPoses().size();
#if 0
        const size_t angle_removed = RemoveOutliers_AngleError(sfm_data_, 2.0);
        const size_t pixel_removed = RemoveOutliers_PixelResidualError(sfm_data_, 4.0);
#else
        const auto filter_result = RemoveOutliers_AngleAndPixelError(sfm_data_, 2.0, 4.0);
        const size_t angle_removed = filter_result.first;
        const size_t pixel_removed = filter_result.second;
#endif
        eraseUnstablePosesAndObservations(sfm_data_);

        OPENMVG_LOG_INFO
          << "-- Round " << resection_round << " filter stats:"
          << " #poses: " << poses_before_filter << " -> " << sfm_data_.GetPoses().size()
          << " | #tracks: " << tracks_before_filter
          << " -angle(" << angle_removed << ")"
          << " -pixel(" << pixel_removed << ")"
          << " -> " << sfm_data_.GetLandmarks().size();

        poses_at_last_ba = sfm_data_.GetPoses().size();
        ++resection_round;
      }
      else
      {
        // Lightweight round: only triangulate new points so the next
        // resection round has more 2D-3D correspondences to work with.
        // Skip the expensive BA + filtering cycle.
        Triangulation();
      }

      // Stop if no cameras have been added
      const IndexT pose_after = sfm_data_.GetPoses().size();
      if (pose_before >= pose_after)
        break;
      pose_before = sfm_data_.GetPoses().size();

      // Early exit: all views have been reconstructed
      if (sfm_data_.GetPoses().size() >= total_views)
        break;

      // Since we have augmented our set of poses we can reset our track inlier ratio iterator
      track_inlier_ratio = track_inlier_ratios.cbegin();
    }
  }

  //--
  //- 3. Final bundle Adjustment (full iterations)
  //--
  BundleAdjustment();

  //-- Reconstruction done.
  //-- Display some statistics
  OPENMVG_LOG_INFO
    << "\n-------------------------------" << "\n"
    << "-- Structure from Motion (statistics):\n"
    << "-- #Camera calibrated: " << sfm_data_.GetPoses().size()
    << " from " << sfm_data_.GetViews().size() << " input images.\n"
    << "-- #Tracks, #3D points: " << sfm_data_.GetLandmarks().size() << "\n"
    << "-- #Poses loop used: " << resection_round << "\n"
    << "-------------------------------";

  return true;
}

bool SequentialSfMReconstructionEngine2::InitTracksAndLandmarks()
{
  // Compute tracks from matches
  tracks::TracksBuilder tracksBuilder;
  {
    tracksBuilder.Build(matches_provider_->pairWise_matches_);
    tracksBuilder.Filter();
    tracksBuilder.ExportToSTL(map_tracks_);

    OPENMVG_LOG_INFO << "\n" << "Track stats";
    {
      std::ostringstream osTrack;
      //-- Display stats :
      //    - number of images
      //    - number of tracks
      std::set<uint32_t> set_imagesId;
      tracks::TracksUtilsMap::ImageIdInTracks(map_tracks_, set_imagesId);
      osTrack
        << "------------------" << "\n"
        << "-- Tracks Stats --" << "\n"
        << " Tracks number: " << tracksBuilder.NbTracks() << "\n"
        << " Images Id: " << "\n";
      std::copy(set_imagesId.cbegin(),
        set_imagesId.cend(),
        std::ostream_iterator<uint32_t>(osTrack, ", "));
      osTrack << "\n------------------" << "\n";

      std::map<uint32_t, uint32_t> map_Occurrence_TrackLength;
      tracks::TracksUtilsMap::TracksLength(map_tracks_, map_Occurrence_TrackLength);
      osTrack << "TrackLength, Occurrence" << "\n";
      for (const auto & it : map_Occurrence_TrackLength)  {
        osTrack << "\t" << it.first << "\t" << it.second << "\n";
      }
      osTrack << "\n";
      OPENMVG_LOG_INFO << osTrack.str();
    }
  }

  // Init the putative landmarks
  {
    // For every track add the observations:
    // - views and feature positions that see this landmark
    for ( const auto & iterT : map_tracks_ )
    {
      Observations obs;
      const size_t track_Length = iterT.second.size();
      for (const auto & track_ids : iterT.second) // {ViewId, FeatureId}
      {
        const auto & view_id = track_ids.first;
        const auto & feat_id = track_ids.second;
        const Vec2 x = features_provider_->feats_per_view[view_id][feat_id].coords().cast<double>();
        obs.insert({view_id, Observation(x, feat_id)});
      }
      landmarks_[iterT.first].obs = std::move(obs);
    }
  }

  // Initialize the shared track visibility helper
  shared_track_visibility_helper_.reset(new openMVG::tracks::SharedTrackVisibilityHelper(map_tracks_));
  return map_tracks_.size() > 0;
}

bool SequentialSfMReconstructionEngine2::Triangulation()
{
  //--
  // Triangulation
  //--
  // Build the structure from putative landmarks, keeping only observations
  // that are linked to valid pose and intrinsic data. This fuses the old
  //   sfm_data_.structure = landmarks_;
  //   eraseObservationsWithMissingPoses(sfm_data_, min_sample_index);
  // into a single pass, avoiding the allocation and deep-copy of landmarks
  // that would be immediately discarded by the filter.

  const double max_reprojection_error = 4.0;
  const IndexT min_required_inliers = 2;
  const IndexT min_sample_index = 2;

  // Pre-build a sorted vector of view ids whose pose exists
  std::vector<IndexT> valid_view_ids;
  valid_view_ids.reserve(sfm_data_.views.size());
  for (const auto & view_it : sfm_data_.views)
  {
    if (sfm_data_.poses.count(view_it.second->id_pose))
      valid_view_ids.push_back(view_it.first);
  }
  std::sort(valid_view_ids.begin(), valid_view_ids.end());

  // Instead of clear + re-insert, we do an incremental update:
  //  1. For landmarks already in structure: update their obs from
  //     landmarks_ to include any newly-posed views, then mark for
  //     re-triangulation (set X to zero).
  //  2. For landmarks NOT in structure: insert if they have enough
  //     posed observations.
  // This avoids all hash node deallocations and most reallocations.

  size_t new_landmarks_added = 0;
  size_t existing_landmarks_updated = 0;
  const size_t landmarks_total = landmarks_.size();

  for (const auto & lm_it : landmarks_)
  {
    // Count valid observations in the master list
    const Observations & src_obs = lm_it.second.obs;
    size_t valid_count = 0;
    for (const auto & obs_it : src_obs)
    {
      if (std::binary_search(valid_view_ids.cbegin(), valid_view_ids.cend(), obs_it.first))
      {
        ++valid_count;
        // Early-out: we only need to know if >= min_sample_index
        if (valid_count >= min_sample_index)
          break;
      }
    }

    if (valid_count < min_sample_index)
      continue;

    // Check if this landmark already exists in structure
    const auto existing_it = sfm_data_.structure.find(lm_it.first);
    if (existing_it != sfm_data_.structure.end())
    {
      // Already exists — check if its observation set actually changed
      // (newly posed views may have been added since last round)
      Landmark & dst = existing_it->second;

      // Count how many valid observations the master list has now
      size_t new_valid_count = 0;
      for (const auto & obs_it : src_obs)
      {
        if (std::binary_search(valid_view_ids.cbegin(), valid_view_ids.cend(), obs_it.first))
          ++new_valid_count;
      }

      // If the observation count hasn't changed, the landmark's X is
      // still valid — skip the expensive re-triangulation.
      if (new_valid_count == dst.obs.size())
      {
        ++existing_landmarks_updated;
        continue;
      }

      // Observation set changed — rebuild and re-triangulate
      dst.obs.clear();
      dst.obs.reserve(src_obs.size());
      for (const auto & obs_it : src_obs)
      {
        if (std::binary_search(valid_view_ids.cbegin(), valid_view_ids.cend(), obs_it.first))
        {
          dst.obs.push_back_unchecked(obs_it);
        }
      }
      // Reset X so robust_triangulation re-triangulates with updated obs
      dst.X = Vec3::Zero();
      ++existing_landmarks_updated;
    }
    else
    {
      // New landmark — this is the only path that allocates a hash node
      Landmark & dst = sfm_data_.structure[lm_it.first];
      dst.X = Vec3::Zero();
      dst.obs.reserve(src_obs.size());
      for (const auto & obs_it : src_obs)
      {
        if (std::binary_search(valid_view_ids.cbegin(), valid_view_ids.cend(), obs_it.first))
        {
          dst.obs.push_back_unchecked(obs_it);
        }
      }
      ++new_landmarks_added;
    }
  }

  // Also remove any structure entries whose landmark was erased from
  // landmarks_ (shouldn't happen, but be safe) or that no longer have
  // enough posed observations. This handles landmarks that lost posed
  // views due to pose removal in eraseUnstablePosesAndObservations.
  {
    auto it = sfm_data_.structure.begin();
    while (it != sfm_data_.structure.end())
    {
      // If the master landmarks_ doesn't contain this track, or if
      // after observation update it has too few observations, remove it.
      if (it->second.obs.size() < min_sample_index)
        it = sfm_data_.structure.erase(it);
      else
        ++it;
    }
  }

  const size_t landmarks_before_triangulation = sfm_data_.structure.size();

  SfM_Data_Structure_Computation_Robust triangulation_engine(
      max_reprojection_error,
      min_required_inliers,
      min_sample_index,
      triangulation_method_);

  triangulation_engine.triangulate(sfm_data_);

  OPENMVG_LOG_INFO
    << "-- Triangulation: #poses: " << sfm_data_.GetPoses().size()
    << " | #landmarks: " << landmarks_total
    << " (updated: " << existing_landmarks_updated
    << ", new: " << new_landmarks_added << ")"
    << " -> " << landmarks_before_triangulation << " (before triangulation)"
    << " -> " << sfm_data_.structure.size() << " (after triangulation)";

  return !sfm_data_.structure.empty();
}

bool SequentialSfMReconstructionEngine2::AddingMissingView
(
  const float & track_inlier_ratio
)
{
  if (sfm_data_.GetLandmarks().empty())
    return false;

  // Collect the views that do not have any 3D pose (sorted vector, no heap per element)
  std::vector<IndexT> views_with_no_pose;
  views_with_no_pose.reserve(sfm_data_.GetViews().size());
  for (const auto & view_it : sfm_data_.GetViews())
  {
    const View * v = view_it.second.get();
    if (sfm_data_.GetPoses().count(v->id_pose) == 0)
      views_with_no_pose.push_back(view_it.first);
  }

  if (views_with_no_pose.empty())
    return false;

  const IndexT pose_before = sfm_data_.GetPoses().size();

  // Get the track ids of the reconstructed landmarks
  const std::vector<IndexT> reconstructed_trackId = [&]
  {
    std::vector<IndexT> tracks_ids;
    tracks_ids.reserve(sfm_data_.GetLandmarks().size());
    std::transform(sfm_data_.GetLandmarks().cbegin(), sfm_data_.GetLandmarks().cend(),
      std::back_inserter(tracks_ids),
      stl::RetrieveKey());
    std::sort(tracks_ids.begin(), tracks_ids.end());
    return tracks_ids;
  }();

  // Phase 1: Gather resection candidates with their 2D-3D match info (parallel)
  struct ResectionCandidate {
    IndexT view_id;
    std::vector<IndexT> track_id_for_resection;  // sorted
    std::vector<IndexT> feature_id_for_resection; // parallel to track_id_for_resection
    double track_ratio;
  };
  std::vector<ResectionCandidate> candidates;

  #ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel
  #endif
  for (const auto & view_id : views_with_no_pose)
  {
  #ifdef OPENMVG_USE_OPENMP
    #pragma omp single nowait
  #endif
    {
      // List the track related to the current view_id
      std::vector<uint32_t> view_track_ids;
      std::vector<uint32_t> view_feat_ids;
      shared_track_visibility_helper_->GetTracksInImages({view_id}, view_track_ids, view_feat_ids);

      // Get the ids of the already reconstructed tracks (sorted vector output)
      std::vector<IndexT> track_id_for_resection;
      std::set_intersection(view_track_ids.cbegin(), view_track_ids.cend(),
        reconstructed_trackId.cbegin(), reconstructed_trackId.cend(),
        std::back_inserter(track_id_for_resection));

      const double track_ratio = track_id_for_resection.size() / static_cast<float>(view_track_ids.size() + 1);
      OPENMVG_LOG_INFO
        << "ViewId: " << view_id
        << "; #number of 2D-3D matches: " << track_id_for_resection.size()
        << "; " << track_ratio * 100 << " % of the view track coverage.";

      if (!track_id_for_resection.empty() && track_ratio > track_inlier_ratio)
      {
        // Build feature ids from the flat vectors using binary search
        std::vector<IndexT> feature_id_for_resection;
        feature_id_for_resection.reserve(track_id_for_resection.size());
        for (const auto & trackId : track_id_for_resection)
        {
          auto it = std::lower_bound(view_track_ids.begin(), view_track_ids.end(), trackId);
          if (it != view_track_ids.end() && *it == trackId)
          {
            feature_id_for_resection.push_back(view_feat_ids[std::distance(view_track_ids.begin(), it)]);
          }
        }

        #ifdef OPENMVG_USE_OPENMP
        #pragma omp critical
        #endif
        {
          candidates.push_back({view_id,
            std::move(track_id_for_resection),
            std::move(feature_id_for_resection),
            track_ratio});
        }
      }
    }
  }

  // Sort candidates: most 2D-3D matches first (deterministic ordering)
  std::sort(candidates.begin(), candidates.end(),
    [](const ResectionCandidate& a, const ResectionCandidate& b) {
      if (a.track_id_for_resection.size() != b.track_id_for_resection.size())
        return a.track_id_for_resection.size() > b.track_id_for_resection.size();
      return a.view_id < b.view_id; // tie-break by view_id for determinism
    });

  // Phase 2 & 3: Resect each candidate and apply results immediately.
  // This is done sequentially because the RefinePose -> Adjust call chain
  // allocates SfM_Data, ceres::Problem, ceres::Solver::Options/Summary,
  // and large Eigen temporaries on the call stack, which exceeds the
  // default OpenMP worker thread stack size (1-4 MB on Windows) and
  // causes silent stack corruption.  Each single-pose BA is trivially
  // fast (DENSE_SCHUR on ~6 parameters), so serialising has negligible
  // impact on total runtime.

  // Construct resection buffers once outside the loop so that:
  //  - vec_inliers retains its heap capacity across iterations
  //  - Mat members (pt2D, pt3D) skip reallocation via resize() when
  //    consecutive candidates have the same match count (Eigen only
  //    reallocates when the total element count changes)
  //  - pose, projection_matrix, etc. are not re-constructed each iteration
  Image_Localizer_Match_Data resection_data;
  Mat pt2D_original;  // Same type as resection_data.pt2D (MatrixXd) for swap
  geometry::Pose3 pose;

  for (size_t ci = 0; ci < candidates.size(); ++ci)
  {
    const auto & candidate = candidates[ci];
    const auto & view_id = candidate.view_id;
    const auto & track_id_for_resection = candidate.track_id_for_resection;
    const auto & feature_id_for_resection = candidate.feature_id_for_resection;
    const auto n_pts = static_cast<Eigen::Index>(track_id_for_resection.size());

    // Resize — Eigen skips reallocation when total element count is unchanged.
    // After the swap below, resection_data.pt2D holds the previous iteration's
    // pt2D_original buffer; if n_pts matches the previous iteration, this is free.
    resection_data.pt2D.resize(2, n_pts);
    resection_data.pt3D.resize(3, n_pts);
    resection_data.vec_inliers.clear();
    resection_data.error_max = std::numeric_limits<double>::infinity();
    pt2D_original.resize(2, n_pts);

    // Look if the intrinsic data is known or not
    const View * view = sfm_data_.GetViews().at(view_id).get();
    std::shared_ptr<cameras::IntrinsicBase> intrinsic;
    {
      const auto intrinsic_it = sfm_data_.GetIntrinsics().find(view->id_intrinsic);
      if (intrinsic_it != sfm_data_.GetIntrinsics().end())
      {
        intrinsic = intrinsic_it->second;
      }
    }

    // Collect the feature observations
    auto track_it = track_id_for_resection.cbegin();
    auto feat_it = feature_id_for_resection.cbegin();
    for (Eigen::Index cpt = 0; cpt < n_pts; ++cpt, ++track_it, ++feat_it)
    {
      resection_data.pt3D.col(cpt) = sfm_data_.GetLandmarks().at(*track_it).X;
      resection_data.pt2D.col(cpt) = pt2D_original.col(cpt) =
        features_provider_->feats_per_view.at(view_id)[*feat_it].coords().cast<double>();
      // Handle image distortion if intrinsic is known (to ease the resection)
      if (intrinsic && intrinsic->have_disto())
      {
        resection_data.pt2D.col(cpt) = intrinsic->get_ud_pixel(resection_data.pt2D.col(cpt));
      }
    }

    const bool bResection = sfm::SfM_Localizer::Localize
    (
      intrinsic ? resection_method_ : resection::SolverType::DLT_6POINTS,
      {view->ui_width, view->ui_height},
      intrinsic ? intrinsic.get() : nullptr,
      resection_data,
      pose
    );
    // Restore original (distorted) image domain points for RefinePose.
    // Use swap instead of move so both Mat buffers survive for potential
    // reuse next iteration (swap is a pointer exchange, zero cost).
    resection_data.pt2D.swap(pt2D_original);

    const float inlier_ratio = resection_data.vec_inliers.size()/static_cast<float>(feature_id_for_resection.size());
    OPENMVG_LOG_INFO
      << std::endl
      << "-------------------------------" << "\n"
      << "-- Robust Resection of camera index: <" << view_id << "> image: "
      <<  view->s_Img_path <<"\n"
      << "-- Threshold: " << resection_data.error_max << "\n"
      << "-- Resection status: " << (bResection ? "OK" : "FAILED") << "\n"
      << "-- Nb points used for Resection: " << feature_id_for_resection.size() << "\n"
      << "-- Nb points validated by robust estimation: " << resection_data.vec_inliers.size() << "\n"
      << "-- % points validated: "
      << inlier_ratio * 100 << "\n"
      << "-------------------------------";

    // Refine the pose of the found camera pose by using a BA and fix 3D points.
    if (bResection && inlier_ratio > 0.5)
    {
      if (!intrinsic)
      {
        Mat3 K, R;
        Vec3 t;
        KRt_From_P(resection_data.projection_matrix, &K, &R, &t);

        const double focal = (K(0,0) + K(1,1))/2.0;
        const Vec2 principal_point(K(0,2), K(1,2));

        switch (cam_type_)
        {
          case PINHOLE_CAMERA:
            intrinsic = std::make_shared<Pinhole_Intrinsic>
              (view->ui_width, view->ui_height, focal, principal_point(0), principal_point(1));
          break;
          case PINHOLE_CAMERA_RADIAL1:
            intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K1>
              (view->ui_width, view->ui_height, focal, principal_point(0), principal_point(1));
          break;
          case PINHOLE_CAMERA_RADIAL3:
            intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K3>
              (view->ui_width, view->ui_height, focal, principal_point(0), principal_point(1));
          break;
          case PINHOLE_CAMERA_BROWN:
            intrinsic = std::make_shared<Pinhole_Intrinsic_Brown_T2>
              (view->ui_width, view->ui_height, focal, principal_point(0), principal_point(1));
          break;
          case PINHOLE_CAMERA_FISHEYE:
            intrinsic = std::make_shared<Pinhole_Intrinsic_Fisheye>
              (view->ui_width, view->ui_height, focal, principal_point(0), principal_point(1));
          break;
          default:
            OPENMVG_LOG_ERROR << "Try to create an unknown camera type.";
        }
      }
      const bool b_refine_pose = true;
      const bool b_refine_intrinsics = false;
      if (intrinsic && sfm::SfM_Localizer::RefinePose(
          intrinsic.get(), pose,
          resection_data, b_refine_pose, b_refine_intrinsics))
      {
        // Validate that the refined pose contains finite values
        // (degenerate resections or numerically unstable BA can produce NaN)
        if (!pose.rotation().allFinite() || !pose.center().allFinite())
        {
          OPENMVG_LOG_WARNING << "Pose for view " << view_id
            << " contains non-finite values after refinement, discarding.";
          continue;
        }

        // Apply result immediately — single find() instead of count()+at()
        const auto existing_intrinsic_it =
          sfm_data_.intrinsics.find(view->id_intrinsic);
        if (existing_intrinsic_it == sfm_data_.intrinsics.end())
        {
          // Need a new intrinsic id
          IndexT new_intrinsic_id = 0;
          if (!sfm_data_.intrinsics.empty())
          {
            // Find max existing id — intrinsics map is small, just iterate
            for (const auto & kv : sfm_data_.intrinsics)
            {
              if (kv.first >= new_intrinsic_id)
                new_intrinsic_id = kv.first + 1;
            }
          }
          sfm_data_.views.at(view_id)->id_intrinsic = new_intrinsic_id;
          sfm_data_.intrinsics[new_intrinsic_id] = intrinsic;
        }

        sfm_data_.poses[view->id_pose] = pose;
      }
    }
  }

  const IndexT pose_after = sfm_data_.GetPoses().size();
  OPENMVG_LOG_INFO
    << "-- AddingMissingView: poses " << pose_before << " -> " << pose_after
    << " | #landmarks: " << sfm_data_.GetLandmarks().size()
    << " | #remaining views: " << views_with_no_pose.size();
  return (pose_after != pose_before);
}

bool SequentialSfMReconstructionEngine2::BundleAdjustment()
{
#if 1
  Bundle_Adjustment_Ceres::BA_Ceres_options options;
  if (sfm_data_.GetPoses().size() > 100 &&
    (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE) ||
      ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE)))
  {
    options.preconditioner_type_ = ceres::JACOBI;
    options.linear_solver_type_ = ceres::SPARSE_SCHUR;
  }
  else
  {
    options.linear_solver_type_ = ceres::DENSE_SCHUR;
  }

  // The original defaults (function_tolerance=0.048, max_iterations=5,
  // trust_region=1e4) caused the final BA to do zero work on a
  // near-converged scene.  The fix is tight tolerances so the solver
  // doesn't stop prematurely.  The trust region stays at the default 1e4
  // — the intermediate BAs use the same value and waste no iterations.
  // Setting it to 1e16 causes ~9 rejected steps as the solver shrinks
  // the radius back down to ~1e5, wasting ~1s per rejected step.
  options.max_num_iterations_ = 20;
  options.parameter_tolerance_ = 1e-8;
  options.function_tolerance_ = 1e-6;
  options.gradient_tolerance_ = 1e-10;
  options.use_nonmonotonic_steps_ = false;

  Bundle_Adjustment_Ceres bundle_adjustment_obj(options);
  const Optimize_Options ba_refine_options
  (ReconstructionEngine::intrinsic_refinement_options_,
    ReconstructionEngine::extrinsic_refinement_options_,
    Structure_Parameter_Type::ADJUST_ALL,
    Control_Point_Parameter(),
    this->b_use_motion_prior_
  );
  return bundle_adjustment_obj.Adjust(sfm_data_, ba_refine_options);
#else
  Bundle_Adjustment_Ceres::BA_Ceres_options options;
  if (sfm_data_.GetPoses().size() > 100 &&
    (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE) ||
      ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
    )
    // Enable sparse BA only if a sparse lib is available and if there more than 100 poses
  {
    options.preconditioner_type_ = ceres::JACOBI;
    options.linear_solver_type_ = ceres::SPARSE_SCHUR;
  }
  else
  {
    options.linear_solver_type_ = ceres::DENSE_SCHUR;
  }
  Bundle_Adjustment_Ceres bundle_adjustment_obj(options);
  const Optimize_Options ba_refine_options
  (ReconstructionEngine::intrinsic_refinement_options_,
    ReconstructionEngine::extrinsic_refinement_options_,
    Structure_Parameter_Type::ADJUST_ALL, // Adjust scene structure
    Control_Point_Parameter(),
    this->b_use_motion_prior_
  );
  return bundle_adjustment_obj.Adjust(sfm_data_, ba_refine_options);
#endif
}

} // namespace sfm
} // namespace openMVG
