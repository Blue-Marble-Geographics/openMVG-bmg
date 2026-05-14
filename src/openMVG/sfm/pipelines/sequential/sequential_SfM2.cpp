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

// Pipeline-health diagnostics. Cheap per-checkpoint, but the per-view
// `#pragma omp critical` writes into `view_diag_records` inside the parallel
// resection loop serialize the loop on heavily threaded runs. Off by default
// for release; flip to 1 to triage pipeline regressions.
#ifndef OPENMVG_SFM_PIPELINE_DIAG
#define OPENMVG_SFM_PIPELINE_DIAG 0
#endif

// Toggle: relaxed angle thresholds for foliage / narrow-baseline scenes.
//   1 = current optimisation (1.5deg in-loop post-BA, 3.0deg post-init).
//   0 = S3 reference values (2.0deg in-loop, Square(2.0)deg post-init).
//   Defined here as a single switch so all four call sites flip together.
#ifndef OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS
#define OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS 0
#endif

// Toggle: freeze intrinsics on intermediate BAs (speed optimisation).
//   1 = intermediate BAs refine extrinsics+structure only; final STRICT BA
//       refines intrinsics end-to-end. ~5-15s faster per run.
//   0 = safe default (refine intrinsics every intermediate BA). Was disabled
//       to diagnose a 'ghost-layer' regression on a dataset with poor EXIF-
//       derived intrinsics; re-enable to test if your inputs are well-
//       calibrated.
#ifndef OPENMVG_SFM2_FREEZE_INTRINSICS_ON_INTERMEDIATE_BA
#define OPENMVG_SFM2_FREEZE_INTRINSICS_ON_INTERMEDIATE_BA 0
#endif

namespace openMVG {
namespace sfm {

using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::matching;

#if OPENMVG_SFM_PIPELINE_DIAG
// NOTE: kept at namespace scope (not in an anonymous namespace) so that
// Hash_Map<IndexT, ViewDiagRecord> can be named from member functions
// without IntelliSense / some compilers tripping on anon-namespace lookup.
// `static` on the free functions gives them internal linkage instead.

// Reasons a view may fail to be calibrated. Tracked per-view so we can give
// a definitive post-mortem for every input image at the end of Process().
enum class ResectionFailReason : uint8_t {
  Unknown                    = 0,
  NoVisibleTracks            = 1,  // view sees zero tracks at all
  NoReconstructedTracks      = 2,  // view sees tracks, but none are triangulated
  LowTrackRatio              = 3,  // 2D-3D ratio below track_inlier_ratio gate
  ResectionFailed            = 4,  // SfM_Localizer::Localize returned false
  LowInlierRatio             = 5,  // resection ran, inlier ratio <= 0.5
  RefinePoseFailed           = 6,  // post-resection BA refinement failed
  Calibrated                 = 7   // success (kept for symmetry / debugging)
};

static inline const char * to_string(ResectionFailReason r)
{
  switch (r) {
    case ResectionFailReason::NoVisibleTracks:       return "NO_VISIBLE_TRACKS";
    case ResectionFailReason::NoReconstructedTracks: return "NO_RECONSTRUCTED_TRACKS";
    case ResectionFailReason::LowTrackRatio:         return "LOW_TRACK_RATIO";
    case ResectionFailReason::ResectionFailed:       return "RESECTION_FAILED";
    case ResectionFailReason::LowInlierRatio:        return "LOW_INLIER_RATIO";
    case ResectionFailReason::RefinePoseFailed:      return "REFINE_POSE_FAILED";
    case ResectionFailReason::Calibrated:            return "CALIBRATED";
    default:                                         return "UNKNOWN";
  }
}

// One entry per input view, accumulated across the run. Updated under
// `#pragma omp critical` because AddingMissingView() is parallelized.
struct ViewDiagRecord {
  ResectionFailReason last_reason = ResectionFailReason::Unknown;
  uint32_t            attempts    = 0;
  uint32_t            best_2d3d   = 0;     // max 2D-3D matches seen
  uint32_t            last_inliers= 0;
  float               last_ratio  = 0.f;   // last inlier ratio
  float               best_track_ratio = 0.f;
};

// --- Checkpoint 1: input sanity ---------------------------------------------
static void DiagInputs(
  const SfM_Data & sfm_data,
  const Features_Provider * feats,
  const Matches_Provider * matches)
{
  const size_t n_views   = sfm_data.GetViews().size();
  const size_t n_intr    = sfm_data.GetIntrinsics().size();
  const size_t n_seed    = sfm_data.GetPoses().size();
  const size_t n_pairs   = matches ? matches->pairWise_matches_.size() : 0;
  const size_t n_feat_v  = feats   ? feats->feats_per_view.size()      : 0;

  size_t total_features = 0;
  size_t total_matches  = 0;
  size_t views_no_feats = 0;
  if (feats)
  {
    for (const auto & v : sfm_data.GetViews())
    {
      auto it = feats->feats_per_view.find(v.first);
      if (it == feats->feats_per_view.end() || it->second.empty())
        ++views_no_feats;
      else
        total_features += it->second.size();
    }
  }
  if (matches)
  {
    for (const auto & pw : matches->pairWise_matches_)
      total_matches += pw.second.size();
  }

  // Per-view pair count
  Hash_Map<IndexT, size_t> pv;
  for (const auto & v : sfm_data.GetViews()) pv[v.first] = 0;
  if (matches)
  {
    for (const auto & p : matches->pairWise_matches_)
    {
      ++pv[p.first.first];
      ++pv[p.first.second];
    }
  }
  size_t isolated = 0;
  std::vector<size_t> pv_counts; pv_counts.reserve(pv.size());
  for (const auto & kv : pv) { pv_counts.push_back(kv.second); if (kv.second <= 1) ++isolated; }
  std::sort(pv_counts.begin(), pv_counts.end());

  OPENMVG_LOG_INFO
    << "[PIPE-DIAG/INPUT]"
    << " views=" << n_views
    << " intrinsics=" << n_intr
    << " seed_poses=" << n_seed
    << " | features: views_with=" << n_feat_v
    << " views_without=" << views_no_feats
    << " total_keypoints=" << total_features
    << " | matches: pairs=" << n_pairs
    << " total=" << total_matches
    << " avg_pairs/view=" << (n_views ? (2.0 * n_pairs) / n_views : 0.0)
    << " | per-view pair count: min=" << (pv_counts.empty() ? 0 : pv_counts.front())
    << " median=" << (pv_counts.empty() ? 0 : pv_counts[pv_counts.size()/2])
    << " max=" << (pv_counts.empty() ? 0 : pv_counts.back())
    << " isolated(<=1)=" << isolated;

  if (n_views == 0)
    OPENMVG_LOG_ERROR << "[PIPE-DIAG/INPUT] FATAL: no views in SfM_Data.";
  if (n_intr == 0)
    OPENMVG_LOG_WARNING << "[PIPE-DIAG/INPUT] No intrinsics defined; resection will use DLT_6POINTS.";
  if (views_no_feats > 0)
    OPENMVG_LOG_WARNING << "[PIPE-DIAG/INPUT] " << views_no_feats
      << " view(s) have NO features. Those images cannot be calibrated."
      << " Re-run ComputeFeatures for the missing views.";
  if (n_pairs == 0)
    OPENMVG_LOG_ERROR << "[PIPE-DIAG/INPUT] FATAL: no pairwise matches loaded.";
  if (n_views > 0 && (2.0 * n_pairs) / n_views < 5.0)
    OPENMVG_LOG_WARNING << "[PIPE-DIAG/INPUT] Sparse match graph: only "
      << (2.0 * n_pairs) / n_views << " pairs/view (typical drone: 15-50)."
      << " Re-run ComputeMatches with more neighbors / EXHAUSTIVE pairing.";
  if (isolated > 0)
    OPENMVG_LOG_WARNING << "[PIPE-DIAG/INPUT] " << isolated
      << " view(s) have <=1 match pair; they almost certainly cannot be calibrated.";
}

// --- Checkpoint 2: per-round summary ----------------------------------------
static void DiagRound(
  IndexT round,
  float track_inlier_ratio,
  IndexT poses_before,
  IndexT poses_after,
  IndexT landmarks_after,
  IndexT remaining_views)
{
  OPENMVG_LOG_INFO
    << "[PIPE-DIAG/ROUND " << round << "]"
    << " ratio_gate=" << track_inlier_ratio
    << " poses: " << poses_before << " -> " << poses_after
    << " (+" << (poses_after - poses_before) << ")"
    << " | tracks=" << landmarks_after
    << " | remaining_views=" << remaining_views;
}

// --- Checkpoint 3: per-view post-mortem -------------------------------------
static void DiagOutputs(
  const SfM_Data & sfm_data,
  const Hash_Map<IndexT, ViewDiagRecord> & records)
{
  const size_t n_views = sfm_data.GetViews().size();
  const size_t n_poses = sfm_data.GetPoses().size();

  // Tally reasons across un-calibrated views
  std::map<ResectionFailReason, size_t> reason_counts;
  std::vector<IndexT> missing;
  missing.reserve(n_views > n_poses ? n_views - n_poses : 0);
  for (const auto & v : sfm_data.GetViews())
  {
    const View * view = v.second.get();
    if (sfm_data.GetPoses().count(view->id_pose) == 0)
    {
      missing.push_back(v.first);
      auto it = records.find(v.first);
      const ResectionFailReason r = (it != records.end())
        ? it->second.last_reason
        : ResectionFailReason::NoVisibleTracks;
      ++reason_counts[r];
    }
  }

  OPENMVG_LOG_INFO
    << "[PIPE-DIAG/OUTPUT]"
    << " input_views=" << n_views
    << " calibrated=" << n_poses
    << " missing=" << missing.size()
    << " (" << (n_views ? (100.0 * missing.size() / n_views) : 0.0) << "%)";

  if (!reason_counts.empty())
  {
    std::ostringstream os;
    os << "[PIPE-DIAG/OUTPUT] failure reasons:";
    for (const auto & rc : reason_counts)
      os << " " << to_string(rc.first) << "=" << rc.second;
    OPENMVG_LOG_INFO << os.str();
  }

  // Per-missing-view detail (cap output to avoid huge logs)
  const size_t max_detail = 60;
  if (!missing.empty())
  {
    std::ostringstream os;
    os << "[PIPE-DIAG/OUTPUT] missing view detail (showing up to "
       << max_detail << "):\n";
    size_t printed = 0;
    for (const IndexT vid : missing)
    {
      if (printed >= max_detail) { os << "    ... (" << (missing.size() - printed) << " more)\n"; break; }
      const auto & view = sfm_data.GetViews().at(vid);
      auto it = records.find(vid);
      os << "    view=" << vid
         << " img=" << view->s_Img_path;
      if (it != records.end())
      {
        const auto & r = it->second;
        os << " reason=" << to_string(r.last_reason)
           << " attempts=" << r.attempts
           << " best_2d3d=" << r.best_2d3d
           << " best_track_ratio=" << r.best_track_ratio
           << " last_inliers=" << r.last_inliers
           << " last_inlier_ratio=" << r.last_ratio;
      }
      else
      {
        os << " reason=NEVER_REACHED_RESECTION";
      }
      os << "\n";
      ++printed;
    }
    OPENMVG_LOG_INFO << os.str();
  }

  // Hard-failure heuristics
  if (n_poses == 0)
    OPENMVG_LOG_ERROR << "[PIPE-DIAG/OUTPUT] FATAL: zero cameras calibrated.";
  else if (missing.size() > n_views / 5)
    OPENMVG_LOG_WARNING << "[PIPE-DIAG/OUTPUT] "
      << missing.size() << "/" << n_views << " views uncalibrated."
      << " Inspect failure reasons above. Common fixes:"
      << " (a) NO_RECONSTRUCTED_TRACKS / LOW_TRACK_RATIO -> sparser match graph;"
      << " add more match neighbors or EXHAUSTIVE pairs."
      << " (b) RESECTION_FAILED -> intrinsic guess wrong; check camera_model."
      << " (c) LOW_INLIER_RATIO -> noisy matches; tighten geometric filter.";
}
#endif // OPENMVG_SFM_PIPELINE_DIAG

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

  // -------------------------------------------------------------------------
  // Analytic Radial3 Jacobian self-test was validated (max err ~9e-12 vs
  // AutoDiff). Disabled in production runs; re-enable manually if the
  // analytic class is modified.
  // -------------------------------------------------------------------------
#if 0
  {
    const double err = RunSelfTest_AnalyticReprojectionCost_Radial3(1000);
    OPENMVG_LOG_INFO
        << "[Analytic-Radial3 self-test] 1000 trials, max err vs AutoDiff = "
        << err
        << "  (pass if < 1e-7)";
  }
#endif

#if OPENMVG_SFM_PIPELINE_DIAG
  // Per-view diagnostic accumulator. Built up by AddingMissingView() and
  // consumed at the end by DiagOutputs(). Kept on the stack of Process() so
  // it doesn't outlive the run.
  Hash_Map<IndexT, ViewDiagRecord> view_diag_records;
  // Checkpoint 1: validate inputs before doing anything.
  DiagInputs(sfm_data_, features_provider_, matches_provider_);
#endif

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

#if OPENMVG_SFM_PIPELINE_DIAG
    OPENMVG_LOG_INFO
      << "[PIPE-DIAG/SEED] seed_poses=" << sfm_data_.GetPoses().size()
      << " of " << sfm_data_.GetViews().size() << " views";
    if (sfm_data_.GetPoses().empty())
      OPENMVG_LOG_ERROR << "[PIPE-DIAG/SEED] FATAL: scene initializer produced 0 poses.";
#endif

    if (!InitTracksAndLandmarks())
      return false;

#if OPENMVG_SFM_PIPELINE_DIAG
    {
      // Track-coverage diagnostic: how many input views are touched by tracks?
      std::set<uint32_t> covered;
      tracks::TracksUtilsMap::ImageIdInTracks(map_tracks_, covered);
      const size_t n_views = sfm_data_.GetViews().size();
      OPENMVG_LOG_INFO
        << "[PIPE-DIAG/TRACKS] tracks=" << map_tracks_.size()
        << " | views_in_tracks=" << covered.size()
        << " / " << n_views
        << " (" << (n_views ? (100.0 * covered.size() / n_views) : 0.0) << "%)";
      if (covered.size() < n_views)
      {
        size_t shown = 0;
        std::ostringstream os;
        os << "[PIPE-DIAG/TRACKS] views NOT touched by any track:";
        for (const auto & v : sfm_data_.GetViews())
        {
          if (covered.count(v.first) == 0)
          {
            if (shown++ < 30) os << " " << v.first;
            // Pre-record the failure so the post-mortem is complete.
            view_diag_records[v.first].last_reason =
              ResectionFailReason::NoVisibleTracks;
          }
        }
        if (shown > 30) os << " ... (+" << (shown - 30) << " more)";
        OPENMVG_LOG_WARNING << os.str();
      }
    }
#endif

    if (!sfm_data_.GetPoses().empty())
    {
      const bool bTriangulation = Triangulation();
      Save(sfm_data_, stlplus::create_filespec(sOut_directory_, "Initialization", ".ply"), ESfM_Data(ALL));
      // Fused angle + pixel filter: one structure traversal, one cache build.
      // Semantically equivalent to the legacy upstream pair at this site:
      //   RemoveOutliers_AngleError(sfm_data_, Square(2.0));        // 4 deg
      //   RemoveOutliers_PixelResidualError(sfm_data_, Square(4.0));// 16 px
      // These deliberately-loose thresholds are correct *here* because the
      // structure was just triangulated and hasn't been BA'd yet -- residuals
      // are coarse, and tighter thresholds would over-prune before BA gets a
      // chance to refine. The in-loop post-BA filter uses 2.0 / 4.0.
#if OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS // Relaxed angle threshold for foliage/narrow-baseline scenes
      RemoveOutliers_PixelAndAngleError(sfm_data_, Square(4.0), 3.0);
#else
      RemoveOutliers_PixelAndAngleError(sfm_data_, Square(4.0), Square(2.0));
#endif

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
  //     b. Perform Bundle Adjustment and cleaning
  //--
  IndexT resection_round = 0;

  // Incrementally estimate the pose of the cameras based on a confidence score.
  // The confidence score is based on the track_inlier_ratio.
  // First the camera with the most of 2D-3D overlap are added then we add
  // ones with lower confidence.
  const std::array<float, 2> track_inlier_ratios = {0.2, 0.0};

  // BA-skip threshold (intermediate BAs only).
  //   Minimum number of poses added since the last intermediate BA before
  //   we run BA again. Each BA pays a per-call setup overhead (~0.5-2.3s
  //   in the slow datasets) plus the minimizer cost; rounds that add 1-2
  //   views move geometry below the next-BA noise floor anyway. Skipping
  //   them lets the next round absorb the work cheaply, and the final
  //   STRICT BA still cleans up unconditionally. We always run BA on the
  //   final round of each track-inlier-ratio band so no skipped work
  //   leaks into the next phase or into the final BA.
  // Bumped from 2 -> 4 after [BA-PERF] showed each intermediate BA on the
  // slow dataset costs ~9-10s wall time (~4M residuals, 4-6 iters). The
  // end-of-band flush in the loop below still runs BA unconditionally on
  // the final pose of each track-inlier-ratio band, so accuracy is
  // protected; we only skip the truly small (1-3 new pose) calls.
  const IndexT kBAMinNewPoses = 4;
  IndexT poses_since_last_ba = 0;

  for (auto track_inlier_ratio = track_inlier_ratios.cbegin();
    track_inlier_ratio < track_inlier_ratios.cend(); ++track_inlier_ratio)
  {
    IndexT pose_before = sfm_data_.GetPoses().size();
    while (AddingMissingView(*track_inlier_ratio
#if OPENMVG_SFM_PIPELINE_DIAG
                              , &view_diag_records
#endif
                              ))
    {
      // Create new 3D points
      Triangulation();
      // Adjust the scene. Three presets are available:
      //   FAST     - aggressive speedup; safe only when no motion priors.
      //   BALANCED - moderate speedup; safe for motion-prior workflows.
      //   STRICT   - reference defaults; used for the final BA below.
      //
      // The fast preset's loose tolerances let LM exit before the motion
      // prior penalty fully balances reprojection, which on prior-bearing
      // scenes initialises the next resection from a slightly-wrong global
      // pose. The error compounds and the final strict BA can settle into
      // a parallel "ghost-layer" basin that no eject filter cleans up.
      // BALANCED gives prior workflows enough LM headroom to converge
      // while still skipping the last few strict-convergence iterations.
      const BAPreset intermediate_preset =
          !b_use_fast_intermediate_ba_ ? BAPreset::STRICT
        :  b_use_motion_prior_         ? BAPreset::BALANCED
        :                                BAPreset::FAST;

      // BA-skip gate. Track newly-added poses since the last successful
      // BA; run BA only when the accumulated delta crosses the threshold.
      // The Triangulation() above must always run (it's how new
      // landmarks are introduced for the next resection round); only the
      // BA itself is gated.
      const IndexT poses_now = sfm_data_.GetPoses().size();
      poses_since_last_ba += (poses_now > pose_before)
                           ? (poses_now - pose_before)
                           : IndexT(0);
      if (poses_since_last_ba >= kBAMinNewPoses)
      {
        BundleAdjustment(intermediate_preset);
        poses_since_last_ba = 0;
        // Remove unstable triangulations and camera poses (fused single pass).
        // Tied to the BA call: the residual statistics needed for outlier
        // pruning are only valid against the just-converged geometry.
#if OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS // Relaxed angle threshold for foliage/narrow-baseline scenes
        RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, 1.5);
#else
        RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, 2.0);
#endif
        eraseUnstablePosesAndObservations(sfm_data_);
      }

      // Per-resection-round PLY snapshot. Diagnostic-only output;
      // disabled by default because each save serialises the full
      // sfm_data scene to disk (5-15s of I/O on multi-round runs)
      // and does not feed any subsequent SfM2 step. Re-enable by
      // flipping the flag below or wiring it to a runtime knob if
      // you need per-round snapshots for debugging.
#if defined(OPENMVG_SFM2_SAVE_PER_RESECTION_PLY)
      std::ostringstream os;
      os << std::setw(8) << std::setfill('0') << resection_round << "_Resection";
      Save(sfm_data_, stlplus::create_filespec(sOut_directory_, os.str(), ".ply"), ESfM_Data(ALL));
#endif

      const IndexT pose_after = sfm_data_.GetPoses().size();
      const IndexT remaining =
        static_cast<IndexT>(sfm_data_.GetViews().size()) - pose_after;
#if OPENMVG_SFM_PIPELINE_DIAG
      DiagRound(resection_round, *track_inlier_ratio, pose_before, pose_after,
                static_cast<IndexT>(sfm_data_.GetLandmarks().size()), remaining);
#endif
      ++resection_round;

      // Stop if no cameras have been added
      // Note: some cameras could have been removed due to instable camera positions.
      if (pose_before >= pose_after)
        break;
      pose_before = sfm_data_.GetPoses().size();
      // Since we have augmented our set of poses we can reset our track inlier ratio iterator
      track_inlier_ratio = track_inlier_ratios.cbegin();
    }
    // End-of-band flush: if we exit the while loop with pending un-baked
    // poses (BA-skip threshold not yet hit), run BA once now so the next
    // ratio band's resection scan reads from converged geometry instead
    // of the stale post-Triangulation parameters. The final STRICT BA
    // still runs unconditionally below; this flush only matters when we
    // still have a looser-threshold band ahead of us.
    if (poses_since_last_ba > 0)
    {
      const BAPreset intermediate_preset =
          !b_use_fast_intermediate_ba_ ? BAPreset::STRICT
        :  b_use_motion_prior_         ? BAPreset::BALANCED
        :                                BAPreset::FAST;
      BundleAdjustment(intermediate_preset);
#if OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS // Relaxed angle threshold for foliage/narrow-baseline scenes
      RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, 1.5);
#else
      RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, 2.0);
#endif
      eraseUnstablePosesAndObservations(sfm_data_);
      poses_since_last_ba = 0;
    }
  }

  //--
  //- 3. Final bundle Adjustment (robust loop):
  //     BA -> eject bad poses (median-residual rule) -> if any ejected,
  //     filter outliers + re-BA. Otherwise we're done.
  //     Cost shape:
  //       - Clean runs: 1 BA + 1 cheap residual sweep (no sqrt) + early
  //         break -> ~same as the legacy single-BA path.
  //       - Bad runs: extra BAs only when poses actually need ejecting,
  //         and only those iterations pay for the O(K^2) angle filter.
  //     The filter is gated behind `ejected > 0` because its angle pass
  //     is O(track_length^2) per landmark over the entire post-BA scene,
  //     so running it unconditionally would force another full BA on
  //     every successful run for marginal benefit.
  //--
  {
    // Healthy reconstructions converge in iter 0 (no poses ejected ->
    // immediate break). The previous cap of 4 was a worst-case safety
    // net for pathological runs; in practice no observed dataset has
    // benefited from iters 2-3 on top of the rest of this pipeline.
    // Bit-identical to the cap=4 path on every run that breaks early.
    const int kMaxRobustIters = 2;
    BundleAdjustment(); // initial BA, same as legacy single-call path

    for (int iter = 0; iter < kMaxRobustIters; ++iter)
    {
      const IndexT prev_poses  = static_cast<IndexT>(sfm_data_.GetPoses().size());
      const IndexT prev_tracks = static_cast<IndexT>(sfm_data_.GetLandmarks().size());

      // EjectPosesByMedianResidual already calls
      // eraseObservationsWithMissingPoses internally for the poses it kills,
      // so we don't need a second eraseUnstablePosesAndObservations pass.
      const IndexT ejected_repro = EjectPosesByMedianResidual(sfm_data_,
                                                              /*k_factor=*/3.0,
                                                              /*abs_floor=*/2.0);
      // Pose-prior eject (EjectPosesByPriorResidual) is intentionally NOT
      // called here. With strict intermediate BA on prior workflows (see
      // resection-loop comment above), the ghost-layer/Z-drift mode that
      // motivated it does not occur, and any heuristic eject on a clean
      // scene only adds risk of culling good cameras. The function is kept
      // available in sfm_data_filters.* for diagnostic use.
      const IndexT ejected = ejected_repro;

      // Common clean-run path: nothing ejected -> done. No extra BA, no
      // expensive outlier filter.
      if (ejected == 0)
      {
        OPENMVG_LOG_INFO
          << "[ROBUST-BA iter " << iter << "]"
          << " ejected_poses=0 (converged) poses=" << prev_poses
          << " tracks=" << prev_tracks;
        break;
      }

      // A pose was ejected. Some observations near the ejected pose's
      // co-visibility may now be high-residual outliers in the surviving
      // scene; clean them before the corrective BA so the optimum isn't
      // pulled by stale features.
#if OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS // Relaxed angle threshold for foliage/narrow-baseline scenes
      const IndexT removed = RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, 1.5);
#else
      const IndexT removed = RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, 2.0);
#endif

      const IndexT cur_poses  = static_cast<IndexT>(sfm_data_.GetPoses().size());
      const IndexT cur_tracks = static_cast<IndexT>(sfm_data_.GetLandmarks().size());

      OPENMVG_LOG_INFO
        << "[ROBUST-BA iter " << iter << "]"
        << " ejected_poses=" << ejected
        << " removed=" << removed
        << " poses=" << prev_poses << "->" << cur_poses
        << " tracks=" << prev_tracks << "->" << cur_tracks;

      // Something changed -> re-BA so the optimum reflects the cleaned scene.
      BundleAdjustment();
    }
  }

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

#if OPENMVG_SFM_PIPELINE_DIAG
  // Checkpoint 3: post-mortem on every uncalibrated view.
  DiagOutputs(sfm_data_, view_diag_records);
#endif

  // [DIAG] Match-graph connectivity analysis. A healthy drone dataset has
  // hundreds of pairs per view and track median-length >= 4. If pairs/view
  // is below ~5 or median track length is 2, the upstream matching stage
  // (PairGenerator / ComputeMatches / GeometricFilter) is the bottleneck
  // - no amount of SfM tuning will recover the missing images.
  {
    const size_t n_views = sfm_data_.GetViews().size();
    const size_t n_pairs = matches_provider_->pairWise_matches_.size();
    const double pairs_per_view = n_views > 0 ? (2.0 * n_pairs) / n_views : 0.0;

    // Per-view pair count
    Hash_Map<IndexT, size_t> pairs_per_view_map;
    for (const auto & v : sfm_data_.GetViews())
      pairs_per_view_map[v.first] = 0;
    for (const auto & pw : matches_provider_->pairWise_matches_)
    {
      ++pairs_per_view_map[pw.first.first];
      ++pairs_per_view_map[pw.first.second];
    }
    std::vector<size_t> pv_counts;
    pv_counts.reserve(pairs_per_view_map.size());
    size_t isolated_views = 0; // views with 0 or 1 pair
    for (const auto & kv : pairs_per_view_map)
    {
      pv_counts.push_back(kv.second);
      if (kv.second <= 1) ++isolated_views;
    }
    std::sort(pv_counts.begin(), pv_counts.end());

    OPENMVG_LOG_INFO
      << "[MATCH-DIAG] n_views=" << n_views
      << " n_pairs=" << n_pairs
      << " avg_pairs/view=" << pairs_per_view
      << " | pair-count per view: min=" << (pv_counts.empty() ? 0 : pv_counts.front())
      << " median=" << (pv_counts.empty() ? 0 : pv_counts[pv_counts.size()/2])
      << " max=" << (pv_counts.empty() ? 0 : pv_counts.back())
      << " | isolated_views(<=1 pair)=" << isolated_views;

    if (pairs_per_view < 5.0)
    {
      OPENMVG_LOG_WARNING
        << "[MATCH-DIAG] CRITICAL: avg pairs/view=" << pairs_per_view
        << " is far below the ~15-50 typical for drone datasets."
        << " The match graph is too sparse for incremental SfM to"
        << " traverse the whole scene. Rerun matching with EXHAUSTIVE"
        << " pair generation or increase neighbor_count to >= 20.";
    }
  }

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
    // Reset the doomed-view filter set; populated alongside the landmarks
    // build below from the same per-observation walk we already do.
    views_with_tracks_.clear();
    views_with_tracks_.reserve(sfm_data_.GetViews().size());
    // Reset the per-view track-id cache. Built in track-id order via the
    // outer loop over `map_tracks_` (which is std::map<size_t, ...>, so
    // iteration order IS sorted track-id order). Each view's vector
    // therefore comes out naturally sorted with no explicit std::sort
    // pass needed -- a sorted track id is appended exactly once per
    // (view, track) pair, in increasing track-id order.
    view_track_ids_cache_.clear();

    // For every track add the observations:
    // - views and feature positions that see this landmark
    for ( const auto & iterT : map_tracks_ )
    {
      const IndexT track_id = static_cast<IndexT>(iterT.first);
      Observations obs;
      const size_t track_Length = iterT.second.size();
      for (const auto & track_ids : iterT.second) // {ViewId, FeatureId}
      {
        const auto & view_id = track_ids.first;
        const auto & feat_id = track_ids.second;
        const Vec2 x = features_provider_->feats_per_view[view_id][feat_id].coords().cast<double>();
        obs.insert({view_id, Observation(x, feat_id)});
        // Same per-observation walk; record that this view has at least
        // one track. unordered_set::insert is O(1) amortized and
        // idempotent so duplicate inserts are cheap.
        views_with_tracks_.insert(view_id);
        // Append this track id to the view's sorted-track-id list. Each
        // (track_id, view_id) pair appears exactly once in map_tracks_,
        // so no dedup needed; outer-loop iteration order over
        // map_tracks_ guarantees track_ids are appended in ascending
        // order, so per-view vectors are already sorted.
        view_track_ids_cache_[view_id].push_back(track_id);
      }
      landmarks_[iterT.first].obs = std::move(obs);
    }
  }

  // Initialize the shared track visibility helper
  shared_track_visibility_helper_.reset(new openMVG::tracks::SharedTrackVisibilityHelper(map_tracks_));

  // Reset the delta-cache state. Sized to max(view_id)+1 so that
  // resection_score_cache_[view_id] is a valid direct-indexed slot for
  // every view in sfm_data_; concurrent thread access in the OMP loop
  // touches only distinct slots and is data-race-free.
  IndexT max_view_id = 0;
  for (const auto & view_it : sfm_data_.GetViews())
    max_view_id = std::max(max_view_id, view_it.first);
  resection_score_cache_.assign(max_view_id + 1, ResectionScoreCache{});
  prev_reconstructed_track_ids_.clear();
  prev_view_with_no_pose_.clear();
  return map_tracks_.size() > 0;
}

bool SequentialSfMReconstructionEngine2::Triangulation()
{
  sfm_data_.structure = landmarks_;

  //--
  // Triangulation
  //--
  // Clean the structure:
  //  - keep observations that are linked to valid pose and intrinsic data.

  const double max_reprojection_error = 4.0;
  const IndexT min_required_inliers = 2;
  const IndexT min_sample_index = 2;
  eraseObservationsWithMissingPoses(sfm_data_, min_sample_index);
  SfM_Data_Structure_Computation_Robust triangulation_engine(
      max_reprojection_error,
      min_required_inliers,
      min_sample_index,
      triangulation_method_);

  triangulation_engine.triangulate(sfm_data_);

  return !sfm_data_.structure.empty();
}

bool SequentialSfMReconstructionEngine2::AddingMissingView
(
  const float & track_inlier_ratio
#if OPENMVG_SFM_PIPELINE_DIAG
  , Hash_Map<IndexT, ViewDiagRecord> * diag_records
#endif
)
{
  if (sfm_data_.GetLandmarks().empty())
    return false;

  // Collect the views that does not have any 3D pose (sorted vector for
  // cache-friendly OMP iteration; keep deterministic order).
  //
  // Output-preserving pre-filter: views that appear in zero entries of
  // `map_tracks_` cannot be resected -- inside the parallel loop below,
  // shared_track_visibility_helper_->GetTracksInImages would return an
  // empty set, view_tracks_ids would be empty, and the resection branch
  // would never be entered. Skipping them up front saves the
  // GetTracksInImages call + set_intersection per iteration of the
  // outer resection loop. The diagnostic accumulator is updated below
  // so per-view records remain identical to the unfiltered behaviour.
  std::vector<IndexT> view_with_no_pose;
  std::vector<IndexT> view_no_tracks;
  {
    view_with_no_pose.reserve(sfm_data_.GetViews().size());
    view_no_tracks.reserve(sfm_data_.GetViews().size());
    for (const auto & view_it : sfm_data_.GetViews())
    {
      const View * v = view_it.second.get();
      const IndexT id_pose = v->id_pose;
      if (sfm_data_.GetPoses().count(id_pose) != 0)
        continue;
      if (views_with_tracks_.count(view_it.first))
        view_with_no_pose.push_back(view_it.first);
      else
        view_no_tracks.push_back(view_it.first);
    }
    std::sort(view_with_no_pose.begin(), view_with_no_pose.end());
  }

#if OPENMVG_SFM_PIPELINE_DIAG
  // Mirror what the inner-loop NoVisibleTracks branch would write: bump
  // attempts and tag last_reason. The two max(..., 0) calls in the
  // original branch are no-ops on the default-zero record so we omit
  // them. Done outside the parallel region; no critical needed.
  if (diag_records)
  {
    for (IndexT id : view_no_tracks)
    {
      auto & rec = (*diag_records)[id];
      rec.last_reason = ResectionFailReason::NoVisibleTracks;
      ++rec.attempts;
    }
  }
#endif

  const IndexT pose_before = sfm_data_.GetPoses().size();

  // Get the track ids of the reconstructed landmarks.
  //
  // Buffer is reused across calls to avoid the per-call allocation of a
  // landmark-count-sized vector (5M+ entries on large scenes -- the
  // alloc + dealloc was a measurable per-AddingMissingView fixed cost).
  // `clear()` preserves capacity, so the `reserve()` below is a no-op
  // once the buffer has grown to the high-water mark.
  //
  // `thread_local` is defensive: AddingMissingView is called serially
  // from Process() (the OMP parallel-for lives *inside* this function),
  // so a plain `static` would also be correct. `thread_local` future-
  // proofs against any caller that might one day invoke this from
  // multiple threads.
  //
  // Output bit-identical to the previous lambda-IIFE form: same
  // transform, same sort, same final contents.
  thread_local static std::vector<IndexT> reconstructed_trackId_buf;
  reconstructed_trackId_buf.clear();
  reconstructed_trackId_buf.reserve(sfm_data_.GetLandmarks().size());
  std::transform(sfm_data_.GetLandmarks().cbegin(), sfm_data_.GetLandmarks().cend(),
    std::back_inserter(reconstructed_trackId_buf),
    stl::RetrieveKey());
  std::sort(reconstructed_trackId_buf.begin(), reconstructed_trackId_buf.end());
  const std::vector<IndexT> & reconstructed_trackId = reconstructed_trackId_buf;

  // === Resection-score delta cache ===
  //
  // The score for view V is `|view_tracks_ids(V) n reconstructed_trackId|`
  // and the corresponding ratio. view_tracks_ids(V) is invariant after
  // InitTracksAndLandmarks (map_tracks_ never changes during the
  // resection loop). reconstructed_trackId changes between successive
  // AddingMissingView() calls via Triangulation() and outlier-erase
  // passes. We diff the current and previous reconstructed-track lists
  // to obtain the (added, removed) delta of track ids; only views whose
  // visible tracks intersect the delta need to be re-scored. All others
  // hold a still-valid cached intersection -- output bit-identical to
  // the unfiltered scan, just with the redundant set_intersection work
  // skipped.
  //
  // Two cases require forced recompute regardless of the delta:
  //   1. First call: prev state is empty, so we have no cache. Treated
  //      naturally by marking all current no-pose views dirty.
  //   2. View V was resected in some past round and later kicked back
  //      to view_with_no_pose (eraseUnstablePosesAndObservations).
  //      V's slot may reflect a much older reconstructed-track set.
  //      Detected via prev_view_with_no_pose_: if V is in current
  //      view_with_no_pose but wasn't last call, it wasn't scored last
  //      call, so we cannot validate its cache from one round of delta.
  //
  // Fast paths to avoid building dirty_views when it can't help:
  //   * First call (no prev state): every view is forced dirty via
  //     `was_scored_last_round=false`; skip the diff entirely.
  //   * Massive delta: if |added|+|removed| >= |reconstructed|/2 the
  //     reconstructed set has materially turned over and most views
  //     will end up dirty anyway. Walking every delta track's view
  //     list would cost more than just letting per-view recompute run.
  std::unordered_set<IndexT> dirty_views;
  const bool first_call = prev_view_with_no_pose_.empty()
                        && prev_reconstructed_track_ids_.empty();
  bool force_all_dirty = first_call;

  std::vector<IndexT> added_tracks, removed_tracks;
  if (!force_all_dirty)
  {
    std::set_difference(
        reconstructed_trackId.cbegin(), reconstructed_trackId.cend(),
        prev_reconstructed_track_ids_.cbegin(), prev_reconstructed_track_ids_.cend(),
        std::back_inserter(added_tracks));
    std::set_difference(
        prev_reconstructed_track_ids_.cbegin(), prev_reconstructed_track_ids_.cend(),
        reconstructed_trackId.cbegin(), reconstructed_trackId.cend(),
        std::back_inserter(removed_tracks));

    const size_t delta_size = added_tracks.size() + removed_tracks.size();
    if (delta_size * 2 >= reconstructed_trackId.size())
    {
      force_all_dirty = true;  // cheaper to just recompute everyone
    }
    else
    {
      auto mark_views_for_track = [&](IndexT track_id)
      {
        auto it = map_tracks_.find(track_id);
        if (it == map_tracks_.end()) return;
        for (const auto & vf : it->second)  // {ViewId, FeatureId}
          dirty_views.insert(vf.first);
      };
      for (IndexT t : added_tracks)   mark_views_for_track(t);
      for (IndexT t : removed_tracks) mark_views_for_track(t);
    }
  }

  // Snapshot current no-pose set for next round's "re-entered" detection.
  std::unordered_set<IndexT> current_view_with_no_pose_set(
      view_with_no_pose.cbegin(), view_with_no_pose.cend());

  // List the view that have a sufficient 2D-3D coverage for robust pose estimation
  //
  // The previous form was `#pragma omp parallel` over a range-based for with
  // `#pragma omp single nowait` inside the loop body. That made every thread
  // walk every iteration of the C++ for loop, with `single` electing one
  // thread per iteration to actually run the body. Even with `nowait`, the
  // lock contention on `single` and the redundant iteration walk cost more
  // than they saved -- the loop was only marginally parallel.
  //
  // Switch to a real `parallel for` over an index. MSVC's OpenMP 2.0 needs a
  // signed int loop counter, hence the explicit cast. Iteration body is
  // unchanged. All shared-state writes inside (sfm_data_.poses,
  // sfm_data_.intrinsics, diag_records) are already wrapped in
  // `#pragma omp critical` -- semantics preserved. Iteration order across
  // threads was already nondeterministic under `single nowait`; that
  // doesn't change. `schedule(dynamic)` matches the previous behaviour
  // (each thread grabs the next available iteration when free) and copes
  // well with the very uneven per-iteration cost (most iterations fail the
  // ratio gate cheaply; a few do a full Localize+RefinePose).
  const int n_no_pose = static_cast<int>(view_with_no_pose.size());
#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (int idx = 0; idx < n_no_pose; ++idx)
  {
    const IndexT view_id = view_with_no_pose[idx];
    {
      // Look up the precomputed sorted track-id vector for this view.
      // Built once in InitTracksAndLandmarks; never changes during the
      // resection loop because `map_tracks_` is invariant after init.
      // The cache replaces the per-iteration GetTracksInImages +
      // GetTracksIdVector pair *for the gate computation*. We still
      // call GetTracksInImages below, but only when the ratio gate
      // passes -- which is the rare case (most no-pose views fail
      // the gate every outer iteration).
      static const std::vector<IndexT> kEmptyTrackIds;
      const auto cache_it = view_track_ids_cache_.find(view_id);
      const std::vector<IndexT>& view_tracks_ids =
          (cache_it != view_track_ids_cache_.end())
              ? cache_it->second
              : kEmptyTrackIds;

      // Delta-cache check. The slot is only safely re-usable if V was
      // scored last round AND no track in V's view_tracks_ids was
      // added/removed from reconstructed_trackId since.
      ResectionScoreCache & slot = resection_score_cache_[view_id];
      const bool was_scored_last_round =
          prev_view_with_no_pose_.count(view_id) > 0;
      const bool is_dirty =
          force_all_dirty
          || !was_scored_last_round
          || dirty_views.count(view_id) > 0;

      if (is_dirty)
      {
        // Recompute the intersection. Both inputs are sorted-ascending;
        // output is bit-identical to the previous std::set-based
        // implementation and to last round's cached value when the
        // delta-relevant track set is unchanged.
        slot.track_id_for_resection.clear();
        slot.track_id_for_resection.reserve(
            std::min(view_tracks_ids.size(), reconstructed_trackId.size()));
        std::set_intersection(
            view_tracks_ids.cbegin(), view_tracks_ids.cend(),
            reconstructed_trackId.cbegin(), reconstructed_trackId.cend(),
            std::back_inserter(slot.track_id_for_resection));
        slot.track_ratio = slot.track_id_for_resection.size()
                         / static_cast<float>(view_tracks_ids.size() + 1);
      }
      // Read-only references for the rest of the iteration.  The slot
      // holds the same bytes the original lambda would have produced.
      const std::vector<IndexT> & track_id_for_resection =
          slot.track_id_for_resection;
      const double track_ratio = slot.track_ratio;

      OPENMVG_LOG_INFO
        << "ViewId: " << view_id
        << "; #number of 2D-3D matches: " << track_id_for_resection.size()
        << "; " << track_ratio * 100 << " % of the view track coverage.";

#if OPENMVG_SFM_PIPELINE_DIAG
      // Pre-classify failure reason; will be overwritten on success.
      ResectionFailReason reason = ResectionFailReason::NoReconstructedTracks;
      if (view_tracks_ids.empty())
        reason = ResectionFailReason::NoVisibleTracks;
      else if (track_id_for_resection.empty())
        reason = ResectionFailReason::NoReconstructedTracks;
      else if (track_ratio <= track_inlier_ratio)
        reason = ResectionFailReason::LowTrackRatio;
#endif

      if (!track_id_for_resection.empty() && track_ratio > track_inlier_ratio)
      {
        // Gate passed: now we genuinely need the STLMAPTracks for the
        // feature-index lookup below. Deferring this call until here
        // (instead of doing it unconditionally at the top of the loop
        // body) avoids the expensive GetTracksInImages work on the
        // overwhelming majority of view-iteration evaluations that fail
        // the gate.
        openMVG::tracks::STLMAPTracks view_tracks;
        shared_track_visibility_helper_->GetTracksInImages({view_id}, view_tracks);

        // Get feat_id for the 2D/3D associations
        std::vector<IndexT> feature_id_for_resection;
        tracks::TracksUtilsMap::GetFeatIndexPerViewAndTrackId(
          view_tracks,
          track_id_for_resection,
          view_id,
          &feature_id_for_resection);

        // Localize the image inside the SfM reconstruction
        Image_Localizer_Match_Data resection_data;
        resection_data.pt2D.resize(2, track_id_for_resection.size());
        resection_data.pt3D.resize(3, track_id_for_resection.size());

        // Look if the intrinsic data is known or not
        const View * view = sfm_data_.GetViews().at(view_id).get();
        std::shared_ptr<cameras::IntrinsicBase> intrinsic;
        if (sfm_data_.GetIntrinsics().count(view->id_intrinsic))
        {
          intrinsic = sfm_data_.GetIntrinsics().at(view->id_intrinsic);
        }

        // Collect the feature observation
        Mat2X pt2D_original(2, track_id_for_resection.size());
        auto track_it = track_id_for_resection.cbegin();
        auto feat_it = feature_id_for_resection.cbegin();
        for (size_t cpt = 0; cpt < track_id_for_resection.size(); ++cpt, ++track_it, ++feat_it)
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

        geometry::Pose3 pose;
        const bool bResection = sfm::SfM_Localizer::Localize
        (
          intrinsic ? resection_method_ : resection::SolverType::DLT_6POINTS,
          {view->ui_width, view->ui_height},
          intrinsic ? intrinsic.get() : nullptr,
          resection_data,
          pose
        );
        resection_data.pt2D = std::move(pt2D_original); // restore original image domain points

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

#if OPENMVG_SFM_PIPELINE_DIAG
        if (!bResection)        reason = ResectionFailReason::ResectionFailed;
        else if (inlier_ratio <= 0.5f) reason = ResectionFailReason::LowInlierRatio;
#endif

        // Refine the pose of the found camera pose by using a BA and fix 3D points.
        if (bResection && inlier_ratio > 0.5)
        {
          // A valid pose has been found (try to refine it):
          // If no valid intrinsic as input:
          //  init a new one from the projection matrix decomposition
          // Else use the existing one and consider it as constant.
          if (!intrinsic)
          {
            // setup a default camera model from the found projection matrix
            Mat3 K, R;
            Vec3 t;
            KRt_From_P(resection_data.projection_matrix, &K, &R, &t);

            const double focal = (K(0,0) + K(1,1))/2.0;
            const Vec2 principal_point(K(0,2), K(1,2));

            // Create the new camera intrinsic
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
          const bool refined = intrinsic && sfm::SfM_Localizer::RefinePose(
              intrinsic.get(), pose,
              resection_data, b_refine_pose, b_refine_intrinsics);
          if (refined)
          {
            // - intrinsic parameters (if the view has no intrinsic group add a new one)
            if (sfm_data_.intrinsics.count(sfm_data_.views.at(view_id)->id_intrinsic) == 0)
            {
              // Since the view have not yet an intrinsic group before, create a new one
              IndexT new_intrinsic_id = 0;
              if (!sfm_data_.GetIntrinsics().empty())
              {
                // Since some intrinsic Id already exists,
                //  we have to create a new unique identifier following the existing one.
                // Single-pass max over the existing keys; output bit-identical
                // to the previous std::set + rbegin() construction (both yield
                // max(existing) + 1) without the per-element red-black-tree
                // insert. GetIntrinsics() is unordered, so a linear scan is
                // the natural way to find the max.
                IndexT max_existing = 0;
                for (const auto & intr_kv : sfm_data_.GetIntrinsics())
                  max_existing = std::max(max_existing, intr_kv.first);
                new_intrinsic_id = max_existing + 1;
              }
              #pragma omp critical
              {
                sfm_data_.views.at(view_id)->id_intrinsic = new_intrinsic_id;
                sfm_data_.intrinsics[new_intrinsic_id] = intrinsic;
              }
            }

            // Update the found camera pose
            #pragma omp critical
            sfm_data_.poses[view->id_pose] = pose;

#if OPENMVG_SFM_PIPELINE_DIAG
            reason = ResectionFailReason::Calibrated;
#endif
          }
#if OPENMVG_SFM_PIPELINE_DIAG
          else
          {
            reason = ResectionFailReason::RefinePoseFailed;
          }
#endif
        }

#if OPENMVG_SFM_PIPELINE_DIAG
        if (diag_records)
        {
          #pragma omp critical
          {
            auto & rec = (*diag_records)[view_id];
            rec.last_reason = reason;
            ++rec.attempts;
            rec.best_2d3d = std::max<uint32_t>(rec.best_2d3d,
              static_cast<uint32_t>(track_id_for_resection.size()));
            rec.best_track_ratio = std::max(rec.best_track_ratio,
              static_cast<float>(track_ratio));
            rec.last_inliers = static_cast<uint32_t>(resection_data.vec_inliers.size());
            rec.last_ratio = inlier_ratio;
          }
        }
#endif
      }
#if OPENMVG_SFM_PIPELINE_DIAG
      else if (diag_records)
      {
        #pragma omp critical
        {
          auto & rec = (*diag_records)[view_id];
          rec.last_reason = reason;
          ++rec.attempts;
          rec.best_2d3d = std::max<uint32_t>(rec.best_2d3d,
            static_cast<uint32_t>(track_id_for_resection.size()));
          rec.best_track_ratio = std::max(rec.best_track_ratio,
            static_cast<float>(track_ratio));
        }
      }
#endif
    }
  }

  const IndexT pose_after = sfm_data_.GetPoses().size();

  // Persist delta-cache state for the next AddingMissingView() call.
  // Done after the parallel loop so writes are race-free.
  prev_reconstructed_track_ids_ = reconstructed_trackId;
  prev_view_with_no_pose_ = std::move(current_view_with_no_pose_set);

  return (pose_after != pose_before);
}

bool SequentialSfMReconstructionEngine2::BundleAdjustment(BAPreset preset)
{
  Bundle_Adjustment_Ceres::BA_Ceres_options options;
  // Linear-solver selector:
  //   * < 100 poses                 -> DENSE_SCHUR (fastest at small scale).
  //   * 100..1500 poses (sparse lib)-> SPARSE_SCHUR + JACOBI preconditioner
  //                                    (direct factorisation of Schur comp.).
  //   * > 1500 poses (sparse lib)   -> ITERATIVE_SCHUR + SCHUR_JACOBI.
  //
  // [BA-PERF] Threshold bumped from 300 to 1500 after measuring on the
  // slow drone dataset (~436 poses, 5.7M residuals): ITERATIVE_SCHUR +
  // SCHUR_JACOBI took 22-36s on the final BA versus 11s for SPARSE_SCHUR
  // with SuiteSparse. CG inner-iteration count balloons at this scale
  // because the Schur complement is sparse-but-not-particularly-banded.
  // Direct factorisation wins until the dense Schur block can no longer
  // fit in cache (empirically ~1500-2000 poses). Quality identical in
  // both cases (same KKT system).
  const std::size_t n_poses = sfm_data_.GetPoses().size();
  const bool sparse_available =
       ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE)
    || ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE);

  if (n_poses > 1500 && sparse_available)
  {
    options.preconditioner_type_ = ceres::SCHUR_JACOBI;
    options.linear_solver_type_  = ceres::ITERATIVE_SCHUR;
  }
  else if (n_poses > 100 && sparse_available)
  {
    options.preconditioner_type_ = ceres::JACOBI;
    options.linear_solver_type_  = ceres::SPARSE_SCHUR;
  }
  else
  {
    options.linear_solver_type_  = ceres::DENSE_SCHUR;
  }

  // Preset for the intermediate BA. Solver / preconditioner / loss function
  // are unchanged -- only the stopping criteria and iteration caps move.
  // The final BA at the end of Process() runs with STRICT and cleans up
  // any slack left by intermediate BAs.
  //
  // All of these are pure Ceres stopping-criteria knobs; none change the
  // residual model, the parameter blocks, or the loss function.
  switch (preset)
  {
    case BAPreset::FAST:
      options.parameter_tolerance_              = 1e-6;   // strict: 1e-8
      options.gradient_tolerance_               = 1e-8;   // strict: 1e-10
      options.function_tolerance_               = 1e-4;   // strict: 1e-6
      options.max_num_iterations_               = 15;     // strict: 50
      options.max_num_consecutive_invalid_steps_= 2;      // strict: 5
      // Plateau early-stop: easy scenes settle in 4-6 iters; relative cost
      // change collapses below 5e-4 well before max_num_iterations on those
      // scenes, so we cut out without affecting hard scenes.
      options.plateau_relative_tolerance_       = 5e-4;
      options.plateau_min_iterations_           = 4;
      options.plateau_patience_                 = 2;
      // Inner iterations (Ceres' post-LM-step refinement subproblem) buy
      // accuracy at intermediate scale that the next intermediate BA or
      // the final STRICT BA would re-derive anyway. Disabling for FAST
      // saves ~6-10% of minimizer wall time per call without changing
      // the model being solved. Final BA keeps it on (STRICT default).
      options.use_inner_iterations_             = false;
      break;
    case BAPreset::BALANCED:
      // Tight enough for motion-prior LM convergence (the prior term needs
      // more iterations to balance reprojection than FAST allows), loose
      // enough to skip the last few strict-convergence iterations that
      // don't measurably change geometry. ~30% faster than STRICT on
      // prior-bearing scenes with no observed quality loss.
      options.parameter_tolerance_              = 1e-7;
      options.gradient_tolerance_               = 1e-9;
      options.function_tolerance_               = 1e-5;
      options.max_num_iterations_               = 30;
      options.max_num_consecutive_invalid_steps_= 4;
      // Plateau early-stop: tighter relative tol than FAST since BALANCED
      // runs on prior workflows where the last few iterations of
      // prior-vs-reprojection rebalancing matter. patience=2 makes the
      // detector robust to single-iteration relative-progress dips.
      options.plateau_relative_tolerance_       = 1e-4;
      options.plateau_min_iterations_           = 5;
      options.plateau_patience_                 = 2;
      // Same rationale as FAST: inner iterations are post-LM refinement
      // that intermediate BAs don't need to keep -- the final STRICT BA
      // refines the same parameters end-to-end. ~6-10% per-call savings.
      options.use_inner_iterations_             = false;
      break;
    case BAPreset::STRICT:
    default:
      // Use BA_Ceres_options defaults (matches reference engine).
      // plateau_relative_tolerance_ stays 0.0 -> callback not installed.
      break;
  }

  Bundle_Adjustment_Ceres bundle_adjustment_obj(options);
  // ---------------------------------------------------------------------
  // TEMPORARILY REVERTED for ghost-layer regression diagnosis.
  // ---------------------------------------------------------------------
  // Optimization "F" (freeze intrinsics on intermediate BAs) is suspected
  // of producing parallel ghost layers in DensifyPointCloud output on at
  // least one dataset whose input intrinsics (likely EXIF-derived) are
  // not accurate enough for the freeze-then-refine-once-at-the-end
  // strategy to land in the correct basin. With intrinsics frozen, every
  // intermediate BA fits poses+landmarks against a wrong focal length;
  // the final STRICT BA refines intrinsics from inside that basin and
  // can leave structure stranded at the old depth alongside structure at
  // the new depth -> ghost layers in the dense cloud.
  //
  // Toggle:
  //   kFreezeIntrinsicsOnIntermediateBA = true   (the speed optimization)
  //   kFreezeIntrinsicsOnIntermediateBA = false  (current; safe default
  //                                               while diagnosing)
  //
  // TODO: once the ghost-layer dataset is verified clean with the freeze
  // disabled, decide whether to:
  //   (a) re-enable the freeze unconditionally if the regression is
  //       traced to a different cause,
  //   (b) gate the freeze behind a "trusted intrinsics" engine flag so
  //       calibrated rigs get the speed win and EXIF-only datasets stay
  //       safe, or
  //   (c) leave it disabled permanently and absorb the ~5-15s cost.
  //
  // The original speed rationale (kept for reference):
  //   Intrinsic refinement on intermediate BAs is wasted work: focal
  //   length / principal-point / distortion drift well below 0.01 px
  //   after the first ~20 poses, while every intermediate BA pays full
  //   Jacobian-eval cost for the intrinsic block. The final STRICT BA at
  //   the end of Process() refines intrinsics end-to-end with the full
  //   reconstructed scene.
  static constexpr bool kFreezeIntrinsicsOnIntermediateBA =
      (OPENMVG_SFM2_FREEZE_INTRINSICS_ON_INTERMEDIATE_BA != 0);
  const cameras::Intrinsic_Parameter_Type intrinsic_refinement_for_ba =
      (preset == BAPreset::STRICT || !kFreezeIntrinsicsOnIntermediateBA)
          ? ReconstructionEngine::intrinsic_refinement_options_
          : cameras::Intrinsic_Parameter_Type::NONE;
  const Optimize_Options ba_refine_options
    ( intrinsic_refinement_for_ba,
      ReconstructionEngine::extrinsic_refinement_options_,
      Structure_Parameter_Type::ADJUST_ALL, // Adjust scene structure
      Control_Point_Parameter(),
      this->b_use_motion_prior_
    );
  return bundle_adjustment_obj.Adjust(sfm_data_, ba_refine_options);
}

} // namespace sfm
} // namespace openMVG
