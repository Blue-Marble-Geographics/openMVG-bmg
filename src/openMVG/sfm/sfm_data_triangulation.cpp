// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data_triangulation.hpp"

#include <deque>
#include <functional>
#include <vector>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include "openMVG/geometry/pose3.hpp"
#include "openMVG/multiview/triangulation_nview.hpp"
#include "openMVG/multiview/triangulation.hpp"
#include "openMVG/robust_estimation/rand_sampling.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_landmark.hpp"
#include "openMVG/system/loggerprogress.hpp"

#include "ceres/internal/fixed_array.h" // Borrow this
using namespace ceres::internal;

namespace openMVG {
namespace sfm {

using namespace openMVG::geometry;
using namespace openMVG::cameras;

SfM_Data_Structure_Computation_Basis::SfM_Data_Structure_Computation_Basis
(
  bool bConsoleVerbose
)
  :bConsole_verbose_(bConsoleVerbose)
{
}

SfM_Data_Structure_Computation_Blind::SfM_Data_Structure_Computation_Blind
(
  bool bConsoleVerbose
)
  :SfM_Data_Structure_Computation_Basis(bConsoleVerbose)
{
}

/// Triangulate a given set of observations
bool track_triangulation
(
  const SfM_Data & sfm_data,
  const Observations & obs,
  Vec3 & X,
  const ETriangulationMethod & etri_method = ETriangulationMethod::DEFAULT
)
{
  if (obs.size() >= 2)
  {
#if 1
    size_t cnt = obs.size();
    FixedArray<Vec3, 16> bearing(cnt); // cnt is not the size of the array, but how many elements it can hold before it becomes dynamic.
    FixedArray<Mat34, 16> poses(cnt);
    FixedArray<const Pose3*, 16> poses_(cnt);

    int trueCnt = 0;

    auto observation = std::begin(obs);
    const auto& intrinsics = sfm_data.GetIntrinsics();
    const auto& allPoses = sfm_data.poses;
    for (size_t i = 0; i != cnt; ++i, ++observation)
    {
      const View* view = sfm_data.views.at(observation->first).get();
      if (!view) return false;
      if (view->id_intrinsic == UndefinedIndexT) return false;
      if (view->id_pose == UndefinedIndexT) return false;

      const auto* cam = intrinsics.at(view->id_intrinsic).get();
      auto it = intrinsics.find(view->id_intrinsic);
      if (it == intrinsics.end()) return false;
      auto itPose = allPoses.find(view->id_pose);
      if (itPose == allPoses.end()) return false;

      const auto tmp = cam->get_ud_pixel(observation->second.x);
      bearing[trueCnt] = cam->oneBearing(tmp);
      poses[trueCnt] = itPose->second.asMatrix();
      poses_[trueCnt] = &itPose->second;
      ++trueCnt;
    }
    if (trueCnt > 2)
    {
      const Eigen::Map<const Mat3X> bearing_matrix(bearing[0].data(), 3, trueCnt);
      Vec4 Xhomogeneous;
      if (TriangulateNViewAlgebraic2
      (
        bearing_matrix,
        poses.get(),
        &Xhomogeneous))
      {
        X = Xhomogeneous.hnormalized();
        return true;
      }
    }
    else
    {
      return Triangulate2View
      (
        poses_[0]->rotation(),
        poses_[0]->translation(),
        bearing[0],
        poses_[trueCnt - 1]->rotation(),
        poses_[trueCnt - 1]->translation(),
        bearing[trueCnt - 1],
        X,
        etri_method
      );
    }
#else
    std::vector<Vec3> bearing;
    std::vector<Mat34> poses;
    std::vector<Pose3> poses_;
    bearing.reserve(obs.size());
    poses.reserve(obs.size());
    for (const auto& observation : obs)
    {
      const View * view = sfm_data.views.at(observation.first).get();
      if (!sfm_data.IsPoseAndIntrinsicDefined(view))
        return false;
      const IntrinsicBase * cam = sfm_data.GetIntrinsics().at(view->id_intrinsic).get();
      const Pose3 pose = sfm_data.GetPoseOrDie(view);
      bearing.emplace_back((*cam)(cam->get_ud_pixel(observation.second.x)));
      poses.emplace_back(pose.asMatrix());
      poses_.emplace_back(pose);
    }
    if (bearing.size() > 2)
    {
      const Eigen::Map<const Mat3X> bearing_matrix(bearing[0].data(), 3, bearing.size());
      Vec4 Xhomogeneous;
      if (TriangulateNViewAlgebraic
      (
        bearing_matrix,
        poses,
        &Xhomogeneous))
      {
        X = Xhomogeneous.hnormalized();
        return true;
      }
    }
    else
    {
      return Triangulate2View
      (
        poses_.front().rotation(),
        poses_.front().translation(),
        bearing.front(),
        poses_.back().rotation(),
        poses_.back().translation(),
        bearing.back(),
        X,
        etri_method
      );
    }
#endif
  }
  return false;
}

// Test if a predicate is true for each observation
// i.e: predicate could be:
// - cheirality test (depth test): cheirality_predicate
// - cheirality and residual error: ResidualAndCheiralityPredicate::predicate
bool track_check_predicate
(
  const Observations & obs,
  const SfM_Data & sfm_data,
  const Vec3 & X,
  std::function<bool(
    const IntrinsicBase&,
    const Pose3&,
    const Vec2&,
    const Vec3&)> predicate
)
{
  bool visibility = false; // assume that no observation has been looked yet
  for (const auto & obs_it : obs)
  {
    const View * view = sfm_data.views.at(obs_it.first).get();
    if (!sfm_data.IsPoseAndIntrinsicDefined(view))
      continue;
    visibility = true; // at least an observation is evaluated
    const IntrinsicBase * cam = sfm_data.intrinsics.at(view->id_intrinsic).get();
    const Pose3 pose = sfm_data.GetPoseOrDie(view);
    if (!predicate(*cam, pose, obs_it.second.x, X))
      return false;
  }
  return visibility;
}

bool cheirality_predicate
(
  const IntrinsicBase& cam,
  const Pose3& pose,
  const Vec2& x,
  const Vec3& X
)
{
  return CheiralityTest(cam(x), pose, X);
}

struct ResidualAndCheiralityPredicate
{
  const double squared_pixel_threshold_;

  ResidualAndCheiralityPredicate(const double squared_pixel_threshold)
    :squared_pixel_threshold_(squared_pixel_threshold){}

  bool predicate
  (
    const IntrinsicBase& cam,
    const Pose3& pose,
    const Vec2& x,
    const Vec3& X
  )
  {
    const Vec2 residual = cam.residual(pose(X), x);
    return CheiralityTest(cam(x), pose, X) &&
           residual.squaredNorm() < squared_pixel_threshold_;
  }
};

void SfM_Data_Structure_Computation_Blind::triangulate
(
  SfM_Data & sfm_data
)
const
{
  std::deque<IndexT> rejectedId;
  std::unique_ptr<system::ProgressInterface> my_progress_bar;
  if (bConsole_verbose_)
    my_progress_bar.reset(
      new system::LoggerProgress(
        sfm_data.structure.size(),
        "Blind triangulation progress" ));
#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel
#endif
  for (auto& tracks_it :sfm_data.structure)
  {
#ifdef OPENMVG_USE_OPENMP
  #pragma omp single nowait
#endif
    {
      if (bConsole_verbose_)
      {
        ++(*my_progress_bar);
      }

      const Observations & obs = tracks_it.second.obs;
      bool bKeep = false;
      {
        // Generate the track 3D hypothesis
        Vec3 X;
        if (track_triangulation(sfm_data, obs, X))
        {
          // Keep the point only if it has a positive depth for all obs
          if (track_check_predicate(obs, sfm_data, X, cheirality_predicate))
          {
            tracks_it.second.X = X;
            bKeep = true;
          }
        }
      }
      if (!bKeep)
      {
#ifdef OPENMVG_USE_OPENMP
        #pragma omp critical
#endif
        rejectedId.push_front(tracks_it.first);
      }
    }
  }
  // Erase the unsuccessful triangulated tracks
  for (auto& it : rejectedId)
  {
    sfm_data.structure.erase(it);
  }
}

SfM_Data_Structure_Computation_Robust::SfM_Data_Structure_Computation_Robust
(
  const double max_reprojection_error,
  const IndexT min_required_inliers,
  const IndexT min_sample_index,
  const ETriangulationMethod etri_method,
  bool bConsoleVerbose
):
  SfM_Data_Structure_Computation_Basis(bConsoleVerbose),
  max_reprojection_error_(max_reprojection_error),
  min_required_inliers_(min_required_inliers),
  min_sample_index_(min_sample_index),
  etri_method_(etri_method)
{
}

void SfM_Data_Structure_Computation_Robust::triangulate
(
  SfM_Data & sfm_data
)
const
{
  robust_triangulation(sfm_data);
}

/// Robust triangulation of track data contained in the structure
/// All observations must have View with valid Intrinsic and Pose data
/// Invalid landmark are removed.
void SfM_Data_Structure_Computation_Robust::robust_triangulation
(
  SfM_Data & sfm_data
)
const
{
  std::unique_ptr<system::ProgressInterface> my_progress_bar;
  if (bConsole_verbose_)
    my_progress_bar.reset(
      new system::LoggerProgress(
        sfm_data.structure.size(),
        "Robust triangulation" ));

  auto& structure = sfm_data.structure;
  const int num_tracks = static_cast<int>(structure.size());

  // Snapshot keys into a vector for random-access parallel iteration
  std::vector<IndexT> track_keys;
  track_keys.reserve(num_tracks);
  for (const auto& entry : structure)
  {
    track_keys.push_back(entry.first);
  }

  // Per-track success flag; failed tracks will be erased after the loop
  std::vector<bool> succeeded(num_tracks, false);

#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 0; i < num_tracks; ++i)
  {
    if (bConsole_verbose_)
    {
      ++(*my_progress_bar);
    }

    const IndexT key = track_keys[i];
    Landmark& landmark = structure.at(key);
    const Observations obs_copy = landmark.obs; // snapshot for RANSAC input

    if (robust_triangulation(sfm_data, obs_copy, landmark))
    {
      succeeded[i] = true;
    }
  }

  // Single-threaded erase of rejected tracks
  for (int i = 0; i < num_tracks; ++i)
  {
    if (!succeeded[i])
    {
      structure.erase(track_keys[i]);
    }
  }
}

Observations ObservationsSampler
(
  const Observations & obs,
  const std::vector<std::uint32_t> & samples
)
{
  Observations sampled_obs;
  sampled_obs.reserve(samples.size());
  for (const auto& idx : samples)
  {
    Observations::const_iterator obs_it = obs.cbegin();
    std::advance(obs_it, idx);
    sampled_obs.insert(*obs_it);
  }
  return sampled_obs;
}

void ObservationsSampler
(
  Observations& sampled_obs,
  const Observations & obs,
  const std::uint32_t* samples,
  size_t cnt
)
{
  sampled_obs.clear();
  for (size_t i = 0; i != cnt; ++i)
  {
    const std::uint32_t idx = samples[i];
    Observations::const_iterator obs_it = obs.cbegin();
    std::advance(obs_it, idx);
    sampled_obs.insert(*obs_it);
  }
}
/// Robustly try to estimate the best 3D point using a ransac scheme
/// A point must be seen in at least min_required_inliers views
/// Return true for a successful triangulation
bool SfM_Data_Structure_Computation_Robust::robust_triangulation
(
  const SfM_Data & sfm_data,
  const Observations & obs,
  Landmark & landmark // X & valid observations
)
const
{
  if (obs.size() < min_required_inliers_ || obs.size() < min_sample_index_)
  {
    return false;
  }

  const double dSquared_pixel_threshold = Square(max_reprojection_error_);

  // Pre-cache per-observation lookups once
  struct ObsCache {
    IndexT obs_key;
    IndexT id_feat;
    const IntrinsicBase* cam;
    const Pose3* pose;
    Vec3 bearing;
    Mat34 pose_matrix;
    Vec2 x;
  };
  const auto& intrinsics = sfm_data.GetIntrinsics();
  const auto& poses = sfm_data.GetPoses();

  FixedArray<ObsCache, 32> obs_cache(obs.size());
  size_t obs_cache_size = 0;
  for (const auto& obs_it : obs)
  {
    const View* view = sfm_data.views.at(obs_it.first).get();
    if (!view) continue;
    if (view->id_intrinsic == UndefinedIndexT) continue;
    if (view->id_pose == UndefinedIndexT) continue;
    auto itCam = intrinsics.find(view->id_intrinsic);
    if (itCam == intrinsics.end()) continue;
    auto itPose = poses.find(view->id_pose);
    if (itPose == poses.end()) continue;

    const IntrinsicBase* cam = itCam->second.get();
    ObsCache& c = obs_cache[obs_cache_size++];
    c.obs_key = obs_it.first;
    c.id_feat = obs_it.second.id_feat;
    c.cam = cam;
    c.pose = &itPose->second;
    c.x = obs_it.second.x;
    const Vec2 ud = cam->get_ud_pixel(obs_it.second.x);
    c.bearing = cam->oneBearing(ud);
    c.pose_matrix = itPose->second.asMatrix();
  }

  if (obs_cache_size < min_required_inliers_ || obs_cache_size < min_sample_index_)
  {
    return false;
  }

  // Handle the case where all observations must be used
  if (min_required_inliers_ == min_sample_index_ &&
      obs_cache_size == min_required_inliers_)
  {
    Vec3 X;
    bool tri_ok = false;
    if (obs_cache_size > 2)
    {
      FixedArray<Vec3, 16> bearings(obs_cache_size);
      FixedArray<Mat34, 16> pm(obs_cache_size);
      for (size_t k = 0; k < obs_cache_size; ++k)
      {
        bearings[k] = obs_cache[k].bearing;
        pm[k] = obs_cache[k].pose_matrix;
      }
      const Eigen::Map<const Mat3X> bearing_matrix(bearings[0].data(), 3, obs_cache_size);
      Vec4 Xhomogeneous;
      tri_ok = TriangulateNViewAlgebraic2(bearing_matrix, pm.get(), &Xhomogeneous);
      if (tri_ok) X = Xhomogeneous.hnormalized();
    }
    else
    {
      tri_ok = Triangulate2View(
        obs_cache[0].pose->rotation(), obs_cache[0].pose->translation(), obs_cache[0].bearing,
        obs_cache[obs_cache_size-1].pose->rotation(), obs_cache[obs_cache_size-1].pose->translation(), obs_cache[obs_cache_size-1].bearing,
        X, etri_method_);
    }
    if (tri_ok)
    {
      for (size_t k = 0; k < obs_cache_size; ++k)
      {
        const ObsCache& c = obs_cache[k];
        if (!CheiralityTest(c.bearing, *c.pose, X))
          return false;
        const Vec2 residual = c.cam->residual((*c.pose)(X), c.x);
        if (residual.squaredNorm() >= dSquared_pixel_threshold)
          return false;
      }
      landmark.X = X;
      landmark.obs = obs;
      return true;
    }
    return false;
  }

  // Robust estimation: more observations than the minimal required sample.

  const IndexT nbIter = obs_cache_size * 2;

  // Ransac variables
  Vec3 best_model = Vec3::Zero();
  FixedArray<IndexT, 32> best_inlier_set(obs_cache_size);
  size_t best_inlier_set_size = 0;
  double best_error = std::numeric_limits<double>::max();

  std::mt19937 random_generator(std::mt19937::default_seed);

  for (IndexT i = 0; i < nbIter; ++i)
  {
    FixedArray<uint32_t, 16> sample_indices(obs_cache_size);
    const size_t numSamples = robust::UniformSample2(
      min_sample_index_, obs_cache_size, random_generator, sample_indices.get());

    Vec3 X;
    bool tri_ok = false;

    // Hypothesis generation directly from obs_cache
    if (numSamples > 2)
    {
      FixedArray<Vec3, 16> sample_bearings(numSamples);
      FixedArray<Mat34, 16> sample_poses(numSamples);
      for (size_t s = 0; s < numSamples; ++s)
      {
        const ObsCache& c = obs_cache[sample_indices[s]];
        sample_bearings[s] = c.bearing;
        sample_poses[s] = c.pose_matrix;
      }
      const Eigen::Map<const Mat3X> bearing_matrix(sample_bearings[0].data(), 3, numSamples);
      Vec4 Xhomogeneous;
      tri_ok = TriangulateNViewAlgebraic2(bearing_matrix, sample_poses.get(), &Xhomogeneous);
      if (tri_ok) X = Xhomogeneous.hnormalized();
    }
    else if (numSamples == 2)
    {
      const ObsCache& c0 = obs_cache[sample_indices[0]];
      const ObsCache& c1 = obs_cache[sample_indices[1]];
      tri_ok = Triangulate2View(
        c0.pose->rotation(), c0.pose->translation(), c0.bearing,
        c1.pose->rotation(), c1.pose->translation(), c1.bearing,
        X, etri_method_);
    }

    if (!tri_ok)
      continue;

    // Validate hypothesis on the sample
    bool sample_valid = true;
    for (size_t s = 0; s < numSamples; ++s)
    {
      const ObsCache& c = obs_cache[sample_indices[s]];
      if (!CheiralityTest(c.bearing, *c.pose, X))
      { sample_valid = false; break; }
      const Vec2 residual = c.cam->residual((*c.pose)(X), c.x);
      if (residual.squaredNorm() >= dSquared_pixel_threshold)
      { sample_valid = false; break; }
    }
    if (!sample_valid)
      continue;

    // Inlier/outlier classification — store cache indices, not obs_keys
    FixedArray<IndexT, 32> inlier_set(obs_cache_size);
    size_t inlier_cnt = 0;
    double current_error = 0.0;
    for (size_t j = 0; j < obs_cache_size; ++j)
    {
      const ObsCache& c = obs_cache[j];
      if (!CheiralityTest(c.bearing, *c.pose, X))
        continue;
      const double residual_sq = c.cam->residual((*c.pose)(X), c.x).squaredNorm();
      if (residual_sq < dSquared_pixel_threshold)
      {
        inlier_set[inlier_cnt++] = static_cast<IndexT>(j);
        current_error += residual_sq;
      }
      else
      {
        current_error += dSquared_pixel_threshold;
      }
    }

    if (current_error < best_error &&
      inlier_cnt >= min_required_inliers_)
    {
      best_model = X;
      std::copy(std::begin(inlier_set), std::begin(inlier_set)+inlier_cnt, std::begin(best_inlier_set));
      best_inlier_set_size = inlier_cnt;
      best_error = current_error;
    }
  }

  if (best_inlier_set_size >= min_required_inliers_)
  {
    landmark.X = best_model;
    landmark.obs.clear();
    landmark.obs.reserve(best_inlier_set_size);
    for (size_t i = 0; i != best_inlier_set_size; ++i)
    {
      const ObsCache& c = obs_cache[best_inlier_set[i]];
      landmark.obs[c.obs_key] = Observation(c.x, c.id_feat);
    }
  }
  return best_inlier_set_size;
}

} // namespace sfm
} // namespace openMVG
