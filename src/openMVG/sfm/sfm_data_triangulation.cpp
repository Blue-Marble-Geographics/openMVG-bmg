// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data_triangulation.hpp"

#include <deque>
#include <functional>

#include "openMVG/geometry/pose3.hpp"
#include "openMVG/multiview/triangulation_nview.hpp"
#include "openMVG/multiview/triangulation.hpp"
#include "openMVG/robust_estimation/rand_sampling.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_landmark.hpp"
#include "openMVG/system/loggerprogress.hpp"

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
  const std::size_t n = obs.size();
  if (n < 2)
    return false;

  const auto & views      = sfm_data.views;
  const auto & intrinsics = sfm_data.GetIntrinsics();
  const auto & poses      = sfm_data.GetPoses();
  const auto  intr_end    = intrinsics.end();
  const auto  pose_end    = poses.end();

  // Resolve one obs into (cam*, pose, bearing). Returns false if undefined.
  auto resolve = [&](const Observations::value_type & o,
                     const IntrinsicBase *& cam,
                     Pose3 & pose,
                     Vec3 & bearing) -> bool
  {
    const auto view_it = views.find(o.first);
    if (view_it == views.end()) return false;
    const View * view = view_it->second.get();
    if (!view ||
        view->id_intrinsic == UndefinedIndexT ||
        view->id_pose      == UndefinedIndexT)
      return false;
    const auto intr_it = intrinsics.find(view->id_intrinsic);
    if (intr_it == intr_end) return false;
    const auto pose_it = poses.find(view->id_pose);
    if (pose_it == pose_end) return false;
    cam     = intr_it->second.get();
    pose    = pose_it->second;
    bearing = (*cam)(cam->get_ud_pixel(o.second.x));
    return true;
  };

  // 2-view fast path: only touch the two observations we need.
  if (n == 2)
  {
    auto it = obs.begin();
    const IntrinsicBase * c0 = nullptr; Pose3 p0; Vec3 b0;
    if (!resolve(*it, c0, p0, b0)) return false;
    ++it;
    const IntrinsicBase * c1 = nullptr; Pose3 p1; Vec3 b1;
    if (!resolve(*it, c1, p1, b1)) return false;
    (void)c0; (void)c1;
    return Triangulate2View(
      p0.rotation(), p0.translation(), b0,
      p1.rotation(), p1.translation(), b1,
      X, etri_method);
  }

  // N > 2: single pass, directly into the buffers the solver consumes.
  std::vector<Vec3> bearing;
  std::vector<Mat34> poses_mat;
  bearing.reserve(n);
  poses_mat.reserve(n);
  for (const auto & obs_it : obs)
  {
    const IntrinsicBase * cam = nullptr; Pose3 pose; Vec3 b;
    if (!resolve(obs_it, cam, pose, b)) return false;
    (void)cam;
    bearing.emplace_back(b);
    poses_mat.emplace_back(pose.asMatrix());
  }
  const Eigen::Map<const Mat3X> bearing_matrix(bearing[0].data(), 3, bearing.size());
  Vec4 Xhomogeneous;
  if (TriangulateNViewAlgebraic(bearing_matrix, poses_mat, &Xhomogeneous))
  {
    X = Xhomogeneous.hnormalized();
    return true;
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
  const std::function<bool(
    const IntrinsicBase&,
    const Pose3&,
    const Vec2&,
    const Vec3&)> & predicate
)
{
  const auto & views      = sfm_data.views;
  const auto & intrinsics = sfm_data.GetIntrinsics();
  const auto & poses      = sfm_data.GetPoses();
  const auto  intr_end    = intrinsics.end();
  const auto  pose_end    = poses.end();

  bool visibility = false; // assume that no observation has been looked yet
  for (const auto & obs_it : obs)
  {
    const auto view_it = views.find(obs_it.first);
    if (view_it == views.end()) continue;
    const View * view = view_it->second.get();
    if (!view ||
        view->id_intrinsic == UndefinedIndexT ||
        view->id_pose      == UndefinedIndexT)
      continue;
    const auto intr_it = intrinsics.find(view->id_intrinsic);
    if (intr_it == intr_end) continue;
    const auto pose_it = poses.find(view->id_pose);
    if (pose_it == pose_end) continue;
    visibility = true; // at least an observation is evaluated
    if (!predicate(*intr_it->second.get(), pose_it->second, obs_it.second.x, X))
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
  std::deque<IndexT> rejectedId;
  std::unique_ptr<system::ProgressInterface> my_progress_bar;
  if (bConsole_verbose_)
    my_progress_bar.reset(
      new system::LoggerProgress(
        sfm_data.structure.size(),
        "Robust triangulation" ));
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
      Landmark landmark;
      if (robust_triangulation(sfm_data, tracks_it.second.obs, landmark))
      {
        tracks_it.second = landmark;
      }
      else
      {
        // Track must be deleted
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

  // Handle the case where all observations must be used.
  // (Fast path: no RANSAC, single triangulation + single predicate pass.)
  if (min_required_inliers_ == min_sample_index_ &&
      obs.size() == min_required_inliers_)
  {
    ResidualAndCheiralityPredicate predicate(dSquared_pixel_threshold);
    auto predicate_binding = std::bind(&ResidualAndCheiralityPredicate::predicate,
                                       predicate,
                                       std::placeholders::_1,
                                       std::placeholders::_2,
                                       std::placeholders::_3,
                                       std::placeholders::_4);
    Vec3 X;
    if (track_triangulation(sfm_data, obs, X, etri_method_) &&
        track_check_predicate(obs, sfm_data, X, predicate_binding))
    {
      landmark.X = X;
      landmark.obs = obs;
      return true;
    }
    return false;
  }

  // else we perform a robust estimation since
  //  there is more observations than the minimal number of required sample.

  //--
  // Per-observation cache: avoid repeated map lookups (views/intrinsics/poses)
  // and repeated bearing computation inside the RANSAC inner loop.
  //
  // Sorted by view_id so the RANSAC sample sequence (indexed positionally into
  // this vector with a fixed-seed PRNG) is stable across Observations backing
  // stores (std::unordered_map vs ankerl::unordered_dense vs std::map) and
  // across standard-library versions. Makes the RANSAC decisions deterministic
  // regardless of map iteration order.
  //
  // `bearing_ud` is the undistorted ray (*cam)(cam->get_ud_pixel(x)) used for
  // triangulation; `bearing` is the raw-pixel ray cam->oneBearing(x) used for
  // the cheirality test (matches the original cheirality_predicate semantics).
  struct ObsCacheEntry {
    IndexT view_id;
    const IntrinsicBase * cam;
    Pose3 pose;
    Vec3 bearing;     // cam(x) -- for cheirality
    Vec3 bearing_ud;  // cam(get_ud_pixel(x)) -- for triangulation
    Vec2 x;
  };
  std::vector<ObsCacheEntry> obs_cache;
  obs_cache.reserve(obs.size());
  {
    const auto & views      = sfm_data.views;
    const auto & intrinsics = sfm_data.GetIntrinsics();
    const auto & poses      = sfm_data.GetPoses();
    for (const auto & obs_it : obs)
    {
      const auto view_it = views.find(obs_it.first);
      if (view_it == views.end()) continue;
      const View * view = view_it->second.get();
      if (!view ||
          view->id_intrinsic == UndefinedIndexT ||
          view->id_pose      == UndefinedIndexT)
        continue;
      const auto intr_it = intrinsics.find(view->id_intrinsic);
      if (intr_it == intrinsics.end()) continue;
      const auto pose_it = poses.find(view->id_pose);
      if (pose_it == poses.end()) continue;
      const IntrinsicBase * cam = intr_it->second.get();
      obs_cache.push_back({
        obs_it.first,
        cam,
        pose_it->second,
        cam->oneBearing(obs_it.second.x),
        (*cam)(cam->get_ud_pixel(obs_it.second.x)),
        obs_it.second.x
      });
    }
  }
  std::sort(obs_cache.begin(), obs_cache.end(),
    [](const ObsCacheEntry & a, const ObsCacheEntry & b) {
      return a.view_id < b.view_id;
    });

  if (obs_cache.size() < min_required_inliers_ ||
      obs_cache.size() < min_sample_index_)
  {
    return false;
  }

  const IndexT nbIter = obs.size() * 2; // TODO: automatic computation of the number of iterations?

  //--
  // Hoisted scratch buffers: allocated once, cleared and reused every iter.
  std::vector<uint32_t> samples;
  samples.reserve(min_sample_index_);

  std::vector<IndexT> inlier_set;              // view_ids of inliers (current hypothesis)
  inlier_set.reserve(obs_cache.size());

  std::vector<IndexT> best_inlier_set;
  best_inlier_set.reserve(obs_cache.size());

  // Scratch buffers only needed for the N-view triangulation path.
  std::vector<Mat34> sample_poses_mat;
  std::vector<Vec3>  sample_bearings_ud;
  if (min_sample_index_ > 2)
  {
    sample_poses_mat.reserve(min_sample_index_);
    sample_bearings_ud.reserve(min_sample_index_);
  }

  Vec3 best_model = Vec3::Zero();
  double best_error = std::numeric_limits<double>::max();

  // Random number generation
  std::mt19937 random_generator(std::mt19937::default_seed);

  // - Ransac loop
  for (IndexT i = 0; i < nbIter; ++i)
  {
    samples.clear();
    robust::UniformSample(min_sample_index_,
                          static_cast<uint32_t>(obs_cache.size()),
                          random_generator, &samples);

    // --- Hypothesis generation from cached data (no map, no view/pose lookup).
    Vec3 X;
    bool tri_ok = false;
    if (min_sample_index_ == 2)
    {
      // Fast path: direct Triangulate2View on cached (pose, bearing_ud).
      const ObsCacheEntry & e0 = obs_cache[samples[0]];
      const ObsCacheEntry & e1 = obs_cache[samples[1]];
      tri_ok = Triangulate2View(
        e0.pose.rotation(), e0.pose.translation(), e0.bearing_ud,
        e1.pose.rotation(), e1.pose.translation(), e1.bearing_ud,
        X, etri_method_);
    }
    else
    {
      sample_poses_mat.clear();
      sample_bearings_ud.clear();
      for (const uint32_t idx : samples)
      {
        sample_poses_mat.emplace_back(obs_cache[idx].pose.asMatrix());
        sample_bearings_ud.emplace_back(obs_cache[idx].bearing_ud);
      }
      const Eigen::Map<const Mat3X> bearing_matrix(
        sample_bearings_ud[0].data(), 3, sample_bearings_ud.size());
      Vec4 Xh;
      if (TriangulateNViewAlgebraic(bearing_matrix, sample_poses_mat, &Xh))
      {
        X = Xh.hnormalized();
        tri_ok = true;
      }
    }
    if (!tri_ok)
      continue;

    // --- Validate the hypothesis on the minimal sample
    // (inline cheirality + residual, no std::function trampoline, no map lookup).
    bool minimal_ok = true;
    for (const uint32_t idx : samples)
    {
      const ObsCacheEntry & e = obs_cache[idx];
      const Vec3 pX = e.pose(X);
      if (e.bearing.dot(pX) <= 0.0)                                   // cheirality
      { minimal_ok = false; break; }
      if (e.cam->residual(pX, e.x).squaredNorm() >= dSquared_pixel_threshold)
      { minimal_ok = false; break; }
    }
    if (!minimal_ok)
      continue;

    // --- Inlier classification on the full observation set.
    inlier_set.clear();
    double current_error = 0.0;
    for (const ObsCacheEntry & entry : obs_cache)
    {
      const Vec3 pX = entry.pose(X);
      if (entry.bearing.dot(pX) <= 0.0)                               // cheirality
        continue;
      const double residual_sq = entry.cam->residual(pX, entry.x).squaredNorm();
      if (residual_sq < dSquared_pixel_threshold)
      {
        inlier_set.push_back(entry.view_id);
        current_error += residual_sq;
      }
      else
      {
        current_error += dSquared_pixel_threshold;
      }
    }
    // Update the best hypothesis.
    if (current_error < best_error &&
        inlier_set.size() >= min_required_inliers_)
    {
      best_model = X;
      best_inlier_set = inlier_set; // vector assignment; capacity is reused
      best_error = current_error;
    }
  }
  if (!best_inlier_set.empty() && best_inlier_set.size() >= min_required_inliers_)
  {
    // Update information (3D landmark position & valid observations)
    landmark.X = best_model;
    landmark.obs.reserve(best_inlier_set.size());
    for (const IndexT view_id : best_inlier_set)
    {
      landmark.obs[view_id] = obs.at(view_id);
    }
  }
  return !best_inlier_set.empty();
}

} // namespace sfm
} // namespace openMVG
