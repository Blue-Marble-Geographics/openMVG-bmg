// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data_filters.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_view_priors.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/tracks/union_find.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

namespace openMVG {
namespace sfm {

/// List the view indexes that have valid camera intrinsic and pose.
std::set<IndexT> Get_Valid_Views
(
  const SfM_Data & sfm_data
)
{
  std::set<IndexT> valid_idx;
  for (const auto & view_it : sfm_data.GetViews())
  {
    const View * v = view_it.second.get();
    if (sfm_data.IsPoseAndIntrinsicDefined(v))
    {
      valid_idx.insert(v->id_view);
    }
  }
  return valid_idx;
}

// Legacy API: pixel-only filter. Implemented as a thin wrapper around the
// fused RemoveOutliers_PixelAndAngleError with a degenerate angle threshold
// (any non-negative angle survives) so callers in the v1 sequential, global,
// and stellar engines keep working without maintaining a duplicate code path.
IndexT RemoveOutliers_PixelResidualError
(
  SfM_Data & sfm_data,
  const double dThresholdPixel,
  const unsigned int minTrackLength
)
{
  IndexT removed_by_angle = 0;
  IndexT removed_by_pixel = 0;
  RemoveOutliers_PixelAndAngleError(
    sfm_data,
    dThresholdPixel,
    /*dMinAcceptedAngle=*/0.0, // disable angle test
    minTrackLength,
    &removed_by_angle,
    &removed_by_pixel);
  // Original return semantics: number of removed observations (pixel pass).
  return removed_by_pixel;
}

// Legacy API: angle-only filter. Implemented as a thin wrapper around the
// fused RemoveOutliers_PixelAndAngleError with a degenerate pixel threshold
// (squared-norm comparison vs +infinity is never true).
IndexT RemoveOutliers_AngleError
(
  SfM_Data & sfm_data,
  const double dMinAcceptedAngle
)
{
  IndexT removed_by_angle = 0;
  IndexT removed_by_pixel = 0;
  RemoveOutliers_PixelAndAngleError(
    sfm_data,
    /*dThresholdPixel=*/std::numeric_limits<double>::infinity(),
    dMinAcceptedAngle,
    /*minTrackLength=*/0, // legacy AngleError has no track-length prune
    &removed_by_angle,
    &removed_by_pixel);
  // Original return semantics: number of removed tracks (angle pass).
  return removed_by_angle;
}

// Fused pass: angle filter (track-level) + pixel residual filter (obs-level).
// Built to be semantically equivalent to:
//   RemoveOutliers_AngleError(...);
//   RemoveOutliers_PixelResidualError(...);
// but with a single per-view cache build and a single sfm_data.structure
// traversal. The angle decision is taken on the *original* obs set (matching
// the legacy ordering); pixel pruning runs only on tracks that survived.
IndexT RemoveOutliers_PixelAndAngleError
(
  SfM_Data & sfm_data,
  const double dThresholdPixel,
  const double dMinAcceptedAngle,
  const unsigned int minTrackLength,
  IndexT * out_removed_by_angle,
  IndexT * out_removed_by_pixel
)
{
  const double dThresholdPixelSq = dThresholdPixel * dThresholdPixel;

  // Shared per-view cache. Store a vector indexed by view_id when ids are
  // dense (typical OpenMVG case: ids are [0..N)), so per-obs lookup is a
  // single bounds check + array index instead of a hash probe. Fall back
  // to the Hash_Map path for sparse id ranges. The angle pass calls back
  // into this cache twice per (obs1, obs2) pair, so the win is real.
  struct ViewCache {
    geometry::Pose3 pose;
    const cameras::IntrinsicBase * intrinsic = nullptr; // nullptr == invalid
  };

  IndexT max_view_id = 0;
  bool any_view = false;
  for (const auto & view_it : sfm_data.views)
  {
    if (view_it.first > max_view_id) max_view_id = view_it.first;
    any_view = true;
  }
  const bool use_flat =
    any_view &&
    max_view_id != UndefinedIndexT &&
    static_cast<size_t>(max_view_id) < sfm_data.views.size() * 8 + 64;

  std::vector<ViewCache> view_cache_flat;
  Hash_Map<IndexT, ViewCache> view_cache_hash;
  if (use_flat)
  {
    view_cache_flat.assign(static_cast<size_t>(max_view_id) + 1, ViewCache{});
  }
  else
  {
    view_cache_hash.reserve(sfm_data.views.size());
  }

  for (const auto & view_it : sfm_data.views)
  {
    const View * v = view_it.second.get();
    if (!v || v->id_intrinsic == UndefinedIndexT || v->id_pose == UndefinedIndexT)
      continue;
    const auto pose_it = sfm_data.poses.find(v->id_pose);
    if (pose_it == sfm_data.poses.end()) continue;
    const auto intr_it = sfm_data.intrinsics.find(v->id_intrinsic);
    if (intr_it == sfm_data.intrinsics.end()) continue;
    if (use_flat)
      view_cache_flat[view_it.first] = { pose_it->second, intr_it->second.get() };
    else
      view_cache_hash[view_it.first] = { pose_it->second, intr_it->second.get() };
  }

  // O(1) lookup. Returns nullptr if the view has no valid (pose,intrinsic).
  //
  // Both caches are bound through const references so the lambda selects the
  // *const* overloads of operator[]/find. This matters because get_cache is
  // called concurrently from the parallel pass below: the standard only
  // guarantees const member functions of a shared container are free of data
  // races ([res.on.data.races]). The non-const find() happens to be
  // read-only in every implementation we build against, but relying on that
  // is exactly the kind of assumption that stops holding after a toolchain
  // bump. The caches are fully populated above and never written again.
  const std::vector<ViewCache> & view_cache_flat_ro = view_cache_flat;
  const Hash_Map<IndexT, ViewCache> & view_cache_hash_ro = view_cache_hash;
  auto get_cache = [&](IndexT view_id) -> const ViewCache * {
    if (use_flat)
    {
      if (view_id > max_view_id) return nullptr;
      const ViewCache & vc = view_cache_flat_ro[view_id];
      return vc.intrinsic ? &vc : nullptr;
    }
    const auto it = view_cache_hash_ro.find(view_id);
    return (it != view_cache_hash_ro.end()) ? &it->second : nullptr;
  };

  IndexT removed_tracks_by_angle = 0;
  IndexT removed_obs_by_pixel = 0;
  // Convert the angle threshold to cosine-domain once. cos is monotone
  // decreasing on [0, pi], so "angle_deg >= threshold_deg" becomes
  // "dot <= cos_threshold" for unit rays. dMinAcceptedAngle is in degrees.
  // Clamp the dot range to [-1+eps, 1-eps] like AngleBetweenRay does.
  const double cos_threshold = std::cos(D2R(dMinAcceptedAngle));

  // ---- Phase 0: snapshot the landmark nodes -----------------------------
  // The loop below used to be a single erase-while-iterating walk, which
  // forces it to be serial. It doesn't need to be: every track is fully
  // independent. The angle pass is read-only, and the pixel pass only
  // mutates that track's *own* Observations map -- a separate container per
  // Landmark. Nothing in either pass touches sfm_data.structure itself, so
  // the only step that has to be serialized is the final track erase.
  //
  // Landmarks is a std::unordered_map, so there is no contiguous value array
  // to index into for a `parallel for`. Node addresses are stable across
  // erase of *other* nodes, so a (key, Landmark*) snapshot is safe to hold
  // across the later erase pass. The snapshot walk is O(N) pointer chasing,
  // negligible next to the per-observation undistortion below.
  std::vector<std::pair<IndexT, Landmark *>> tracks;
  tracks.reserve(sfm_data.structure.size());
  for (auto & it : sfm_data.structure)
    tracks.emplace_back(it.first, &it.second);
  const int n_tracks = static_cast<int>(tracks.size());

  // Per-track verdict + per-track pixel-rejection count. Counts are summed
  // in the serial pass afterwards rather than via an OpenMP reduction:
  // MSVC is stuck on OpenMP 2.0 where reductions on unsigned/typedef'd
  // integer types are a portability trap, and this also keeps the totals
  // deterministic regardless of thread scheduling.
  enum : uint8_t { kKeep = 0, kEraseByAngle = 1, kEraseByPixel = 2 };
  std::vector<uint8_t>  verdict(n_tracks, kKeep);
  std::vector<uint32_t> pixel_removed(n_tracks, 0);

  // Threading gate -- two conditions must hold before we fork:
  //  1) Enough work to pay for fork/join. A stellar pod reconstruction has
  //     a few hundred landmarks; forking there costs more than the loop.
  //     A v2 sequential scene has 100k+, which is where this matters.
  //  2) We are not already inside a parallel region. This function IS
  //     reachable from one: sfm_stellar_engine.cpp's
  //     `#pragma omp parallel for` over stellar pods calls
  //     Stellar_Solver::Solve, which calls the RemoveOutliers_* wrappers.
  //     Every core is already busy there; nesting would oversubscribe, or
  //     (with nesting disabled, the default) pay fork overhead for a
  //     one-thread team. omp_in_parallel() is OpenMP 2.0, so it is
  //     available on MSVC.
#ifdef OPENMVG_USE_OPENMP
  static const int kParallelMinTracks = 2048;
  const bool use_threads = (n_tracks >= kParallelMinTracks) && !omp_in_parallel();
  // schedule(dynamic) with a coarse chunk: per-track cost is uneven (the
  // angle pass is O(K^2) for tracks that fail and typically O(K) for tracks
  // that pass early), and 1024-track chunks keep the scheduling overhead
  // down to ~N/1024 atomic grabs.
  #pragma omp parallel for schedule(dynamic, 1024) if(use_threads)
#endif
  for (int track_idx = 0; track_idx < n_tracks; ++track_idx)
  {
    Landmark & landmark = *tracks[track_idx].second;
    Observations & obs = landmark.obs;

    // Per-thread scratch for the angle pass: avoid reallocating on each of
    // the millions of landmarks. Each entry is a precomputed world-space
    // *unit* ray; pair angle then collapses to a single dot product (cosine
    // domain comparison, no acos in the inner loop). thread_local so the
    // capacity survives both the loop iterations and the call, and so each
    // OpenMP worker gets its own buffer.
    thread_local static std::vector<Vec3> ray_entries;

    // ---- Angle pass (track-level) -------------------------------------
    // Three key wins vs. the legacy nested loop:
    //   1) Undistort each obs's pixel ONCE up front (the inner loop used to
    //      recompute get_ud_pixel(obs2.x) for every (obs1, obs2) pair, i.e.
    //      O(K^2) undistortion calls per track; for Brown/Radial/Fisheye
    //      intrinsics that's an iterative root-find each time).
    //   2) Precompute the K world-space rays ONCE per track. The legacy
    //      AngleBetweenRay regenerates BOTH rays per pair (K(K-1) ray ops
    //      total) even though each ray only depends on a single obs. We do
    //      K ray ops total + cheap dot products in the inner loop.
    //   3) Compare in cosine domain. acos is the most expensive op in
    //      AngleBetweenRay; we skip it entirely by precomputing
    //      cos(threshold). "angle >= threshold" ⇔ "dot <= cos_threshold"
    //      since cos is monotone decreasing on [0, pi].
    //   Plus: early-exit as soon as ANY pair meets the threshold. The
    //   track is kept iff max_angle >= dMinAcceptedAngle (== min_dot
    //   <= cos_threshold). For well-conditioned scenes most tracks pass
    //   within the first few pairs, turning the worst-case O(K^2) into
    //   typical O(K).
    ray_entries.clear();
    ray_entries.reserve(obs.size());
    for (const auto & ob : obs)
    {
      const ViewCache * vc = get_cache(ob.first);
      if (!vc) continue;
      const Vec2 ud = vc->intrinsic->get_ud_pixel(ob.second.x);
      // ray in world space: R^T * bearing(ud), normalized. AngleBetweenRay
      // internally does the same; we hoist it so each obs pays once.
      // emplace_back, not push_back: the Eigen expression is evaluated
      // straight into the vector's storage. push_back of a braced RayEntry
      // wrapper materialized a temporary Vec3 first and then copied it in.
      ray_entries.emplace_back(
        (vc->pose.rotation().transpose() * vc->intrinsic->oneBearing(ud)).normalized());
    }

    bool angle_ok = false;
    {
      const std::size_t n = ray_entries.size();
      // Track the min observed dot (== max observed angle) across all
      // pairs we got to before any early-exit. Used to preserve the
      // legacy "n <= 1 with dMinAcceptedAngle == 0" keep-track behavior.
      double min_dot = 1.0; // cos(0) -- no pair has been seen yet.
      for (std::size_t i = 0; i < n && !angle_ok; ++i)
      {
        const Vec3 & ra = ray_entries[i];
        for (std::size_t j = i + 1; j < n; ++j)
        {
          const double dot = ra.dot(ray_entries[j]);
          if (dot <= cos_threshold)
          {
            angle_ok = true;
            break;
          }
          if (dot < min_dot) min_dot = dot;
        }
      }
      // Mirror the legacy "if max_angle >= dMinAcceptedAngle keep" check
      // in cosine domain. Real purpose: when dMinAcceptedAngle == 0
      // (legacy pixel-only wrapper, cos_threshold == 1.0) AND n <= 1
      // (no pair was checked, min_dot stays 1.0), this keeps the track.
      // For dMinAcceptedAngle > 0, cos_threshold < 1.0 < min_dot=1.0, so
      // the test fails as it should -- a track with no valid pair has
      // no parallax and is removed.
      if (!angle_ok && min_dot <= cos_threshold)
        angle_ok = true;
    }
    if (!angle_ok)
    {
      // Track dies on parallax. Matching the legacy ordering, its
      // observations are never seen by the pixel pass.
      verdict[track_idx] = kEraseByAngle;
      continue;
    }

    // ---- Pixel residual pass (obs-level) ------------------------------
    // Identical to RemoveOutliers_PixelResidualError's inner body. Safe to
    // run concurrently: `obs` is this landmark's own Observations map, not
    // shared with any other iteration.
    const Vec3 & X = landmark.X;
    uint32_t n_pixel_removed = 0;
    Observations::iterator itObs = obs.begin();
    while (itObs != obs.end())
    {
      const ViewCache * vc = get_cache(itObs->first);
      if (vc)
      {
        const Vec2 residual = vc->intrinsic->residual(
          vc->pose(X), itObs->second.x);
        if (residual.squaredNorm() > dThresholdPixelSq)
        {
          ++n_pixel_removed;
          itObs = obs.erase(itObs);
          continue;
        }
      }
      ++itObs;
    }
    pixel_removed[track_idx] = n_pixel_removed;
    if (obs.empty() || obs.size() < minTrackLength)
      verdict[track_idx] = kEraseByPixel;
  }

  // ---- Phase 2: apply the verdicts (serial) -----------------------------
  // The only step that mutates sfm_data.structure. Walking the snapshot in
  // order means the surviving container is identical to what the old
  // erase-while-iterating loop produced, and the totals are summed in a
  // fixed order so they don't depend on thread scheduling.
  for (int track_idx = 0; track_idx < n_tracks; ++track_idx)
  {
    removed_obs_by_pixel += pixel_removed[track_idx];
    switch (verdict[track_idx])
    {
      case kEraseByAngle:
        ++removed_tracks_by_angle;
        sfm_data.structure.erase(tracks[track_idx].first);
        break;
      case kEraseByPixel:
        // Legacy semantics: tracks dropped for falling under minTrackLength
        // are not counted in either total (their removed observations
        // already were).
        sfm_data.structure.erase(tracks[track_idx].first);
        break;
      default:
        break;
    }
  }

  if (out_removed_by_angle) *out_removed_by_angle = removed_tracks_by_angle;
  if (out_removed_by_pixel) *out_removed_by_pixel = removed_obs_by_pixel;
  return removed_tracks_by_angle + removed_obs_by_pixel;
}

// COLMAP-style "bad pose" ejection. After a BA, any pose whose median
// reprojection residual is much worse than the scene-wide median-of-medians
// is almost certainly a wrongly-resectioned camera or one whose intrinsic
// drifted. Removing it lets the next BA round re-optimize without its
// poison, which is the dominant cure for run-to-run "good vs unusable"
// reconstructions.
IndexT EjectPosesByMedianResidual
(
  SfM_Data & sfm_data,
  const double k_factor,
  const double abs_floor_pixels,
  const IndexT min_points_per_landmark
)
{
  if (sfm_data.poses.empty() || sfm_data.structure.empty())
    return 0;

  // Per-view (pose, intrinsic*) cache. View ids and pose ids in OpenMVG
  // are typically dense small integers ([0..N) per image); use the same
  // flat-vector trick as RemoveOutliers_PixelAndAngleError so the per-obs
  // lookup over sfm_data.structure (the hot loop) is an array index
  // instead of a hash probe. Fall back to Hash_Map for sparse id ranges.
  struct ViewCache {
    geometry::Pose3 pose;
    const cameras::IntrinsicBase * intrinsic = nullptr; // nullptr == invalid
    IndexT pose_id = UndefinedIndexT;
  };

  IndexT max_view_id = 0;
  IndexT max_pose_id = 0;
  bool any_view = false;
  for (const auto & view_it : sfm_data.views)
  {
    if (view_it.first > max_view_id) max_view_id = view_it.first;
    any_view = true;
  }
  for (const auto & pose_it : sfm_data.poses)
  {
    if (pose_it.first > max_pose_id) max_pose_id = pose_it.first;
  }
  const bool use_flat_view =
    any_view &&
    max_view_id != UndefinedIndexT &&
    static_cast<size_t>(max_view_id) < sfm_data.views.size() * 8 + 64;
  const bool use_flat_pose =
    max_pose_id != UndefinedIndexT &&
    static_cast<size_t>(max_pose_id) < sfm_data.poses.size() * 8 + 64;

  // Buffers below are thread_local-static so allocations amortize across
  // calls (this function runs once per robust-BA iteration, so the
  // capacity of the per-pose residual vectors is the dominant win).
  // ViewCache has no heap (Pose3 is a fixed-size Eigen aggregate), so a
  // straight assign() doesn't lose anything we wanted to keep.
  thread_local static std::vector<ViewCache> view_cache_flat;
  thread_local static Hash_Map<IndexT, ViewCache> view_cache_hash;
  if (use_flat_view)
  {
    view_cache_flat.assign(static_cast<size_t>(max_view_id) + 1, ViewCache{});
  }
  else
  {
    view_cache_hash.clear();
    view_cache_hash.reserve(sfm_data.views.size());
  }

  for (const auto & view_it : sfm_data.views)
  {
    const View * v = view_it.second.get();
    if (!v || v->id_intrinsic == UndefinedIndexT || v->id_pose == UndefinedIndexT)
      continue;
    const auto pose_it = sfm_data.poses.find(v->id_pose);
    if (pose_it == sfm_data.poses.end()) continue;
    const auto intr_it = sfm_data.intrinsics.find(v->id_intrinsic);
    if (intr_it == sfm_data.intrinsics.end()) continue;
    if (use_flat_view)
      view_cache_flat[view_it.first] = { pose_it->second, intr_it->second.get(), v->id_pose };
    else
      view_cache_hash[view_it.first] = { pose_it->second, intr_it->second.get(), v->id_pose };
  }

  auto get_cache = [&](IndexT view_id) -> const ViewCache * {
    if (use_flat_view)
    {
      if (view_id > max_view_id) return nullptr;
      const ViewCache & vc = view_cache_flat[view_id];
      return vc.intrinsic ? &vc : nullptr;
    }
    const auto it = view_cache_hash.find(view_id);
    return (it != view_cache_hash.end()) ? &it->second : nullptr;
  };

  // Accumulate squared residuals per pose_id. We compare squared values all
  // the way through (median is monotonic under squaring for non-negatives,
  // so median(r^2) == (median(r))^2). This avoids one sqrt per observation
  // -- this loop runs over every obs in sfm_data.structure.
  //
  // thread_local-static so the per-pose inner vectors retain their
  // capacity across calls. We deliberately do NOT use assign(N+1, {})
  // -- that would destroy every inner vector and throw away exactly the
  // allocations we want to amortize. Pattern: grow outer if needed, then
  // clear() each inner in place (size->0, capacity preserved).
  thread_local static std::vector<std::vector<double>> residuals_sq_per_pose_flat;
  thread_local static Hash_Map<IndexT, std::vector<double>> residuals_sq_per_pose_hash;
  if (use_flat_pose)
  {
    if (residuals_sq_per_pose_flat.size() < static_cast<size_t>(max_pose_id) + 1)
      residuals_sq_per_pose_flat.resize(static_cast<size_t>(max_pose_id) + 1);
    for (auto & v : residuals_sq_per_pose_flat) v.clear();
  }
  else
  {
    // Keep inner storage; the entry set may shift across calls but for
    // overlapping pose ids we recycle the existing buffer.
    for (auto & kv : residuals_sq_per_pose_hash) kv.second.clear();
    residuals_sq_per_pose_hash.reserve(sfm_data.poses.size());
  }

  auto push_residual = [&](IndexT pose_id, double r_sq) {
    if (use_flat_pose)
      residuals_sq_per_pose_flat[pose_id].push_back(r_sq);
    else
      residuals_sq_per_pose_hash[pose_id].push_back(r_sq);
  };

  for (const auto & landmark_it : sfm_data.structure)
  {
    const Vec3 & X = landmark_it.second.X;
    for (const auto & obs_it : landmark_it.second.obs)
    {
      const ViewCache * vc = get_cache(obs_it.first);
      if (!vc) continue;
      const double r_sq = vc->intrinsic->residual(vc->pose(X), obs_it.second.x).squaredNorm();
      push_residual(vc->pose_id, r_sq);
    }
  }

  // Per-pose median (of squared residuals). Iterate over actual poses so
  // we don't waste work on empty flat slots.
  thread_local static Hash_Map<IndexT, double> median_sq_per_pose;
  median_sq_per_pose.clear();
  median_sq_per_pose.reserve(sfm_data.poses.size());
  thread_local static std::vector<double> medians_sq;
  medians_sq.clear();
  medians_sq.reserve(sfm_data.poses.size());
  auto consume_residuals = [&](IndexT pose_id, std::vector<double> & v) {
    if (v.empty()) return;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    const double m_sq = v[v.size() / 2];
    median_sq_per_pose[pose_id] = m_sq;
    medians_sq.push_back(m_sq);
  };
  if (use_flat_pose)
  {
    for (const auto & pose_it : sfm_data.poses)
      consume_residuals(pose_it.first, residuals_sq_per_pose_flat[pose_it.first]);
  }
  else
  {
    for (auto & kv : residuals_sq_per_pose_hash)
      consume_residuals(kv.first, kv.second);
  }
  if (medians_sq.empty())
    return 0;

  // Global median-of-medians (robust scale estimator), still squared.
  std::nth_element(medians_sq.begin(), medians_sq.begin() + medians_sq.size() / 2, medians_sq.end());
  const double global_median_sq = medians_sq[medians_sq.size() / 2];

  // Threshold in squared space. Original (linear-space) test was:
  //   r > max(k_factor * global_median, abs_floor_pixels)
  // Squaring both sides (both non-negative) and using
  //   median(r^2) == (median(r))^2  =>  k^2 * median(r^2) == (k * median(r))^2
  // gives the equivalent test:
  //   r^2 > max(k_factor^2 * global_median_sq, abs_floor_pixels^2)
  const double threshold_sq = std::max(
    k_factor * k_factor * global_median_sq,
    abs_floor_pixels * abs_floor_pixels);

  // Collect bad pose ids.
  thread_local static std::vector<IndexT> bad_poses;
  bad_poses.clear();
  bad_poses.reserve(median_sq_per_pose.size());
  for (const auto & kv : median_sq_per_pose)
  {
    if (kv.second > threshold_sq)
      bad_poses.push_back(kv.first);
  }
  if (bad_poses.empty())
    return 0;

  // Erase the offending poses; observations referencing them become
  // orphans and are cleaned by eraseObservationsWithMissingPoses.
  for (const IndexT pose_id : bad_poses)
    sfm_data.poses.erase(pose_id);

  eraseObservationsWithMissingPoses(sfm_data, min_points_per_landmark);

  return static_cast<IndexT>(bad_poses.size());
}

// Pose-prior outlier ejection. With GPS / motion priors active, a pose can
// have a clean reprojection median yet sit far from its prior center -- the
// optimizer locked onto local features for that camera but the global
// position drifted. EjectPosesByMedianResidual won't catch these; this
// helper does.
//
// Threshold rule (MAD-based for bimodal robustness):
//     d_i = || pose_i.center - prior_i.pose_center ||
//     med = median(d_i)
//     mad = median( | d_i - med | )                    -- spread of good cluster
//     thresh = max( med + k_factor * 1.4826 * mad, abs_floor_units )
// MAD is preferred over k * median because a coherent drifted cohort
// (the "ghost-layer" failure mode) shifts the median upward so much that
// k * median accepts the drift. MAD reflects the spread of the inlier
// cluster only and stays small even when up to ~50% of poses are biased
// in the same direction.
//
// Safety cap: never eject more than max_eject_fraction of available poses
// in a single call (default 15%). If more would be flagged, only the
// worst max-cap are ejected and a warning is logged. Prevents cascade
// blowups when the threshold is too tight or the data is genuinely
// unprior-able.
IndexT EjectPosesByPriorResidual
(
  SfM_Data & sfm_data,
  const double k_factor,
  const double abs_floor_units,
  const IndexT min_points_per_landmark
)
{
  if (sfm_data.poses.empty())
    return 0;

  // pose_id -> distance between optimized center and prior center.
  // We work in distance (not squared) so the MAD computation is meaningful.
  // One entry per pose (first ViewPriors that references it wins).
  Hash_Map<IndexT, double> dist_per_pose;
  dist_per_pose.reserve(sfm_data.poses.size());
  for (const auto & view_it : sfm_data.views)
  {
    const ViewPriors * prior =
      dynamic_cast<const ViewPriors *>(view_it.second.get());
    if (!prior || !prior->b_use_pose_center_)
      continue;
    if (prior->id_pose == UndefinedIndexT) continue;
    const auto pose_it = sfm_data.poses.find(prior->id_pose);
    if (pose_it == sfm_data.poses.end()) continue;
    if (dist_per_pose.count(prior->id_pose)) continue;
    const double d =
      (pose_it->second.center() - prior->pose_center_).norm();
    dist_per_pose[prior->id_pose] = d;
  }

  if (dist_per_pose.size() < 6) // not enough samples for robust med + MAD
    return 0;

  // --- median of distances
  std::vector<double> dists;
  dists.reserve(dist_per_pose.size());
  for (const auto & kv : dist_per_pose)
    dists.push_back(kv.second);
  std::nth_element(dists.begin(),
                   dists.begin() + dists.size() / 2,
                   dists.end());
  const double med = dists[dists.size() / 2];

  // --- MAD: median of |d_i - med|. 1.4826 is the consistency factor that
  // makes 1.4826*MAD an unbiased estimator of stddev for Gaussian data.
  std::vector<double> abs_dev;
  abs_dev.reserve(dist_per_pose.size());
  for (const auto & kv : dist_per_pose)
    abs_dev.push_back(std::abs(kv.second - med));
  std::nth_element(abs_dev.begin(),
                   abs_dev.begin() + abs_dev.size() / 2,
                   abs_dev.end());
  const double mad = abs_dev[abs_dev.size() / 2];
  const double sigma = 1.4826 * mad;

  const double threshold = std::max(med + k_factor * sigma, abs_floor_units);

  // Collect (distance, pose_id) for everything above threshold; we may need
  // to sort by severity if the safety cap kicks in.
  std::vector<std::pair<double, IndexT>> flagged;
  flagged.reserve(dist_per_pose.size());
  for (const auto & kv : dist_per_pose)
  {
    if (kv.second > threshold)
      flagged.emplace_back(kv.second, kv.first);
  }
  if (flagged.empty())
    return 0;

  // Safety cap: never eject more than 15% of priored poses in one call.
  // Cascade-of-ejection across BA iterations is the failure mode this
  // guards against.
  const std::size_t max_eject =
    std::max<std::size_t>(1, dist_per_pose.size() * 15 / 100);
  if (flagged.size() > max_eject)
  {
    // Keep only the worst max_eject by distance (descending).
    std::partial_sort(
      flagged.begin(),
      flagged.begin() + max_eject,
      flagged.end(),
      [](const std::pair<double, IndexT> & a,
         const std::pair<double, IndexT> & b) { return a.first > b.first; });
    OPENMVG_LOG_WARNING
      << "[EjectPosesByPriorResidual] flagged=" << flagged.size()
      << " > cap=" << max_eject
      << " (med=" << med << " sigma=" << sigma
      << " thresh=" << threshold << "); ejecting only the worst.";
    flagged.resize(max_eject);
  }

  for (const auto & f : flagged)
    sfm_data.poses.erase(f.second);

  eraseObservationsWithMissingPoses(sfm_data, min_points_per_landmark);

  return static_cast<IndexT>(flagged.size());
}

bool eraseMissingPoses
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_pose
)
{
  IndexT removed_elements = 0;
  const Landmarks & landmarks = sfm_data.structure;

  // Both lookups below are hit once per observation in sfm_data.structure
  // (potentially millions of probes), so use the same flat-vector trick as
  // RemoveOutliers_PixelAndAngleError / EjectPosesByMedianResidual: array
  // index instead of hash probe when ids are dense, fall back to Hash_Map
  // for sparse id ranges.

  // viewId -> poseId
  IndexT max_view_id = 0;
  bool any_view = false;
  for (const auto & view_it : sfm_data.views)
  {
    if (view_it.first > max_view_id) max_view_id = view_it.first;
    any_view = true;
  }
  const bool use_flat_view =
    any_view &&
    max_view_id != UndefinedIndexT &&
    static_cast<size_t>(max_view_id) < sfm_data.views.size() * 8 + 64;

  std::vector<IndexT> view_to_pose_flat;
  Hash_Map<IndexT, IndexT> view_to_pose_hash;
  if (use_flat_view)
    view_to_pose_flat.assign(static_cast<size_t>(max_view_id) + 1, UndefinedIndexT);
  else
    view_to_pose_hash.reserve(sfm_data.views.size());
  for (const auto & view_it : sfm_data.views)
  {
    if (use_flat_view)
      view_to_pose_flat[view_it.first] = view_it.second->id_pose;
    else
      view_to_pose_hash[view_it.first] = view_it.second->id_pose;
  }

  // poseId -> observation count. Init to 0 for every existing pose so we
  // can detect non-referenced poses (count remains 0 -> removed).
  IndexT max_pose_id = 0;
  for (const auto & pose_it : sfm_data.GetPoses())
  {
    if (pose_it.first > max_pose_id) max_pose_id = pose_it.first;
  }
  const bool use_flat_pose =
    !sfm_data.poses.empty() &&
    max_pose_id != UndefinedIndexT &&
    static_cast<size_t>(max_pose_id) < sfm_data.poses.size() * 8 + 64;

  // Sentinel = std::numeric_limits<IndexT>::max() means "this slot is not
  // a real pose"; real poses are initialized to 0 below.
  static constexpr IndexT kNotAPose = std::numeric_limits<IndexT>::max();
  std::vector<IndexT> count_flat;
  Hash_Map<IndexT, IndexT> count_hash;
  if (use_flat_pose)
  {
    count_flat.assign(static_cast<size_t>(max_pose_id) + 1, kNotAPose);
    for (const auto & pose_it : sfm_data.GetPoses())
      count_flat[pose_it.first] = 0;
  }
  else
  {
    count_hash.reserve(sfm_data.poses.size());
    for (const auto & pose_it : sfm_data.GetPoses())
      count_hash[pose_it.first] = 0;
  }

  // Count occurrence of the poses in the Landmark observations.
  for (const auto & lanmark_it : landmarks)
  {
    const Observations & obs = lanmark_it.second.obs;
    for (const auto & obs_it : obs)
    {
      IndexT pose_id;
      if (use_flat_view)
      {
        if (obs_it.first > max_view_id) continue;
        pose_id = view_to_pose_flat[obs_it.first];
        if (pose_id == UndefinedIndexT) continue;
      }
      else
      {
        const auto it = view_to_pose_hash.find(obs_it.first);
        if (it == view_to_pose_hash.end()) continue;
        pose_id = it->second;
      }
      if (use_flat_pose)
      {
        if (pose_id > max_pose_id) continue;
        IndexT & c = count_flat[pose_id];
        if (c != kNotAPose) ++c;
      }
      else
      {
        const auto it = count_hash.find(pose_id);
        if (it != count_hash.end()) ++it->second;
      }
    }
  }
  // If usage count is smaller than the threshold, remove the Pose. Iterate
  // poses (not the count container) so the flat path skips empty slots.
  if (use_flat_pose)
  {
    // Snapshot ids first: we mutate sfm_data.poses inside the loop.
    std::vector<IndexT> pose_ids;
    pose_ids.reserve(sfm_data.poses.size());
    for (const auto & pose_it : sfm_data.GetPoses())
      pose_ids.push_back(pose_it.first);
    for (const IndexT pid : pose_ids)
    {
      if (count_flat[pid] < min_points_per_pose)
      {
        sfm_data.poses.erase(pid);
        ++removed_elements;
      }
    }
  }
  else
  {
    for (const auto & it : count_hash)
    {
      if (it.second < min_points_per_pose)
      {
        sfm_data.poses.erase(it.first);
        ++removed_elements;
      }
    }
  }
  return removed_elements > 0;
}

bool eraseObservationsWithMissingPoses
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_landmark
)
{
  IndexT removed_elements = 0;

  // Build a presence lookup over view ids whose pose exists. View ids in
  // OpenMVG are typically dense small integers ([0..N)), so a flat
  // std::vector<uint8_t> indexed by view id gives O(1) presence checks
  // with much better cache behavior than std::binary_search on a sorted
  // vector. Fall back to the sorted-vector path if the id range turns out
  // to be too sparse (heuristic: max_id > 8 * count + 64 so we don't
  // allocate a 100MB bitmap for a handful of ids).
  std::vector<IndexT> valid_view_ids;
  valid_view_ids.reserve(sfm_data.views.size());
  IndexT max_view_id = 0;
  for (const auto & view_it : sfm_data.views)
  {
    if (sfm_data.poses.count(view_it.second->id_pose))
    {
      valid_view_ids.push_back(view_it.first);
      if (view_it.first > max_view_id) max_view_id = view_it.first;
    }
  }

  const bool use_presence_vector =
    !valid_view_ids.empty() &&
    max_view_id != UndefinedIndexT &&
    static_cast<size_t>(max_view_id) < valid_view_ids.size() * 8 + 64;

  std::vector<uint8_t> view_present;
  if (use_presence_vector)
  {
    view_present.assign(static_cast<size_t>(max_view_id) + 1, 0);
    for (const IndexT id : valid_view_ids) view_present[id] = 1;
  }
  else
  {
    std::sort(valid_view_ids.begin(), valid_view_ids.end());
  }

  auto is_valid = [&](IndexT view_id) -> bool {
    if (use_presence_vector)
    {
      return view_id <= max_view_id && view_present[view_id] != 0;
    }
    return std::binary_search(valid_view_ids.cbegin(), valid_view_ids.cend(), view_id);
  };

  // For each landmark:
  //  - Check if we need to keep the observations & the track
  Landmarks::iterator itLandmarks = sfm_data.structure.begin();
  while (itLandmarks != sfm_data.structure.end())
  {
    Observations & obs = itLandmarks->second.obs;
    Observations::iterator itObs = obs.begin();
    while (itObs != obs.end())
    {
      if (!is_valid(itObs->first))
      {
        itObs = obs.erase(itObs);
        ++removed_elements;
      }
      else
        ++itObs;
    }
    if (obs.empty() || obs.size() < min_points_per_landmark)
      itLandmarks = sfm_data.structure.erase(itLandmarks);
    else
      ++itLandmarks;
  }
  return removed_elements > 0;
}

/// Remove unstable content from analysis of the sfm_data structure
bool eraseUnstablePosesAndObservations
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_pose,
  const IndexT min_points_per_landmark
)
{
  // First remove orphan observation(s) (observation using an undefined pose)
  eraseObservationsWithMissingPoses(sfm_data, min_points_per_landmark);
  // Then iteratively remove orphan poses & observations
  IndexT remove_iteration = 0;
  bool bRemovedContent = false;
  do
  {
    bRemovedContent = false;
    if (eraseMissingPoses(sfm_data, min_points_per_pose))
    {
      bRemovedContent = eraseObservationsWithMissingPoses(sfm_data, min_points_per_landmark);
      // Erase some observations can make some Poses index disappear so perform the process in a loop
    }
    remove_iteration += bRemovedContent ? 1 : 0;
  }
  while (bRemovedContent);

  return remove_iteration > 0;
}

/// Tell if the sfm_data structure is one CC or not
bool IsTracksOneCC
(
  const SfM_Data & sfm_data
)
{
  // Compute the Connected Component from the tracks

  // Build a table to have contiguous view index in [0,n]
  // (Use only the view index used in the observations)
  Hash_Map<IndexT, IndexT> view_renumbering;
  IndexT cpt = 0;
  const Landmarks & landmarks = sfm_data.structure;
  for (const auto & Landmark_it : landmarks)
  {
    const Observations & obs = Landmark_it.second.obs;
    for (const auto & obs_it : obs)
    {
      if (view_renumbering.count(obs_it.first) == 0)
      {
        view_renumbering[obs_it.first] = cpt++;
      }
    }
  }

  UnionFind uf_tree;
  uf_tree.InitSets(view_renumbering.size());

  // Link track observations in connected component
  for (const auto & Landmark_it : landmarks)
  {
    const Observations & obs = Landmark_it.second.obs;
    std::set<IndexT> id_to_link;
    for (const auto & obs_it : obs)
    {
      id_to_link.insert(view_renumbering.at(obs_it.first));
    }
    std::set<IndexT>::const_iterator iterI = id_to_link.cbegin();
    std::set<IndexT>::const_iterator iterJ = id_to_link.cbegin();
    std::advance(iterJ, 1);
    while (iterJ != id_to_link.cend())
    {
      // Link I => J
      uf_tree.Union(*iterI, *iterJ);
      ++iterJ;
    }
  }

  // Run path compression to identify all the CC id belonging to every item
  for (unsigned int i = 0; i < uf_tree.GetNumNodes(); ++i)
  {
    uf_tree.Find(i);
  }

  // Count the number of CC
  const std::set<unsigned int> parent_id(uf_tree.m_cc_parent.cbegin(), uf_tree.m_cc_parent.cend());
  return parent_id.size() == 1;
}

/// Keep the largest connected component of tracks from the sfm_data structure
void KeepLargestViewCCTracks
(
  SfM_Data & sfm_data
)
{
  // Compute the Connected Component from the tracks

  // Build a table to have contiguous view index in [0,n]
  // (Use only the view index used in the observations)
  Hash_Map<IndexT, IndexT> view_renumbering;
  {
    IndexT cpt = 0;
    const Landmarks & landmarks = sfm_data.structure;
    for (const auto & Landmark_it : landmarks)
    {
      const Observations & obs = Landmark_it.second.obs;
      for (const auto & obs_it : obs)
      {
        if (view_renumbering.count(obs_it.first) == 0)
        {
          view_renumbering[obs_it.first] = cpt++;
        }
      }
    }
  }

  UnionFind uf_tree;
  uf_tree.InitSets(view_renumbering.size());

  // Link track observations in connected component
  Landmarks & landmarks = sfm_data.structure;
  for (const auto & Landmark_it : landmarks)
  {
    const Observations & obs = Landmark_it.second.obs;
    std::set<IndexT> id_to_link;
    for (const auto & obs_it : obs)
    {
      id_to_link.insert(view_renumbering.at(obs_it.first));
    }
    std::set<IndexT>::const_iterator iterI = id_to_link.cbegin();
    std::set<IndexT>::const_iterator iterJ = id_to_link.cbegin();
    std::advance(iterJ, 1);
    while (iterJ != id_to_link.cend())
    {
      // Link I => J
      uf_tree.Union(*iterI, *iterJ);
      ++iterJ;
    }
  }

  // Count the number of CC
  const std::set<unsigned int> parent_id(uf_tree.m_cc_parent.cbegin(), uf_tree.m_cc_parent.cend());
  if (parent_id.size() > 1)
  {
    // There is many CC, look the largest one
    // (if many CC have the same size, export the first that have been seen)
    std::pair<IndexT, unsigned int> max_cc( UndefinedIndexT, std::numeric_limits<unsigned int>::min());
    {
      for (const unsigned int parent_id_it : parent_id)
      {
        if (uf_tree.m_cc_size[parent_id_it] > max_cc.second) // Update the component parent id and size
        {
          max_cc = {parent_id_it, uf_tree.m_cc_size[parent_id_it]};
        }
      }
    }
    // Delete track ids that are not contained in the largest CC
    if (max_cc.first != UndefinedIndexT)
    {
      const unsigned int parent_id_largest_cc = max_cc.first;
      Landmarks::iterator itLandmarks = landmarks.begin();
      while (itLandmarks != landmarks.end())
      {
        // Since we built a view 'track' graph thanks to the UF tree,
        //  checking the CC of each track is equivalent to check the CC of any observation of it.
        // So we check only the first
        const Observations & obs = itLandmarks->second.obs;
        Observations::const_iterator itObs = obs.begin();
        if (!obs.empty())
        {
          if (uf_tree.Find(view_renumbering.at(itObs->first)) != parent_id_largest_cc)
          {
            itLandmarks = landmarks.erase(itLandmarks);
          }
          else
          {
            ++itLandmarks;
          }
        }
      }
    }
  }
}

/**
* @brief Implement a statistical Structure filter that remove 3D points that have:
* - a depth that is too large (threshold computed as factor * median ~= X84)
* @param sfm_data The sfm scene to filter (inplace filtering)
* @param k_factor The factor applied to the median depth per view
* @param k_min_point_per_pose Keep only poses that have at least this amount of points
* @param k_min_track_length Keep only tracks that have at least this length
* @return The min_median_value observed for all the view
*/
double DepthCleaning
(
  SfM_Data & sfm_data,
  const double k_factor,
  const IndexT k_min_point_per_pose,
  const IndexT k_min_track_length
)
{
  // Per-view (pose, view_id) cache. The original code did views.at() +
  // IsPoseAndIntrinsicDefined() + GetPoseOrDie() per observation in *both*
  // passes (and used std::map for the depth/median accumulators). We use
  // the same flat-vector trick as the rest of this file: array index by
  // view_id when ids are dense, Hash_Map fallback for sparse id ranges.
  // view_cache, depth accumulator and median-depth all share the same
  // sparseness decision since they're all keyed by view_id.
  struct ViewCache {
    geometry::Pose3 pose;
    IndexT view_id = UndefinedIndexT; // == sentinel "invalid slot" on flat path
  };

  IndexT max_view_id = 0;
  bool any_view = false;
  for (const auto & view_it : sfm_data.views)
  {
    if (view_it.first > max_view_id) max_view_id = view_it.first;
    any_view = true;
  }
  const bool use_flat =
    any_view &&
    max_view_id != UndefinedIndexT &&
    static_cast<size_t>(max_view_id) < sfm_data.views.size() * 8 + 64;

  // thread_local-static buffers: DepthCleaning is called multiple times
  // per stellar reconstruction; per-view depth-accumulator inner vectors
  // are the main capacity we want to retain across calls. ViewCache and
  // median_depth contain no heap, so plain assign() is fine for those.
  thread_local static std::vector<ViewCache> view_cache_flat;
  thread_local static Hash_Map<IndexT, ViewCache> view_cache_hash;
  if (use_flat)
  {
    view_cache_flat.assign(static_cast<size_t>(max_view_id) + 1, ViewCache{});
  }
  else
  {
    view_cache_hash.clear();
    view_cache_hash.reserve(sfm_data.views.size());
  }

  for (const auto & view_it : sfm_data.views)
  {
    const View * v = view_it.second.get();
    if (!v || v->id_intrinsic == UndefinedIndexT || v->id_pose == UndefinedIndexT)
      continue;
    const auto pose_it = sfm_data.poses.find(v->id_pose);
    if (pose_it == sfm_data.poses.end()) continue;
    if (sfm_data.intrinsics.find(v->id_intrinsic) == sfm_data.intrinsics.end())
      continue;
    // Store the MAP KEY, not v->id_view. These are the same value in every
    // scene OpenMVG builds itself, but nothing enforces it -- Views are
    // shared_ptr'd into sfm_data.views under a caller-chosen key, and a
    // scene assembled by hand or by subsetting can hold a View whose
    // id_view field disagrees with the slot it sits in.
    //
    // That distinction is load-bearing on the flat path: max_view_id (and
    // therefore the size of depth_accum_flat / median_depth_flat below) is
    // derived from the map KEYS, while push_depth() indexes those arrays by
    // ViewCache::view_id with no bounds check. Keying the cache off id_view
    // made "id_view > max(keys)" an out-of-bounds push_back through a bogus
    // std::vector header -- heap corruption, not a clean crash.
    if (use_flat)
      view_cache_flat[view_it.first] = { pose_it->second, view_it.first };
    else
      view_cache_hash[view_it.first] = { pose_it->second, view_it.first };
  }

  auto get_cache = [&](IndexT view_id) -> const ViewCache * {
    if (use_flat)
    {
      if (view_id > max_view_id) return nullptr;
      const ViewCache & vc = view_cache_flat[view_id];
      return (vc.view_id != UndefinedIndexT) ? &vc : nullptr;
    }
    const auto it = view_cache_hash.find(view_id);
    return (it != view_cache_hash.end()) ? &it->second : nullptr;
  };

  // Per-view depth accumulator, indexed by ViewCache::view_id -- which is
  // the sfm_data.views map key, the same quantity max_view_id was measured
  // over, so the flat index below is in range by construction.
  // thread_local-static + clear() each inner so the per-view capacity
  // (typically thousands of depths) is retained across calls. Avoid
  // assign(N+1, {}) which would destroy every inner vector.
  thread_local static std::vector<std::vector<double>> depth_accum_flat;
  thread_local static Hash_Map<IndexT, std::vector<double>> depth_accum_hash;
  if (use_flat)
  {
    if (depth_accum_flat.size() < static_cast<size_t>(max_view_id) + 1)
      depth_accum_flat.resize(static_cast<size_t>(max_view_id) + 1);
    for (auto & v : depth_accum_flat) v.clear();
  }
  else
  {
    for (auto & kv : depth_accum_hash) kv.second.clear();
    depth_accum_hash.reserve(sfm_data.views.size());
  }

  auto push_depth = [&](IndexT view_id, double d) {
    if (use_flat) depth_accum_flat[view_id].push_back(d);
    else          depth_accum_hash[view_id].push_back(d);
  };

  // Pass 1: accumulate per-view depths.
  for (const auto & landmark_it : sfm_data.structure)
  {
    const Vec3 & X = landmark_it.second.X;
    for (const auto & obs_it : landmark_it.second.obs)
    {
      const ViewCache * vc = get_cache(obs_it.first);
      if (!vc) continue;
      const double depth = Depth(vc->pose.rotation(), vc->pose.translation(), X);
      if (depth > 0)
        push_depth(vc->view_id, depth);
    }
  }

  // Per-view median depth -> threshold (k_factor * median). Sentinel
  // kNoMedian (negative) marks views with no usable accumulator. Negative
  // sentinel is safe because all real thresholds are k_factor*positive
  // depths > 0.
  static constexpr double kNoMedian = -1.0;
  thread_local static std::vector<double> median_depth_flat;
  thread_local static Hash_Map<IndexT, double> median_depth_hash;
  if (use_flat)
  {
    median_depth_flat.assign(static_cast<size_t>(max_view_id) + 1, kNoMedian);
  }
  else
  {
    median_depth_hash.clear();
    median_depth_hash.reserve(sfm_data.views.size());
  }

  double min_median_value = std::numeric_limits<double>::max();
  auto consume_acc = [&](IndexT view_id, std::vector<double> & acc) {
    if (acc.empty()) return;
    double mn, mx, mean, median;
    if (minMaxMeanMedian(acc.begin(), acc.end(), mn, mx, mean, median))
    {
      min_median_value = std::min(min_median_value, median);
      const double thresh = k_factor * median;
      if (use_flat) median_depth_flat[view_id] = thresh;
      else          median_depth_hash[view_id] = thresh;
    }
  };
  if (use_flat)
  {
    for (size_t vid = 0; vid <= static_cast<size_t>(max_view_id); ++vid)
      consume_acc(static_cast<IndexT>(vid), depth_accum_flat[vid]);
  }
  else
  {
    for (auto & kv : depth_accum_hash)
      consume_acc(kv.first, kv.second);
  }
  // Note: the accumulator is intentionally NOT freed between passes here.
  // It's a thread_local cache reused across calls; freeing it would
  // defeat the amortization. Peak working set is the same as before since
  // the cache would have been re-allocated by the next call anyway.

  auto get_median = [&](IndexT view_id) -> double {
    if (use_flat)
    {
      if (view_id > max_view_id) return kNoMedian;
      return median_depth_flat[view_id];
    }
    const auto it = median_depth_hash.find(view_id);
    return (it != median_depth_hash.end()) ? it->second : kNoMedian;
  };

  // Pass 2: in-place erase. Original semantics:
  //   - obs without a valid view cache entry -> drop.
  //   - obs with depth <= 0, missing median, or depth >= threshold -> drop.
  //   - else keep.
  // The original built a fresh Observations map per landmark and swapped
  // it in, paying an insert per kept obs. In-place erase is O(1) per
  // kept obs.
  for (auto & landmark_it : sfm_data.structure)
  {
    const Vec3 & X = landmark_it.second.X;
    Observations & obs = landmark_it.second.obs;
    auto itObs = obs.begin();
    while (itObs != obs.end())
    {
      bool keep = false;
      const ViewCache * vc = get_cache(itObs->first);
      if (vc)
      {
        const double depth = Depth(vc->pose.rotation(), vc->pose.translation(), X);
        const double thresh = get_median(vc->view_id);
        keep = (depth > 0) && (thresh >= 0.0) && (depth < thresh);
      }
      if (keep)
        ++itObs;
      else
        itObs = obs.erase(itObs);
    }
  }

  // Remove orphans
  eraseUnstablePosesAndObservations(sfm_data, k_min_point_per_pose, k_min_track_length);

  return min_median_value;
}

} // namespace sfm
} // namespace openMVG
