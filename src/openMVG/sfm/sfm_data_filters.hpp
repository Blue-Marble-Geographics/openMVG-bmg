// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_DATA_FILTERS_HPP
#define OPENMVG_SFM_SFM_DATA_FILTERS_HPP

#include <set>

#include "openMVG/types.hpp"

namespace openMVG { namespace sfm { struct SfM_Data; } }

namespace openMVG {
namespace sfm {

/// List the view indexes that have valid camera intrinsic and pose.
std::set<IndexT> Get_Valid_Views
(
  const SfM_Data & sfm_data
);

/// Filter a list of pair: Keep only the pairs that are defined by the index list
template <typename IterablePairs, typename IterableIndex>
Pair_Set Pair_filter
(
  const IterablePairs & pairs,
  const IterableIndex & index
)
{
  Pair_Set kept_pairs;
  for (auto& it : pairs)
  {
    if (index.count(it.first) > 0 &&
      index.count(it.second) > 0)
    kept_pairs.insert(it);
  }
  return kept_pairs;
}

// Remove tracks that have a small angle (tracks with tiny angle leads to instable 3D points)
// Return the number of removed tracks
IndexT RemoveOutliers_PixelResidualError
(
  SfM_Data & sfm_data,
  const double dThresholdPixel,
  const unsigned int minTrackLength = 2
);

// Remove tracks that have a small angle (tracks with tiny angle leads to instable 3D points)
// Return the number of removed tracks
IndexT RemoveOutliers_AngleError
(
  SfM_Data & sfm_data,
  const double dMinAcceptedAngle
);

/// Fused angle + pixel-residual outlier filter. Equivalent to calling
///   RemoveOutliers_AngleError(sfm_data, dMinAcceptedAngle);
///   RemoveOutliers_PixelResidualError(sfm_data, dThresholdPixel, minTrackLength);
/// in that order, but does it in a single pass over sfm_data.structure with
/// the per-view (pose, intrinsic) cache built only once. Returns the total
/// number of erased observations + erased-track placeholders (semantically
/// the sum of what the two separate calls would have returned).
IndexT RemoveOutliers_PixelAndAngleError
(
  SfM_Data & sfm_data,
  const double dThresholdPixel,
  const double dMinAcceptedAngle,
  const unsigned int minTrackLength = 2,
  IndexT * out_removed_by_angle = nullptr,
  IndexT * out_removed_by_pixel = nullptr
);

/// Eject poses whose median reprojection residual is much worse than the
/// rest of the reconstruction. Implements the "one bad pose poisons the BA"
/// safety net used by COLMAP-style robust pipelines.
///
/// For each pose, the median pixel residual over its observations is
/// computed. The global median over poses is used as the reference. Any
/// pose whose median residual exceeds
///     max( k_factor * global_median_of_medians, abs_floor_pixels )
/// is removed (along with its observations); short tracks get pruned as a
/// side effect via the same minTrackLength rule used elsewhere.
///
/// @param sfm_data        scene to clean (modified in place)
/// @param k_factor        multiplier on the median-of-medians (default 3.0)
/// @param abs_floor_pixels minimum residual a pose must exceed to be ejected
///                         even when the scene is uniformly noisy (default 2.0)
/// @param min_points_per_landmark forwarded to the orphan cleanup pass
/// @return number of poses ejected
IndexT EjectPosesByMedianResidual
(
  SfM_Data & sfm_data,
  const double k_factor = 3.0,
  const double abs_floor_pixels = 2.0,
  const IndexT min_points_per_landmark = 2
);

/// Pose-prior outlier ejection. Complements EjectPosesByMedianResidual for
/// scenes with motion priors: a pose can have a clean reprojection median
/// yet sit far from its prior center. Each pose's distance from its prior
/// center is compared against
///     max( k_factor * median_distance, abs_floor_units )
/// (user units, e.g. meters). Poses exceeding the threshold are erased
/// along with their orphaned observations. Safe to call when no priors
/// are active (returns 0).
IndexT EjectPosesByPriorResidual
(
  SfM_Data & sfm_data,
  const double k_factor = 3.0,
  const double abs_floor_units = 1.0,
  const IndexT min_points_per_landmark = 2
);

/// Erase pose with insufficient track observations
bool eraseMissingPoses
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_pose = 6
);

/// Erase observations with no defined pose
bool eraseObservationsWithMissingPoses
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_landmark = 2
);

/// Remove unstable content from analysis of the sfm_data structure
bool eraseUnstablePosesAndObservations
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_pose = 6,
  const IndexT min_points_per_landmark = 2
);

/// Tell if the sfm_data structure is one CC or not
bool IsTracksOneCC
(
  const SfM_Data & sfm_data
);

/// Keep the largest connected component of tracks from the sfm_data structure
void KeepLargestViewCCTracks
(
  SfM_Data & sfm_data
);

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
  const double k_factor =  5.2,    //5.2 * median ~= X84,
  const IndexT k_min_point_per_pose = 12,  // 6 min
  const IndexT k_min_track_length = 2      // 2 min
);

} // namespace sfm
} // namespace openMVG

#endif // OPENMVG_SFM_SFM_DATA_FILTERS_HPP
