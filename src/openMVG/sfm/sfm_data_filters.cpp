// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data_filters.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/tracks/union_find.hpp"

#include <utility>

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

// Remove tracks that have a small angle (tracks with tiny angle leads to instable 3D points)
// Return the number of removed tracks
IndexT RemoveOutliers_PixelResidualError
(
  SfM_Data& sfm_data,
  const double dThresholdPixel,
  const unsigned int minTrackLength
)
{
  // Precompute squared threshold to avoid sqrt per residual
  const double dThresholdPixelSq = dThresholdPixel * dThresholdPixel;

  // Build a per-view cache of pose + intrinsic (small, ~186 entries)
  struct ViewCache {
    geometry::Pose3 pose;
    const cameras::IntrinsicBase* intrinsic;
  };
  Hash_Map<IndexT, ViewCache> view_cache;
  view_cache.reserve(sfm_data.views.size());
  for (const auto& view_it : sfm_data.views)
  {
    const View* v = view_it.second.get();
    if (v->id_intrinsic == UndefinedIndexT || v->id_pose == UndefinedIndexT)
      continue;
    const auto pose_it = sfm_data.poses.find(v->id_pose);
    if (pose_it == sfm_data.poses.end())
      continue;
    const auto intrinsic_it = sfm_data.intrinsics.find(v->id_intrinsic);
    if (intrinsic_it == sfm_data.intrinsics.end())
      continue;
    view_cache[view_it.first] = {
      pose_it->second,
      intrinsic_it->second.get()
    };
  }

  IndexT outlier_count = 0;
  Landmarks::iterator iterTracks = sfm_data.structure.begin();
  while (iterTracks != sfm_data.structure.end())
  {
    Observations& obs = iterTracks->second.obs;
    Observations::iterator itObs = obs.begin();
    const Vec3& X = iterTracks->second.X;
    while (itObs != obs.end())
    {
      const auto cache_it = view_cache.find(itObs->first);
      if (cache_it != view_cache.end())
      {
        const Vec2 residual = cache_it->second.intrinsic->residual(
          cache_it->second.pose(X), itObs->second.x);
        if (residual.squaredNorm() > dThresholdPixelSq)
        {
          ++outlier_count;
          itObs = obs.erase(itObs);
          continue;
        }
      }
      ++itObs;
    }
    if (obs.empty() || obs.size() < minTrackLength)
      iterTracks = sfm_data.structure.erase(iterTracks);
    else
      ++iterTracks;
  }
  return outlier_count;
}

// Remove tracks that have a small angle (tracks with tiny angle leads to instable 3D points)
// Return the number of removed tracks
IndexT RemoveOutliers_AngleError
(
  SfM_Data& sfm_data,
  const double dMinAcceptedAngle
)
{
  IndexT removedTrack_count = 0;

  // Precompute the cosine threshold once (angle decreasing => cosine increasing)
  const double cosAngleThreshold = cos(D2R(dMinAcceptedAngle));

  // Build a per-view cache of R^T + intrinsic (small, ~186 entries)
  struct ViewCache {
    Mat3 Rt;  // rotation transposed
    const cameras::IntrinsicBase* intrinsic;
  };
  Hash_Map<IndexT, ViewCache> view_cache;
  view_cache.reserve(sfm_data.views.size());
  for (const auto& view_it : sfm_data.views)
  {
    const View* v = view_it.second.get();
    if (v->id_intrinsic == UndefinedIndexT || v->id_pose == UndefinedIndexT)
      continue;
    const auto pose_it = sfm_data.poses.find(v->id_pose);
    if (pose_it == sfm_data.poses.end())
      continue;
    const auto intrinsic_it = sfm_data.intrinsics.find(v->id_intrinsic);
    if (intrinsic_it == sfm_data.intrinsics.end())
      continue;
    view_cache[view_it.first] = {
      pose_it->second.rotation().transpose(),
      intrinsic_it->second.get()
    };
  }

  // Pre-fetch bearing ray for each observation
  std::vector<Vec3> rays;

  Landmarks::iterator iterTracks = sfm_data.structure.begin();
  while (iterTracks != sfm_data.structure.end())
  {
    Observations& obs = iterTracks->second.obs;
    bool convergent = false;

    rays.clear();
    rays.reserve(obs.size());
    for (const auto& obs_it : obs)
    {
      const auto cache_it = view_cache.find(obs_it.first);
      if (cache_it != view_cache.end())
      {
        const auto& vc = cache_it->second;
        // ray = R^T * bearing, normalized
        rays.emplace_back(
          (vc.Rt * vc.intrinsic->operator()(
            vc.intrinsic->get_ud_pixel(obs_it.second.x))).normalized());
      }
    }

    // Compare pairs via dot product, early-exit once angle exceeds threshold
    // angle >= threshold  <=>  dot <= cos(threshold)  (for angles in [0, pi])
    const size_t rays_size = rays.size();
    for (size_t i = 0; i < rays_size && !convergent; ++i)
    {
      for (size_t j = i + 1; j < rays_size; ++j)
      {
        if (rays[i].dot(rays[j]) <= cosAngleThreshold)
        {
          convergent = true;
          break;
        }
      }
    }

    if (!convergent)
    {
      iterTracks = sfm_data.structure.erase(iterTracks);
      ++removedTrack_count;
    }
    else
      ++iterTracks;
  }
  return removedTrack_count;
}

bool eraseMissingPoses
(
  SfM_Data& sfm_data,
  const IndexT min_points_per_pose
)
{
  IndexT removed_elements = 0;
  const Landmarks& landmarks = sfm_data.structure;

  // Build a flat viewId -> poseId lookup (small, ~186 entries)
  Hash_Map<IndexT, IndexT> view_to_pose;
  view_to_pose.reserve(sfm_data.views.size());
  for (const auto& view_it : sfm_data.views)
  {
    view_to_pose[view_it.first] = view_it.second->id_pose;
  }

  // Count the observation poses occurrence
  Hash_Map<IndexT, IndexT> map_PoseId_Count;
  // Init with 0 count (in order to be able to remove non referenced elements)
  for (const auto& pose_it : sfm_data.GetPoses())
  {
    map_PoseId_Count[pose_it.first] = 0;
  }

  // Count occurrence of the poses in the Landmark observations
  for (const auto& lanmark_it : landmarks)
  {
    const Observations& obs = lanmark_it.second.obs;
    for (const auto& obs_it : obs)
    {
      const auto it = view_to_pose.find(obs_it.first);
      if (it != view_to_pose.end())
        map_PoseId_Count[it->second] += 1;
    }
  }
  // If usage count is smaller than the threshold, remove the Pose
  for (const auto& it : map_PoseId_Count)
  {
    if (it.second < min_points_per_pose)
    {
      sfm_data.poses.erase(it.first);
      ++removed_elements;
    }
  }
  return removed_elements > 0;
}

bool eraseObservationsWithMissingPoses
(
  SfM_Data& sfm_data,
  const IndexT min_points_per_landmark
)
{
  IndexT removed_elements = 0;

  // Build a sorted vector of view ids whose pose exists (small, ~186 elements)
  std::vector<IndexT> valid_view_ids;
  valid_view_ids.reserve(sfm_data.views.size());
  for (const auto& view_it : sfm_data.views)
  {
    if (sfm_data.poses.count(view_it.second->id_pose))
      valid_view_ids.push_back(view_it.first);
  }
  std::sort(valid_view_ids.begin(), valid_view_ids.end());

  // For each landmark:
  //  - Check if we need to keep the observations & the track
  Landmarks::iterator itLandmarks = sfm_data.structure.begin();
  while (itLandmarks != sfm_data.structure.end())
  {
    Observations& obs = itLandmarks->second.obs;
    Observations::iterator itObs = obs.begin();
    while (itObs != obs.end())
    {
      if (!std::binary_search(valid_view_ids.cbegin(), valid_view_ids.cend(), itObs->first))
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
  using DepthAccumulatorT = std::vector<double>;
  std::map<IndexT, DepthAccumulatorT > map_depth_accumulator;

  // For each landmark accumulate the camera/point depth info for each view
  for (const auto & landmark_it : sfm_data.structure)
  {
    const Observations & obs = landmark_it.second.obs;
    for (const auto & obs_it : obs)
    {
      const View * view = sfm_data.views.at(obs_it.first).get();
      if (sfm_data.IsPoseAndIntrinsicDefined(view))
      {
        const Pose3 pose = sfm_data.GetPoseOrDie(view);
        const double depth = Depth(pose.rotation(), pose.translation(), landmark_it.second.X);
        if (depth > 0)
        {
          map_depth_accumulator[view->id_view].push_back(depth);
        }
      }
    }
  }

  double min_median_value = std::numeric_limits<double>::max();
  std::map<IndexT, double > map_median_depth;
  for (const auto & iter : sfm_data.GetViews())
  {
    const View * v = iter.second.get();
    const IndexT view_id = v->id_view;
    if (map_depth_accumulator.count(view_id) == 0)
      continue;
    // Compute median from the depth distribution
    const auto & acc = map_depth_accumulator.at(view_id);
    double min, max, mean, median;
    if (minMaxMeanMedian(acc.begin(), acc.end(), min, max, mean, median))
    {

      min_median_value = std::min(min_median_value, median);
      // Compute depth threshold for each view: factor * medianDepth
      map_median_depth[view_id] = k_factor * median;
    }
  }
  map_depth_accumulator.clear();

  // Delete invalid observations
  size_t cpt = 0;
  for (auto & landmark_it : sfm_data.structure)
  {
    Observations obs;
    for (auto & obs_it : landmark_it.second.obs)
    {
      const View * view = sfm_data.views.at(obs_it.first).get();
      if (sfm_data.IsPoseAndIntrinsicDefined(view))
      {
        const Pose3 pose = sfm_data.GetPoseOrDie(view);
        const double depth = Depth(pose.rotation(), pose.translation(), landmark_it.second.X);
        if ( depth > 0
            && map_median_depth.count(view->id_view)
            && depth < map_median_depth[view->id_view])
          obs.insert(obs_it);
        else
          ++cpt;
      }
    }
    landmark_it.second.obs.swap(obs);
  }

  // Remove orphans
  eraseUnstablePosesAndObservations(sfm_data, k_min_point_per_pose, k_min_track_length);

  return min_median_value;
}

} // namespace sfm
} // namespace openMVG
