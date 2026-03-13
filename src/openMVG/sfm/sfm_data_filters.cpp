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
      const SfM_Data& sfm_data
    )
    {
      std::set<IndexT> valid_idx;
      for (const auto& view_it : sfm_data.GetViews())
      {
        const View* v = view_it.second.get();
        if (sfm_data.IsPoseAndIntrinsicDefined(v))
        {
          valid_idx.insert(v->id_view);
        }
      }
      return valid_idx;
    }

#if 1 // original
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
#else

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

      // Collect iterators for parallel random-access
      std::vector<Landmarks::iterator> workItems;
      workItems.reserve(sfm_data.structure.size());
      for (auto it = sfm_data.structure.begin(); it != sfm_data.structure.end(); ++it)
      {
        workItems.push_back(it);
      }

      // Per-landmark result: count of outliers removed, and whether to erase the landmark
      struct LandmarkResult {
        IndexT outliers_removed;
        bool erase_landmark;
      };
      std::vector<LandmarkResult> results(workItems.size());

#ifdef OPENMVG_USE_OPENMP
#pragma omp parallel for schedule(static, 128)
#endif
      for (int i = 0; i < static_cast<int>(workItems.size()); ++i)
      {
        auto& tracks_it = *workItems[i];
        Observations& obs = tracks_it.second.obs;
        const Vec3& X = tracks_it.second.X;
        IndexT local_outliers = 0;

        Observations::iterator itObs = obs.begin();
        while (itObs != obs.end())
        {
          const auto cache_it = view_cache.find(itObs->first);
          if (cache_it != view_cache.end())
          {
            const Vec2 residual = cache_it->second.intrinsic->residual(
              cache_it->second.pose(X), itObs->second.x);
            if (residual.squaredNorm() > dThresholdPixelSq)
            {
              ++local_outliers;
              itObs = obs.erase(itObs);
              continue;
            }
          }
          ++itObs;
        }

        results[i].outliers_removed = local_outliers;
        results[i].erase_landmark = (obs.empty() || obs.size() < minTrackLength);
      }

      // Sequential accumulation and erasure
      IndexT outlier_count = 0;
      for (int i = static_cast<int>(workItems.size()) - 1; i >= 0; --i)
      {
        outlier_count += results[i].outliers_removed;
        if (results[i].erase_landmark)
          sfm_data.structure.erase(workItems[i]);
      }
      return outlier_count;
    }
#endif

#if 1 // original
    IndexT RemoveOutliers_AngleError
    (
      SfM_Data& sfm_data,
      const double dMinAcceptedAngle
    )
    {
      IndexT removedTrack_count = 0;

      // Precompute the cosine squared threshold
      // angle >= threshold  <=>  cos(angle) <= cos(threshold)
      // For unit-free rays: dot(a,b)/(|a|*|b|) <= cos(threshold)
      // Squared (avoiding sqrt): dot^2 >= |a|^2*|b|^2*cos^2(threshold) means angle < threshold (reject)
      // So a track is convergent when we find a pair where:
      //   dot < 0  (angle > 90°, always convergent), OR
      //   dot^2 < |a|^2 * |b|^2 * cos^2(threshold)  (angle > threshold)
      const double cosThreshold = cos(D2R(dMinAcceptedAngle));
      const double cos2Threshold = cosThreshold * cosThreshold;

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

      // Per-track ray storage (unnormalized) + precomputed squared norms
      std::vector<Vec3> rays;
      std::vector<double> sqNorms;

      Landmarks::iterator iterTracks = sfm_data.structure.begin();
      while (iterTracks != sfm_data.structure.end())
      {
        Observations& obs = iterTracks->second.obs;
        bool convergent = false;

        rays.clear();
        sqNorms.clear();
        rays.reserve(obs.size());
        sqNorms.reserve(obs.size());
        for (const auto& obs_it : obs)
        {
          const auto cache_it = view_cache.find(obs_it.first);
          if (cache_it != view_cache.end())
          {
            const auto& vc = cache_it->second;
            const Vec2 cam_pt = vc.intrinsic->ima2cam(obs_it.second.x);
            const Vec2 undist_pt = vc.intrinsic->remove_disto(cam_pt);
            // ray = R^T * [undist_x, undist_y, 1]^T  (unnormalized)
            rays.emplace_back(vc.Rt * undist_pt.homogeneous());
            sqNorms.push_back(rays.back().squaredNorm());
          }
        }

        // Compare pairs — no sqrt needed:
        //  convergent when dot < 0  OR  dot^2 < sqNorm_i * sqNorm_j * cos^2(threshold)
        const size_t rays_size = rays.size();
        for (size_t i = 0; i < rays_size && !convergent; ++i)
        {
          const double sqNorm_i_cos2 = sqNorms[i] * cos2Threshold;
          for (size_t j = i + 1; j < rays_size; ++j)
          {
            const double d = rays[i].dot(rays[j]);
            if (d < 0.0 || d * d <= sqNorm_i_cos2 * sqNorms[j])
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
#else

    // Remove tracks that have a small angle (tracks with tiny angle leads to instable 3D points)
    // Return the number of removed tracks
    IndexT RemoveOutliers_AngleError
    (
      SfM_Data& sfm_data,
      const double dMinAcceptedAngle
    )
    {
      // Precompute the cosine squared threshold
      // angle >= threshold  <=>  cos(angle) <= cos(threshold)
      // For unit-free rays: dot(a,b)/(|a|*|b|) <= cos(threshold)
      // Squared (avoiding sqrt): dot^2 >= |a|^2*|b|^2*cos^2(threshold) means angle < threshold (reject)
      // So a track is convergent when we find a pair where:
      //   dot < 0  (angle > 90°, always convergent), OR
      //   dot^2 < |a|^2 * |b|^2 * cos^2(threshold)  (angle > threshold)
      const double cosThreshold = cos(D2R(dMinAcceptedAngle));
      const double cos2Threshold = cosThreshold * cosThreshold;

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

      // Collect iterators for parallel random-access
      std::vector<Landmarks::iterator> workItems;
      workItems.reserve(sfm_data.structure.size());
      for (auto it = sfm_data.structure.begin(); it != sfm_data.structure.end(); ++it)
      {
        workItems.push_back(it);
      }

      // Parallel classification: determine which tracks to reject
      std::vector<IndexT> rejectedIds;
      rejectedIds.reserve(workItems.size());

#ifdef OPENMVG_USE_OPENMP
#pragma omp parallel
#endif
      {
        // Thread-local storage to avoid contention
        std::vector<IndexT> localRejected;
        std::vector<Vec3> rays;
        std::vector<double> sqNorms;

#ifdef OPENMVG_USE_OPENMP
#pragma omp for schedule(static, 128)
#endif
        for (int i = 0; i < static_cast<int>(workItems.size()); ++i)
        {
          const auto& tracks_it = *workItems[i];
          const Observations& obs = tracks_it.second.obs;
          bool convergent = false;

          rays.clear();
          sqNorms.clear();
          rays.reserve(obs.size());
          sqNorms.reserve(obs.size());
          for (const auto& obs_it : obs)
          {
            const auto cache_it = view_cache.find(obs_it.first);
            if (cache_it != view_cache.end())
            {
              const auto& vc = cache_it->second;
              const Vec2 cam_pt = vc.intrinsic->ima2cam(obs_it.second.x);
              const Vec2 undist_pt = vc.intrinsic->remove_disto(cam_pt);
              // ray = R^T * [undist_x, undist_y, 1]^T  (unnormalized)
              rays.emplace_back(vc.Rt * undist_pt.homogeneous());
              sqNorms.push_back(rays.back().squaredNorm());
            }
          }

          // Compare pairs — no sqrt needed
          const size_t rays_size = rays.size();
          for (size_t ri = 0; ri < rays_size && !convergent; ++ri)
          {
            const double sqNorm_i_cos2 = sqNorms[ri] * cos2Threshold;
            for (size_t rj = ri + 1; rj < rays_size; ++rj)
            {
              const double d = rays[ri].dot(rays[rj]);
              if (d < 0.0 || d * d <= sqNorm_i_cos2 * sqNorms[rj])
              {
                convergent = true;
                break;
              }
            }
          }

          if (!convergent)
          {
            localRejected.push_back(tracks_it.first);
          }
        }

#ifdef OPENMVG_USE_OPENMP
#pragma omp critical
#endif
        {
          rejectedIds.insert(rejectedIds.end(), localRejected.begin(), localRejected.end());
        }
      }

      // Sequential erasure
      for (const auto& id : rejectedIds)
      {
        sfm_data.structure.erase(id);
      }

      return static_cast<IndexT>(rejectedIds.size());
    }
#endif
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
      SfM_Data& sfm_data,
      const IndexT min_points_per_pose,
      const IndexT min_points_per_landmark
    )
    {
      const size_t poses_before = sfm_data.poses.size();
      const size_t tracks_before = sfm_data.structure.size();

      // First remove orphan observation(s) (observation using an undefined pose)
      eraseObservationsWithMissingPoses(sfm_data, min_points_per_landmark);

      // Early-out: if no poses were removed by the caller (outlier filtering),
      // and eraseObservationsWithMissingPoses didn't remove anything,
      // then the cascade loop below cannot remove anything either.
      if (sfm_data.poses.size() == poses_before &&
          sfm_data.structure.size() == tracks_before)
        return false;

      // Then iteratively remove orphan poses & observations
      IndexT remove_iteration = 0;
      bool bRemovedContent = false;
      do
      {
        bRemovedContent = false;
        if (eraseMissingPoses(sfm_data, min_points_per_pose))
        {
          bRemovedContent = eraseObservationsWithMissingPoses(sfm_data, min_points_per_landmark);
        }
        remove_iteration += bRemovedContent ? 1 : 0;
      } while (bRemovedContent);

      if (remove_iteration > 0)
      {
        OPENMVG_LOG_INFO
          << "-- eraseUnstablePosesAndObservations: "
          << remove_iteration << " cascade iteration(s)"
          << " | #poses: " << poses_before << " -> " << sfm_data.poses.size()
          << " | #tracks: " << tracks_before << " -> " << sfm_data.structure.size();
      }

      return remove_iteration > 0;
    }

    /// Tell if the sfm_data structure is one CC or not
    bool IsTracksOneCC
    (
      const SfM_Data& sfm_data
    )
    {
      // Compute the Connected Component from the tracks

      // Build a table to have contiguous view index in [0,n]
      // (Use only the view index used in the observations)
      Hash_Map<IndexT, IndexT> view_renumbering;
      IndexT cpt = 0;
      const Landmarks& landmarks = sfm_data.structure;
      for (const auto& Landmark_it : landmarks)
      {
        const Observations& obs = Landmark_it.second.obs;
        for (const auto& obs_it : obs)
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
      for (const auto& Landmark_it : landmarks)
      {
        const Observations& obs = Landmark_it.second.obs;
        std::set<IndexT> id_to_link;
        for (const auto& obs_it : obs)
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
      SfM_Data& sfm_data
    )
    {
      // Compute the Connected Component from the tracks

      // Build a table to have contiguous view index in [0,n]
      // (Use only the view index used in the observations)
      Hash_Map<IndexT, IndexT> view_renumbering;
      {
        IndexT cpt = 0;
        const Landmarks& landmarks = sfm_data.structure;
        for (const auto& Landmark_it : landmarks)
        {
          const Observations& obs = Landmark_it.second.obs;
          for (const auto& obs_it : obs)
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
      Landmarks& landmarks = sfm_data.structure;
      for (const auto& Landmark_it : landmarks)
      {
        const Observations& obs = Landmark_it.second.obs;
        std::set<IndexT> id_to_link;
        for (const auto& obs_it : obs)
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
        std::pair<IndexT, unsigned int> max_cc(UndefinedIndexT, std::numeric_limits<unsigned int>::min());
        {
          for (const unsigned int parent_id_it : parent_id)
          {
            if (uf_tree.m_cc_size[parent_id_it] > max_cc.second) // Update the component parent id and size
            {
              max_cc = { parent_id_it, uf_tree.m_cc_size[parent_id_it] };
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
            const Observations& obs = itLandmarks->second.obs;
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
      SfM_Data& sfm_data,
      const double k_factor,
      const IndexT k_min_point_per_pose,
      const IndexT k_min_track_length
    )
    {
      using DepthAccumulatorT = std::vector<double>;
      std::map<IndexT, DepthAccumulatorT > map_depth_accumulator;

      // For each landmark accumulate the camera/point depth info for each view
      for (const auto& landmark_it : sfm_data.structure)
      {
        const Observations& obs = landmark_it.second.obs;
        for (const auto& obs_it : obs)
        {
          const View* view = sfm_data.views.at(obs_it.first).get();
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
      for (const auto& iter : sfm_data.GetViews())
      {
        const View* v = iter.second.get();
        const IndexT view_id = v->id_view;
        if (map_depth_accumulator.count(view_id) == 0)
          continue;
        // Compute median from the depth distribution
        const auto& acc = map_depth_accumulator.at(view_id);
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
      for (auto& landmark_it : sfm_data.structure)
      {
        Observations obs;
        for (auto& obs_it : landmark_it.second.obs)
        {
          const View* view = sfm_data.views.at(obs_it.first).get();
          if (sfm_data.IsPoseAndIntrinsicDefined(view))
          {
            const Pose3 pose = sfm_data.GetPoseOrDie(view);
            const double depth = Depth(pose.rotation(), pose.translation(), landmark_it.second.X);
            if (depth > 0
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

    // Fused filter: runs both angle and pixel-residual outlier removal in a single
    // pass over the structure, avoiding redundant unordered_map traversals.
    // Returns {angle_removed, pixel_removed}.
    std::pair<IndexT, IndexT> RemoveOutliers_AngleAndPixelError
    (
      SfM_Data& sfm_data,
      const double dMinAcceptedAngle,
      const double dThresholdPixel,
      const unsigned int minTrackLength
    )
    {
      // --- Shared precomputation (done once instead of twice) ---

      const double dThresholdPixelSq = dThresholdPixel * dThresholdPixel;
      const double cosThreshold = cos(D2R(dMinAcceptedAngle));
      const double cos2Threshold = cosThreshold * cosThreshold;

      // Unified per-view cache with precomputed R, t = -RC, R^T, and intrinsic.
      // Storing R and t directly avoids the R*(X-C) subtraction in Pose3::operator().
      struct ViewCache {
        Mat3 R;
        Vec3 t;               // t = -R*C, so R*X+t == R*(X-C)
        Mat3 Rt;              // R^T for ray computation
        const cameras::IntrinsicBase* intrinsic;
      };

      // Find the max view id to size a flat lookup table
      IndexT max_view_id = 0;
      for (const auto& view_it : sfm_data.views)
      {
        if (view_it.first > max_view_id)
          max_view_id = view_it.first;
      }

      // Flat vector indexed by view_id — O(1) lookup, no hashing
      std::vector<ViewCache*> view_lut(max_view_id + 1, nullptr);
      std::vector<ViewCache> view_cache_storage;
      view_cache_storage.reserve(sfm_data.views.size());
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
        const Mat3& R = pose_it->second.rotation();
        view_cache_storage.push_back({
          R,
          -(R * pose_it->second.center()),
          R.transpose(),
          intrinsic_it->second.get()
        });
        view_lut[view_it.first] = &view_cache_storage.back();
      }

      // Single traversal of structure to collect workItems
      std::vector<Landmarks::iterator> workItems;
      workItems.reserve(sfm_data.structure.size());
      for (auto it = sfm_data.structure.begin(); it != sfm_data.structure.end(); ++it)
      {
        workItems.push_back(it);
      }

      // --- Per-landmark results ---
      struct LandmarkResult {
        IndexT pixel_outliers_removed;
        bool erase_angle;       // failed angle test -> erase entire landmark
        bool erase_pixel;       // too few obs after pixel filtering -> erase
      };
      std::vector<LandmarkResult> results(workItems.size());

      // --- Parallel pass: angle check + pixel residual filtering ---

#ifdef OPENMVG_USE_OPENMP
#pragma omp parallel
#endif
      {
        std::vector<Vec3> rays;
        std::vector<double> sqNorms;
        std::vector<const ViewCache*> obs_vc;

#ifdef OPENMVG_USE_OPENMP
#pragma omp for schedule(static, 128)
#endif
        for (int i = 0; i < static_cast<int>(workItems.size()); ++i)
        {
          auto& tracks_it = *workItems[i];
          Observations& obs = tracks_it.second.obs;
          const Vec3& X = tracks_it.second.X;
          LandmarkResult& res = results[i];
          res.pixel_outliers_removed = 0;
          res.erase_angle = false;
          res.erase_pixel = false;

          // --- Resolve ViewCache pointers once, reuse in both phases ---
          obs_vc.clear();
          obs_vc.reserve(obs.size());
          for (const auto& obs_it : obs)
          {
            const IndexT view_id = obs_it.first;
            obs_vc.push_back(
              view_id <= max_view_id ? view_lut[view_id] : nullptr);
          }

          // --- Phase 1: Angle check ---
          bool convergent = false;
          rays.clear();
          sqNorms.clear();
          rays.reserve(obs.size());
          sqNorms.reserve(obs.size());

          size_t obs_idx = 0;
          for (const auto& obs_it : obs)
          {
            const ViewCache* vc = obs_vc[obs_idx++];
            if (vc)
            {
              const Vec2 cam_pt = vc->intrinsic->ima2cam(obs_it.second.x);
              const Vec2 undist_pt = vc->intrinsic->remove_disto(cam_pt);
              rays.emplace_back(vc->Rt * undist_pt.homogeneous());
              sqNorms.push_back(rays.back().squaredNorm());
            }
          }

          const size_t rays_size = rays.size();
          for (size_t ri = 0; ri < rays_size && !convergent; ++ri)
          {
            const double sqNorm_i_cos2 = sqNorms[ri] * cos2Threshold;
            for (size_t rj = ri + 1; rj < rays_size; ++rj)
            {
              const double d = rays[ri].dot(rays[rj]);
              if (d < 0.0 || d * d <= sqNorm_i_cos2 * sqNorms[rj])
              {
                convergent = true;
                break;
              }
            }
          }

          if (!convergent)
          {
            res.erase_angle = true;
            continue;
          }

          // --- Phase 2: Pixel residual filtering ---
          // Use push_back_unchecked to skip the O(n) find() per element
          // since we know keys are unique (copying from an existing SmallMap).
          Observations inlier_obs;
          inlier_obs.reserve(obs.size());
          IndexT local_outliers = 0;

          obs_idx = 0;
          for (const auto& obs_it : obs)
          {
            const ViewCache* vc = obs_vc[obs_idx++];
            if (vc)
            {
              // R*X + t avoids the R*(X-C) subtraction in Pose3::operator()
              const Vec3 Xc = vc->R * X + vc->t;
              const Vec2 residual = vc->intrinsic->residual(Xc, obs_it.second.x);
              if (residual.squaredNorm() > dThresholdPixelSq)
              {
                ++local_outliers;
                continue;
              }
            }
            inlier_obs.push_back_unchecked(obs_it);
          }

          res.pixel_outliers_removed = local_outliers;
          if (local_outliers > 0)
          {
            obs.swap(inlier_obs);
          }
          res.erase_pixel = (obs.empty() || obs.size() < minTrackLength);
        }
      }

      // --- Sequential erasure and accumulation ---
      IndexT angle_removed = 0;
      IndexT pixel_outlier_count = 0;
      for (int i = static_cast<int>(workItems.size()) - 1; i >= 0; --i)
      {
        const auto& res = results[i];
        if (res.erase_angle)
        {
          sfm_data.structure.erase(workItems[i]);
          ++angle_removed;
        }
        else
        {
          pixel_outlier_count += res.pixel_outliers_removed;
          if (res.erase_pixel)
            sfm_data.structure.erase(workItems[i]);
        }
      }
      return { angle_removed, pixel_outlier_count };
    }
  }
}
