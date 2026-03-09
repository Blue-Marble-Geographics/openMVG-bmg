// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Implementation of [1] an efficient algorithm to compute track from pairwise
//  correspondences.
//
//  [1] Pierre Moulon and Pascal Monasse,
//    "Unordered feature tracking made fast and easy" CVMP 2012.
//
// It tracks the position of features along the series of image from pairwise
//  correspondences.
//
// From map<[imageI,ImageJ], [indexed matches array] > it builds tracks.
//
// Usage :
//  PairWiseMatches map_Matches;
//  PairedIndMatchImport(sMatchFile, map_Matches); // Load series of pairwise matches
//  //---------------------------------------
//  // Compute tracks from matches
//  //---------------------------------------
//  TracksBuilder tracksBuilder;
//  tracks::STLMAPTracks map_tracks;
//  tracksBuilder.Build(map_Matches); // Build: Efficient fusion of correspondences
//  tracksBuilder.Filter();           // Filter: Remove tracks that have conflict
//  tracksBuilder.ExportToSTL(map_tracks); // Build tracks with STL compliant type
//

#ifndef OPENMVG_TRACKS_TRACKS_HPP
#define OPENMVG_TRACKS_TRACKS_HPP

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "openMVG/matching/indMatch.hpp"
#include "openMVG/tracks/flat_pair_map.hpp"
#include "openMVG/tracks/union_find.hpp"

namespace openMVG  {

namespace tracks  {

// Data structure to store a track: collection of {ImageId,FeatureId}
//  The corresponding image points with their imageId and FeatureId.
using submapTrack = std::map<uint32_t, uint32_t>;
// A track is a collection of {trackId, submapTrack}
using STLMAPTracks = std::map<uint32_t, submapTrack>;

struct TracksBuilder
{
  using indexedFeaturePair = std::pair<uint32_t, uint32_t>;

  flat_pair_map<indexedFeaturePair, uint32_t> map_node_to_index;
  UnionFind uf_tree;

  /// Build tracks for a given series of pairWise matches
  void Build( const matching::PairWiseMatches &  map_pair_wise_matches)
  {
    // 1. We need to know how much single set we will have.
    //   i.e each set is made of a tuple : (imageIndex, featureIndex)
    std::set<indexedFeaturePair> allFeatures;
    // For each couple of images list the used features
    for ( const auto & iter : map_pair_wise_matches )
    {
      const auto & I = iter.first.first;
      const auto & J = iter.first.second;
      const std::vector<matching::IndMatch> & vec_FilteredMatches = iter.second;

      // Retrieve all shared features and add them to a set
      for ( const auto & cur_filtered_match : vec_FilteredMatches )
      {
        allFeatures.emplace(I,cur_filtered_match.i_);
        allFeatures.emplace(J,cur_filtered_match.j_);
      }
    }

    // 2. Build the 'flat' representation where a tuple (the node)
    //  is attached to a unique index.
    map_node_to_index.reserve(allFeatures.size());
    uint32_t cpt = 0;
    for (const auto & feat : allFeatures)
    {
      map_node_to_index.emplace_back(feat, cpt);
      ++cpt;
    }
    // Sort the flat_pair_map
    map_node_to_index.sort();
    // Clean some memory
    allFeatures.clear();

    // 3. Add the node and the pairwise correspondences in the UF tree.
    uf_tree.InitSets(map_node_to_index.size());

    // 4. Union of the matched features corresponding UF tree sets
    for ( const auto & iter : map_pair_wise_matches )
    {
      const auto & I = iter.first.first;
      const auto & J = iter.first.second;
      const std::vector<matching::IndMatch> & vec_FilteredMatches = iter.second;
      for (const matching::IndMatch & match : vec_FilteredMatches)
      {
        const indexedFeaturePair pairI(I, match.i_);
        const indexedFeaturePair pairJ(J, match.j_);
        // Link feature correspondences to the corresponding containing sets.
        uf_tree.Union(map_node_to_index[pairI], map_node_to_index[pairJ]);
      }
    }
  }

  /// Remove bad tracks (too short or track with ids collision)
  bool Filter(uint32_t nLengthSupTo = 2)
  {
    // Build the Track observations & mark tracks that have id collision:
    std::map<uint32_t, std::set<uint32_t>> tracks; // {track_id, {image_id, image_id, ...}}
    std::set<uint32_t> problematic_track_id; // {track_id, ...}

    // For each node retrieve its track id from the UF tree and add the node to the track
    // - if an image id is observed multiple time, then mark the track as invalid
    //   - a track cannot list many times the same image index
    for (uint32_t k = 0; k < map_node_to_index.size(); ++k)
    {
      const uint32_t & track_id = uf_tree.Find(k);
      const auto & feat = map_node_to_index[k];

      // Augment the track and mark if invalid (an image can only be listed once)
      if (tracks[track_id].insert(feat.first.first).second == false)
      {
        problematic_track_id.insert(track_id); // invalid
      }
    }

    // Reject tracks that have too few observations
    for (const auto & val : tracks)
    {
      if (val.second.size() < nLengthSupTo)
      {
        problematic_track_id.insert(val.first);
      }
    }

    // Reset the marked invalid track ids in the UF Tree
    for (uint32_t & root_index : uf_tree.m_cc_parent)
    {
      if (problematic_track_id.count(root_index) > 0)
      {
        // reset selected root
        uf_tree.m_cc_size[root_index] = 1;
        root_index = std::numeric_limits<uint32_t>::max();
      }
    }
    return false;
  }

  /// Return the number of connected set in the UnionFind structure (tree forest)
  size_t NbTracks() const
  {
    std::set<uint32_t> parent_id(uf_tree.m_cc_parent.cbegin(), uf_tree.m_cc_parent.cend());
    // Erase the "special marker" that depicted rejected tracks
    parent_id.erase(std::numeric_limits<uint32_t>::max());
    return parent_id.size();
  }

  /// Export tracks as a map (each entry is a sequence of imageId and featureIndex):
  ///  {TrackIndex => {(imageIndex, featureIndex), ... ,(imageIndex, featureIndex)}
  void ExportToSTL(STLMAPTracks & map_tracks)
  {
    map_tracks.clear();
    for (uint32_t k = 0; k < map_node_to_index.size(); ++k)
    {
      const auto & feat = map_node_to_index[k];
      const uint32_t & track_id = uf_tree.m_cc_parent[k];
      if
      (
        // ensure never add rejected elements (track marked as invalid)
        track_id != std::numeric_limits<uint32_t>::max()
        // ensure never add 1-length track element (it's not a track)
        && uf_tree.m_cc_size[track_id] > 1
      )
      {
        map_tracks[track_id].insert(feat.first);
      }
    }
  }
};

// This structure help to store the track visibility per view.
// Computing the tracks in common between many view can then be done
//  by computing the intersection of the track visibility for the asked view index.
// Thank to an additional array in memory this solution is faster than TracksUtilsMap::GetTracksInImages.
struct SharedTrackVisibilityHelper
{
private:
  // Sorted vectors are much cheaper than std::set for iteration and
  // binary-search lookups, and avoid per-element heap allocations.
  using TrackIdsPerView = std::map<uint32_t, std::vector<uint32_t>>;

  TrackIdsPerView track_ids_per_view_;
  const STLMAPTracks & tracks_;

public:

  explicit SharedTrackVisibilityHelper
  (
    const STLMAPTracks & tracks
  ): tracks_(tracks)
  {
    // First pass: collect unsorted
    for (const auto & tracks_it : tracks_)
    {
      for (const auto & track_obs_it : tracks_it.second)
      {
        track_ids_per_view_[track_obs_it.first].push_back(tracks_it.first);
      }
    }
    // Second pass: sort & deduplicate each per-view vector
    for (auto & per_view : track_ids_per_view_)
    {
      auto & vec = per_view.second;
      std::sort(vec.begin(), vec.end());
      vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
      vec.shrink_to_fit();
    }
  }

  /**
   * @brief Find the shared tracks between some images ids.
   *
   * @param[in] image_ids: images id to consider
   * @param[out] tracks: tracks shared by the input images id
   */
  bool GetTracksInImages
  (
    const std::set<uint32_t>& image_ids,
    STLMAPTracks& tracks
  )
  {
    tracks.clear();
    if (image_ids.empty())
      return false;

    std::vector<const std::vector<uint32_t>*> per_view_vecs;
    per_view_vecs.reserve(image_ids.size());

    for (const auto & image_id : image_ids)
    {
      const auto ids_per_view_it = track_ids_per_view_.find(image_id);
      if (ids_per_view_it == track_ids_per_view_.end())
      {
        if (image_ids.size() > 1)
          return false;
        continue;
      }
      per_view_vecs.push_back(&ids_per_view_it->second);
    }

    if (per_view_vecs.empty())
      return false;

    // Sort pointers by vector size so the smallest is iterated first
    std::sort(
      per_view_vecs.begin(),
      per_view_vecs.end(),
      [](const std::vector<uint32_t>* a, const std::vector<uint32_t>* b)
      {
        return a->size() < b->size();
      });

    const std::vector<uint32_t>& candidate_track_ids = *per_view_vecs[0];

    for (const uint32_t track_id : candidate_track_ids)
    {
      bool present_in_all = true;
      for (size_t i = 1; i < per_view_vecs.size(); ++i)
      {
        if (!std::binary_search(per_view_vecs[i]->begin(), per_view_vecs[i]->end(), track_id))
        {
          present_in_all = false;
          break;
        }
      }

      if (!present_in_all)
        continue;

      const auto track_it = tracks_.find(track_id);
      if (track_it == tracks_.end())
        continue;

      const auto& track = track_it->second;
      submapTrack& trackFeatsOut = tracks[track_id];

      for (const auto & img_id : image_ids)
      {
        const auto track_view_info = track.find(img_id);
        if (track_view_info != track.end())
        {
          trackFeatsOut[img_id] = track_view_info->second;
        }
      }
    }

    return !tracks.empty();
  }

  /**
   * @brief Flat output for single-view or multi-view track queries.
   *
   * Instead of building a nested std::map<uint32_t, std::map<...>>
   * (which does thousands of heap allocations), this returns two
   * parallel sorted vectors: track_ids and feat_ids.
   *
   * For single-view queries this is O(N) with zero heap allocation
   * beyond the vector growth (which callers can pre-reserve).
   *
   * @param[in]  image_ids  images to consider
   * @param[out] track_ids  sorted track ids visible in all requested images
   * @param[out] feat_ids   per-image feature ids, interleaved in image_ids order:
   *                         for single image: feat_ids[i] is the feature for track_ids[i]
   *                         for N images: feat_ids[i*N + j] is feat for track_ids[i], image j
   * @return true if any tracks found
   */
  bool GetTracksInImages
  (
    const std::set<uint32_t>& image_ids,
    std::vector<uint32_t>& track_ids,
    std::vector<uint32_t>& feat_ids
  )
  {
    track_ids.clear();
    feat_ids.clear();
    if (image_ids.empty())
      return false;

    const size_t num_images = image_ids.size();

    std::vector<const std::vector<uint32_t>*> per_view_vecs;
    per_view_vecs.reserve(num_images);

    for (const auto & image_id : image_ids)
    {
      const auto it = track_ids_per_view_.find(image_id);
      if (it == track_ids_per_view_.end())
      {
        if (num_images > 1)
          return false;
        continue;
      }
      per_view_vecs.push_back(&it->second);
    }

    if (per_view_vecs.empty())
      return false;

    // For single-view queries, just copy the track list and look up features
    if (num_images == 1)
    {
      const auto & candidate_ids = *per_view_vecs[0];
      const uint32_t image_id = *image_ids.begin();
      track_ids.reserve(candidate_ids.size());
      feat_ids.reserve(candidate_ids.size());

      for (const uint32_t track_id : candidate_ids)
      {
        const auto track_it = tracks_.find(track_id);
        if (track_it == tracks_.end())
          continue;
        const auto obs_it = track_it->second.find(image_id);
        if (obs_it == track_it->second.end())
          continue;
        track_ids.push_back(track_id);
        feat_ids.push_back(obs_it->second);
      }
      return !track_ids.empty();
    }

    // Multi-view: find intersection, then collect features
    // Sort pointers by size
    // We need to remember which image_id maps to which per_view_vec
    // Since image_ids is a set (sorted), and we inserted in order,
    // per_view_vecs[i] corresponds to the i-th element of image_ids.
    // But we need to sort by size for intersection efficiency.
    std::vector<size_t> sorted_indices(per_view_vecs.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
      [&per_view_vecs](size_t a, size_t b) {
        return per_view_vecs[a]->size() < per_view_vecs[b]->size();
      });

    const auto & candidate_ids = *per_view_vecs[sorted_indices[0]];
    track_ids.reserve(candidate_ids.size());
    feat_ids.reserve(candidate_ids.size() * num_images);

    // Build a quick lookup: image_ids as a vector for indexed access
    std::vector<uint32_t> image_ids_vec(image_ids.begin(), image_ids.end());

    for (const uint32_t track_id : candidate_ids)
    {
      bool present_in_all = true;
      for (size_t i = 1; i < sorted_indices.size(); ++i)
      {
        if (!std::binary_search(
              per_view_vecs[sorted_indices[i]]->begin(),
              per_view_vecs[sorted_indices[i]]->end(),
              track_id))
        {
          present_in_all = false;
          break;
        }
      }
      if (!present_in_all)
        continue;

      const auto track_it = tracks_.find(track_id);
      if (track_it == tracks_.end())
        continue;

      // Collect feature ids for each requested image
      bool all_found = true;
      size_t base = feat_ids.size();
      for (size_t j = 0; j < num_images; ++j)
      {
        const auto obs_it = track_it->second.find(image_ids_vec[j]);
        if (obs_it != track_it->second.end())
        {
          feat_ids.push_back(obs_it->second);
        }
        else
        {
          all_found = false;
          feat_ids.resize(base); // rollback
          break;
        }
      }
      if (all_found)
      {
        track_ids.push_back(track_id);
      }
    }

    return !track_ids.empty();
  }
};

struct TracksUtilsMap
{
  /**
   * @brief Find common tracks between images.
   *
   * @param[in] set_imageIndex: set of images we are looking for common tracks
   * @param[in] map_tracksIn: all tracks of the scene
   * @param[out] map_tracksOut: output with only the common tracks
   */
  static bool GetTracksInImages
  (
    const std::set<uint32_t> & set_imageIndex,
    const STLMAPTracks & map_tracksIn,
    STLMAPTracks & map_tracksOut
  )
  {
    map_tracksOut.clear();

    // Go along the tracks
    for ( const auto & iterT : map_tracksIn )
    {
      // Look if the track contains the provided view index & save the point ids
      submapTrack map_temp;
      bool bTest = true;
      for (auto iterIndex = set_imageIndex.begin();
        iterIndex != set_imageIndex.end() && bTest; ++iterIndex)
      {
        auto iterSearch = iterT.second.find(*iterIndex);
        if (iterSearch != iterT.second.end())
          map_temp[iterSearch->first] = iterSearch->second;
        else
          bTest = false;
      }

      if (!map_temp.empty() && map_temp.size() == set_imageIndex.size())
        map_tracksOut[iterT.first] = std::move(map_temp);
    }
    return !map_tracksOut.empty();
  }

  /// Return the tracksId as a set (sorted increasing)
  static void GetTracksIdVector
  (
    const STLMAPTracks & map_tracks,
    std::set<uint32_t> * set_tracksIds
  )
  {
    set_tracksIds->clear();
    for ( const auto & iterT : map_tracks )
    {
      set_tracksIds->insert(iterT.first);
    }
  }

  /// Get feature index PerView and TrackId
  static bool GetFeatIndexPerViewAndTrackId
  (
    const STLMAPTracks & tracks,
    const std::set<uint32_t> & track_ids,
    uint32_t nImageIndex,
    std::vector<uint32_t> * feat_ids
  )
  {
    feat_ids->reserve(tracks.size());
    for (const uint32_t & trackId: track_ids)
    {
      const auto iterT = tracks.find(trackId);
      if (iterT != tracks.end())
      {
        // Look if the desired image index exists in the track visibility
        const auto iterSearch = iterT->second.find(nImageIndex);
        if (iterSearch != iterT->second.end())
        {
          feat_ids->emplace_back(iterSearch->second);
        }
      }
    }
    return !feat_ids->empty();
  }

  /// Return the occurrence of tracks length.
  static void TracksLength
  (
    const STLMAPTracks & map_tracks,
    std::map<uint32_t, uint32_t> & map_Occurrence_TrackLength
  )
  {
    for ( const auto & iterT : map_tracks )
    {
      const size_t trLength = iterT.second.size();
      if (map_Occurrence_TrackLength.count(trLength) == 0)
      {
        map_Occurrence_TrackLength[trLength] = 1;
      }
      else
      {
        map_Occurrence_TrackLength[trLength] += 1;
      }
    }
  }

  /// Return a set containing the image Id considered in the tracks container.
  static void ImageIdInTracks
  (
    const STLMAPTracks & map_tracks,
    std::set<uint32_t> & set_imagesId
  )
  {
    for ( const auto & iterT : map_tracks )
    {
      const submapTrack & map_ref = iterT.second;
      for ( const auto & iter : map_ref )
      {
        set_imagesId.insert(iter.first);
      }
    }
  }
};

} // namespace tracks
} // namespace openMVG

#endif // OPENMVG_TRACKS_TRACKS_HPP
