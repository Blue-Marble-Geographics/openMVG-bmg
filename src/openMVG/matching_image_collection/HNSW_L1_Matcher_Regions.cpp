// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/matching_image_collection/HNSW_L1_Matcher_Regions.hpp"

#include "openMVG/features/feature.hpp"
#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching/indMatchDecoratorXY.hpp"
#include "openMVG/matching/matching_filters.hpp"
#include "openMVG/matching/matcher_hnsw.hpp"
#include "openMVG/matching/metric.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/system/progressinterface.hpp"
#include "openMVG/types.hpp"

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace openMVG {
namespace matching_image_collection {

using namespace openMVG::matching;
using namespace openMVG::features;

HNSW_L1_Matcher_Regions::HNSW_L1_Matcher_Regions
(
  float distRatio
): Matcher(), f_dist_ratio_(distRatio)
{
}

namespace impl_hnsw_l1
{

void Match
(
  const sfm::Regions_Provider & regions_provider,
  const Pair_Set & pairs,
  float fDistRatio,
  PairWiseMatchesContainer & map_PutativeMatches,
  system::ProgressInterface * my_progress_bar
)
{
  using ScalarT = unsigned char;
  using MetricT = L1<ScalarT>;
  using DistanceType = typename MetricT::ResultType; // int

  if (!my_progress_bar)
    my_progress_bar = &system::ProgressInterface::dummy();
  my_progress_bar->Restart(pairs.size(), "- Matching (HNSW L1) -");

  // Collect all used view indices
  std::set<IndexT> used_index;
  using Map_vectorT = std::map<IndexT, std::vector<IndexT>>;
  Map_vectorT map_Pairs;
  for (const auto & pair_idx : pairs)
  {
    map_Pairs[pair_idx.first].push_back(pair_idx.second);
    used_index.insert(pair_idx.first);
    used_index.insert(pair_idx.second);
  }

  // Convert to a vector for indexed parallel access
  const std::vector<IndexT> used_index_vec(used_index.begin(), used_index.end());
  const int num_views = static_cast<int>(used_index_vec.size());

  if (num_views == 0)
    return;

  // --- Pre-fetch all region data (read-only after this point) ---
  struct ViewData {
    std::shared_ptr<features::Regions> regions;
    const ScalarT* descriptors;
    size_t regionCount;
    size_t dimension;
  };
  std::map<IndexT, ViewData> view_data;
  for (const auto idx : used_index)
  {
    ViewData vd;
    vd.regions = regions_provider.get(idx);
    vd.descriptors = reinterpret_cast<const ScalarT*>(vd.regions->DescriptorRawData());
    vd.regionCount = vd.regions->RegionCount();
    vd.dimension = vd.regions->DescriptorLength();
    view_data[idx] = std::move(vd);
  }

  // --- Build one HNSW index per view, all in parallel ---
  // Each HNSWMatcher is independent and thread-safe for building.
  using HNSWMatcherT = HNSWMatcher<ScalarT, MetricT, HNSWMETRIC::L1_HNSW>;
  std::map<IndexT, std::unique_ptr<HNSWMatcherT>> hnsw_indices;

  // Prepare storage indexed by position for parallel fill
  std::unique_ptr<std::pair<IndexT, std::unique_ptr<HNSWMatcherT>>[]>
    build_results(new std::pair<IndexT, std::unique_ptr<HNSWMatcherT>>[num_views]);

  OPENMVG_LOG_INFO << "Building " << num_views << " HNSW L1 indices in parallel...";

#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 0; i < num_views; ++i)
  {
    const IndexT idx = used_index_vec[i];
    const ViewData& vd = view_data.at(idx);
    auto matcher = std::unique_ptr<HNSWMatcherT>(new HNSWMatcherT());
    if (vd.regionCount > 0)
    {
      matcher->Build(vd.descriptors,
                     static_cast<int>(vd.regionCount),
                     static_cast<int>(vd.dimension));
    }
    build_results[i] = {idx, std::move(matcher)};
  }

  // Collect built indices into the map (single-threaded, fast)
  for (int i = 0; i < num_views; ++i)
  {
    auto& r = build_results[i];
    hnsw_indices[r.first] = std::move(r.second);
  }

  OPENMVG_LOG_INFO << "HNSW L1 index construction complete.";

  // --- Flatten all (I, J) pairs for fine-grained parallel scheduling ---
  struct PairTask {
    IndexT I;
    IndexT J;
  };
  std::vector<PairTask> all_tasks;
  all_tasks.reserve(pairs.size());
  for (const auto& pair_it : map_Pairs)
  {
    for (const auto J : pair_it.second)
    {
      all_tasks.push_back({pair_it.first, J});
    }
  }
  const int numTasks = static_cast<int>(all_tasks.size());

  // --- Match all pairs in parallel ---
  // Each thread queries the prebuilt HNSW index for view I with
  // the descriptors of view J, finding 2-NN for distance ratio test.
  // HNSW searchKnn is thread-safe for concurrent reads.
  // The only critical section is the final insert into map_PutativeMatches.

#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (int t = 0; t < numTasks; ++t)
  {
    if (my_progress_bar->hasBeenCanceled())
      continue;

    const IndexT I = all_tasks[t].I;
    const IndexT J = all_tasks[t].J;

    const ViewData& vdI = view_data.at(I);
    const ViewData& vdJ = view_data.at(J);

    if (vdI.regionCount == 0 || vdJ.regionCount == 0)
    {
      ++(*my_progress_bar);
      continue;
    }

    if (vdI.regions->Type_id() != vdJ.regions->Type_id())
    {
      ++(*my_progress_bar);
      continue;
    }

    // Query the HNSW index of view I with all descriptors of view J
    const auto& hnsw_I = hnsw_indices.at(I);
    const size_t NN = 2;

    IndMatches nn_matches;
    std::vector<DistanceType> nn_distances;
    nn_matches.resize(vdJ.regionCount * NN);
    nn_distances.resize(vdJ.regionCount * NN);

    // HNSW searchKnn is thread-safe for concurrent read access.
    // We call it per-descriptor to avoid internal OpenMP nesting issues.
    for (size_t q = 0; q < vdJ.regionCount; ++q)
    {
      const ScalarT* query = vdJ.descriptors + q * vdJ.dimension;
      auto result = hnsw_I->SearchKnn(query, NN);

      size_t result_id = NN - 1;
      while (!result.empty())
      {
        const auto& res = result.top();
        const size_t match_id = q * NN + result_id;
        nn_matches[match_id] = {static_cast<IndexT>(q), static_cast<IndexT>(res.second)};
        nn_distances[match_id] = res.first;
        result.pop();
        if (result_id > 0) --result_id;
      }
    }

    // Distance ratio filtering (L1 is not squared, so use fDistRatio directly)
    std::vector<int> vec_nn_ratio_idx;
    NNdistanceRatio(
      nn_distances.cbegin(),
      nn_distances.cend(),
      2,
      vec_nn_ratio_idx,
      fDistRatio);

    IndMatches vec_putative_matches;
    vec_putative_matches.reserve(vec_nn_ratio_idx.size());
    for (const auto & index : vec_nn_ratio_idx)
    {
      vec_putative_matches.emplace_back(
        nn_matches[index * 2].j_,  // database (I) index
        nn_matches[index * 2].i_); // query (J) index
    }

    // Remove duplicates
    IndMatch::getDeduplicated(vec_putative_matches);

    // Remove matches that have the same (X,Y) coordinates
    const std::vector<PointFeature> pointFeaturesI = vdI.regions->GetRegionsPositions();
    const std::vector<PointFeature> pointFeaturesJ = vdJ.regions->GetRegionsPositions();
    IndMatchDecorator<float> matchDeduplicator(vec_putative_matches,
      pointFeaturesI, pointFeaturesJ);
    matchDeduplicator.getDeduplicated(vec_putative_matches);

    if (!vec_putative_matches.empty())
    {
#ifdef OPENMVG_USE_OPENMP
      #pragma omp critical
#endif
      {
        map_PutativeMatches.insert(
          {
            {I, J},
            std::move(vec_putative_matches)
          });
      }
    }
    ++(*my_progress_bar);
  }
}

} // namespace impl_hnsw_l1

void HNSW_L1_Matcher_Regions::Match
(
  const std::shared_ptr<sfm::Regions_Provider> & regions_provider,
  const Pair_Set & pairs,
  PairWiseMatchesContainer & map_PutativeMatches,
  system::ProgressInterface * my_progress_bar
) const
{
#ifdef OPENMVG_USE_OPENMP
  OPENMVG_LOG_INFO << "Using the OPENMP thread interface";
#endif
  if (!regions_provider)
    return;

  if (regions_provider->IsBinary())
    return;

  if (regions_provider->Type_id() == typeid(unsigned char).name())
  {
    impl_hnsw_l1::Match(
      *regions_provider.get(),
      pairs,
      f_dist_ratio_,
      map_PutativeMatches,
      my_progress_bar);
  }
  else
  {
    OPENMVG_LOG_ERROR << "HNSW L1 matcher only supports unsigned char (uint8) regions. "
                      << "Got: " << regions_provider->Type_id();
  }
}

} // namespace matching_image_collection
} // namespace openMVG