// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp"

#include "openMVG/matching/cascade_hasher.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/matching/matching_filters.hpp"
#include "openMVG/matching/indMatchDecoratorXY.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/system/progressinterface.hpp"
#include "openMVG/system/timer.hpp"
#include "openMVG/types.hpp"

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

namespace openMVG {
namespace matching_image_collection {

using namespace openMVG::matching;
using namespace openMVG::features;

Cascade_Hashing_Matcher_Regions
::Cascade_Hashing_Matcher_Regions
(
  float distRatio
):Matcher(), f_dist_ratio_(distRatio)
{
}

namespace impl
{
template <typename ScalarT>
void Match
(
  const sfm::Regions_Provider & regions_provider,
  const Pair_Set & pairs,
  float fDistRatio,
  PairWiseMatchesContainer & map_PutativeMatches, // the pairwise photometric corresponding points
  system::ProgressInterface * my_progress_bar
)
{
  if (!my_progress_bar)
    my_progress_bar = &system::ProgressInterface::dummy();
  my_progress_bar->Restart(pairs.size(), "- Matching -");

  // Collect used view indexes
  std::set<IndexT> used_index;
  // Sort pairs according the first index to minimize later memory swapping
  using Map_vectorT = std::map<IndexT, std::vector<IndexT>>;
  Map_vectorT map_Pairs;
  for (const auto & pair_idx : pairs)
  {
    map_Pairs[pair_idx.first].push_back(pair_idx.second);
    used_index.insert(pair_idx.first);
    used_index.insert(pair_idx.second);
  }

  using BaseMat = Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  // Init the cascade hasher
  CascadeHasher cascade_hasher;
  if (!used_index.empty())
  {
    const IndexT I = *used_index.begin();
    const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
    const size_t dimension = regionsI->DescriptorLength();
    cascade_hasher.Init(dimension);
  }

  std::map<IndexT, HashedDescriptions> hashed_base_;

  // --- Diagnostics: feature counts ---
  {
    size_t totalFeatures = 0;
    size_t minFeatures = std::numeric_limits<size_t>::max();
    size_t maxFeatures = 0;
    for (const auto idx : used_index)
    {
      const size_t cnt = regions_provider.get(idx)->RegionCount();
      totalFeatures += cnt;
      if (cnt < minFeatures) minFeatures = cnt;
      if (cnt > maxFeatures) maxFeatures = cnt;
    }
    OPENMVG_LOG_INFO
      << "[Diag] Views: " << used_index.size()
      << ", Pairs: " << pairs.size()
      << ", Features — min: " << minFeatures
      << ", max: " << maxFeatures
      << ", avg: " << (used_index.empty() ? 0 : totalFeatures / used_index.size())
      << ", total: " << totalFeatures;
  }

#ifdef OPENMVG_USE_OPENMP
  OPENMVG_LOG_INFO << "[Diag] OpenMP max threads: " << omp_get_max_threads();
#else
  OPENMVG_LOG_INFO << "[Diag] OpenMP DISABLED — single threaded";
#endif

  // --- Phase 1: Zero mean descriptor ---
  system::Timer timer_phase;
  Eigen::VectorXf zero_mean_descriptor;
  {
    Eigen::MatrixXf matForZeroMean;
    for (int i =0; i < used_index.size(); ++i)
    {
      std::set<IndexT>::const_iterator iter = used_index.begin();
      std::advance(iter, i);
      const IndexT I = *iter;
      const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
      const ScalarT * tabI =
        reinterpret_cast<const ScalarT*>(regionsI->DescriptorRawData());
      const size_t dimension = regionsI->DescriptorLength();
      if (i==0)
      {
        matForZeroMean.resize(used_index.size(), dimension);
        matForZeroMean.fill(0.0f);
      }
      if (regionsI->RegionCount() > 0)
      {
        Eigen::Map<BaseMat> mat_I( (ScalarT*)tabI, regionsI->RegionCount(), dimension);
        matForZeroMean.row(i) = CascadeHasher::GetZeroMeanDescriptor(mat_I);
      }
    }
    zero_mean_descriptor = CascadeHasher::GetZeroMeanDescriptor(matForZeroMean);
  }
  OPENMVG_LOG_INFO << "[Diag] Phase 1 — Zero mean descriptor: " << timer_phase.elapsed() << " s";

  // --- Phase 2: Hashing ---
  timer_phase.reset();
#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (int i =0; i < used_index.size(); ++i)
  {
    std::set<IndexT>::const_iterator iter = used_index.begin();
    std::advance(iter, i);
    const IndexT I = *iter;
    const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
    const ScalarT * tabI =
      reinterpret_cast<const ScalarT*>(regionsI->DescriptorRawData());
    const size_t dimension = regionsI->DescriptorLength();

    Eigen::Map<BaseMat> mat_I( (ScalarT*)tabI, regionsI->RegionCount(), dimension);

    // Hash OUTSIDE the critical section — this is the expensive part
    HashedDescriptions hashed =
      cascade_hasher.CreateHashedDescriptions(mat_I, zero_mean_descriptor);

    // Only the map insertion needs serialization
#ifdef OPENMVG_USE_OPENMP
    #pragma omp critical
#endif
    {
      hashed_base_[I] = std::move(hashed);
    }
  }
  OPENMVG_LOG_INFO << "[Diag] Phase 2 — Hashing " << used_index.size() << " views: " << timer_phase.elapsed() << " s";

  // --- Phase 3: Matching ---
  timer_phase.reset();

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

  // Find max region count and estimate max candidates per query for scratch sizing.
  // Each query hits one bucket per group; the candidates per query is bounded
  // by the sum of the largest bucket sizes across groups — typically ~500-2000,
  // far smaller than the total region count (100K+).
  size_t maxRegionCount = 0;
  int maxCandidatesPerQuery = 0;
  for (const auto idx : used_index)
  {
    const std::shared_ptr<features::Regions> r = regions_provider.get(idx);
    const size_t rc = r->RegionCount();
    if (rc > maxRegionCount)
      maxRegionCount = rc;
  }
  for (const auto& kv : hashed_base_)
  {
    const auto& hd = kv.second;
    int viewTotal = 0;
    for (int g = 0; g < static_cast<int>(hd.buckets.size()); ++g)
    {
      int groupMax = 0;
      for (const auto& bucket : hd.buckets[g])
      {
        if (static_cast<int>(bucket.size()) > groupMax)
          groupMax = static_cast<int>(bucket.size());
      }
      viewTotal += groupMax;
    }
    if (viewTotal > maxCandidatesPerQuery)
      maxCandidatesPerQuery = viewTotal;
  }
  const size_t dimension = regions_provider.get(*used_index.begin())->DescriptorLength();
  const int nb_hash_code = static_cast<int>(dimension);

  OPENMVG_LOG_INFO
    << "[Diag] maxRegionCount: " << maxRegionCount
    << ", maxCandidatesPerQuery: " << maxCandidatesPerQuery
    << ", scratch matrix per thread: "
    << (maxCandidatesPerQuery * (nb_hash_code + 1) * 4 / 1024) << " KB"
    << " (was " << (maxRegionCount * (nb_hash_code + 1) * 4 / 1048576) << " MB)";

#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel
#endif
  {
    // --- Per-thread scratch: allocated once, reused for every task ---
    std::vector<int> scratch_candidates;
    scratch_candidates.reserve(maxCandidatesPerQuery);
    std::vector<std::pair<typename Accumulator<ScalarT>::Type, int>> scratch_euclidean;
    scratch_euclidean.reserve(10);
    std::vector<char> scratch_used(maxRegionCount, 0);

    // Per-thread output buffers — avoid 2.5 MB heap alloc per pair
    IndMatches pvec_indices;
    using ResultType = typename Accumulator<ScalarT>::Type;
    std::vector<ResultType> pvec_distances;

#ifdef OPENMVG_USE_OPENMP
    #pragma omp for schedule(dynamic)
#endif
    for (int t = 0; t < numTasks; ++t)
    {
      if (my_progress_bar->hasBeenCanceled())
        continue;

      const IndexT I = all_tasks[t].I;
      const IndexT J = all_tasks[t].J;

      const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
      if (regionsI->RegionCount() == 0)
      {
        ++(*my_progress_bar);
        continue;
      }

      const std::shared_ptr<features::Regions> regionsJ = regions_provider.get(J);

      if (regionsI->Type_id() != regionsJ->Type_id())
      {
        ++(*my_progress_bar);
        continue;
      }

      const ScalarT * tabI =
        reinterpret_cast<const ScalarT*>(regionsI->DescriptorRawData());
      Eigen::Map<BaseMat> mat_I( (ScalarT*)tabI, regionsI->RegionCount(), dimension);

      const ScalarT * tabJ = reinterpret_cast<const ScalarT*>(regionsJ->DescriptorRawData());
      Eigen::Map<BaseMat> mat_J( (ScalarT*)tabJ, regionsJ->RegionCount(), dimension);

      // Reuse per-thread buffers — clear, don't reallocate
      pvec_indices.clear();
      pvec_distances.clear();

      cascade_hasher.Match_HashedDescriptions<BaseMat, ResultType>(
        hashed_base_.at(J), mat_J,
        hashed_base_.at(I), mat_I,
        &pvec_indices, &pvec_distances,
        2,
        scratch_candidates,
        scratch_euclidean,
        scratch_used);

      std::vector<int> vec_nn_ratio_idx;
      matching::NNdistanceRatio(
        pvec_distances.begin(),
        pvec_distances.end(),
        2,
        vec_nn_ratio_idx,
        Square(fDistRatio));

      matching::IndMatches vec_putative_matches;
      vec_putative_matches.reserve(vec_nn_ratio_idx.size());
      for (size_t k=0; k < vec_nn_ratio_idx.size(); ++k)
      {
        const size_t index = vec_nn_ratio_idx[k];
        vec_putative_matches.emplace_back(pvec_indices[index*2].j_, pvec_indices[index*2].i_);
      }

      matching::IndMatch::getDeduplicated(vec_putative_matches);

      // Only do the expensive XY dedup if there are enough matches to justify
      // copying 107K PointFeatures. For small match sets the index dedup above
      // is sufficient — XY duplicates are extremely rare.
      if (vec_putative_matches.size() > 2)
      {
        const std::vector<features::PointFeature> pointFeaturesI = regionsI->GetRegionsPositions();
        const std::vector<features::PointFeature> pointFeaturesJ = regionsJ->GetRegionsPositions();
        matching::IndMatchDecorator<float> matchDeduplicator(vec_putative_matches,
          pointFeaturesI, pointFeaturesJ);
        matchDeduplicator.getDeduplicated(vec_putative_matches);
      }
      if (!vec_putative_matches.empty())
      {
#ifdef OPENMVG_USE_OPENMP
        #pragma omp critical
#endif
        {
          map_PutativeMatches.insert(
            {
              {I,J},
              std::move(vec_putative_matches)
            });
        }
      }
      ++(*my_progress_bar);
    }
  } // end omp parallel
  OPENMVG_LOG_INFO << "[Diag] Phase 3 — Matching " << numTasks << " pairs: " << timer_phase.elapsed() << " s";
}
} // namespace impl

void Cascade_Hashing_Matcher_Regions::Match
(
  const std::shared_ptr<sfm::Regions_Provider> & regions_provider,
  const Pair_Set & pairs,
  PairWiseMatchesContainer & map_PutativeMatches, // the pairwise photometric corresponding points
  system::ProgressInterface * my_progress_bar
)const
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
    impl::Match<unsigned char>(
      *regions_provider.get(),
      pairs,
      f_dist_ratio_,
      map_PutativeMatches,
      my_progress_bar);
  }
  else
  if (regions_provider->Type_id() == typeid(float).name())
  {
    impl::Match<float>(
      *regions_provider.get(),
      pairs,
      f_dist_ratio_,
      map_PutativeMatches,
      my_progress_bar);
  }
  else
  {
    OPENMVG_LOG_ERROR << "Matcher not implemented for this region type: " << regions_provider->Type_id();
  }
}

} // namespace openMVG
} // namespace matching_image_collection
