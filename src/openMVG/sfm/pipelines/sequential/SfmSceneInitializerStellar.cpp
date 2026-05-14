// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2018 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/pipelines/sequential/SfmSceneInitializerStellar.hpp"

#include "openMVG/sfm/pipelines/relative_pose_engine.hpp"
#include "openMVG/sfm/pipelines/sfm_matches_provider.hpp"
#include "openMVG/sfm/pipelines/stellar/stellar_solver.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/stl/stlMap.hpp"
#include "openMVG/system/logger.hpp"

namespace openMVG {

using namespace matching;

namespace sfm {

SfMSceneInitializerStellar::SfMSceneInitializerStellar(
  SfM_Data & sfm_data,
  const Features_Provider * features_provider,
  const Matches_Provider * matches_provider)
  :SfMSceneInitializer(sfm_data, features_provider, matches_provider)
{
  sfm_data_.poses.clear();
}

bool find_largest_stellar_configuration
(
  const Pair_Set & pairs,
  const Matches_Provider * matches_provider,
  Pair_Set & stellar_pod
)
{
  // List all possible stellar configurations
  using StellarPods = Hash_Map<IndexT, Pair_Set>;
  StellarPods stellar_pods;
  for (const auto & it : pairs)
  {
    stellar_pods[it.first].insert(it);
    stellar_pods[it.second].insert(it);
  }

  // Find the stellar configuration with the most correspondences support
  std::vector<float> matches_count_per_stellar_pod(stellar_pods.size(), 0);
  IndexT id = 0;
  for (const auto & stellar_pod_it : stellar_pods)
  {
    const Pair_Set & pairs = stellar_pod_it.second;
    for (const auto & pair_it : pairs)
    {
      const matching::IndMatches & matches = matches_provider->pairWise_matches_.at(pair_it);
      matches_count_per_stellar_pod[id] += matches.size();
    }
    matches_count_per_stellar_pod[id] /= pairs.size();
    ++id;
  }
  const auto stellar_pod_max_matches_iterator =
    std::max_element(matches_count_per_stellar_pod.cbegin(),
                     matches_count_per_stellar_pod.cend());

  if (stellar_pod_max_matches_iterator == matches_count_per_stellar_pod.cend())
    return false;

  auto stellar_pod_it =
    std::next(stellar_pods.cbegin(),
      std::distance(matches_count_per_stellar_pod.cbegin(),
                    stellar_pod_max_matches_iterator));

  stellar_pod = stellar_pod_it->second;
  return true;
}

bool SfMSceneInitializerStellar::Process()
{
  if (sfm_data_.GetIntrinsics().empty())
  {
    OPENMVG_LOG_ERROR << "Stellar init failed: no intrinsics defined.";
    return false;
  }

  // List the pairs that are valid for relative pose estimation:
  // - Each pair must have different pose ids and defined intrinsic data.
  const Pair_Set pairs = matches_provider_->getPairs();
  OPENMVG_LOG_INFO << "Stellar init: total match pairs = " << pairs.size();

  Pair_Set relative_pose_pairs;
  for (const auto & pair_it : pairs)
  {
    const View * v1 = sfm_data_.GetViews().at(pair_it.first).get();
    const View * v2 = sfm_data_.GetViews().at(pair_it.second).get();
    if (v1->id_pose != v2->id_pose
        && sfm_data_.GetIntrinsics().count(v1->id_intrinsic)
        && sfm_data_.GetIntrinsics().count(v2->id_intrinsic))
      relative_pose_pairs.insert({v1->id_pose, v2->id_pose});
  }
  OPENMVG_LOG_INFO << "Stellar init: valid relative pose pairs = " << relative_pose_pairs.size();

  if (relative_pose_pairs.empty())
  {
    OPENMVG_LOG_ERROR << "Stellar init failed: no valid relative pose pairs."
      << " Check that matches exist and views have valid intrinsics.";
    return false;
  }

  // Find the stellar configuration with the most pair candidate.
  Pair_Set selected_putative_stellar_pod;
  if (!find_largest_stellar_configuration(
        relative_pose_pairs,
        matches_provider_,
        selected_putative_stellar_pod))
  {
    OPENMVG_LOG_ERROR << "Stellar init failed: unable to find a valid stellar configuration"
      << " from " << relative_pose_pairs.size() << " relative pose pairs.";
    return false;
  }
  OPENMVG_LOG_INFO << "Stellar init: selected putative pod has " << selected_putative_stellar_pod.size() << " pairs.";

  // Compute a relative pose for each selected edge of the pose pair graph
  const Relative_Pose_Engine::Relative_Pair_Poses relative_poses = [&]
  {
    Relative_Pose_Engine relative_pose_engine;
    if (!relative_pose_engine.Process(
          selected_putative_stellar_pod,
          sfm_data_,
          matches_provider_,
          features_provider_))
      return Relative_Pose_Engine::Relative_Pair_Poses();
    else
      return relative_pose_engine.Get_Relative_Poses();
  }();

  OPENMVG_LOG_INFO << "Stellar init: computed " << relative_poses.size()
    << " relative poses from " << selected_putative_stellar_pod.size() << " pairs.";

  if (relative_poses.empty())
  {
    OPENMVG_LOG_ERROR << "Stellar init failed: relative pose estimation produced no valid poses.";
    return false;
  }

  Pair_Set relative_poses_pairs;
  // Retrieve all keys
  std::transform(relative_poses.begin(), relative_poses.end(),
    std::inserter(relative_poses_pairs, relative_poses_pairs.begin()),
    stl::RetrieveKey());

  Pair_Set selected_stellar_pod;
  if (!find_largest_stellar_configuration(
        relative_pose_pairs,
        matches_provider_,
        selected_stellar_pod))
  {
    OPENMVG_LOG_ERROR << "Stellar init failed: unable to select a valid stellar configuration"
      << " from the computed relative poses (" << relative_poses.size() << " poses).";
    return false;
  }

  OPENMVG_LOG_INFO << "The chosen stellar pod has " << selected_stellar_pod.size() << " pairs.";

  // Configure the stellar pod optimization to use all the matches relating to the considered view id.
  // The found camera poses will be better thanks to the larger point support.
  const bool use_all_matches = true;
  const bool use_threading = true;
  Stellar_Solver stellar_pod_solver(
    selected_stellar_pod,
    relative_poses,
    sfm_data_,
    matches_provider_,
    features_provider_,
    use_all_matches,
    use_threading);

  if (stellar_pod_solver.Solve(sfm_data_.poses))
  {
    OPENMVG_LOG_INFO << "Stellar init: solved with " << sfm_data_.poses.size() << " poses.";
    return true;
  }

  OPENMVG_LOG_ERROR << "Stellar init failed: Stellar_Solver::Solve returned false.";
  return false;
}

} // namespace sfm
} // namespace openMVG
