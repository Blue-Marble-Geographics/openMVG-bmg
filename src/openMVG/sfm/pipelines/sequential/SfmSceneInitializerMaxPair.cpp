
// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2018 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/pipelines/sequential/SfmSceneInitializerMaxPair.hpp"

#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/multiview/triangulation.hpp"
#include "openMVG/numeric/numeric.h"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_matches_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_robust_model_estimation.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/system/logger.hpp"

#include <algorithm>
#include <cmath>

// --- Seed-pair validation thresholds (see the gate in Process()) -----------
// A candidate pair is only accepted as the SfM seed if, using the pose
// recovered by robustRelativePose, at least MIN_TRIANGULATED_POINTS inlier
// correspondences triangulate in front of both cameras (cheirality) with a
// parallax angle >= MIN_PARALLAX_DEG. This rejects high-match-count but
// low-parallax / dominant-plane pairs whose essential matrix is well-supported
// yet whose pose cannot triangulate stable structure. Set MIN_TRIANGULATED
// _POINTS to 0 to disable the gate (revert to legacy "first valid pose wins").
#ifndef OPENMVG_MAXPAIR_SEED_MIN_PARALLAX_DEG
#define OPENMVG_MAXPAIR_SEED_MIN_PARALLAX_DEG 2.0
#endif
#ifndef OPENMVG_MAXPAIR_SEED_MIN_TRIANGULATED_POINTS
#define OPENMVG_MAXPAIR_SEED_MIN_TRIANGULATED_POINTS 50
#endif

// Number of top pairs (ranked by match count) to evaluate when choosing the
// seed. Among these, the pair with the WIDEST baseline (highest median inlier
// parallax angle) that still triangulates >= MIN_TRIANGULATED_POINTS well-
// parallaxed points is chosen. Ranking by match count alone favours along-track
// / low-parallax pairs on aerial flight data, which seed a poorly-conditioned
// (dome-prone) reconstruction; scanning a handful of the strongest-matched
// candidates and picking the widest baseline yields a far better-conditioned
// seed at a one-time init cost of up to this many robustRelativePose calls.
#ifndef OPENMVG_MAXPAIR_SEED_CANDIDATES
#define OPENMVG_MAXPAIR_SEED_CANDIDATES 15
#endif

// Fast-path threshold (degrees). If the strongest-matched valid pair already
// has a median inlier parallax >= this, it is accepted IMMEDIATELY without the
// widest-baseline scan. This keeps well-conditioned datasets (terrestrial /
// oblique, whose top pair is high-parallax) on their historical top-match seed
// -- identical result, only ONE robustRelativePose spent -- so nothing that
// worked before is changed or slowed. The multi-candidate scan engages only
// when the top pair is LOW-parallax (aerial / nadir), i.e. exactly the case
// that needs a better-conditioned seed.
#ifndef OPENMVG_MAXPAIR_SEED_GOOD_PARALLAX_DEG
#define OPENMVG_MAXPAIR_SEED_GOOD_PARALLAX_DEG 5.0
#endif

namespace openMVG {

using namespace cameras;
using namespace geometry;
using namespace matching;

namespace sfm {

SfMSceneInitializerMaxPair::SfMSceneInitializerMaxPair(
  SfM_Data & sfm_data,
  const Features_Provider * features_provider,
  const Matches_Provider * matches_provider)
  :SfMSceneInitializer(sfm_data, features_provider, matches_provider)
{
  sfm_data_.poses.clear();
}

bool SfMSceneInitializerMaxPair::Process()
{
  if (sfm_data_.GetIntrinsics().empty())
    return false;

  //
  // Sort the PairWiseMatches by matches count.
  // Keep the first pair that provides a valid relative pose.
  //
  std::vector<IndexT> matches_count_per_pair;
  matches_count_per_pair.reserve(matches_provider_->pairWise_matches_.size());
  std::transform(matches_provider_->pairWise_matches_.cbegin(), matches_provider_->pairWise_matches_.cend(),
    std::back_inserter(matches_count_per_pair),
    [](const std::pair<Pair, IndMatches> & match_pair) -> IndexT { return match_pair.second.size(); });

  // sort the Pairs in descending order according their matches count
  using namespace stl::indexed_sort;
  std::vector<sort_index_packet_descend<IndexT, IndexT>> packet_vec(matches_count_per_pair.size());
  const size_t n_candidates =
    std::min<size_t>(OPENMVG_MAXPAIR_SEED_CANDIDATES, matches_count_per_pair.size());
  sort_index_helper(packet_vec, &matches_count_per_pair[0],
                    std::min((IndexT)OPENMVG_MAXPAIR_SEED_CANDIDATES,
                             (IndexT)matches_count_per_pair.size()));

  // Evaluate the top n_candidates pairs (by match count) and pick the one whose
  // recovered relative pose has the WIDEST baseline -- i.e. the highest median
  // triangulation (parallax) angle over its inliers -- among those that still
  // triangulate at least MIN_TRIANGULATED_POINTS well-parallaxed points.
  //
  // MaxPair by match count alone favours along-track / low-parallax pairs on
  // aerial flight data; those pass the essential-matrix test yet seed a poorly-
  // conditioned reconstruction that BA warps into a dome. Choosing the widest-
  // baseline pair among the strongest-matched candidates gives a much better-
  // conditioned seed while still requiring enough matches to bootstrap. The
  // first geometrically-valid pair is kept as a fallback so init never fails
  // when no candidate clears the parallax bar (e.g. genuinely near-planar
  // scenes) -- there the adaptive GPS prior weighting carries the geometry.
  double best_median_parallax_deg = -1.0;
  bool   have_best = false;
  IndexT best_I = 0, best_J = 0;
  geometry::Pose3 best_pose_J;

  bool   have_fallback = false;
  IndexT fb_I = 0, fb_J = 0;
  geometry::Pose3 fb_pose_J;

  for (size_t i = 0; i < n_candidates; ++i)
  {
    const IndexT index = packet_vec[i].index;
    openMVG::matching::PairWiseMatches::const_iterator iter = matches_provider_->pairWise_matches_.cbegin();
    std::advance(iter, index);

    const IndexT
      I = iter->first.first,
      J = iter->first.second;

    const View
      * view_I = sfm_data_.views[I].get(),
      * view_J = sfm_data_.views[J].get();

    // Check that the pair has valid intrinsic
    if (sfm_data_.GetIntrinsics().count(view_I->id_intrinsic) == 0 ||
        sfm_data_.GetIntrinsics().count(view_J->id_intrinsic) == 0)
      continue;

    const IntrinsicBase
      * cam_I = sfm_data_.GetIntrinsics().at(view_I->id_intrinsic).get(),
      * cam_J = sfm_data_.GetIntrinsics().at(view_J->id_intrinsic).get();

    // Compute for each feature the un-distorted camera coordinates
    const matching::IndMatches & matches = matches_provider_->pairWise_matches_.at(iter->first);
    size_t number_matches = matches.size();
    Mat2X x1(2, number_matches), x2(2, number_matches);
    number_matches = 0;
    for (const auto & match : matches)
    {
      x1.col(number_matches) = cam_I->get_ud_pixel(features_provider_->feats_per_view.at(I)[match.i_].coords().cast<double>());
      x2.col(number_matches) = cam_J->get_ud_pixel(features_provider_->feats_per_view.at(J)[match.j_].coords().cast<double>());
      ++number_matches;
    }

    RelativePose_Info relativePose_info;
    relativePose_info.initial_residual_tolerance = Square(2.5);
    if (!robustRelativePose(cam_I, cam_J,
                            x1, x2, relativePose_info,
                            {cam_I->w(), cam_I->h()},
                            {cam_J->w(), cam_J->h()},
                            2048))
    {
      continue;
    }

    // First geometrically-valid pair -> keep as the fallback seed.
    if (!have_fallback)
    {
      have_fallback = true;
      fb_I = I; fb_J = J; fb_pose_J = relativePose_info.relativePose;
    }

    // Triangulate the inliers with the recovered pose; collect the parallax
    // (apical) angle of every cheirality-valid point. The count above the
    // MIN_PARALLAX bar drives the acceptance gate; the median drives the
    // widest-baseline selection.
    const geometry::Pose3 & pose_J = relativePose_info.relativePose;
    const Mat3 R1 = pose_J.rotation();
    const Vec3 t1 = pose_J.translation();
    const Vec3 C1 = pose_J.center();
    const double min_parallax_rad = D2R(double(OPENMVG_MAXPAIR_SEED_MIN_PARALLAX_DEG));

    std::vector<double> parallax_rad;
    parallax_rad.reserve(relativePose_info.vec_inliers.size());
    IndexT well_triangulated = 0;
    for (const uint32_t inlier_idx : relativePose_info.vec_inliers)
    {
      const matching::IndMatch & match = matches[inlier_idx];
      const Vec3 bearing0 =
        (*cam_I)(cam_I->get_ud_pixel(
          features_provider_->feats_per_view.at(I)[match.i_].coords().cast<double>()));
      const Vec3 bearing1 =
        (*cam_J)(cam_J->get_ud_pixel(
          features_provider_->feats_per_view.at(J)[match.j_].coords().cast<double>()));
      Vec3 X;
      if (!Triangulate2View(Mat3::Identity(), Vec3::Zero(), bearing0, R1, t1, bearing1, X))
        continue; // cheirality failed
      const Vec3 ray0 = (Vec3::Zero() - X).normalized();
      const Vec3 ray1 = (C1 - X).normalized();
      const double cos_parallax =
        std::max(-1.0, std::min(1.0, ray0.dot(ray1)));
      const double angle = std::acos(cos_parallax);
      parallax_rad.push_back(angle);
      if (angle >= min_parallax_rad)
        ++well_triangulated;
    }

    if (OPENMVG_MAXPAIR_SEED_MIN_TRIANGULATED_POINTS > 0 &&
        well_triangulated < IndexT(OPENMVG_MAXPAIR_SEED_MIN_TRIANGULATED_POINTS))
    {
      OPENMVG_LOG_INFO
        << " -> seed candidate (" << I << "," << J << "): only "
        << well_triangulated << " pts with parallax >= "
        << OPENMVG_MAXPAIR_SEED_MIN_PARALLAX_DEG << " deg (need "
        << OPENMVG_MAXPAIR_SEED_MIN_TRIANGULATED_POINTS << "); skipping." << std::endl;
      continue;
    }

    // Median parallax = baseline-quality score for this pair.
    double median_parallax_deg = 0.0;
    if (!parallax_rad.empty())
    {
      const size_t mid = parallax_rad.size() / 2;
      std::nth_element(parallax_rad.begin(), parallax_rad.begin() + mid, parallax_rad.end());
      median_parallax_deg = R2D(parallax_rad[mid]);
    }

    OPENMVG_LOG_INFO
      << " -> seed candidate (" << I << "," << J << "): " << matches.size()
      << " matches, " << well_triangulated << " well-parallaxed pts, median parallax "
      << median_parallax_deg << " deg" << std::endl;

    // Fast path: the strongest-matched pair that is ALREADY well-conditioned
    // (median parallax >= GOOD_PARALLAX_DEG) is accepted immediately. Because
    // we iterate in descending match-count order and have not started tracking
    // a best pair yet, this is exactly the historical top-match seed for
    // well-conditioned datasets -- their result is unchanged and only one
    // robustRelativePose is spent. The widest-baseline scan below engages only
    // when the top pair is low-parallax (aerial / nadir).
    if (!have_best &&
        median_parallax_deg >= double(OPENMVG_MAXPAIR_SEED_GOOD_PARALLAX_DEG))
    {
      OPENMVG_LOG_INFO
        << "Selected seed pair (" << I << "," << J
        << ") -- strongest-matched well-conditioned pair (median parallax "
        << median_parallax_deg << " deg >= "
        << OPENMVG_MAXPAIR_SEED_GOOD_PARALLAX_DEG << ")." << std::endl;
      sfm_data_.poses[view_I->id_pose] = Pose3(Mat3::Identity(), Vec3::Zero());
      sfm_data_.poses[view_J->id_pose] = relativePose_info.relativePose;
      return true;
    }

    if (median_parallax_deg > best_median_parallax_deg)
    {
      best_median_parallax_deg = median_parallax_deg;
      have_best = true;
      best_I = I; best_J = J; best_pose_J = relativePose_info.relativePose;
    }
  }

  if (have_best)
  {
    OPENMVG_LOG_INFO
      << "Selected seed pair (" << best_I << "," << best_J
      << ") -- widest baseline among top " << n_candidates
      << " candidates (median parallax " << best_median_parallax_deg << " deg)." << std::endl;
    sfm_data_.poses[sfm_data_.views[best_I]->id_pose] = Pose3(Mat3::Identity(), Vec3::Zero());
    sfm_data_.poses[sfm_data_.views[best_J]->id_pose] = best_pose_J;
    return true;
  }
  if (have_fallback)
  {
    OPENMVG_LOG_WARNING
      << "No seed pair cleared the parallax bar in the top " << n_candidates
      << " candidates; falling back to highest-match valid pair ("
      << fb_I << "," << fb_J << ")." << std::endl;
    sfm_data_.poses[sfm_data_.views[fb_I]->id_pose] = Pose3(Mat3::Identity(), Vec3::Zero());
    sfm_data_.poses[sfm_data_.views[fb_J]->id_pose] = fb_pose_J;
    return true;
  }
  return false;
}

} // namespace sfm
} // namespace openMVG
