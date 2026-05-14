// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2016 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data_transform.hpp"

#include "openMVG/geometry/Similarity3.hpp"
#include "openMVG/sfm/sfm_data.hpp"

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include <vector>

namespace openMVG {
namespace sfm {

/// Apply a similarity to the SfM_Data scene (transform landmarks & camera poses)
void ApplySimilarity
(
  const geometry::Similarity3 & sim,
  SfM_Data & sfm_data,
  bool transform_priors
)
{
  // Hoist the affine map once. The original per-point form expanded as
  //   X' = sim(X) = scale * R * (X - center)
  // which costs a Vec3 subtract + Mat3*Vec3 + scalar broadcast per point.
  // We rewrite as
  //   X' = M * X + t,  where  M = scale * R,  t = -M * center
  // (algebraically identical: M*X + t = scale*R*X - scale*R*center
  //                                   = scale*R*(X - center)).
  // Per-point cost drops to one Mat3*Vec3 + Vec3 add.
  const Mat3 M = sim.scale_ * sim.pose_.rotation();
  const Vec3 t = -(M * sim.pose_.center());

  // Transform the landmark positions. On stellar/global scenes structure
  // is the dominant N here (millions of landmarks), so parallelize. Each
  // landmark write is independent. Hash_Map isn't random-access, so
  // snapshot pointers into a flat vector first; the snapshot pays off
  // above ~50k landmarks (single-threaded snapshot << parallel transform
  // of millions of points). Below that, fall through to the serial loop.
  constexpr size_t kParallelLandmarkThreshold = 50000;
  if (sfm_data.structure.size() >= kParallelLandmarkThreshold)
  {
    std::vector<Vec3 *> X_ptrs;
    X_ptrs.reserve(sfm_data.structure.size());
    for (auto & iterLandMark : sfm_data.structure)
      X_ptrs.push_back(&iterLandMark.second.X);
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(X_ptrs.size());
#ifdef OPENMVG_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i)
    {
      Vec3 & X = *X_ptrs[i];
      // Use temporary: X is both source and destination. Eigen's small
      // fixed-size Mat3*Vec3 materializes a temp internally, but being
      // explicit makes the no-aliasing semantics obvious to the reader.
      const Vec3 X_new = M * X + t;
      X = X_new;
    }
  }
  else
  {
    for (auto & iterLandMark : sfm_data.structure)
    {
      Vec3 & X = iterLandMark.second.X;
      const Vec3 X_new = M * X + t;
      X = X_new;
    }
  }

  // Transform the camera positions. Pose composition is not just an
  // affine map (it touches rotation_ * pose_.rotation_.transpose()), so
  // the M/t hoist doesn't apply here -- keep the original sim(Pose3)
  // path. Counts here are O(N_views), small.
  for (auto & iterPose : sfm_data.poses)
  {
    iterPose.second = sim(iterPose.second);
  }

  if (transform_priors)
  {
    for (auto & iterView : sfm_data.views)
    {
      // Transform the camera position priors
      if (sfm::ViewPriors * prior = dynamic_cast<sfm::ViewPriors*>(iterView.second.get()))
      {
        prior->pose_center_ = sim(prior->pose_center_);
      }
    }

    // Transform the control points
    for (auto & iterControlPoint : sfm_data.control_points)
    {
      iterControlPoint.second.X = sim(iterControlPoint.second.X);
    }
  }
}

} // namespace sfm
} // namespace openMVG
