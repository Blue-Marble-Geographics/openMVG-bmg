// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_LANDMARK_HPP
#define OPENMVG_SFM_SFM_LANDMARK_HPP

#include "openMVG/types.hpp"
#include "third_party/ankerl/unordered_dense.h"

namespace openMVG {
namespace sfm {

/// Define 3D-2D tracking data: 3D landmark with its 2D observations
struct Observation
{
  Observation():id_feat(UndefinedIndexT) {  }
  Observation(const Vec2 & p, IndexT idFeat): x(p), id_feat(idFeat) {}

  Vec2 x;
  IndexT id_feat;

  // Serialization
  template <class Archive>
  void save( Archive & ar) const;

  // Serialization
  template <class Archive>
  void load( Archive & ar);
};

/// Observations are indexed by their View_id.
/// Backed by ankerl::unordered_dense::map: same API as std::unordered_map, but
/// values live in a contiguous std::vector (one allocation per map rather than
/// one per entry). Triangulation and filter passes over every landmark's
/// `obs` see a significant allocation-count reduction.
///
/// IMPORTANT: iterator/reference invalidation semantics differ from
/// std::unordered_map -- insert/erase may invalidate iterators and references
/// (like std::vector). Call sites in OpenMVG have been audited; none hold an
/// iterator or reference across a mutation of the same Observations map.
using Observations = ankerl::unordered_dense::map<IndexT, Observation>;

/// Define a landmark (a 3D point, with its 2d observations)
struct Landmark
{
  Vec3 X;
  Observations obs;

  // Serialization
  template <class Archive>
  void save( Archive & ar) const;

  template <class Archive>
  void load( Archive & ar);
};

/// Define a collection of landmarks are indexed by their TrackId
using Landmarks = Hash_Map<IndexT, Landmark>;

} // namespace sfm
} // namespace openMVG

#endif // OPENMVG_SFM_SFM_LANDMARK_HPP
