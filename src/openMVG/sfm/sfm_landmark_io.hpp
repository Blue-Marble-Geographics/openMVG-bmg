// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_LANDMARK_IO_HPP
#define OPENMVG_SFM_SFM_LANDMARK_IO_HPP

#include "openMVG/sfm/sfm_landmark.hpp"

#include <cereal/cereal.hpp> // Serialization
#include <cereal/details/traits.hpp> // cereal::traits::is_text_archive
#include <cereal/types/map.hpp>

#include <map>
#include <type_traits>
#include <utility>
#include <vector>

template <class Archive>
void openMVG::sfm::Observation::save( Archive & ar) const
{
  ar(cereal::make_nvp("id_feat", id_feat ));
  const std::vector<double> pp { x(0), x(1) };
  ar(cereal::make_nvp("x", pp));
}

template <class Archive>
void openMVG::sfm::Observation::load( Archive & ar)
{
  if constexpr (cereal::traits::is_text_archive<Archive>::value)
  {
    ar(cereal::make_nvp("id_feat", id_feat ));
    std::vector<double> p(2);
    ar(cereal::make_nvp("x", p));
    x = Eigen::Map<const Vec2>(&p[0]);
  }
  else
  {
    // Binary fast path: avoid the std::vector<double>(2) heap alloc that
    // happens once per observation (~5.7M allocs on a 568k-landmark
    // scene). The on-wire format produced by save() is identical -- a
    // vector<double> of length 2 is just a size_tag(=2) followed by two
    // doubles, which is exactly what we read here.
    ar(id_feat);
    cereal::size_type sz = 0;
    ar(cereal::make_size_tag(sz));
    // sz is always 2 on a well-formed file; reading exactly two doubles
    // keeps stream position correct regardless. Portable archives
    // byte-swap each scalar individually, so reading two doubles is
    // equivalent to (and byte-identical with) reading a vector of two.
    double x0 = 0.0, x1 = 0.0;
    ar(x0, x1);
    x(0) = x0;
    x(1) = x1;
  }
}


template <class Archive>
void openMVG::sfm::Landmark::save( Archive & ar) const
{
  const std::vector<double> point { X(0), X(1), X(2) };
  ar(cereal::make_nvp("X", point ));
  // Serialize through a std::map so cereal's built-in ordered-map support is
  // used (the Observations backing store is ankerl::unordered_dense::map which
  // cereal does not know about). Using std::map also gives a deterministic,
  // view_id-sorted on-disk order.
  const std::map<IndexT, Observation> obs_ordered(obs.begin(), obs.end());
  ar(cereal::make_nvp("observations", obs_ordered));
}


template <class Archive>
void openMVG::sfm::Landmark::load( Archive & ar)
{
  if constexpr (cereal::traits::is_text_archive<Archive>::value)
  {
    // Text archives (JSON / XML): keep the std::map intermediate so cereal's
    // tagged map deserialization matches the on-disk schema (<observations>
    // wrapper + per-entry <key>/<value> nodes). Then move-merge into the
    // ankerl::unordered_dense backing store. This path is unchanged from
    // the legacy implementation.
    std::vector<double> point(3);
    ar(cereal::make_nvp("X", point ));
    X = Eigen::Map<const Vec3>(&point[0]);

    std::map<IndexT, Observation> obs_ordered;
    ar(cereal::make_nvp("observations", obs_ordered));
    obs.clear();
    obs.reserve(obs_ordered.size());
    for (auto & kv : obs_ordered)
      obs.emplace(kv.first, std::move(kv.second));
  }
  else
  {
    // Binary archives (Portable / non-portable BinaryInputArchive): drive
    // the map deserialization protocol manually, writing observations
    // straight into the ankerl::unordered_dense map. The on-wire format
    // (size_tag + 3 doubles for X, size_tag + N (key, value) pairs for obs)
    // is byte-identical to what cereal emits for Save_Cereal<...
    // BinaryOutputArchive> via std::vector<double> + std::map<K,V>, so
    // this is a pure load-path optimisation -- no file format change.
    //
    // Two heap allocations are eliminated per landmark vs the legacy path:
    //   1. The std::vector<double>(3) wrapping X.
    //   2. Each RB-tree node in the std::map<IndexT, Observation>.
    // On a 568k-landmark, ~10-obs-per-landmark scene this removes ~568k
    // + ~5.7M heap allocations from the load path.

    // X: stored as a vector<double> of size 3 on disk. Read the size tag
    // (always 3 on a well-formed file) then the three doubles directly
    // into the Vec3 storage. Portable archives byte-swap each scalar
    // individually so this is wire-equivalent to a vector<double> read.
    cereal::size_type x_sz = 0;
    ar(cereal::make_size_tag(x_sz));
    double x0 = 0.0, x1 = 0.0, x2 = 0.0;
    ar(x0, x1, x2);
    X(0) = x0;
    X(1) = x1;
    X(2) = x2;

    cereal::size_type sz = 0;
    ar(cereal::make_size_tag(sz));
    obs.clear();
    obs.reserve(static_cast<std::size_t>(sz));
    for (cereal::size_type i = 0; i < sz; ++i)
    {
      // Read key, then try_emplace so the Observation is default-
      // constructed directly in the ankerl values vector and we
      // deserialize into it in place -- no stack-local Observation, no
      // move per observation.
      IndexT key;
      ar(key);
      auto [it, inserted] = obs.try_emplace(key);
      (void)inserted;
      ar(it->second);
    }
  }
}

#endif // OPENMVG_SFM_SFM_LANDMARK_IO_HPP
