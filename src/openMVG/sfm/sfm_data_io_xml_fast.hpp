// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.
//
// Fast XML writer for SfM_Data scenes.
//
// Produces a file byte-readable by Load_Cereal<cereal::XMLInputArchive>, but
// avoids cereal's rapidxml DOM construction (and its tens of millions of small
// allocations) for the dominant `structure` section. Speed-up is roughly an
// order of magnitude on scenes with hundreds of thousands of landmarks.

#ifndef OPENMVG_SFM_SFM_DATA_IO_XML_FAST_HPP
#define OPENMVG_SFM_SFM_DATA_IO_XML_FAST_HPP

#include "openMVG/sfm/sfm_data_io.hpp"

#include <string>

namespace openMVG {
namespace sfm {

struct SfM_Data;

/// Save `data` to `filename` as XML. Output is identical (modulo whitespace)
/// to Save_Cereal<cereal::XMLOutputArchive> and can be loaded back via the
/// regular Load() / Load_Cereal<XMLInputArchive> path.
bool Save_XML_Fast(
  const SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part);

} // namespace sfm
} // namespace openMVG

#endif // OPENMVG_SFM_SFM_DATA_IO_XML_FAST_HPP
