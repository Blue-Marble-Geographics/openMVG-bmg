// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.
//
// Compact binary writer for Global Mapper "Pixels to Points".
//
// Emits a purpose-built little-endian binary file (extension ".gmpcam")
// carrying ONLY the fields Global Mapper consumes from the SfM_Data scene
// (root path, views, intrinsics, poses, and -- optionally -- structure).
//
// This replaces the XML round-trip (sfm_data.bin -> final_camera_data.xml ->
// SAX parse) on the GM side: GM reads this file directly with
// P2P_FinalCameraData_t::readBinary(), avoiding the multi-million-element
// rapidxml/SAX traversal entirely.
//
// The format is consumed ONLY by GM on the same machine, so native
// little-endian doubles / uint32 are written as-is (no portability layer).

#ifndef OPENMVG_SFM_SFM_DATA_IO_GMP_HPP
#define OPENMVG_SFM_SFM_DATA_IO_GMP_HPP

#include "openMVG/sfm/sfm_data_io.hpp"

#include <string>

namespace openMVG {
namespace sfm {

struct SfM_Data;

/// Save `data` to `filename` as a compact GM-camera binary (".gmpcam").
/// The structure section is written only when (flags_part & STRUCTURE).
bool Save_GMP_Binary(
  const SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part);

} // namespace sfm
} // namespace openMVG

#endif // OPENMVG_SFM_SFM_DATA_IO_GMP_HPP
