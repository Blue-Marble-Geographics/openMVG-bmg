// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2016 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// The <cereal/archives> headers are special and must be included first.
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>

#include "openMVG/sfm/sfm_data_io_cereal.hpp"

#include "openMVG/cameras/cameras_io.hpp"
#include "openMVG/geometry/pose3_io.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_landmark_io.hpp"
#include "openMVG/sfm/sfm_view_io.hpp"
#include "openMVG/sfm/sfm_view_priors_io.hpp"
#include "openMVG/types.hpp"
#include "openMVG/system/logger.hpp"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

namespace openMVG {
namespace sfm {

/// Here a View description object that was used before OpenMVG v1.1
/// This object is used in order to be able to load old project files.
struct View_version_1
{
  std::string s_Img_path; // image path on disk
  IndexT id_view; // Id of the view
  IndexT id_intrinsic, id_pose; // Index of intrinsics and the pose
  IndexT ui_width, ui_height; // image size

  // Constructor (use unique index for the view_id)
  View_version_1(
    const std::string & sImgPath = "",
    IndexT view_id = UndefinedIndexT,
    IndexT intrinsic_id = UndefinedIndexT,
    IndexT pose_id = UndefinedIndexT,
    IndexT width = UndefinedIndexT, IndexT height = UndefinedIndexT)
    :s_Img_path(sImgPath), id_view(view_id), id_intrinsic(intrinsic_id),
    id_pose(pose_id), ui_width(width), ui_height(height)
    {}

  /**
  * @brief Serialization in
  * @param ar Archive
  */
  template <class Archive>
  void load( Archive & ar )
  {
    //Define a view with two string (base_path & basename)
    std::string local_path = s_Img_path;
    std::string filename = s_Img_path;

    ar(cereal::make_nvp("local_path", local_path),
       cereal::make_nvp("filename", filename),
       cereal::make_nvp("width", ui_width),
       cereal::make_nvp("height", ui_height),
       cereal::make_nvp("id_view", id_view),
       cereal::make_nvp("id_intrinsic", id_intrinsic),
       cereal::make_nvp("id_pose", id_pose));

    s_Img_path = stlplus::create_filespec(local_path, filename);
  }
};

template <
// JSONInputArchive/ ...
typename archiveType
>
bool Load_Cereal(
  SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part)
{
  const bool bBinary = stlplus::extension_part(filename) == "bin";

  // List which part of the file must be considered
  const bool b_views = (flags_part & VIEWS) == VIEWS;
  const bool b_intrinsics = (flags_part & INTRINSICS) == INTRINSICS;
  const bool b_extrinsics = (flags_part & EXTRINSICS) == EXTRINSICS;
  const bool b_structure = (flags_part & STRUCTURE) == STRUCTURE;
  const bool b_control_point = (flags_part & CONTROL_POINTS) == CONTROL_POINTS;

  // cereal binary archives read one primitive at a time. Going through
  // std::filebuf for tens of millions of tiny reads (every double / IndexT
  // in the structure block triggers an sgetn) is the dominant cost on
  // large scenes. Slurp the whole file into memory and feed cereal a
  // std::istringstream so all subsequent reads are pointer arithmetic
  // inside a stringbuf instead of buffered file I/O.
  //
  // Text archives benefit from the same trick (they tokenize through the
  // same streambuf interface) so we use this path unconditionally.
  std::string file_buffer;
  {
    std::ifstream stream(filename, std::ios::binary | std::ios::in | std::ios::ate);
    if (!stream)
      return false;
    const std::streamoff sz = stream.tellg();
    if (sz < 0)
      return false;
    stream.seekg(0, std::ios::beg);
    file_buffer.resize(static_cast<std::size_t>(sz));
    if (sz > 0 && !stream.read(file_buffer.data(), sz))
      return false;
  }
  std::istringstream stream(std::move(file_buffer), std::ios::binary | std::ios::in);

  // Data serialization
  try
  {
    archiveType archive(stream);

    std::string version;
    archive(cereal::make_nvp("sfm_data_version", version));
    archive(cereal::make_nvp("root_path", data.s_root_path));

    if (b_views)
    {
      if ( version >= "0.3" )
      {
        archive(cereal::make_nvp("views", data.views));
      }
      else // sfm_data version is < to v0.3 => Previous to OpenMVG v1.1
      {
        // It means that it use the View that does not support inheritance
        using Views_v1 = Hash_Map<IndexT, std::shared_ptr<View_version_1>>;
        Views_v1 views;
        try
        {
          archive(cereal::make_nvp("views", views));
          // Convert views to old to new format
          for (const auto v_it : views)
          {
            data.views[v_it.first] =
              std::make_shared<View>(
                v_it.second->s_Img_path,
                v_it.second->id_view,
                v_it.second->id_intrinsic,
                v_it.second->id_pose,
                v_it.second->ui_width,
                v_it.second->ui_height);
          }
        }
        catch (cereal::Exception& e)
        {
          OPENMVG_LOG_ERROR << e.what();
          return false;
        }
      }
    }
    else
    {
      if (bBinary)
      {
        // Binary file requires to read all the member,
        // read in a temporary object since the data is not needed.
        if ( version >= "0.3" )
        {
          Views views;
          archive(cereal::make_nvp("views", views));
        }
        else // sfm_data version is < to v0.3 => Previous to OpenMVG v1.1
        {
          // It means that it use the View that does not support inheritance
          using Views_v1 = Hash_Map<IndexT, std::shared_ptr<View_version_1>>;
          Views_v1 views;
          archive(cereal::make_nvp("views", views));
        }
      }
    }
    if (b_intrinsics)
      archive(cereal::make_nvp("intrinsics", data.intrinsics));
    else
      if (bBinary)
      {
        // Binary file requires to read all the member,
        // read in a temporary object since the data is not needed.
        Intrinsics intrinsics;
        archive(cereal::make_nvp("intrinsics", intrinsics));
      }

    if (b_extrinsics)
      archive(cereal::make_nvp("extrinsics", data.poses));
    else
      if (bBinary)
      {
        // Binary file requires to read all the member,
        // read in a temporary object since the data is not needed.
        Poses poses;
        archive(cereal::make_nvp("extrinsics", poses));
      }

    if (b_structure)
    {
      // cereal's generic pair_associative_container loader emplace_hint's
      // one item at a time WITHOUT reserving the unordered_map first, so on
      // a 568k-landmark scene the map rehashes ~20 times and each rehash
      // moves every Landmark (which itself owns an ankerl obs map). That
      // dominates the binary-load wall time. Drive the map protocol
      // manually so we can reserve() exactly once. Binary on-wire format is
      // a size_tag followed by N (key, value) pairs -- byte-identical to
      // what the generic loader consumes, so this is a pure load-path
      // optimisation (no file format change).
      if (bBinary)
      {
        cereal::size_type sz = 0;
        archive(cereal::make_size_tag(sz));
        data.structure.clear();
        data.structure.reserve(static_cast<std::size_t>(sz));
        for (cereal::size_type i = 0; i < sz; ++i)
        {
          // Read the key first, then try_emplace default-constructs the
          // Landmark directly inside the unordered_map node. Deserialize
          // into it->second so the Landmark (Vec3 + ankerl obs map) is
          // never move-constructed -- saves one move per landmark, which
          // matters because the ankerl map has non-trivial move cost.
          IndexT key;
          archive(key);
          auto [it, inserted] = data.structure.try_emplace(key);
          (void)inserted;
          archive(it->second);
        }
      }
      else
      {
        archive(cereal::make_nvp("structure", data.structure));
      }
    }
    else
      if (bBinary)
      {
        // Binary file requires to read all the member,
        // read in a temporary object since the data is not needed.
        Landmarks structure;
        archive(cereal::make_nvp("structure", structure));
      }

    if (version != "0.1") // fast check to assert we are at least using version 0.2
    {
      if (b_control_point)
        archive(cereal::make_nvp("control_points", data.control_points));
      else
        if (bBinary)
        {
          // Binary file requires to read all the member,
          // read in a temporary object since the data is not needed.
          Landmarks control_points;
          archive(cereal::make_nvp("control_points", control_points));
        }
    }
  }
  catch (const cereal::Exception & e)
  {
    OPENMVG_LOG_ERROR << e.what();
    return false;
  }
  return true;
}

// When OPENMVG_SFM_XML_COMPACT is enabled, the XML output archive is
// constructed with indentation disabled so the emitted document contains no
// inter-element whitespace. This keeps the file byte-for-byte parseable while
// eliminating the flood of whitespace-only character callbacks that a SAX
// consumer (e.g. Global Mapper's XmlDecoder) would otherwise have to process.
// JSON and binary archives are unaffected.
#ifndef OPENMVG_SFM_XML_COMPACT
#define OPENMVG_SFM_XML_COMPACT 1
#endif

namespace {
// Factory that constructs the requested output archive and runs the
// serialization body. The primary template uses cereal's default options.
template <typename Archive>
struct OutputArchiveFactory
{
  template <typename Fn>
  static void Run(std::ofstream & stream, Fn && fn)
  {
    Archive archive(stream);
    fn(archive);
  }
};

#if OPENMVG_SFM_XML_COMPACT
// XML specialization: disable indentation (no inter-element whitespace).
template <>
struct OutputArchiveFactory<cereal::XMLOutputArchive>
{
  template <typename Fn>
  static void Run(std::ofstream & stream, Fn && fn)
  {
    cereal::XMLOutputArchive archive(
      stream, cereal::XMLOutputArchive::Options().indent(false));
    fn(archive);
  }
};
#endif // OPENMVG_SFM_XML_COMPACT
} // namespace

template <
// JSONOutputArchive/ ...
typename archiveType
>
bool Save_Cereal(
  const SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part)
{
  // List which part of the file must be considered
  const bool b_views = (flags_part & VIEWS) == VIEWS;
  const bool b_intrinsics = (flags_part & INTRINSICS) == INTRINSICS;
  const bool b_extrinsics = (flags_part & EXTRINSICS) == EXTRINSICS;
  const bool b_structure = (flags_part & STRUCTURE) == STRUCTURE;
  const bool b_control_point = (flags_part & CONTROL_POINTS) == CONTROL_POINTS;

  //Create the stream and check it is ok
  std::ofstream stream(filename.c_str(), std::ios::binary | std::ios::out);
  if (!stream)
    return false;

  // Data serialization
  OutputArchiveFactory<archiveType>::Run(stream, [&](archiveType & archive)
  {
    // since OpenMVG 0.9, the sfm_data version 0.2 is introduced
    //  - it adds control_points storage
    // since OpenMVG 1.1, the sfm_data version 0.3 is introduced
    //  - it introduce polymorphic View data
    const std::string version = "0.3";
    archive(cereal::make_nvp("sfm_data_version", version));
    archive(cereal::make_nvp("root_path", data.s_root_path));

    if (b_views)
      archive(cereal::make_nvp("views", data.views));
    else
      archive(cereal::make_nvp("views", Views()));

    if (b_intrinsics)
      archive(cereal::make_nvp("intrinsics", data.intrinsics));
    else
      archive(cereal::make_nvp("intrinsics", Intrinsics()));

    if (b_extrinsics)
      archive(cereal::make_nvp("extrinsics", data.poses));
    else
      archive(cereal::make_nvp("extrinsics", Poses()));

    // Structure -> See for export in another file
    if (b_structure)
      archive(cereal::make_nvp("structure", data.structure));
    else
      archive(cereal::make_nvp("structure", Landmarks()));

    if (version != "0.1") // fast check to assert we are at least using version 0.2
    {
      if (b_control_point)
        archive(cereal::make_nvp("control_points", data.control_points));
      else
        archive(cereal::make_nvp("control_points", Landmarks()));
    }
  });
  stream.close();
  return true;
}

//
// Explicit template instantiation
//

template bool Load_Cereal
<cereal::BinaryInputArchive>
(
  SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part
);

template bool Load_Cereal
<cereal::PortableBinaryInputArchive>
(
  SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part
);

template bool Load_Cereal
<cereal::JSONInputArchive>
(
  SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part
);

template bool Load_Cereal
<cereal::XMLInputArchive>
(
  SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part
);

template bool Save_Cereal
<cereal::BinaryOutputArchive>
(
  const SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part
);

template bool Save_Cereal
<cereal::PortableBinaryOutputArchive>
(
  const SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part
);

template bool Save_Cereal
<cereal::JSONOutputArchive>
(
  const SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part
);

template bool Save_Cereal
<cereal::XMLOutputArchive>
(
  const SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part
);

} // namespace sfm
} // namespace openMVG
