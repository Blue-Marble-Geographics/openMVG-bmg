// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.
//
// Compact binary writer for Global Mapper "Pixels to Points" -- see
// sfm_data_io_gmp.hpp for the rationale.
//
// On-disk layout (all integers little-endian; strings are uint32 byte-length
// followed by raw UTF-8 bytes with no terminator):
//
//   char     magic[4]        = 'G','M','P','C'
//   uint32   version         = 1
//   uint32   section_flags   bit0 = views present
//                            bit1 = intrinsics present
//                            bit2 = extrinsics present
//                            bit3 = structure present
//   string   root_path
//
//   uint32   view_count
//     repeat:
//       int64   map_key          (Views map key == id_view)
//       int64   id_view
//       int64   id_intrinsic
//       string  filename         (View::s_Img_path, relative)
//       int32   width
//       int32   height
//       uint8   has_center_prior
//       if has_center_prior: double center[3]   (ViewPriors::pose_center_)
//
//   uint32   intrinsic_count
//     repeat:
//       int64   map_key          (Intrinsics map key == id_intrinsic)
//       double  focal
//       uint8   has_principal_point
//       if has_principal_point: double pp[2]
//       int32   width
//       int32   height
//       uint8   disto_count
//       double  disto[disto_count]
//       string  model_name
//
//   uint32   pose_count
//     repeat:
//       int64   map_key          (Poses map key == id_pose)
//       double  center[3]
//       double  rotation[9]      (row-major: rotation(r,c))
//
//   if (section_flags bit3):
//     uint32  landmark_count
//       repeat:
//         int64   key
//         double  pos[3]
//         uint32  obs_count
//           repeat:
//             int64   view_id
//             int64   id_feat
//             double  pix[2]

// cereal not needed here -- this is a hand-rolled binary stream.
#include "openMVG/sfm/sfm_data_io_gmp.hpp"

#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/geometry/pose3.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_landmark.hpp"
#include "openMVG/sfm/sfm_view.hpp"
#include "openMVG/sfm/sfm_view_priors.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/types.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace openMVG {
namespace sfm {

namespace {

// --- little-endian append helpers (Windows x64 host; native LE) -------------

inline void put_bytes(std::string & buf, const void * p, std::size_t n)
{
  buf.append(static_cast<const char *>(p), n);
}

inline void put_u32(std::string & buf, std::uint32_t v)
{
  put_bytes(buf, &v, sizeof(v));
}

inline void put_i64(std::string & buf, std::int64_t v)
{
  put_bytes(buf, &v, sizeof(v));
}

inline void put_i32(std::string & buf, std::int32_t v)
{
  put_bytes(buf, &v, sizeof(v));
}

inline void put_u8(std::string & buf, std::uint8_t v)
{
  put_bytes(buf, &v, sizeof(v));
}

inline void put_f64(std::string & buf, double v)
{
  put_bytes(buf, &v, sizeof(v));
}

inline void put_string(std::string & buf, const std::string & s)
{
  put_u32(buf, static_cast<std::uint32_t>(s.size()));
  if (!s.empty())
    put_bytes(buf, s.data(), s.size());
}

// Map an intrinsic type enum to a human/cereal-ish model name. GM stores this
// in P2P_CameraIntrinsic_t::mName (currently informational only).
const char * IntrinsicModelName(cameras::EINTRINSIC type)
{
  using namespace cameras;
  switch (type)
  {
    case PINHOLE_CAMERA:         return "pinhole";
    case PINHOLE_CAMERA_RADIAL1: return "pinhole_radial_k1";
    case PINHOLE_CAMERA_RADIAL3: return "pinhole_radial_k3";
    case PINHOLE_CAMERA_BROWN:   return "pinhole_brown_t2";
    case PINHOLE_CAMERA_FISHEYE: return "fisheye";
    case CAMERA_SPHERICAL:       return "spherical";
    default:                     return "";
  }
}

} // anonymous namespace

bool Save_GMP_Binary(
  const SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part)
{
  const bool b_views      = (flags_part & VIEWS) == VIEWS;
  const bool b_intrinsics = (flags_part & INTRINSICS) == INTRINSICS;
  const bool b_extrinsics = (flags_part & EXTRINSICS) == EXTRINSICS;
  const bool b_structure  = (flags_part & STRUCTURE) == STRUCTURE;

  std::string buf;
  // Rough reserve: header + per-view/intrinsic records are small; structure
  // dominates when present. Grow lazily otherwise.
  buf.reserve(64 + data.views.size() * 96 + data.intrinsics.size() * 96
              + data.poses.size() * 112);

  // Header.
  static const char kMagic[4] = { 'G', 'M', 'P', 'C' };
  put_bytes(buf, kMagic, 4);
  put_u32(buf, 1u); // version

  std::uint32_t section_flags = 0;
  if (b_views)      section_flags |= 1u << 0;
  if (b_intrinsics) section_flags |= 1u << 1;
  if (b_extrinsics) section_flags |= 1u << 2;
  if (b_structure)  section_flags |= 1u << 3;
  put_u32(buf, section_flags);

  put_string(buf, data.s_root_path);

  // Views.
  if (b_views)
  {
    put_u32(buf, static_cast<std::uint32_t>(data.views.size()));
    for (const auto & kv : data.views)
    {
      const IndexT map_key = kv.first;
      const View * const view = kv.second.get();

      put_i64(buf, static_cast<std::int64_t>(map_key));
      put_i64(buf, static_cast<std::int64_t>(view->id_view));
      put_i64(buf, static_cast<std::int64_t>(view->id_intrinsic));
      put_string(buf, view->s_Img_path);
      put_i32(buf, static_cast<std::int32_t>(view->ui_width));
      put_i32(buf, static_cast<std::int32_t>(view->ui_height));

      // EXIF/GPS center prior, present only for ViewPriors with a center.
      const ViewPriors * const vp = dynamic_cast<const ViewPriors *>(view);
      if (vp != nullptr && vp->b_use_pose_center_)
      {
        put_u8(buf, 1);
        put_f64(buf, vp->pose_center_(0));
        put_f64(buf, vp->pose_center_(1));
        put_f64(buf, vp->pose_center_(2));
      }
      else
      {
        put_u8(buf, 0);
      }
    }
  }
  else
  {
    put_u32(buf, 0u);
  }

  // Intrinsics.
  if (b_intrinsics)
  {
    put_u32(buf, static_cast<std::uint32_t>(data.intrinsics.size()));
    for (const auto & kv : data.intrinsics)
    {
      const IndexT map_key = kv.first;
      const cameras::IntrinsicBase * const cam = kv.second.get();

      put_i64(buf, static_cast<std::int64_t>(map_key));

      // getParams() ordering for every pinhole model: [focal, ppx, ppy, disto...].
      const std::vector<double> params = cam->getParams();
      const double focal = (params.size() >= 1) ? params[0] : 0.0;
      put_f64(buf, focal);

      if (params.size() >= 3)
      {
        put_u8(buf, 1);
        put_f64(buf, params[1]);
        put_f64(buf, params[2]);
      }
      else
      {
        put_u8(buf, 0);
      }

      put_i32(buf, static_cast<std::int32_t>(cam->w()));
      put_i32(buf, static_cast<std::int32_t>(cam->h()));

      // Distortion coefficients (everything past focal + principal point).
      std::size_t disto_n = (params.size() > 3) ? (params.size() - 3) : 0;
      if (disto_n > 255) disto_n = 255;
      put_u8(buf, static_cast<std::uint8_t>(disto_n));
      for (std::size_t i = 0; i < disto_n; ++i)
        put_f64(buf, params[3 + i]);

      put_string(buf, IntrinsicModelName(cam->getType()));
    }
  }
  else
  {
    put_u32(buf, 0u);
  }

  // Extrinsics (poses).
  if (b_extrinsics)
  {
    put_u32(buf, static_cast<std::uint32_t>(data.poses.size()));
    for (const auto & kv : data.poses)
    {
      const IndexT map_key = kv.first;
      const geometry::Pose3 & pose = kv.second;

      put_i64(buf, static_cast<std::int64_t>(map_key));

      const Vec3 & c = pose.center();
      put_f64(buf, c(0));
      put_f64(buf, c(1));
      put_f64(buf, c(2));

      const Mat3 & r = pose.rotation();
      for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
          put_f64(buf, r(row, col));
    }
  }
  else
  {
    put_u32(buf, 0u);
  }

  // Structure (optional).
  if (b_structure)
  {
    put_u32(buf, static_cast<std::uint32_t>(data.structure.size()));
    for (const auto & kv : data.structure)
    {
      const IndexT key = kv.first;
      const Landmark & lm = kv.second;

      put_i64(buf, static_cast<std::int64_t>(key));
      put_f64(buf, lm.X(0));
      put_f64(buf, lm.X(1));
      put_f64(buf, lm.X(2));

      put_u32(buf, static_cast<std::uint32_t>(lm.obs.size()));
      for (const auto & ob_kv : lm.obs)
      {
        const IndexT view_id = ob_kv.first;
        const Observation & ob = ob_kv.second;
        put_i64(buf, static_cast<std::int64_t>(view_id));
        put_i64(buf, static_cast<std::int64_t>(ob.id_feat));
        put_f64(buf, ob.x(0));
        put_f64(buf, ob.x(1));
      }
    }
  }

  // Stream to disk in one shot with a large streambuf.
  std::vector<char> bigbuf(1u << 20);
  std::ofstream stream;
  stream.rdbuf()->pubsetbuf(bigbuf.data(), static_cast<std::streamsize>(bigbuf.size()));
  stream.open(filename.c_str(), std::ios::binary | std::ios::out | std::ios::trunc);
  if (!stream)
  {
    OPENMVG_LOG_ERROR << "Save_GMP_Binary: cannot open output \"" << filename << "\".";
    return false;
  }
  stream.write(buf.data(), static_cast<std::streamsize>(buf.size()));
  stream.flush();
  const bool ok = static_cast<bool>(stream);
  stream.close();
  return ok;
}

} // namespace sfm
} // namespace openMVG
