// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.
//
// Fast XML writer -- see sfm_data_io_xml_fast.hpp.
//
// Strategy:
//   1. Use cereal to serialize EVERYTHING EXCEPT `structure` into an in-memory
//      "skeleton" string. This keeps cereal's polymorphic / NVP / size-tag
//      machinery for views (shared_ptr<View>), intrinsics (shared_ptr<IntrinsicBase>),
//      and poses, all of which require type-registration metadata we do not
//      want to reproduce by hand.
//   2. Locate the placeholder `<structure ... />` element in the skeleton.
//   3. Stream to disk: bytes-before-structure, then a hand-built
//      `<structure size="dynamic"> ... </structure>` block formatted with
//      snprintf (no DOM, no per-value std::ostringstream), then bytes-after.
//
// The hand-built block uses an OpenMP-parallel formatter: landmarks are split
// into a few chunks, each thread fills its own std::string, and the final
// write concatenates them in order. Per-thread formatting is independent
// (no shared state) so this scales near-linearly with cores until the
// pubsetbuf'd ofstream becomes the bottleneck.
//
// cereal's XMLInputArchive is invoked with `parse_trim_whitespace |
// parse_no_data_nodes`, so the absence of indentation in our hand-written
// section is irrelevant on load.

// cereal archive headers must be included first.
#include <cereal/archives/xml.hpp>

#include "openMVG/sfm/sfm_data_io_xml_fast.hpp"

#include "openMVG/cameras/cameras_io.hpp"
#include "openMVG/geometry/pose3_io.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_landmark_io.hpp"
#include "openMVG/sfm/sfm_view_io.hpp"
#include "openMVG/sfm/sfm_view_priors_io.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/types.hpp"

#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace openMVG {
namespace sfm {

namespace {

// %.17g matches std::setprecision(std::numeric_limits<double>::max_digits10)
// with std::defaultfloat, i.e. exactly what cereal::XMLOutputArchive emits.
inline int format_double(char * buf, std::size_t cap, double v)
{
  return std::snprintf(buf, cap, "%.17g", v);
}

inline int format_uint(char * buf, std::size_t cap, std::uint32_t v)
{
  return std::snprintf(buf, cap, "%u", v);
}

// Append a single landmark element body (i.e. the inside of `<valueN>...</valueN>`).
// `wrapper_idx` is the landmark's position inside the `<structure>` map.
void AppendLandmark(
  std::string & out,
  std::size_t wrapper_idx,
  IndexT id,
  const Landmark & lm)
{
  char nb[48];
  int n;

  // <valueN>
  n = std::snprintf(nb, sizeof(nb), "<value%zu>", wrapper_idx);
  out.append(nb, n);

  // <key>id</key><value>
  out.append("<key>");
  n = format_uint(nb, sizeof(nb), static_cast<std::uint32_t>(id));
  out.append(nb, n);
  out.append("</key><value>");

  // <X size="dynamic"><value0>..</value0><value1>..</value1><value2>..</value2></X>
  out.append("<X size=\"dynamic\">");
  for (int i = 0; i < 3; ++i)
  {
    const char idx_ch = static_cast<char>('0' + i);
    out.append("<value", 6).append(1, idx_ch).push_back('>');
    n = format_double(nb, sizeof(nb), lm.X(i));
    out.append(nb, n);
    out.append("</value", 7).append(1, idx_ch).push_back('>');
  }
  out.append("</X>");

  // <observations size="dynamic"> ... </observations>
  out.append("<observations size=\"dynamic\">");

  // Sort observations by view id for deterministic output (matches the
  // legacy Landmark::save which routes through std::map).
  std::vector<IndexT> obs_keys;
  obs_keys.reserve(lm.obs.size());
  for (const auto & kv : lm.obs) obs_keys.push_back(kv.first);
  std::sort(obs_keys.begin(), obs_keys.end());

  for (std::size_t k = 0; k < obs_keys.size(); ++k)
  {
    const IndexT view_id = obs_keys[k];
    const Observation & ob = lm.obs.find(view_id)->second;

    n = std::snprintf(nb, sizeof(nb), "<value%zu><key>", k);
    out.append(nb, n);
    n = format_uint(nb, sizeof(nb), static_cast<std::uint32_t>(view_id));
    out.append(nb, n);
    out.append("</key><value><id_feat>");
    n = format_uint(nb, sizeof(nb), static_cast<std::uint32_t>(ob.id_feat));
    out.append(nb, n);
    out.append("</id_feat><x size=\"dynamic\"><value0>");
    n = format_double(nb, sizeof(nb), ob.x(0));
    out.append(nb, n);
    out.append("</value0><value1>");
    n = format_double(nb, sizeof(nb), ob.x(1));
    out.append(nb, n);
    out.append("</value1></x></value>");
    n = std::snprintf(nb, sizeof(nb), "</value%zu>", k);
    out.append(nb, n);
  }
  out.append("</observations></value>");

  // </valueN>
  n = std::snprintf(nb, sizeof(nb), "</value%zu>\n", wrapper_idx);
  out.append(nb, n);
}

// Build the inside-of-`<structure>` text in parallel.
std::string BuildStructureBody(const Landmarks & structure)
{
  const std::size_t N = structure.size();
  if (N == 0) return {};

  // Sort landmark ids for deterministic output.
  std::vector<IndexT> ids;
  ids.reserve(N);
  for (const auto & kv : structure) ids.push_back(kv.first);
  std::sort(ids.begin(), ids.end());

  // Choose chunk count -- target ~num_threads, but keep each chunk meaty
  // enough that thread startup / final concat dominates nothing.
  std::size_t chunks = 1;
#ifdef _OPENMP
  chunks = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#endif
  const std::size_t kMinChunk = 4096;
  while (chunks > 1 && N / chunks < kMinChunk) --chunks;

  const std::size_t chunk_size = (N + chunks - 1) / chunks;
  std::vector<std::string> parts(chunks);

  // Rough average per-landmark size: 3 doubles + ~5 observations * (id + 2 doubles).
  // Empirically ~250-400 bytes; reserve 300 to keep reallocs few.
  const std::size_t kBytesPerLandmark = 320;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (long long c = 0; c < static_cast<long long>(chunks); ++c)
  {
    const std::size_t cu = static_cast<std::size_t>(c);
    const std::size_t begin = cu * chunk_size;
    const std::size_t end   = std::min(N, begin + chunk_size);
    std::string & buf = parts[cu];
    buf.reserve((end - begin) * kBytesPerLandmark);

    for (std::size_t i = begin; i < end; ++i)
    {
      const IndexT id = ids[i];
      const Landmark & lm = structure.find(id)->second;
      AppendLandmark(buf, i, id, lm);
    }
  }

  // Concatenate; release each chunk's memory as we go.
  std::size_t total = 0;
  for (const auto & p : parts) total += p.size();
  std::string out;
  out.reserve(total);
  for (auto & p : parts)
  {
    out.append(p);
    std::string().swap(p);
  }
  return out;
}

// Serialize everything EXCEPT structure to an in-memory string via cereal.
// `structure` is replaced with an empty Landmarks() placeholder; we locate it
// later and splice in the hand-built body.
bool BuildSkeleton(const SfM_Data & data, ESfM_Data flags_part, std::string & out)
{
  const bool b_views      = (flags_part & VIEWS) == VIEWS;
  const bool b_intrinsics = (flags_part & INTRINSICS) == INTRINSICS;
  const bool b_extrinsics = (flags_part & EXTRINSICS) == EXTRINSICS;
  const bool b_cp         = (flags_part & CONTROL_POINTS) == CONTROL_POINTS;

  std::ostringstream ss(std::ios::binary | std::ios::out);
  try
  {
    cereal::XMLOutputArchive archive(ss);
    const std::string version = "0.3";
    archive(cereal::make_nvp("sfm_data_version", version));
    archive(cereal::make_nvp("root_path", data.s_root_path));

    if (b_views) archive(cereal::make_nvp("views", data.views));
    else         archive(cereal::make_nvp("views", Views()));

    if (b_intrinsics) archive(cereal::make_nvp("intrinsics", data.intrinsics));
    else              archive(cereal::make_nvp("intrinsics", Intrinsics()));

    if (b_extrinsics) archive(cereal::make_nvp("extrinsics", data.poses));
    else              archive(cereal::make_nvp("extrinsics", Poses()));

    // Empty placeholder; spliced in Save_XML_Fast() below.
    archive(cereal::make_nvp("structure", Landmarks()));

    if (b_cp) archive(cereal::make_nvp("control_points", data.control_points));
    else      archive(cereal::make_nvp("control_points", Landmarks()));
  }
  catch (const cereal::Exception & e)
  {
    OPENMVG_LOG_ERROR << "Save_XML_Fast: cereal skeleton failed: " << e.what();
    return false;
  }
  out = ss.str();
  return true;
}

} // anonymous namespace

bool Save_XML_Fast(
  const SfM_Data & data,
  const std::string & filename,
  ESfM_Data flags_part)
{
  const bool b_structure = (flags_part & STRUCTURE) == STRUCTURE;

  // Phase 1: skeleton from cereal.
  std::string skeleton;
  if (!BuildSkeleton(data, flags_part, skeleton))
    return false;

  // Phase 2: locate the placeholder `<structure ...>` ... `</structure>` region.
  // It may be emitted self-closing (`<structure size="dynamic"/>`) or as an
  // open/close pair, depending on rapidxml's print implementation.
  const std::size_t open_pos = skeleton.find("<structure");
  if (open_pos == std::string::npos)
  {
    OPENMVG_LOG_ERROR << "Save_XML_Fast: could not locate <structure> in skeleton.";
    return false;
  }
  const std::size_t open_end = skeleton.find('>', open_pos);
  if (open_end == std::string::npos) return false;
  const bool self_closing = (open_end > 0 && skeleton[open_end - 1] == '/');

  std::size_t suffix_begin;
  if (self_closing)
  {
    suffix_begin = open_end + 1;
  }
  else
  {
    static const char kClose[] = "</structure>";
    const std::size_t close_pos = skeleton.find(kClose, open_end + 1);
    if (close_pos == std::string::npos) return false;
    suffix_begin = close_pos + (sizeof(kClose) - 1);
  }

  // Phase 3: build the structure body in parallel (skipped when STRUCTURE
  // flag was not requested -- we still emit an empty container).
  std::string body;
  if (b_structure)
    body = BuildStructureBody(data.structure);

  // Phase 4: stream to disk with a large streambuf (4 MB) to keep the
  // syscall count low on the gigabyte-scale outputs typical of large scenes.
  // pubsetbuf must be called before any I/O on the stream, so we open
  // separately rather than via the ofstream ctor.
  std::vector<char> bigbuf(1u << 22);
  std::ofstream stream;
  stream.rdbuf()->pubsetbuf(bigbuf.data(), static_cast<std::streamsize>(bigbuf.size()));
  stream.open(filename.c_str(), std::ios::binary | std::ios::out);
  if (!stream)
  {
    OPENMVG_LOG_ERROR << "Save_XML_Fast: cannot open output \"" << filename << "\".";
    return false;
  }

  // Bytes before the placeholder structure element.
  stream.write(skeleton.data(), static_cast<std::streamsize>(open_pos));
  // Hand-written replacement structure element.
  static const char kOpen[] = "<structure size=\"dynamic\">";
  stream.write(kOpen, sizeof(kOpen) - 1);
  if (!body.empty())
  {
    stream.put('\n');
    stream.write(body.data(), static_cast<std::streamsize>(body.size()));
  }
  static const char kClose2[] = "</structure>";
  stream.write(kClose2, sizeof(kClose2) - 1);
  // Bytes after the placeholder.
  stream.write(skeleton.data() + suffix_begin,
               static_cast<std::streamsize>(skeleton.size() - suffix_begin));

  stream.flush();
  const bool ok = static_cast<bool>(stream);
  stream.close();
  return ok;
}

} // namespace sfm
} // namespace openMVG
