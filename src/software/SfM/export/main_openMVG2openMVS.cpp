// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2016 cDc <cdc.seacave@gmail.com>, Pierre MOULON

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/cameras/Camera_Pinhole.hpp"
#include "openMVG/cameras/Camera_undistort_image.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/image/sample.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/system/loggerprogress.hpp"

#define _USE_EIGEN
#include "InterfaceMVS.h"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::image;
using namespace openMVG::sfm;

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

// SSE2 SIMD bilinear: process all 3 colour channels of one output pixel as
// a single 4-lane float vector. SSE2 is the MSVC x64 ABI baseline (and is
// otherwise guarded by __SSE2__), so this path is always taken on x64.
// We deliberately do NOT use any SSE4.1+ intrinsics here so the binary
// runs on any SSE2-capable CPU without the project needing /arch:SSE4 or
// /arch:AVX. Higher-ISA paths (AVX2 below) are kept as dead helpers for
// future runtime-dispatched use; they are not entered at compile time.
#if defined(_M_X64) || defined(__SSE2__)
  #define OPENMVG_MVS_HAS_SSE2 1
  #include <emmintrin.h>
#else
  #define OPENMVG_MVS_HAS_SSE2 0
#endif

// AVX2 SIMD bilinear: process two RGB output pixels per call in a single
// 256-bit YMM register. CMake compiles this TU with /arch:AVX2 (MSVC) or
// -mavx2 -mfma (gcc/clang), so __AVX2__ is reliably set.
#if defined(__AVX2__)
  #define OPENMVG_MVS_HAS_AVX2 0
  #include <immintrin.h>
#else
  #define OPENMVG_MVS_HAS_AVX2 0
#endif

// A/B toggle: 1 = use the explicit 256-bit BilinearInterior2PixelsRGB helper
// in the pair loop; 0 = two back-to-back SSE4.1 1-pixel calls (auto VEX-encoded
// under /arch:AVX2). Previous measurements: 0 was faster (~33.5% vs ~35.5%
// kernel self-time) on this workload, but the helper may win on different
// hardware / image patterns. Set to 1 to re-test.
#define OPENMVG_MVS_USE_AVX2_2PIX 0

namespace {

// Two-stream bilinear tap layout. The hot inner loop is dominated by the
// interior case (>=99% of output pixels for typical lenses); the rim/OOB
// case is <1%. Splitting into two vectors lets the interior loop run with
// zero branches and uniform memory access (great for SIMD / prefetch),
// while the tiny edge loop pays a branch but only over a few thousand
// entries.
//
// Interior bilinear tap. 8 bytes.
//
// dst[implicit_out_idx] = bilinear(src[tl_idx], +1, +stride, +stride+1)
// with fractional weights (wx, wy) stored as Q0.16 fixed-point.
//
// The output index is *implicit*: interior taps belong to per-row spans
// (InteriorRowSpan) and are stored consecutively for each span. The kernel
// walks each span as a flat raster of `length` pixels into the destination
// row, so no out_idx field is needed. Weights run through the kernel as
// float -- one mul+cvtsi at the call site is hidden behind the source loads.
// 16 bits of weight precision is overkill against 8-bit pixels.
struct InteriorTap
{
  int32_t  tl_idx;
  uint16_t wx_q16;   // wx * 65535, rounded
  uint16_t wy_q16;   // wy * 65535, rounded
};
static_assert(sizeof(InteriorTap) == 8, "InteriorTap must be tightly packed");

// One contiguous run of interior pixels in a destination row. Most lenses
// (radial barrel / pincushion) give a single run per row -- the rim falls
// into the edge stream at the row ends and possibly mid-row for severe
// distortion. Builder emits one span per run; kernel iterates spans and
// then walks `length` consecutive (mini-tap, dst pixel) positions, giving
// sequential write-streams and a 33% tap-bandwidth cut versus carrying a
// per-pixel out_idx.
struct InteriorRowSpan
{
  uint32_t out_base;     // = row*W + x_begin, first destination index
  uint32_t tap_offset;   // first interior-tap index for this span
  uint16_t x_begin;      // (kept for debugging / asserts)
  uint16_t length;       // number of interior taps in this span
};
static_assert(sizeof(InteriorRowSpan) == 12, "InteriorRowSpan must be tightly packed");

// Scale factor for converting Q0.16 weights back to float in [0,1].
// 1/65535 (not 1/65536) so an input weight of exactly 1.0 round-trips.
inline constexpr float kQ16ToFloat = 1.0f / 65535.0f;

// Edge / partial-OOB tap: dst[out_idx] = sampler(src, sy, sx) if
// Contains(sy, sx) else fillcolor. 12 bytes.
struct EdgeTap
{
  int32_t out_idx;
  float   sx;
  float   sy;
};
static_assert(sizeof(EdgeTap) == 12, "EdgeTap must be tightly packed");

// Per-(intrinsic, W, H) precomputed distortion lookup. W, H are the
// undistorted dims. The interior spans + edge stream together cover every
// output pixel exactly once -- caller does NOT need to memset the
// destination buffer before processing.
struct DistoMap
{
  uint32_t W = 0;
  uint32_t H = 0;
  std::vector<InteriorTap>     interior;
  std::vector<InteriorRowSpan> spans;
  std::vector<EdgeTap>         edge;
};

// Fast 2x2 bilinear for the interior case. The 4 taps are addressed directly
// via the precomputed top-left index `tl` plus the source row stride; no
// per-tap bounds branch, no total_weight divide, no double precision. The
// separable lerp form `a + (b - a) * t` fuses cleanly to FMA on AVX2.
inline openMVG::image::RGBColor
BilinearInteriorRGB(
  const openMVG::image::RGBColor * __restrict src_data,
  const int32_t tl,
  const int32_t src_stride,
  const float u, const float v)
{
#if OPENMVG_MVS_HAS_SSE2
  // SIMD path: process R, G, B in parallel as one float4 vector per pixel.
  // Each pixel is 3 contiguous bytes. We load 4 bytes -- the 4th is the R
  // byte of the next horizontal pixel (or 1 byte past the BR pixel for the
  // very last cell). BuildDistortionMaps tightens the interior bound to
  // guarantee at least one valid byte after the bottom-right tap, so this
  // 1-byte over-read is always inside imageIn's buffer.
  const uint8_t * base = reinterpret_cast<const uint8_t *>(src_data)
                       + static_cast<size_t>(tl) * 3u;
  const uint8_t * row0 = base;
  const uint8_t * row1 = base + static_cast<size_t>(src_stride) * 3u;

  // 4-byte unaligned loads into the low dword of an xmm register, then
  // expand the 4 u8s to 4 i32 lanes (lane 3 is garbage R of the next pixel
  // and is ignored after the final pack). Uses pure SSE2: a two-step
  // zero-extend (u8 -> u16 -> u32) against a zero vector instead of the
  // SSE4.1 PMOVZXBD. Both lower 8-byte halves of the unpack results carry
  // the four lanes we want, so a single unpacklo at each step is enough.
  const __m128i z = _mm_setzero_si128();
  uint32_t u00, u10, u01, u11;
  std::memcpy(&u00, row0,     4);
  std::memcpy(&u10, row0 + 3, 4);
  std::memcpy(&u01, row1,     4);
  std::memcpy(&u11, row1 + 3, 4);

  const __m128 p00 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(u00), z), z));
  const __m128 p10 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(u10), z), z));
  const __m128 p01 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(u01), z), z));
  const __m128 p11 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(u11), z), z));

  const __m128 wx = _mm_set1_ps(u);
  const __m128 wy = _mm_set1_ps(v);

  // Separable lerp: top = p00 + (p10 - p00) * wx, etc. Compiler emits
  // mulps+addps under SSE2 (the dependency chain is the same length as
  // the FMA variant; we just lose the fused-rounding micro-op).
  const __m128 top = _mm_add_ps(p00, _mm_mul_ps(_mm_sub_ps(p10, p00), wx));
  const __m128 bot = _mm_add_ps(p01, _mm_mul_ps(_mm_sub_ps(p11, p01), wx));
  const __m128 res = _mm_add_ps(top, _mm_mul_ps(_mm_sub_ps(bot, top), wy));

  // Round-to-nearest-even (MXCSR default) instead of the scalar `+0.5f`
  // truncation: differs only when the residual is *exactly* 0.5, which on
  // a [0,255] convex combination is vanishingly rare and within bilinear
  // resampling noise. Saves an add + a sub-cycle on the final pack.
  const __m128i resi = _mm_cvtps_epi32(res);
  // Pack 4 i32 -> 4 i16 (saturate) -> 4 u8 (saturate). Saturation is a no-op
  // here -- inputs are in [0,255] -- but it gives the hardware a single-uop
  // path. The low 32 bits of the result are [R, G, B, garbage].
  const __m128i res_i16 = _mm_packs_epi32(resi, resi);
  const __m128i res_u8  = _mm_packus_epi16(res_i16, res_i16);
  const uint32_t packed = static_cast<uint32_t>(_mm_cvtsi128_si32(res_u8));
  return openMVG::image::RGBColor(
    static_cast<unsigned char>(packed         & 0xFFu),
    static_cast<unsigned char>((packed >> 8)  & 0xFFu),
    static_cast<unsigned char>((packed >> 16) & 0xFFu));
#else
  // Scalar fallback (pre-SSE4.1 or non-x86). Same math as the SIMD path.
  const openMVG::image::RGBColor & p00 = src_data[tl];
  const openMVG::image::RGBColor & p10 = src_data[tl + 1];
  const openMVG::image::RGBColor & p01 = src_data[tl + src_stride];
  const openMVG::image::RGBColor & p11 = src_data[tl + src_stride + 1];
  const float r00 = static_cast<float>(p00.r());
  const float r10 = static_cast<float>(p10.r());
  const float r01 = static_cast<float>(p01.r());
  const float r11 = static_cast<float>(p11.r());
  const float g00 = static_cast<float>(p00.g());
  const float g10 = static_cast<float>(p10.g());
  const float g01 = static_cast<float>(p01.g());
  const float g11 = static_cast<float>(p11.g());
  const float b00 = static_cast<float>(p00.b());
  const float b10 = static_cast<float>(p10.b());
  const float b01 = static_cast<float>(p01.b());
  const float b11 = static_cast<float>(p11.b());
  const float r_top = r00 + (r10 - r00) * u;
  const float r_bot = r01 + (r11 - r01) * u;
  const float g_top = g00 + (g10 - g00) * u;
  const float g_bot = g01 + (g11 - g01) * u;
  const float b_top = b00 + (b10 - b00) * u;
  const float b_bot = b01 + (b11 - b01) * u;
  const float r = r_top + (r_bot - r_top) * v;
  const float g = g_top + (g_bot - g_top) * v;
  const float b = b_top + (b_bot - b_top) * v;
  return openMVG::image::RGBColor(
    static_cast<unsigned char>(r + 0.5f),
    static_cast<unsigned char>(g + 0.5f),
    static_cast<unsigned char>(b + 0.5f));
#endif
}

inline uint8_t
BilinearInteriorGray(
  const uint8_t * __restrict src_data,
  const int32_t tl,
  const int32_t src_stride,
  const float u, const float v)
{
  const float p00 = static_cast<float>(src_data[tl]);
  const float p10 = static_cast<float>(src_data[tl + 1]);
  const float p01 = static_cast<float>(src_data[tl + src_stride]);
  const float p11 = static_cast<float>(src_data[tl + src_stride + 1]);
  const float top = p00 + (p10 - p00) * u;
  const float bot = p01 + (p11 - p01) * u;
  const float v_  = top + (bot - top) * v;
  return static_cast<uint8_t>(v_ + 0.5f);
}

#if OPENMVG_MVS_HAS_AVX2
// Two-pixel AVX2 RGB bilinear. Both pixels' 4 taps and weights are packed
// into a single YMM, the bilinear math runs in 8-wide float lanes
// (R0,G0,B0,_, R1,G1,B1,_), and the two RGB results are extracted from the
// low and high 128-bit halves. Compared to two back-to-back SSE4.1 calls
// this roughly halves the per-pixel SIMD instruction count, doubles the
// effective FMA throughput, and gives the OoO scheduler one independent
// dataflow graph per pixel pair instead of two serialised ones.
inline void BilinearInterior2PixelsRGB(
  const openMVG::image::RGBColor * __restrict src_data,
  const int32_t tl0, const float u0, const float v0,
  const int32_t tl1, const float u1, const float v1,
  const int32_t src_stride,
  openMVG::image::RGBColor & __restrict out0,
  openMVG::image::RGBColor & __restrict out1)
{
  // Compute byte base addresses for each pixel's row0 / row1.
  const uint8_t * const src_b = reinterpret_cast<const uint8_t *>(src_data);
  const size_t   stride_b   = static_cast<size_t>(src_stride) * 3u;
  const uint8_t * row0_0 = src_b + static_cast<size_t>(tl0) * 3u;
  const uint8_t * row1_0 = row0_0 + stride_b;
  const uint8_t * row0_1 = src_b + static_cast<size_t>(tl1) * 3u;
  const uint8_t * row1_1 = row0_1 + stride_b;

  // Load four 4-byte chunks per pixel (4 byte over-read into the next
  // horizontal pixel; BuildDistortionMaps guarantees this stays in buffer).
  uint32_t u00_0, u10_0, u01_0, u11_0;
  uint32_t u00_1, u10_1, u01_1, u11_1;
  std::memcpy(&u00_0, row0_0,     4);
  std::memcpy(&u10_0, row0_0 + 3, 4);
  std::memcpy(&u01_0, row1_0,     4);
  std::memcpy(&u11_0, row1_0 + 3, 4);
  std::memcpy(&u00_1, row0_1,     4);
  std::memcpy(&u10_1, row0_1 + 3, 4);
  std::memcpy(&u01_1, row1_1,     4);
  std::memcpy(&u11_1, row1_1 + 3, 4);

  // Pack {pix0_u8s, pix1_u8s} into low 8 bytes of an xmm, then widen 8 u8
  // -> 8 i32 across a ymm. Each 256-bit register holds both pixels' taps.
  auto pack_two = [](uint32_t a, uint32_t b) {
    return _mm_unpacklo_epi32(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b));
  };
  const __m256i p00_i32 = _mm256_cvtepu8_epi32(pack_two(u00_0, u00_1));
  const __m256i p10_i32 = _mm256_cvtepu8_epi32(pack_two(u10_0, u10_1));
  const __m256i p01_i32 = _mm256_cvtepu8_epi32(pack_two(u01_0, u01_1));
  const __m256i p11_i32 = _mm256_cvtepu8_epi32(pack_two(u11_0, u11_1));
  const __m256  p00 = _mm256_cvtepi32_ps(p00_i32);
  const __m256  p10 = _mm256_cvtepi32_ps(p10_i32);
  const __m256  p01 = _mm256_cvtepi32_ps(p01_i32);
  const __m256  p11 = _mm256_cvtepi32_ps(p11_i32);

  // Lane layout: [u0 u0 u0 u0 | u1 u1 u1 u1] etc.
  const __m256 wx = _mm256_setr_ps(u0, u0, u0, u0, u1, u1, u1, u1);
  const __m256 wy = _mm256_setr_ps(v0, v0, v0, v0, v1, v1, v1, v1);

  // Separable lerp. With -mfma / /arch:AVX2 the compiler folds add+mul into
  // VFMADD231PS.
  const __m256 top = _mm256_add_ps(p00, _mm256_mul_ps(_mm256_sub_ps(p10, p00), wx));
  const __m256 bot = _mm256_add_ps(p01, _mm256_mul_ps(_mm256_sub_ps(p11, p01), wx));
  const __m256 res = _mm256_add_ps(top, _mm256_mul_ps(_mm256_sub_ps(bot, top), wy));

  // Pack 8 i32 -> 8 u8. _mm256_packs_epi32 works per-128-bit-lane, so the
  // result's low xmm contains pixel 0's RGB+garbage in the first dword and
  // the high xmm contains pixel 1's RGB+garbage in its first dword.
  const __m256i resi    = _mm256_cvtps_epi32(res);
  const __m256i res_i16 = _mm256_packs_epi32(resi, resi);
  const __m256i res_u8  = _mm256_packus_epi16(res_i16, res_i16);

  const uint32_t packed0 = static_cast<uint32_t>(
    _mm_cvtsi128_si32(_mm256_castsi256_si128(res_u8)));
  const uint32_t packed1 = static_cast<uint32_t>(
    _mm_cvtsi128_si32(_mm256_extracti128_si256(res_u8, 1)));

  out0 = openMVG::image::RGBColor(
    static_cast<unsigned char>(packed0         & 0xFFu),
    static_cast<unsigned char>((packed0 >> 8)  & 0xFFu),
    static_cast<unsigned char>((packed0 >> 16) & 0xFFu));
  out1 = openMVG::image::RGBColor(
    static_cast<unsigned char>(packed1         & 0xFFu),
    static_cast<unsigned char>((packed1 >> 8)  & 0xFFu),
    static_cast<unsigned char>((packed1 >> 16) & 0xFFu));
}
#endif // OPENMVG_MVS_HAS_AVX2

// Faster replacement for openMVG::cameras::UndistortImage that consults a
// precomputed two-stream tap table. The interior loop is a tight, branch-free
// stream over ~99% of output pixels -- one InteriorTap read, four source taps,
// one bilinear call, one store. The edge loop walks the rim (~1%) with the
// sampler-or-fillcolor branch. No per-pixel `if (interior?)` check, and no
// destination buffer pre-fill: every pixel is written exactly once across the
// two loops combined.
template <typename Image>
void UndistortImageWithMap(
  const Image & imageIn,
  const DistoMap & dm,
  Image & image_ud,
  typename Image::Tpixel fillcolor = typename Image::Tpixel(0))
{
  static_assert(
    std::is_same_v<typename Image::Tpixel, openMVG::image::RGBColor> ||
    std::is_same_v<typename Image::Tpixel, uint8_t>,
    "UndistortImageWithMap only supports RGBColor and uint8_t pixel types");

  // No-init resize: every output pixel is covered by interior+edge streams.
  image_ud.resize(static_cast<int>(dm.W), static_cast<int>(dm.H), false);

  // __restrict: thread-local imageIn / image_ud are distinct buffers,
  // dm.interior / dm.edge are owned by a separate DistoMap. Lets MSVC keep
  // src/tap loads in registers across the bilinear inline.
  const int32_t src_stride = imageIn.Width();
  const auto * __restrict const src_data = imageIn.data();
  auto       * __restrict const dst_data = image_ud.data();

  // ----- Interior spans: branch-free, the hot path. Each span walks a
  // contiguous run of destination pixels with sequential tap and sequential
  // output writes -- prefetcher-friendly for both streams.
  {
    const InteriorTap      * __restrict const tap_base  = dm.interior.data();
    const InteriorRowSpan  * __restrict       sp        = dm.spans.data();
    const InteriorRowSpan  * __restrict const sp_end    = sp + dm.spans.size();
    if constexpr (std::is_same_v<typename Image::Tpixel, openMVG::image::RGBColor>)
    {
      for (; sp != sp_end; ++sp)
      {
        const InteriorTap * __restrict mt    = tap_base + sp->tap_offset;
        auto              * __restrict drow  = dst_data + sp->out_base;
        const uint32_t                 N     = sp->length;
#if OPENMVG_MVS_HAS_AVX2
        // Two back-to-back SSE4.1 1-pixel calls per iteration. Under
        // /arch:AVX2 these get VEX-encoded automatically (no AVX/SSE
        // transition penalty), the two independent dataflow chains give
        // the OoO scheduler enough ILP to hide the FMA latency without
        // the register pressure of a wider unroll. We avoid the explicit
        // 256-bit helper's cross-lane shuffles and 8 scalar inserts.
        const uint32_t N_pair = N & ~uint32_t(1);
        for (uint32_t k = 0; k < N_pair; k += 2)
        {
          const float a_wx = static_cast<float>(mt[k    ].wx_q16) * kQ16ToFloat;
          const float a_wy = static_cast<float>(mt[k    ].wy_q16) * kQ16ToFloat;
          const float b_wx = static_cast<float>(mt[k + 1].wx_q16) * kQ16ToFloat;
          const float b_wy = static_cast<float>(mt[k + 1].wy_q16) * kQ16ToFloat;
#if OPENMVG_MVS_USE_AVX2_2PIX
          BilinearInterior2PixelsRGB(
            src_data,
            mt[k    ].tl_idx, a_wx, a_wy,
            mt[k + 1].tl_idx, b_wx, b_wy,
            src_stride,
            drow[k    ],
            drow[k + 1]);
#else
          drow[k    ] = BilinearInteriorRGB(src_data, mt[k    ].tl_idx, src_stride, a_wx, a_wy);
          drow[k + 1] = BilinearInteriorRGB(src_data, mt[k + 1].tl_idx, src_stride, b_wx, b_wy);
#endif
        }
        // Scalar tail for odd count.
        for (uint32_t k = N_pair; k < N; ++k)
        {
          const float wx = static_cast<float>(mt[k].wx_q16) * kQ16ToFloat;
          const float wy = static_cast<float>(mt[k].wy_q16) * kQ16ToFloat;
          drow[k] = BilinearInteriorRGB(src_data, mt[k].tl_idx, src_stride, wx, wy);
        }
#else
        for (uint32_t k = 0; k < N; ++k)
        {
          const float wx = static_cast<float>(mt[k].wx_q16) * kQ16ToFloat;
          const float wy = static_cast<float>(mt[k].wy_q16) * kQ16ToFloat;
          drow[k] = BilinearInteriorRGB(src_data, mt[k].tl_idx, src_stride, wx, wy);
        }
#endif
      }
    }
    else // uint8_t
    {
      for (; sp != sp_end; ++sp)
      {
        const InteriorTap * __restrict mt   = tap_base + sp->tap_offset;
        auto              * __restrict drow = dst_data + sp->out_base;
        const uint32_t                 N    = sp->length;
        for (uint32_t k = 0; k < N; ++k)
        {
          const float wx = static_cast<float>(mt[k].wx_q16) * kQ16ToFloat;
          const float wy = static_cast<float>(mt[k].wy_q16) * kQ16ToFloat;
          drow[k] = BilinearInteriorGray(src_data, mt[k].tl_idx, src_stride, wx, wy);
        }
      }
    }
  }

  // ----- Edge stream: short, per-entry branch on Contains(). -----
  if (!dm.edge.empty())
  {
    const openMVG::image::Sampler2d<openMVG::image::SamplerLinear> sampler;
    const EdgeTap * __restrict it  = dm.edge.data();
    const EdgeTap * __restrict end = it + dm.edge.size();
    for (; it != end; ++it)
    {
      const double dx = static_cast<double>(it->sx);
      const double dy = static_cast<double>(it->sy);
      dst_data[it->out_idx] =
        imageIn.Contains(dy, dx) ? sampler(imageIn, dy, dx) : fillcolor;
    }
  }
}

// Build distortion lookup tables for every distorted intrinsic that will be
// consumed by the undistort-image loop. Per-table construction is row-parallel
// via OpenMP so each table is built using all cores. Tables are keyed by
// intrinsic id; we additionally verify (W, H) at use site and fall back to the
// uncached UndistortImage if the source image dimensions don't match the
// intrinsic, so this is purely an optimization with no semantic change.
std::unordered_map<openMVG::IndexT, DistoMap>
BuildDistortionMaps(const SfM_Data & sfm_data)
{
  // Collect unique distorted intrinsic ids actually referenced by views with
  // a defined pose+intrinsic (matches the predicate used in the image loop).
  std::vector<openMVG::IndexT> needed;
  needed.reserve(sfm_data.intrinsics.size());
  {
    std::unordered_map<openMVG::IndexT, char> seen;
    seen.reserve(sfm_data.intrinsics.size() * 2);
    for (const auto & v : sfm_data.views)
    {
      const View * view = v.second.get();
      if (!sfm_data.IsPoseAndIntrinsicDefined(view)) continue;
      const auto it = sfm_data.GetIntrinsics().find(view->id_intrinsic);
      if (it == sfm_data.GetIntrinsics().end()) continue;
      const auto * cam = it->second.get();
      if (!cam->have_disto()) continue;
      if (seen.emplace(view->id_intrinsic, 1).second)
        needed.push_back(view->id_intrinsic);
    }
  }

  std::unordered_map<openMVG::IndexT, DistoMap> out;
  out.reserve(needed.size() * 2);
  for (const auto id : needed) (void)out[id];

  // Build each table. We partition output pixels into an interior stream
  // (interior bilinear taps -- the hot, branch-free path consumed by the
  // image loop) and a much smaller edge stream (rim / OOB pixels).
  // Construction is parallelized by rows within each intrinsic: every thread
  // builds per-row local vectors, then a serial merge concatenates them in
  // row order so the final tap/span/edge streams are identical to the old
  // serial builder. This is the right granularity because typically there
  // are 1-3 unique intrinsics and thousands of rows each.
  for (const auto id : needed)
  {
    const auto * cam = sfm_data.GetIntrinsics().at(id).get();
    DistoMap & m = out[id];
    m.W = cam->w();
    m.H = cam->h();
    if (m.W == 0 || m.H == 0) continue;

    const int W = static_cast<int>(m.W);
    const int H = static_cast<int>(m.H);

#if OPENMVG_MVS_HAS_SSE2
    const float xmax = static_cast<float>(W - 2);
    const float ymax = static_cast<float>(H - 2);
    const float xmax_tight = static_cast<float>(W - 3);
    const float ymax_tight = static_cast<float>(H - 3);
#else
    const float xmax = static_cast<float>(W - 2);
    const float ymax = static_cast<float>(H - 2);
#endif

    // Per-row results: each row produces its own taps, spans, edges.
    struct RowResult
    {
      std::vector<InteriorTap>     taps;
      std::vector<InteriorRowSpan> spans;
      std::vector<EdgeTap>         edges;
    };
    std::vector<RowResult> rows(static_cast<size_t>(H));

#ifdef OPENMVG_USE_OPENMP
    #pragma omp parallel for schedule(dynamic, 32)
#endif
    for (int j = 0; j < H; ++j)
    {
      RowResult & rr = rows[static_cast<size_t>(j)];
      rr.taps.reserve(static_cast<size_t>(W));
      // edges/spans typically tiny — default capacity is fine.

      const int32_t row_off = j * W;
      int    run_begin   = -1;
      size_t run_tap_off = 0;

      for (int i = 0; i < W; ++i)
      {
        const Vec2 d = cam->get_d_pixel(Vec2(i, j));
        const float fx = static_cast<float>(d.x());
        const float fy = static_cast<float>(d.y());

        const bool in_bounds =
          fx >= 0.0f && fy >= 0.0f && fx <= xmax && fy <= ymax;
#if OPENMVG_MVS_HAS_SSE2
        const bool simd_safe = fx <= xmax_tight || fy <= ymax_tight;
        const bool is_interior = in_bounds && simd_safe;
#else
        const bool is_interior = in_bounds;
#endif
        if (is_interior)
        {
          const int32_t x0 = static_cast<int32_t>(fx);
          const int32_t y0 = static_cast<int32_t>(fy);
          const float fwx = fx - static_cast<float>(x0);
          const float fwy = fy - static_cast<float>(y0);
          const uint32_t qx = static_cast<uint32_t>(fwx * 65535.0f + 0.5f);
          const uint32_t qy = static_cast<uint32_t>(fwy * 65535.0f + 0.5f);
          if (run_begin < 0)
          {
            run_begin   = i;
            run_tap_off = rr.taps.size();
          }
          InteriorTap t;
          t.tl_idx  = y0 * W + x0;
          t.wx_q16  = static_cast<uint16_t>(qx > 0xFFFFu ? 0xFFFFu : qx);
          t.wy_q16  = static_cast<uint16_t>(qy > 0xFFFFu ? 0xFFFFu : qy);
          rr.taps.push_back(t);
        }
        else
        {
          if (run_begin >= 0)
          {
            InteriorRowSpan sp;
            sp.out_base   = static_cast<uint32_t>(row_off + run_begin);
            sp.tap_offset = static_cast<uint32_t>(run_tap_off);  // local offset, fixed up later
            sp.x_begin    = static_cast<uint16_t>(run_begin);
            sp.length     = static_cast<uint16_t>(i - run_begin);
            rr.spans.push_back(sp);
            run_begin = -1;
          }
          EdgeTap t;
          t.out_idx = row_off + i;
          t.sx      = fx;
          t.sy      = fy;
          rr.edges.push_back(t);
        }
      }
      if (run_begin >= 0)
      {
        InteriorRowSpan sp;
        sp.out_base   = static_cast<uint32_t>(row_off + run_begin);
        sp.tap_offset = static_cast<uint32_t>(run_tap_off);
        sp.x_begin    = static_cast<uint16_t>(run_begin);
        sp.length     = static_cast<uint16_t>(W - run_begin);
        rr.spans.push_back(sp);
      }
    } // end parallel for

    // Serial merge: concatenate per-row results in row order, fixing up
    // tap_offset in each span to be a global index into m.interior.
    {
      size_t total_taps  = 0, total_spans = 0, total_edges = 0;
      for (const auto & rr : rows)
      {
        total_taps  += rr.taps.size();
        total_spans += rr.spans.size();
        total_edges += rr.edges.size();
      }
      m.interior.reserve(total_taps);
      m.spans.reserve(total_spans);
      m.edge.reserve(total_edges);

      for (auto & rr : rows)
      {
        const uint32_t tap_base = static_cast<uint32_t>(m.interior.size());
        // Fix up span tap_offsets from row-local to global.
        for (auto & sp : rr.spans)
          sp.tap_offset += tap_base;
        m.interior.insert(m.interior.end(),
          std::make_move_iterator(rr.taps.begin()),
          std::make_move_iterator(rr.taps.end()));
        m.spans.insert(m.spans.end(),
          std::make_move_iterator(rr.spans.begin()),
          std::make_move_iterator(rr.spans.end()));
        m.edge.insert(m.edge.end(),
          std::make_move_iterator(rr.edges.begin()),
          std::make_move_iterator(rr.edges.end()));
        // Release per-row memory eagerly.
        rr.taps.clear(); rr.taps.shrink_to_fit();
        rr.spans.clear(); rr.spans.shrink_to_fit();
        rr.edges.clear(); rr.edges.shrink_to_fit();
      }
    }
  }
  return out;
}

} // namespace

bool exportToOpenMVS(
  const SfM_Data & sfm_data,
  const std::string & sOutFile,
  const std::string & sOutDir,
  const int iNumThreads = 0
  )
{
  // Create undistorted images directory structure
  if (!stlplus::is_folder(sOutDir))
  {
    stlplus::folder_create(sOutDir);
    if (!stlplus::is_folder(sOutDir))
    {
      OPENMVG_LOG_ERROR << "Cannot access to one of the desired output directory";
      return false;
    }
  }
  const std::string sOutSceneDir = stlplus::folder_part(sOutFile);
  const std::string sOutImagesDir = stlplus::folder_to_relative_path(sOutSceneDir, sOutDir);

  // Export data :
  _INTERFACE_NAMESPACE::Interface scene;
  size_t nPoses(0);
  const uint32_t nViews((uint32_t)sfm_data.GetViews().size());

  system::LoggerProgress my_progress_bar(nViews,"- PROCESS VIEWS -");

  // OpenMVG can have not contiguous index, use a map to create the required OpenMVS contiguous ID index
  std::map<openMVG::IndexT, uint32_t> map_intrinsic, map_view;

  // define a platform with all the intrinsic group
  for (const auto& intrinsic: sfm_data.GetIntrinsics())
  {
    if (isPinhole(intrinsic.second->getType()))
    {
      const Pinhole_Intrinsic * cam = dynamic_cast<const Pinhole_Intrinsic*>(intrinsic.second.get());
      if (map_intrinsic.count(intrinsic.first) == 0)
        map_intrinsic.insert(std::make_pair(intrinsic.first, scene.platforms.size()));
      _INTERFACE_NAMESPACE::Interface::Platform platform;
      // add the camera
      _INTERFACE_NAMESPACE::Interface::Platform::Camera camera;
      camera.width = cam->w();
      camera.height = cam->h();
      camera.K = cam->K();
      // sub-pose
      camera.R = Mat3::Identity();
      camera.C = Vec3::Zero();
      platform.cameras.push_back(camera);
      scene.platforms.push_back(platform);
    }
  }

  // define images & poses
  scene.images.reserve(nViews);
  for (const auto& view : sfm_data.GetViews())
  {
    ++my_progress_bar;

    const std::string srcImage = stlplus::create_filespec(sfm_data.s_root_path, view.second->s_Img_path);
    if (!stlplus::is_file(srcImage))
    {
      OPENMVG_LOG_INFO << "Cannot read the corresponding image: " << srcImage;
      return false;
    }

    if (sfm_data.IsPoseAndIntrinsicDefined(view.second.get())) 
    {
      map_view[view.first] = scene.images.size();

      _INTERFACE_NAMESPACE::Interface::Image image;
      image.name = stlplus::create_filespec(sOutImagesDir, view.second->s_Img_path);
      image.platformID = map_intrinsic.at(view.second->id_intrinsic);
      _INTERFACE_NAMESPACE::Interface::Platform& platform = scene.platforms[image.platformID];
      image.cameraID = 0;

      _INTERFACE_NAMESPACE::Interface::Platform::Pose pose;
      image.poseID = platform.poses.size();
      const openMVG::geometry::Pose3 poseMVG(sfm_data.GetPoseOrDie(view.second.get()));
      pose.R = poseMVG.rotation();
      pose.C = poseMVG.center();
      platform.poses.push_back(pose);
      ++nPoses;

      scene.images.emplace_back(image);
    }
    else
    {
      OPENMVG_LOG_INFO << "Cannot read the corresponding pose or intrinsic of view " << view.first;
    }
  }

  // Export undistorted images
  // Precompute per-intrinsic distortion lookup tables ONCE so the inner image
  // loop doesn't recompute cam->get_d_pixel(...) (iterative inverse-distortion
  // solve for radial models) for every pixel of every image. With one shared
  // intrinsic across hundreds of views this is the difference between
  // billions of solves and a few million.
  const std::unordered_map<openMVG::IndexT, DistoMap> disto_maps =
    BuildDistortionMaps(sfm_data);

  system::LoggerProgress my_progress_bar_images(sfm_data.views.size(), "- UNDISTORT IMAGES " );
  std::atomic<bool> bOk(true); // Use a boolean to track the status of the loop process
#ifdef OPENMVG_USE_OPENMP
  const unsigned int nb_max_thread = (iNumThreads > 0)? iNumThreads : omp_get_max_threads();
#endif

  // ---- OS file-cache prewarm ------------------------------------------------
  // The main image loop below decodes + undistorts + encodes each view in
  // parallel. On a cold OS file cache the threads serialize on disk I/O
  // (NtReadFile waits) because OpenMP schedule(dynamic) interleaves reads
  // with compute, leaving the storage queue half-empty. This pass issues a
  // tight parallel sequence of full-file reads into a discard buffer, which
  // saturates the SSD/NVMe queue and forces every byte of every input JPEG
  // into the Windows standby cache. The subsequent ReadJpgRawInto then hits
  // RAM. Goal is *consistent* profiles: on a warm cache this loop returns
  // in microseconds (the reads are already cached), on a cold cache it
  // makes the I/O phase predictable and short.
  {
    std::vector<std::string> warm_paths;
    warm_paths.reserve(sfm_data.views.size());
    for (const auto & v : sfm_data.views)
    {
      const View * view = v.second.get();
      if (!sfm_data.IsPoseAndIntrinsicDefined(view)) continue;
      warm_paths.push_back(
        stlplus::create_filespec(sfm_data.s_root_path, view->s_Img_path));
    }
    const int N = static_cast<int>(warm_paths.size());
#ifdef OPENMVG_USE_OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nb_max_thread)
#endif
    for (int i = 0; i < N; ++i)
    {
      // 64 KiB per-thread discard buffer; libc fread reuses its own
      // FILE-internal buffer too, so this never needs to be large.
      unsigned char buf[64 * 1024];
      if (FILE * fp = std::fopen(warm_paths[i].c_str(), "rb"))
      {
        while (std::fread(buf, 1, sizeof(buf), fp) == sizeof(buf)) { /* drain */ }
        std::fclose(fp);
      }
    }
  }
  // --------------------------------------------------------------------------

  // Open a parallel region so each thread owns persistent scratch image
  // buffers across iterations. Image::resize is a no-op when (W, H) match,
  // so after the first iteration on each thread the per-image allocation
  // cost for these four buffers drops to zero. Falls back to a single-
  // threaded reuse pattern when OpenMP is disabled.
  //
  // NOTE: a 3-stage pipeline (decode -> undistort -> encode with bounded
  // queues) was prototyped and measured ~20% slower than this design. Two
  // reasons it lost:
  //   1. Stage-decoupled buffers can't reuse the Image<>::resize() no-op
  //      fast path -- each in-flight item ends up allocating a fresh
  //      ~3*W*H buffer for decode and another for undistort, which
  //      Windows zero-fills on first touch (1 page fault per 4 KiB).
  //      The per-thread scratch-buffer reuse here amortises that cost to
  //      O(distinct (W,H)) instead of O(views).
  //   2. Splitting the nb_max_thread budget across three stages
  //      under-feeds the libjpeg-turbo codec (decode + encode are ~60%
  //      of the per-image work) -- the OS scheduler was already doing a
  //      better job overlapping JPEG I/O against undistort compute when
  //      every thread ran the whole sequence.
  // If revisiting, both costs would need to be addressed first: a
  // buffer pool with size-keyed reuse, and a wider undistort-stage
  // staffing skew. Until then, keep the flat OpenMP loop.
#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel num_threads(nb_max_thread)
#endif
  {
    Image<openMVG::image::RGBColor> imageRGB, imageRGB_ud;
    Image<uint8_t> image_gray, image_gray_ud;

#ifdef OPENMVG_USE_OPENMP
    #pragma omp for schedule(dynamic)
#endif
  for (int i = 0; i < static_cast<int>(sfm_data.views.size()); ++i)
  {
    ++my_progress_bar_images;

    if (!bOk)
      continue;

    Views::const_iterator iterViews = sfm_data.views.begin();
    std::advance(iterViews, i);
    const View * view = iterViews->second.get();

    // Get image paths
    const std::string srcImage = stlplus::create_filespec(sfm_data.s_root_path, view->s_Img_path);
    const std::string imageName = stlplus::create_filespec(sOutDir, view->s_Img_path);

    if (sfm_data.IsPoseAndIntrinsicDefined(view))
    {
      // export undistorted images
      const openMVG::cameras::IntrinsicBase * cam = sfm_data.GetIntrinsics().at(view->id_intrinsic).get();
      if (cam->have_disto())
      {
        // Thread-local scratch buffers reused from the enclosing parallel
        // region -- declarations moved out so resize() can hit the no-op
        // fast path across iterations.
        // Look up a precomputed distortion table for this intrinsic. If it
        // exists AND its size matches the actual image we just read, use the
        // fast cached-undistort path. Otherwise fall back to the original
        // UndistortImage (which recomputes per pixel) for full safety.
        const auto dm_it = disto_maps.find(view->id_intrinsic);
        try
        {
          if (ReadImage(srcImage.c_str(), &imageRGB))
          {
            if (dm_it != disto_maps.end() &&
                static_cast<int>(dm_it->second.W) == imageRGB.Width() &&
                static_cast<int>(dm_it->second.H) == imageRGB.Height())
            {
              UndistortImageWithMap(imageRGB, dm_it->second, imageRGB_ud, BLACK);
            }
            else
            {
              UndistortImage(imageRGB, cam, imageRGB_ud, BLACK);
            }
            bOk = WriteImage(imageName.c_str(), imageRGB_ud);
          }
          else // If RGBColor reading fails, try to read as gray image
          if (ReadImage(srcImage.c_str(), &image_gray))
          {
            if (dm_it != disto_maps.end() &&
                static_cast<int>(dm_it->second.W) == image_gray.Width() &&
                static_cast<int>(dm_it->second.H) == image_gray.Height())
            {
              UndistortImageWithMap(image_gray, dm_it->second, image_gray_ud, BLACK);
            }
            else
            {
              UndistortImage(image_gray, cam, image_gray_ud, BLACK);
            }
            const bool bRes = WriteImage(imageName.c_str(), image_gray_ud);
            bOk = bOk & bRes;
          }
          else
          {
            bOk = false;
          }
        }
        catch (const std::bad_alloc& e)
        {
          bOk = false;
        }
      }
      else
      {
        // just copy image
        stlplus::file_copy(srcImage, imageName);
      }
    }
    else
    {
      // just copy the image
      stlplus::file_copy(srcImage, imageName);
    }
  }
  } // end omp parallel region (thread-local scratch buffers)

  if (!bOk)
  {
    OPENMVG_LOG_ERROR << "Catched a memory error in the image conversion."
     << " Please consider to use less threads ([-n|--numThreads])." << std::endl;
    return false;
  }

  // define structure -- parallel build, preserving the same per-landmark
  // semantics as the original sequential loop:
  //   * obs filtered to those whose view survived map_view,
  //   * landmarks with < 2 surviving views are dropped,
  //   * surviving views sorted by imageID,
  //   * Vertex::X = landmark.X.cast<float>().
  // Output order matches the iteration order of sfm_data.GetLandmarks() and
  // is therefore identical (modulo unordered_map bucket order, which is
  // non-deterministic in both versions) to the sequential code.
  {
    const auto & landmarks = sfm_data.GetLandmarks();
    const size_t L = landmarks.size();

    // Materialize landmark pointers once -- iterating an unordered_map by
    // index requires linear advance, which we avoid by snapshotting.
    std::vector<const Landmark *> lm_ptrs;
    lm_ptrs.reserve(L);
    for (const auto & kv : landmarks) lm_ptrs.push_back(&kv.second);

    int nthreads = 1;
#ifdef OPENMVG_USE_OPENMP
    nthreads = (iNumThreads > 0) ? iNumThreads : omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#endif
    std::vector<std::vector<_INTERFACE_NAMESPACE::Interface::Vertex>> per_thread(nthreads);
    // Rough reserve: a small fraction of L per thread bucket avoids early
    // reallocs without overcommitting when most landmarks survive.
    for (auto & v : per_thread) v.reserve(L / static_cast<size_t>(nthreads) + 1);

#ifdef OPENMVG_USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (long long i = 0; i < static_cast<long long>(L); ++i)
    {
      const Landmark & landmark = *lm_ptrs[static_cast<size_t>(i)];
      _INTERFACE_NAMESPACE::Interface::Vertex vert;
      _INTERFACE_NAMESPACE::Interface::Vertex::ViewArr & views = vert.views;
      for (const auto & observation : landmark.obs)
      {
        const auto it = map_view.find(observation.first); // read-only: safe to share
        if (it != map_view.end())
        {
          _INTERFACE_NAMESPACE::Interface::Vertex::View view;
          view.imageID = it->second;
          view.confidence = 0;
          views.push_back(view);
        }
      }
      if (views.size() < 2)
        continue;
      std::sort(
        views.begin(), views.end(),
        [] (const _INTERFACE_NAMESPACE::Interface::Vertex::View & view0,
            const _INTERFACE_NAMESPACE::Interface::Vertex::View & view1)
        {
          return view0.imageID < view1.imageID;
        }
      );
      vert.X = landmark.X.cast<float>();
      int tid = 0;
#ifdef OPENMVG_USE_OPENMP
      tid = omp_get_thread_num();
#endif
      per_thread[static_cast<size_t>(tid)].push_back(std::move(vert));
    }

    // Concatenate per-thread buckets in thread-id order. With schedule(static)
    // this reproduces the sequential iteration order of lm_ptrs[].
    size_t total = 0;
    for (const auto & v : per_thread) total += v.size();
    scene.vertices.reserve(total);
    for (auto & v : per_thread)
    {
      for (auto & vert : v) scene.vertices.push_back(std::move(vert));
      std::vector<_INTERFACE_NAMESPACE::Interface::Vertex>().swap(v);
    }
  }

  // write OpenMVS data
  if (!_INTERFACE_NAMESPACE::ARCHIVE::SerializeSave(scene, sOutFile))
    return false;

  OPENMVG_LOG_INFO
    << "Scene saved to OpenMVS interface format:\n"
    << " #platforms: " << scene.platforms.size();
    for (int i = 0; i < scene.platforms.size(); ++i)
    {
      OPENMVG_LOG_INFO << "  platform ( " << i << " ) #cameras: " << scene.platforms[i].cameras.size();
    }
  OPENMVG_LOG_INFO
    << "  " << scene.images.size() << " images (" << nPoses << " calibrated)\n"
    << "  " << scene.vertices.size() << " Landmarks";
  return true;
}

int main(int argc, char *argv[])
{
  CmdLine cmd;
  std::string sSfM_Data_Filename;
  std::string sOutFile = "scene.mvs";
  std::string sOutDir = "undistorted_images";
  int iNumThreads = 0;

  cmd.add( make_option('i', sSfM_Data_Filename, "sfmdata") );
  cmd.add( make_option('o', sOutFile, "outfile") );
  cmd.add( make_option('d', sOutDir, "outdir") );
#ifdef OPENMVG_USE_OPENMP
  cmd.add( make_option('n', iNumThreads, "numThreads") );
#endif

  try {
    if (argc == 1) throw std::string("Invalid command line parameter.");
    cmd.process(argc, argv);
  } catch (const std::string& s) {
    OPENMVG_LOG_INFO << "Usage: " << argv[0] << '\n'
      << "[-i|--sfmdata] filename, the SfM_Data file to convert\n"
      << "[-o|--outfile] OpenMVS scene file\n"
      << "[-d|--outdir] undistorted images path\n"
#ifdef OPENMVG_USE_OPENMP
      << "[-n|--numThreads] number of thread(s)\n"
#endif
      ;

    OPENMVG_LOG_ERROR << s;
    return EXIT_FAILURE;
  }

  if (stlplus::extension_part(sOutFile) != "mvs") {
    OPENMVG_LOG_ERROR
      << "Invalid output file extension: " << sOutFile << "."
      << "You must use a filename with a .mvs extension.";
      return EXIT_FAILURE;
  }

  // Read the input SfM scene
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(ALL))) {
    OPENMVG_LOG_ERROR << "The input SfM_Data file \""<< sSfM_Data_Filename << "\" cannot be read.";
    return EXIT_FAILURE;
  }

  // Export OpenMVS data structure
  if (!exportToOpenMVS(sfm_data, sOutFile, sOutDir, iNumThreads))
  {
    OPENMVG_LOG_ERROR << "The output openMVS scene file cannot be written";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
