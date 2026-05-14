#ifndef OPENMVG_MATCHING_METRIC_SIMD_HPP
#define OPENMVG_MATCHING_METRIC_SIMD_HPP

#include <array>
#include <numeric>

#include <cstdint>
#include <emmintrin.h> // SSE2 — always available on x64

#if defined(OPENMVG_USE_AVX2) || defined(OPENMVG_USE_AVX)
#include <immintrin.h>
#endif

namespace openMVG {
namespace matching {

#ifdef _MSC_VER
#define ALIGNED32 __declspec(align(32))
#else
#define ALIGNED32 __attribute__((aligned(32)))
#endif

// SSE2 L2 squared distance for 128-byte uint8_t descriptors (SIFT).
inline int L2_SSE2_uint8
(
  const uint8_t * a,
  const uint8_t * b,
  size_t size
)
{
  const __m128i zero = _mm_setzero_si128();
  __m128i acc0 = zero;
  __m128i acc1 = zero;

  for (size_t i = 0; i < size; i += 32)
  {
    const __m128i va0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
    const __m128i vb0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
    const __m128i va1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 16));
    const __m128i vb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 16));

    const __m128i d0  = _mm_sub_epi8(_mm_max_epu8(va0, vb0), _mm_min_epu8(va0, vb0));
    const __m128i d0l = _mm_unpacklo_epi8(d0, zero);
    const __m128i d0h = _mm_unpackhi_epi8(d0, zero);
    acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(d0l, d0l));
    acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(d0h, d0h));

    const __m128i d1  = _mm_sub_epi8(_mm_max_epu8(va1, vb1), _mm_min_epu8(va1, vb1));
    const __m128i d1l = _mm_unpacklo_epi8(d1, zero);
    const __m128i d1h = _mm_unpackhi_epi8(d1, zero);
    acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(d1l, d1l));
    acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(d1h, d1h));
  }

  __m128i acc = _mm_add_epi32(acc0, acc1);
  acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1)));
  acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2)));
  return _mm_cvtsi128_si32(acc);
}

#ifdef OPENMVG_USE_AVX2
inline int L2_AVX2
(
  const uint8_t * a,
  const uint8_t * b,
  size_t size
)
{
  __m256i acc (_mm256_setzero_si256());

  const ALIGNED32 __m256i* ad = reinterpret_cast<const ALIGNED32 __m256i*>(a);
  const ALIGNED32 __m256i* bd = reinterpret_cast<const ALIGNED32 __m256i*>(b);

  for (int i = 0; i < 4; ++i) {
    const __m256i min = _mm256_min_epu8(ad[i], bd[i]);
    const __m256i max = _mm256_max_epu8(ad[i], bd[i]);
    const __m256i d = _mm256_sub_epi8(max, min);

    __m256i dl = _mm256_unpacklo_epi8(d, _mm256_setzero_si256());
    dl = _mm256_madd_epi16(dl, dl);
    __m256i dh = _mm256_unpackhi_epi8(d, _mm256_setzero_si256());
    dh = _mm256_madd_epi16(dh, dh);
    acc = _mm256_add_epi32(acc, _mm256_add_epi32(dl, dh));
  }
  __m128i l = _mm256_extracti128_si256(acc, 0);
  __m128i h = _mm256_extracti128_si256(acc, 1);
  __m128i r = _mm_hadd_epi32(_mm_add_epi32(h, l), _mm_setzero_si128());
  return _mm_extract_epi32(r, 0) + _mm_extract_epi32(r, 1);
}
#endif // OPENMVG_USE_AVX2

#ifdef OPENMVG_USE_AVX
inline float L2_AVX
(
  const float * a,
  const float * b,
  size_t size
)
{
  __m256 acc (_mm256_setzero_ps());

  for (int j = 0; j < size; j += 8)
  {
    const __m256 t0 = _mm256_sub_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j));
    acc = _mm256_add_ps(acc, _mm256_mul_ps(t0, t0));
  }
  float ALIGNED32 acc_float[8];
  _mm256_store_ps(acc_float, acc);
  return std::accumulate(acc_float, acc_float + 8, 0.f);
}
#endif // OPENMVG_USE_AVX

}  // namespace matching
}  // namespace openMVG

#endif // OPENMVG_MATCHING_METRIC_SIMD_HPP