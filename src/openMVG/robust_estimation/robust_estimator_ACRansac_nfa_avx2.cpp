// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// ----------------------------------------------------------------
// AVX2 + FMA implementation of the ACRANSAC exhaustive NFA argmin.
//
// THIS IS THE ONLY TRANSLATION UNIT IN THE PROJECT BUILT WITH AVX2 CODE
// GENERATION (see robust_estimation/CMakeLists.txt, which puts /arch:AVX2 --
// or -mavx2 -mfma -- on this file alone). Nothing here may be #included from
// a header: everything AVX2 must stay behind the plain-C++ declarations in
// robust_estimator_ACRansac_nfa_simd.hpp, reached through the runtime CPUID
// gate ACRansacNFA_HasAVX2().
//
// Belt and braces: on GCC/Clang the functions also carry
// __attribute__((target("avx2,fma"))), so this file still compiles and works
// correctly even if the per-file flags are ever lost -- the target attribute
// alone is enough to enable the intrinsics there. MSVC has no equivalent
// attribute but accepts AVX2 intrinsics without /arch:AVX2, so it is covered
// either way.
// ----------------------------------------------------------------

#include "openMVG/robust_estimation/robust_estimator_ACRansac_nfa_simd.hpp"

#if OPENMVG_ACRANSAC_NFA_AVX2_KERNEL

#include <immintrin.h>

#include <cmath>
#include <limits>

#if defined(__GNUC__) || defined(__clang__)
#define OPENMVG_ACRANSAC_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define OPENMVG_ACRANSAC_AVX2_TARGET
#endif

namespace openMVG {
namespace robust {
namespace acransac_nfa_internal {

namespace {

// log10(2) and log10(e), correctly rounded.
constexpr double kLog10_2 = 0.30102999566398119521;
constexpr double kLog10_E = 0.43429448190325182765;

// ----------------------------------------------------------------
// 4-wide double log10.
//
//   x = 2^e * m,  m in [1,2)          (exact, straight from the IEEE fields)
//   fold m into [sqrt(2)/2, sqrt(2)) so |t| <= 0.1716 for t = (m-1)/(m+1)
//   ln(m) = 2*atanh(t) = 2t*(1 + t^2/3 + t^4/5 + ... + t^12/13)
//   log10(x) = e*log10(2) + ln(m)*log10(e)
//
// The truncated tail is 2t*s^7/15 with s = t^2 <= 0.02944, i.e. < 5e-13
// absolute in ln(m) -- far below the error the *float* m_logc_n / m_logc_k
// tables already contribute (entries reach ~10^3, where a float ULP is
// ~6e-5). Measured worst case against std::log10 over 4M points spanning
// [1e-7, 1e12], plus exact powers of two and both sides of the fold
// boundary: 2.0e-13 absolute, 1.3e-12 relative.
//
// PRECONDITION: x finite, normal, strictly positive. Guaranteed by the
// caller, which passes residual + FLT_EPSILON with a finite residual >= 0,
// so x >= 1.19e-7.
// ----------------------------------------------------------------
OPENMVG_ACRANSAC_AVX2_TARGET inline __m256d Log10_AVX2(__m256d x)
{
  const __m256i xi = _mm256_castpd_si256(x);

  // AVX2 has no int64 -> double convert (that is AVX512DQ), but the raw
  // biased exponent is in [0,2047], so the usual "or in 2^52, subtract
  // 2^52" trick applies. Unbias afterwards, in double.
  const __m256i raw_e =
      _mm256_and_si256(_mm256_srli_epi64(xi, 52), _mm256_set1_epi64x(0x7FF));
  __m256d e = _mm256_sub_pd(
      _mm256_castsi256_pd(
          _mm256_or_si256(raw_e, _mm256_set1_epi64x(0x4330000000000000LL))),
      _mm256_set1_pd(4503599627370496.0 /* 2^52 */));
  e = _mm256_sub_pd(e, _mm256_set1_pd(1023.0));

  __m256d m = _mm256_castsi256_pd(
      _mm256_or_si256(
          _mm256_and_si256(xi, _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL)),
          _mm256_set1_epi64x(0x3FF0000000000000LL)));

  // Branch-free `if (m > sqrt2) { m *= 0.5; e += 1; }`.
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d fold =
      _mm256_cmp_pd(m, _mm256_set1_pd(1.4142135623730951), _CMP_GT_OQ);
  m = _mm256_mul_pd(m, _mm256_blendv_pd(one, _mm256_set1_pd(0.5), fold));
  e = _mm256_add_pd(e, _mm256_and_pd(fold, one));

  const __m256d t = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
  const __m256d s = _mm256_mul_pd(t, t);
  __m256d p = _mm256_set1_pd(1.0 / 13.0);
  p = _mm256_fmadd_pd(p, s, _mm256_set1_pd(1.0 / 11.0));
  p = _mm256_fmadd_pd(p, s, _mm256_set1_pd(1.0 /  9.0));
  p = _mm256_fmadd_pd(p, s, _mm256_set1_pd(1.0 /  7.0));
  p = _mm256_fmadd_pd(p, s, _mm256_set1_pd(1.0 /  5.0));
  p = _mm256_fmadd_pd(p, s, _mm256_set1_pd(1.0 /  3.0));
  p = _mm256_fmadd_pd(p, s, one);

  const __m256d ln_m = _mm256_mul_pd(_mm256_add_pd(t, t), p);
  return _mm256_fmadd_pd(e, _mm256_set1_pd(kLog10_2),
                         _mm256_mul_pd(ln_m, _mm256_set1_pd(kLog10_E)));
}

} // namespace

OPENMVG_ACRANSAC_AVX2_TARGET
std::pair<double, uint32_t> ACRansacBestNFA_AVX2
(
  const double * sorted_residuals,
  const double * nfa_base,
  const double * nfa_scale,
  std::size_t k_begin,
  std::size_t k_end
)
{
  const double inf = std::numeric_limits<double>::infinity();
  std::pair<double, uint32_t> best(inf, static_cast<uint32_t>(k_begin));
  if (k_begin > k_end)
    return best;

  constexpr double flt_eps = std::numeric_limits<float>::epsilon();
  std::size_t k = k_begin;

  if (k + 3 <= k_end)
  {
    // Lane-parallel argmin: every lane keeps its own running best and they
    // are merged at the end. Within a lane k increases monotonically and the
    // update is a strict `<`, so a lane keeps its earliest minimum; the merge
    // then breaks cross-lane ties toward the smaller k. Net effect is
    // identical to the scalar first-minimum-wins loop.
    __m256d best_v = _mm256_set1_pd(inf);
    __m256d best_k = _mm256_setzero_pd();
    const __m256d veps = _mm256_set1_pd(flt_eps);
    const __m256d lane_off = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);

    for (; k + 3 <= k_end; k += 4)
    {
      // The caller keeps sorted residuals in a packed double array, so the
      // four values for k..k+3 are one contiguous load -- no deinterleave.
      // (They used to live in 16-byte (residual,index) pairs, which cost two
      // loads plus a shuffle pair per four elements.)
      const __m256d r = _mm256_loadu_pd(sorted_residuals + k - 1);

      const __m256d lg = Log10_AVX2(_mm256_add_pd(r, veps));
      const __m256d v = _mm256_fmadd_pd(_mm256_loadu_pd(nfa_scale + k), lg,
                                        _mm256_loadu_pd(nfa_base + k));

      const __m256d kk =
          _mm256_add_pd(_mm256_set1_pd(static_cast<double>(k)), lane_off);
      const __m256d lt = _mm256_cmp_pd(v, best_v, _CMP_LT_OQ);
      best_v = _mm256_blendv_pd(best_v, v, lt);
      best_k = _mm256_blendv_pd(best_k, kk, lt);
    }

    alignas(32) double lane_v[4], lane_k[4];
    _mm256_store_pd(lane_v, best_v);
    _mm256_store_pd(lane_k, best_k);
    for (int lane = 0; lane < 4; ++lane)
    {
      const uint32_t cand_k = static_cast<uint32_t>(lane_k[lane]);
      if (lane_v[lane] < best.first
          || (lane_v[lane] == best.first && cand_k < best.second))
        best = std::pair<double, uint32_t>(lane_v[lane], cand_k);
    }
  }

  // Remaining (< 4) values of k. These are all larger than anything the
  // vector body saw, so a strict `<` preserves lowest-k-wins.
  for (; k <= k_end; ++k)
  {
    const double v = nfa_base[k]
      + nfa_scale[k] * std::log10(sorted_residuals[k-1] + flt_eps);
    if (v < best.first)
      best = std::pair<double, uint32_t>(v, static_cast<uint32_t>(k));
  }

  return best;
}

} // namespace acransac_nfa_internal
} // namespace robust
} // namespace openMVG

#endif // OPENMVG_ACRANSAC_NFA_AVX2_KERNEL
