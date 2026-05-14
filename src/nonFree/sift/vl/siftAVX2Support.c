#include <immintrin.h>
#include <math.h>

#include "generic.h"

#define VL_RESTRICT __restrict

/* ================================================================
 * AVX2 fast exp(x) for 8 floats — cubic spline approximation
 * matching the scalar fastExp3S version in P2PUtils.h
 * ================================================================ */
static __forceinline __m256
FastExp3S_AVX2(__m256 x)
{
  __m256  vScale = _mm256_set1_ps(12102203.0f);
  __m256i vBias  = _mm256_set1_epi32(127 * (1 << 23));

  __m256i vi = _mm256_add_epi32(
    _mm256_cvtps_epi32(_mm256_mul_ps(vScale, x)),
    vBias);

  /* m = (vi >> 7) & 0xFFFF */
  __m256i vm = _mm256_and_si256(
    _mm256_srai_epi32(vi, 7),
    _mm256_set1_epi32(0xFFFF));

  /* Cubic correction */
  __m256i vCorr = _mm256_mullo_epi32(_mm256_set1_epi32(1277), vm);
  vCorr = _mm256_srai_epi32(vCorr, 14);
  vCorr = _mm256_add_epi32(vCorr, _mm256_set1_epi32(14825));
  vCorr = _mm256_mullo_epi32(vCorr, vm);
  vCorr = _mm256_srai_epi32(vCorr, 14);
  vCorr = _mm256_sub_epi32(vCorr, _mm256_set1_epi32(79749));
  vCorr = _mm256_mullo_epi32(vCorr, vm);
  vCorr = _mm256_srai_epi32(vCorr, 11);
  vCorr = _mm256_sub_epi32(vCorr, _mm256_set1_epi32(626));

  vi = _mm256_add_epi32(vi, vCorr);
  return _mm256_castsi256_ps(vi);
}

/* ================================================================
 * AVX2 Mod2PI for angles in [-4pi, +4pi] range.
 * ================================================================ */
static __forceinline __m256
Mod2PILimited_AVX2(__m256 x)
{
  __m256 vTwoPI = _mm256_set1_ps(6.2831853071795864f);
  __m256 vZero  = _mm256_setzero_ps();

  __m256 needsSub = _mm256_cmp_ps(x, vTwoPI, _CMP_GE_OS);
  __m256 result   = _mm256_sub_ps(x, _mm256_and_ps(needsSub, vTwoPI));

  __m256 needsAdd = _mm256_cmp_ps(x, vZero, _CMP_LT_OS);
  result = _mm256_add_ps(result, _mm256_and_ps(needsAdd, vTwoPI));

  return result;
}

/* ================================================================
 * AVX2 descriptor normalize + truncate + renormalize (128 floats)
 * ================================================================ */
static __forceinline float
NormalizeHistogramAVX2(float* VL_RESTRICT descr)
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();

  for (int i = 0; i < 128; i += 16) {
    __m256 v0 = _mm256_loadu_ps(descr + i);
    __m256 v1 = _mm256_loadu_ps(descr + i + 8);
    acc0 = _mm256_fmadd_ps(v0, v0, acc0);
    acc1 = _mm256_fmadd_ps(v1, v1, acc1);
  }

  acc0 = _mm256_add_ps(acc0, acc1);

  /* Horizontal sum of 8 floats */
  __m128 lo   = _mm256_castps256_ps128(acc0);
  __m128 hi   = _mm256_extractf128_ps(acc0, 1);
  __m128 sum4 = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(sum4);
  __m128 sums = _mm_add_ps(sum4, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  float normSq = _mm_cvtss_f32(sums);

  float norm = sqrtf(normSq) + 1.19209290E-07f; /* VL_EPSILON_F */
  __m256 vInvNorm = _mm256_set1_ps(1.0f / norm);

  for (int i = 0; i < 128; i += 16) {
    __m256 v0 = _mm256_loadu_ps(descr + i);
    __m256 v1 = _mm256_loadu_ps(descr + i + 8);
    _mm256_storeu_ps(descr + i,     _mm256_mul_ps(v0, vInvNorm));
    _mm256_storeu_ps(descr + i + 8, _mm256_mul_ps(v1, vInvNorm));
  }

  return norm;
}

void
SiftDescriptorNormalizeTruncateAVX2(
  float* VL_RESTRICT descr,
  float norm_thresh)
{
  float norm = NormalizeHistogramAVX2(descr);

  if (norm_thresh != 0.0f && norm < norm_thresh) {
    __m256 vZero = _mm256_setzero_ps();
    for (int i = 0; i < 128; i += 8)
      _mm256_storeu_ps(descr + i, vZero);
    return;
  }

  /* Truncate at 0.2 */
  {
    __m256 vClamp = _mm256_set1_ps(0.2f);
    for (int i = 0; i < 128; i += 16) {
      __m256 v0 = _mm256_loadu_ps(descr + i);
      __m256 v1 = _mm256_loadu_ps(descr + i + 8);
      _mm256_storeu_ps(descr + i,     _mm256_min_ps(v0, vClamp));
      _mm256_storeu_ps(descr + i + 8, _mm256_min_ps(v1, vClamp));
    }
  }

  /* Renormalize */
  NormalizeHistogramAVX2(descr);
}

/* ================================================================
 * AVX2 descriptor inner-row accumulation.
 *
 * Processes 8 pixels at a time from the gradient buffer.
 * Computes Gaussian weight, angle bin, and scatters trilinearly
 * into the 128-bin descriptor histogram.
 *
 * The gradient buffer is interleaved: mod0,ang0, mod1,ang1, ...
 * so 8 pixels = 16 floats, stride = 2 per pixel.
 *
 * *pixelsConsumed is set to the number of pixels processed (multiple of 8).
 * The caller handles the scalar tail.
 * ================================================================ */
void
SiftDescriptorRowAVX2(
  float* VL_RESTRICT dpt,
  const float* VL_RESTRICT pMA,
  int count,
  float nx0,
  float ny0,
  float nxInc,
  float nyInc,
  float angle0,
  float negInvSigma2,
  float ntFactor,
  int* VL_RESTRICT pixelsConsumed)
{
  int const simdCount = count & ~7;
  *pixelsConsumed = simdCount;

  if (simdCount == 0)
    return;

  __m256 vAngle0       = _mm256_set1_ps(angle0);
  __m256 vNegInvSigma2 = _mm256_set1_ps(negInvSigma2);
  __m256 vNtFactor     = _mm256_set1_ps(ntFactor);
  __m256 vHalf         = _mm256_set1_ps(0.5f);
  __m256 vOne          = _mm256_set1_ps(1.0f);

  /* Build nx and ny vectors for the 8 pixels */
  __m256 vNx = _mm256_set_ps(
    nx0 + 7*nxInc, nx0 + 6*nxInc, nx0 + 5*nxInc, nx0 + 4*nxInc,
    nx0 + 3*nxInc, nx0 + 2*nxInc, nx0 + 1*nxInc, nx0);
  __m256 vNy = _mm256_set_ps(
    ny0 + 7*nyInc, ny0 + 6*nyInc, ny0 + 5*nyInc, ny0 + 4*nyInc,
    ny0 + 3*nyInc, ny0 + 2*nyInc, ny0 + 1*nyInc, ny0);

  __m256 vNxStep = _mm256_set1_ps(8.0f * nxInc);
  __m256 vNyStep = _mm256_set1_ps(8.0f * nyInc);

  /* Permutation index to fix lane ordering after shuffle-based deinterleave */
  __m256i vPerm = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

  for (int p = 0; p < simdCount; p += 8,
       pMA += 16,
       vNx = _mm256_add_ps(vNx, vNxStep),
       vNy = _mm256_add_ps(vNy, vNyStep))
  {
    /* Load 16 interleaved floats: m0,a0, m1,a1, ... m7,a7
     * chunk0 = pMA[0..7]  = m0,a0,m1,a1,m2,a2,m3,a3
     * chunk1 = pMA[8..15] = m4,a4,m5,a5,m6,a6,m7,a7
     */
    __m256 chunk0 = _mm256_loadu_ps(pMA);
    __m256 chunk1 = _mm256_loadu_ps(pMA + 8);

    /* Deinterleave: extract even (mods) and odd (angles) elements.
     * _mm256_shuffle_ps within 128-bit lanes, then cross-lane permute. */
    __m256 mods_raw = _mm256_shuffle_ps(chunk0, chunk1, _MM_SHUFFLE(2, 0, 2, 0));
    __m256 angs_raw = _mm256_shuffle_ps(chunk0, chunk1, _MM_SHUFFLE(3, 1, 3, 1));

    __m256 vMods = _mm256_permutevar8x32_ps(mods_raw, vPerm);
    __m256 vAngs = _mm256_permutevar8x32_ps(angs_raw, vPerm);

    /* theta = Mod2PI(angle - angle0) */
    __m256 vTheta = Mod2PILimited_AVX2(_mm256_sub_ps(vAngs, vAngle0));

    /* nt = theta * ntFactor */
    __m256 vNt = _mm256_mul_ps(vTheta, vNtFactor);

    /* Gaussian window: win = fastExp3S(r2 * negInvSigma2) */
    __m256 vR2 = _mm256_fmadd_ps(vNx, vNx, _mm256_mul_ps(vNy, vNy));
    __m256 vWin = FastExp3S_AVX2(_mm256_mul_ps(vR2, vNegInvSigma2));
    __m256 vWinMod = _mm256_mul_ps(vWin, vMods);

    /* Bin indices */
    __m256 vBinxF = _mm256_floor_ps(_mm256_sub_ps(vNx, vHalf));
    __m256 vBinyF = _mm256_floor_ps(_mm256_sub_ps(vNy, vHalf));
    __m256 vBintF = _mm256_floor_ps(vNt);

    __m256i vBinx = _mm256_cvtps_epi32(vBinxF);
    __m256i vBiny = _mm256_cvtps_epi32(vBinyF);
    __m256i vBint = _mm256_cvtps_epi32(vBintF);

    /* Fractional remainders */
    __m256 vRbinx = _mm256_sub_ps(vNx, _mm256_add_ps(vBinxF, vHalf));
    __m256 vRbiny = _mm256_sub_ps(vNy, _mm256_add_ps(vBinyF, vHalf));
    __m256 vRbint = _mm256_sub_ps(vNt, vBintF);

    /* Trilinear weights */
    __m256 vW1x = _mm256_sub_ps(vOne, vRbinx);
    __m256 vW1y = _mm256_sub_ps(vOne, vRbiny);
    __m256 vW1t = _mm256_sub_ps(vOne, vRbint);

    /* bint % 8 (NBO=8) */
    __m256i vSeven = _mm256_set1_epi32(7);
    __m256i vBint0 = _mm256_and_si256(vBint, vSeven);
    __m256i vBint1 = _mm256_and_si256(
      _mm256_add_epi32(vBint, _mm256_set1_epi32(1)), vSeven);

    /* Extract to scalar arrays for the histogram scatter.
     * The scatter itself is inherently serial (random writes to 128-bin hist)
     * but all the SIMD math above has been vectorized. */
    __declspec(align(32)) float aWinMod[8], aW1x[8], aW1y[8], aW1t[8];
    __declspec(align(32)) float aRbinx[8], aRbiny[8], aRbint[8];
    __declspec(align(32)) int   aBinx[8], aBiny[8], aBint0[8], aBint1[8];

    _mm256_store_ps(aWinMod, vWinMod);
    _mm256_store_ps(aW1x, vW1x);
    _mm256_store_ps(aW1y, vW1y);
    _mm256_store_ps(aW1t, vW1t);
    _mm256_store_ps(aRbinx, vRbinx);
    _mm256_store_ps(aRbiny, vRbiny);
    _mm256_store_ps(aRbint, vRbint);
    _mm256_store_si256((__m256i*)aBinx, vBinx);
    _mm256_store_si256((__m256i*)aBiny, vBiny);
    _mm256_store_si256((__m256i*)aBint0, vBint0);
    _mm256_store_si256((__m256i*)aBint1, vBint1);

    for (int j = 0; j < 8; ++j) {
      float wm    = aWinMod[j];
      float w1x   = aW1x[j];
      float w1y   = aW1y[j];
      float w1t   = aW1t[j];
      float rbinx = aRbinx[j];
      float rbiny = aRbiny[j];
      float rbint = aRbint[j];
      int   binx  = aBinx[j];
      int   biny  = aBiny[j];
      int   bt0   = aBint0[j];
      int   bt1   = aBint1[j];

      float w_00t0 = wm * w1x * w1y * w1t;
      float w_00t1 = wm * w1x * w1y * rbint;
      float w_01t0 = wm * w1x * rbiny * w1t;
      float w_01t1 = wm * w1x * rbiny * rbint;
      float w_10t0 = wm * rbinx * w1y * w1t;
      float w_10t1 = wm * rbinx * w1y * rbint;
      float w_11t0 = wm * rbinx * rbiny * w1t;
      float w_11t1 = wm * rbinx * rbiny * rbint;

      /* NBP=4, NBP/2=2, NBO=8, binxo=8, binyo=32 */
      int xb0 = binx + 2;
      int yb0 = biny + 2;
      int xb1 = xb0 + 1;
      int yb1 = yb0 + 1;

      if ((unsigned)xb0 < 4) {
        int offx0 = binx * 8;
        if ((unsigned)yb0 < 4) {
          int addr = biny * 32 + offx0;
          dpt[addr + bt0] += w_00t0;
          dpt[addr + bt1] += w_00t1;
        }
        if ((unsigned)yb1 < 4) {
          int addr = (biny + 1) * 32 + offx0;
          dpt[addr + bt0] += w_01t0;
          dpt[addr + bt1] += w_01t1;
        }
      }
      if ((unsigned)xb1 < 4) {
        int offx1 = (binx + 1) * 8;
        if ((unsigned)yb0 < 4) {
          int addr = biny * 32 + offx1;
          dpt[addr + bt0] += w_10t0;
          dpt[addr + bt1] += w_10t1;
        }
        if ((unsigned)yb1 < 4) {
          int addr = (biny + 1) * 32 + offx1;
          dpt[addr + bt0] += w_11t0;
          dpt[addr + bt1] += w_11t1;
        }
      }
    }
  }
}

/* ================================================================
 * Existing functions below
 * ================================================================ */

void
GaussianRowSymmetricClampAVX2(
  float* VL_RESTRICT dst,
  const float* VL_RESTRICT src,
  size_t width,
  const float* VL_RESTRICT k,
  int W)
{
#if 1
  size_t x = 0;

  // Left edge (scalar clamp)
  for (; x < (size_t)W && x < width; ++x) {
    float acc = k[0] * src[x];
    for (int i = 1; i <= W; ++i) {
      size_t xm = (x < (size_t)i) ? 0 : (x - (size_t)i);
      size_t xp = (x + (size_t)i >= width) ? (width - 1) : (x + (size_t)i);
      acc += k[i] * (src[xm] + src[xp]);
    }
    dst[x] = acc;
  }

  // Nothing to vectorize if width <= W
  if (width <= (size_t)W) {
    return;
  }

  // Pre-broadcast kernel taps once
  // kVec has W+1 entries (k[0..W])
  __m256* kVec = (__m256*)_alloca((size_t)(W + 1) * sizeof(__m256));
  for (int i = 0; i <= W; ++i) {
    kVec[i] = _mm256_set1_ps(k[i]);
  }

  // Center region is x in [W, width - W)
  {
    const size_t xEnd = width - (size_t)W;
    const size_t simdEnd = xEnd & ~(size_t)7;

    // Optional prefetch distance (floats). Tune 64..256 or disable.
    const size_t pfDist = 128;

#if GAUSS_USE_STREAM_STORE
    const int canStream = IsAligned32(dst + x);
#endif

    for (; x + 7 < simdEnd; x += 8) {
      // Light prefetch. Can help on huge rows, can be neutral.
      _mm_prefetch((const char*)(src + x + pfDist), _MM_HINT_T0);

      __m256 acc0 = _mm256_mul_ps(kVec[0], _mm256_loadu_ps(src + x));
      __m256 acc1 = _mm256_setzero_ps();

      int i = 1;

      // Unroll taps by 2 to increase ILP
      for (; i + 1 <= W; i += 2) {
        const size_t ii0 = (size_t)i;
        const size_t ii1 = (size_t)(i + 1);

        __m256 a0 = _mm256_loadu_ps(src + x - ii0);
        __m256 b0 = _mm256_loadu_ps(src + x + ii0);
        acc0 = _mm256_fmadd_ps(kVec[i], _mm256_add_ps(a0, b0), acc0);

        __m256 a1 = _mm256_loadu_ps(src + x - ii1);
        __m256 b1 = _mm256_loadu_ps(src + x + ii1);
        acc1 = _mm256_fmadd_ps(kVec[i + 1], _mm256_add_ps(a1, b1), acc1);
      }

      // Odd leftover tap
      for (; i <= W; ++i) {
        const size_t ii = (size_t)i;
        __m256 a = _mm256_loadu_ps(src + x - ii);
        __m256 b = _mm256_loadu_ps(src + x + ii);
        acc0 = _mm256_fmadd_ps(kVec[i], _mm256_add_ps(a, b), acc0);
      }

      {
        __m256 acc = _mm256_add_ps(acc0, acc1);

#if GAUSS_USE_STREAM_STORE
        // Streaming stores require 32-byte aligned destination.
        // If dst is not aligned, fall back to normal storeu.
        if (canStream && IsAligned32(dst + x)) {
          _mm256_stream_ps(dst + x, acc);
        }
        else {
          _mm256_storeu_ps(dst + x, acc);
        }
#else
        _mm256_storeu_ps(dst + x, acc);
#endif
      }
    }

#if GAUSS_USE_STREAM_STORE
    // Make streaming stores visible before any dependent phase on other cores.
    _mm_sfence();
#endif
  }

  // Right edge (scalar clamp) + any leftover < 8 in center range
  for (; x < width; ++x) {
    float acc = k[0] * src[x];
    for (int i = 1; i <= W; ++i) {
      size_t xm = (x < (size_t)i) ? 0 : (x - (size_t)i);
      size_t xp = (x + (size_t)i >= width) ? (width - 1) : (x + (size_t)i);
      acc += k[i] * (src[xm] + src[xp]);
    }
    dst[x] = acc;
  }
#else
  size_t x = 0;

  /* left edge */
  for (; x < (size_t)W && x < width; ++x) {
    float acc = k[0] * src[x];
    int i;
    for (i = 1; i <= W; ++i) {
      size_t xm = (x < (size_t)i) ? 0 : x - (size_t)i;
      size_t xp = (x + (size_t)i >= width) ? width - 1 : x + (size_t)i;
      acc += k[i] * (src[xm] + src[xp]);
    }
    dst[x] = acc;
  }

  /* center */
  for (; x + 7 < width - (size_t)W; x += 8) {
    __m256 acc = _mm256_mul_ps(_mm256_set1_ps(k[0]),
      _mm256_loadu_ps(src + x));

    int i;
    for (i = 1; i <= W; ++i) {
      __m256 km = _mm256_set1_ps(k[i]);
      __m256 a = _mm256_loadu_ps(src + x - i);
      __m256 b = _mm256_loadu_ps(src + x + i);
      acc = _mm256_fmadd_ps(km, _mm256_add_ps(a, b), acc);
    }

    _mm256_storeu_ps(dst + x, acc);
  }

  /* right edge */
  for (; x < width; ++x) {
    float acc = k[0] * src[x];
    int i;
    for (i = 1; i <= W; ++i) {
      size_t xm = (x < (size_t)i) ? 0 : x - (size_t)i;
      size_t xp = (x + (size_t)i >= width) ? width - 1 : x + (size_t)i;
      acc += k[i] * (src[xm] + src[xp]);
    }
    dst[x] = acc;
  }
#endif
}

static inline float AccumulateW13_scalar(
  float k0,
  const float* VL_RESTRICT rowCenter,
  float* const VL_RESTRICT rowUp[64],
  float* const VL_RESTRICT rowDn[64],
  const float* VL_RESTRICT kw,
  size_t x)
{
  float acc = k0 * rowCenter[x];

  // taps 1..13
  for (int i = 1; i <= 13; ++i) {
    acc += (rowUp[i][x] + rowDn[i][x]) * kw[i];
  }

  return acc;
}

void AccumulateW13_AVX2_Block(
  float k0,
  const float* VL_RESTRICT rowCenter,
  float* const VL_RESTRICT rowUp[64],
  float* const VL_RESTRICT rowDn[64],
  const float* VL_RESTRICT kw,
  float* VL_RESTRICT out,
  int width)
{
  int x = 0;

  for (; x + 7 < width; x += 8) {

    __m256 acc = _mm256_mul_ps(
      _mm256_set1_ps(k0),
      _mm256_loadu_ps(rowCenter + x));

    // taps 1..13
    for (int i = 1; i <= 13; ++i) {
      __m256 up = _mm256_loadu_ps(rowUp[i] + x);
      __m256 dn = _mm256_loadu_ps(rowDn[i] + x);
      __m256 k = _mm256_set1_ps(kw[i]);

      acc = _mm256_fmadd_ps(
        _mm256_add_ps(up, dn),
        k,
        acc);
    }

    _mm256_storeu_ps(out + x, acc);
  }

  // scalar tail
  for (; x < width; ++x) {
    out[x] = AccumulateW13_scalar(k0, rowCenter, rowUp, rowDn, kw, x);
  }
}

static inline int IsAligned32(const void* p) {
  return (((uintptr_t)p) & 31) == 0;
}

void DogSubtractAvx2(
  float* __restrict dst,
  const float* __restrict a,
  const float* __restrict b,
  int count)
{
  // Streaming stores require aligned dst. Aligned loads are optional, but we
  // keep loads aligned when possible.
  if (!IsAligned32(dst)) {
    int i = 0;
    int simdEnd = count & ~7;
    for (; i < simdEnd; i += 8) {
      __m256 va = _mm256_loadu_ps(a + i);
      __m256 vb = _mm256_loadu_ps(b + i);
      _mm256_storeu_ps(dst + i, _mm256_sub_ps(vb, va));
    }
    for (; i < count; ++i) {
      dst[i] = b[i] - a[i];
    }
    return;
  }

  int i = 0;
  int simdEnd = count & ~7;
  int unrollEnd = simdEnd & ~31; // 32 floats per iter

  for (; i < unrollEnd; i += 32) {
    __m256 a0 = _mm256_loadu_ps(a + i + 0);
    __m256 b0 = _mm256_loadu_ps(b + i + 0);
    _mm256_store_ps(dst + i + 0, _mm256_sub_ps(b0, a0));

    __m256 a1 = _mm256_loadu_ps(a + i + 8);
    __m256 b1 = _mm256_loadu_ps(b + i + 8);
    _mm256_store_ps(dst + i + 8, _mm256_sub_ps(b1, a1));

    __m256 a2 = _mm256_loadu_ps(a + i + 16);
    __m256 b2 = _mm256_loadu_ps(b + i + 16);
    _mm256_store_ps(dst + i + 16, _mm256_sub_ps(b2, a2));

    __m256 a3 = _mm256_loadu_ps(a + i + 24);
    __m256 b3 = _mm256_loadu_ps(b + i + 24);
    _mm256_store_ps(dst + i + 24, _mm256_sub_ps(b3, a3));
  }

  for (; i < simdEnd; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    _mm256_store_ps(dst + i, _mm256_sub_ps(vb, va));
  }

  for (; i < count; ++i) {
    dst[i] = b[i] - a[i];
  }
}

void DogSubtract130TileAvx2(
  float* VL_RESTRICT dst,
  const float* VL_RESTRICT a,
  const float* VL_RESTRICT b,
  int height,
  int stride)
{
  const int aligned =
    ((((uintptr_t)dst |
      (uintptr_t)a |
      (uintptr_t)b) & 31) == 0);

  for (int y = 0; y < height; ++y) {

    float* VL_RESTRICT d = dst;
    const float* VL_RESTRICT pa = a;
    const float* VL_RESTRICT pb = b;

    if (aligned) {

#define DOG_SUB_8(offset)                                       \
      do {                                                      \
        __m256 va = _mm256_load_ps(pa + (offset));              \
        __m256 vb = _mm256_load_ps(pb + (offset));              \
        _mm256_store_ps(d + (offset), _mm256_sub_ps(vb, va));   \
      } while (0)

      DOG_SUB_8(0);  DOG_SUB_8(8);
      DOG_SUB_8(16);  DOG_SUB_8(24);
      DOG_SUB_8(32);  DOG_SUB_8(40);
      DOG_SUB_8(48);  DOG_SUB_8(56);
      DOG_SUB_8(64);  DOG_SUB_8(72);
      DOG_SUB_8(80);  DOG_SUB_8(88);
      DOG_SUB_8(96);  DOG_SUB_8(104);
      DOG_SUB_8(112);  DOG_SUB_8(120);

#undef DOG_SUB_8
    }
    else {

#define DOG_SUB_8U(offset)                                      \
      do {                                                      \
        __m256 va = _mm256_loadu_ps(pa + (offset));             \
        __m256 vb = _mm256_loadu_ps(pb + (offset));             \
        _mm256_storeu_ps(d + (offset), _mm256_sub_ps(vb, va));  \
      } while (0)

      DOG_SUB_8U(0);  DOG_SUB_8U(8);
      DOG_SUB_8U(16);  DOG_SUB_8U(24);
      DOG_SUB_8U(32);  DOG_SUB_8U(40);
      DOG_SUB_8U(48);  DOG_SUB_8U(56);
      DOG_SUB_8U(64);  DOG_SUB_8U(72);
      DOG_SUB_8U(80);  DOG_SUB_8U(88);
      DOG_SUB_8U(96);  DOG_SUB_8U(104);
      DOG_SUB_8U(112);  DOG_SUB_8U(120);

#undef DOG_SUB_8U
    }

    // scalar tail (129,130)
    d[128] = pb[128] - pa[128];
    d[129] = pb[129] - pa[129];

    // advance to next row
    dst += stride;
    a += stride;
    b += stride;
  }
}