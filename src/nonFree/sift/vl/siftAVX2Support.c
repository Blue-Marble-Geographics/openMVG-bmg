#include <immintrin.h>
#define VL_RESTRICT __restrict

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

  int aAligned = IsAligned32(a);
  int bAligned = IsAligned32(b);

  if (aAligned && bAligned) {
    for (; i < unrollEnd; i += 32) {
      __m256 a0 = _mm256_load_ps(a + i + 0);
      __m256 b0 = _mm256_load_ps(b + i + 0);
      _mm256_stream_ps(dst + i + 0, _mm256_sub_ps(b0, a0));

      __m256 a1 = _mm256_load_ps(a + i + 8);
      __m256 b1 = _mm256_load_ps(b + i + 8);
      _mm256_stream_ps(dst + i + 8, _mm256_sub_ps(b1, a1));

      __m256 a2 = _mm256_load_ps(a + i + 16);
      __m256 b2 = _mm256_load_ps(b + i + 16);
      _mm256_stream_ps(dst + i + 16, _mm256_sub_ps(b2, a2));

      __m256 a3 = _mm256_load_ps(a + i + 24);
      __m256 b3 = _mm256_load_ps(b + i + 24);
      _mm256_stream_ps(dst + i + 24, _mm256_sub_ps(b3, a3));
    }

    for (; i < simdEnd; i += 8) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      _mm256_stream_ps(dst + i, _mm256_sub_ps(vb, va));
    }
  }
  else {
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
  }

  for (; i < count; ++i) {
    dst[i] = b[i] - a[i];
  }

  // Ensure streaming stores are globally visible before a dependent phase
  // that might run on another core. If you only consume dst on the same
  // thread later, you can usually omit this.
  // We are only going to use this data on our thread. _mm_sfence();
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