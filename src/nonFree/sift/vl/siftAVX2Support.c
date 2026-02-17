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

void DogSubtractAvx2(
  float* VL_RESTRICT dst,
  const float* VL_RESTRICT a,
  const float* VL_RESTRICT b,
  int count)
{
  int i = 0;
  int simdEnd = count & ~7;

  // Fast aligned path
  if ((((uintptr_t)dst |
    (uintptr_t)a |
    (uintptr_t)b) & 31) == 0) {

    for (; i < simdEnd; i += 8) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      _mm256_store_ps(dst + i, _mm256_sub_ps(vb, va));
    }
  }
  else {
    // Unaligned fallback
    for (; i < simdEnd; i += 8) {
      __m256 va = _mm256_loadu_ps(a + i);
      __m256 vb = _mm256_loadu_ps(b + i);
      _mm256_storeu_ps(dst + i, _mm256_sub_ps(vb, va));
    }
  }

  for (; i < count; ++i) {
    dst[i] = b[i] - a[i];
  }
}

#define DOG_SUB_8(offset, load, store)               \
  do {                                                \
    __m256 va = load(a + (offset));                   \
    __m256 vb = load(b + (offset));                   \
    store(dst + (offset), _mm256_sub_ps(vb, va));     \
  } while (0)

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