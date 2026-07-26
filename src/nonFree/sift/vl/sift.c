#include "../../../P2PUtils.h"

#include "sift.h"
#include "imopv.h"
#include "mathop.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/** @internal @brief Use bilinear interpolation to compute orientations */
#define VL_SIFT_BILINEAR_ORIENTATIONS 1

#define EXPN_MAX 25.0         /**< ::fast_expn table max  @internal */

#define NBO 8
#define NBP 4

#define log2(x) (log(x)/VL_LOG_OF_2)

#define REDUCE_MEMORY /* Share buffer space to reduce memory use 25% */

extern int hasAVX2;
extern int hasSSE41;


#if defined(_MSC_VER)
#define VL_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define VL_RESTRICT __restrict__
#else
#define VL_RESTRICT
#endif


/** ------------------------------------------------------------------
 ** @internal
 ** @brief Fast @f$exp(-x)@f$ approximation
 **
 ** @param x argument.
 **
 ** The argument must be in the range [0, ::EXPN_MAX] .
 **
 ** @return approximation of @f$exp(-x)@f$.
 **/

VL_INLINE double
fast_expn (VlSiftFilt const * filter, double x)
{
  double a,b,r ;
  int i ;
  /*assert(0 <= x && x <= EXPN_MAX) ;*/

  if (x > EXPN_MAX) return 0.0 ;

  x *= EXPN_SZ / EXPN_MAX ;
  i = (int)vl_floor_d (x) ;
  r = x - i ;
  a = filter->expn_tab [i    ] ;
  b = filter->expn_tab [i + 1] ;
  return a + r * (b - a) ;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Initialize tables for ::fast_expn
 **/

VL_INLINE void
fast_expn_init (VlSiftFilt * filter)
{
  int k  ;
  for(k = 0 ; k < EXPN_SZ + 1 ; ++ k) {
    filter->expn_tab [k] = exp (- (double) k * (EXPN_MAX / EXPN_SZ)) ;
  }
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Copy image, upsample rows and take transpose
 **
 ** @param dst     output image buffer.
 ** @param src     input image buffer.
 ** @param width   input image width.
 ** @param height  input image height.
 **
 ** The output image has dimensions @a height by 2 @a width (so the
 ** destination buffer must be at least as big as two times the
 ** input buffer).
 **
 ** Upsampling is performed by linear interpolation.
 **/

static void
copy_and_upsample_rows
(vl_sift_pix       *dst,
 vl_sift_pix const *src, int width, int height)
{
  int x, y ;
  vl_sift_pix a, b ;

  for(y = 0 ; y < height ; ++y) {
    b = a = *src++ ;
    for(x = 0 ; x < width - 1 ; ++x) {
      b = *src++ ;
      *dst = a ;             dst += height ;
      *dst = 0.5 * (a + b) ; dst += height ;
      a = b ;
    }
    *dst = b ; dst += height ;
    *dst = b ; dst += height ;
    dst += 1 - width * 2 * height ;
  }
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Smooth an image
 ** @param self        SIFT filter.
 ** @param outputImage output imgae buffer.
 ** @param tempImage   temporary image buffer.
 ** @param inputImage  input image buffer.
 ** @param width       input image width.
 ** @param height      input image height.
 ** @param sigma       smoothing.
 **/

#if 1 // AVX2 optimized
void
GaussianRowSymmetricClampSSE2(
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
  for (; x + 3 < width - (size_t)W; x += 4) {
    __m128 acc = _mm_mul_ps(_mm_set1_ps(k[0]),
      _mm_loadu_ps(src + x));

    int i;
    for (i = 1; i <= W; ++i) {
      __m128 km = _mm_set1_ps(k[i]);
      __m128 a = _mm_loadu_ps(src + x - i);
      __m128 b = _mm_loadu_ps(src + x + i);
      acc = _mm_add_ps(acc, _mm_mul_ps(km, _mm_add_ps(a, b)));
    }

    _mm_storeu_ps(dst + x, acc);
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

void
GaussianRowSymmetricClampAVX2(
  float* VL_RESTRICT dst,
  const float* VL_RESTRICT src,
  size_t width,
  const float* VL_RESTRICT k,
  int W);

void AccumulateW13_AVX2_Block(
  float k0,
  const float* VL_RESTRICT rowCenter,
  float* const VL_RESTRICT rowUp[64],
  float* const VL_RESTRICT rowDn[64],
  const float* VL_RESTRICT kw,
  float* VL_RESTRICT out,
  int width);

typedef void (*GaussianRowFn)(
  float*,
  const float*,
  vl_size,
  const float*,
  int);

static GaussianRowFn gGaussianRowFn = 0;

static void
_vl_sift_smooth(
  VlSiftFilt* self,
  vl_sift_pix* VL_RESTRICT outputImage,
  vl_sift_pix* VL_RESTRICT tempImage, /* unused, ABI only */
  vl_sift_pix const* VL_RESTRICT inputImage,
  vl_size width,
  vl_size height,
  double sigma)
{
  /* Handle in-place case: the ring-buffer approach reads inputImage
   * rows lazily while writing outputImage rows.  When they alias,
   * earlier output overwrites not-yet-read input, causing double-
   * smoothing.  Break the alias with the caller-provided scratch
   * buffer (self->temp, already sized to the full octave) instead of
   * a per-call heap allocation; fall back to malloc only if that
   * scratch is missing or itself aliases the images. */
  vl_sift_pix* inputBuf = NULL; /* set only when we heap-allocate */
  if (inputImage == outputImage) {
    size_t nbytes = sizeof(vl_sift_pix) * width * height;
    vl_sift_pix* alias = tempImage;
    if (alias == NULL || alias == inputImage || alias == outputImage) {
      alias = inputBuf = (vl_sift_pix*)vl_malloc(nbytes);
    }
    memcpy(alias, inputImage, nbytes);
    inputImage = alias;
  }

  if (hasAVX2) {
    gGaussianRowFn = GaussianRowSymmetricClampAVX2;
  }
  else {
    gGaussianRowFn = GaussianRowSymmetricClampSSE2;
  }

  /* ------------------------------------------------------------
   * Build symmetric Gaussian kernel (half only)
   * ------------------------------------------------------------ */
  if (self->gaussFilterSigma != sigma) {
    int i;
    float acc = 0.0f;

    self->gaussFilterWidth = VL_MAX((int)ceil(4.0 * sigma), 1);
    {
      int W = self->gaussFilterWidth;
      size_t needed = sizeof(vl_sift_pix) * (size_t)(W + 1);

      if (needed > self->gaussFilterSize) {
        if (self->gaussFilter) vl_free(self->gaussFilter);
        self->gaussFilter = vl_malloc(needed);
        self->gaussFilterSize = needed;
      }

      self->gaussFilterSigma = sigma;

      for (i = 0; i <= W; ++i) {
        float d = (float)i / (float)sigma;
        float v = expf(-0.5f * d * d);
        self->gaussFilter[i] = v;
        acc += (i == 0) ? v : 2.0f * v;
      }

      for (i = 0; i <= W; ++i) {
        self->gaussFilter[i] /= acc;
      }
    }
  }

  {
    const int W = self->gaussFilterWidth;
    const float* VL_RESTRICT k = self->gaussFilter;

    if (W == 0) {
      if (!inputBuf) /* only copy if not already aliased */
        memcpy(outputImage, inputImage,
          sizeof(vl_sift_pix) * width * height);
      if (inputBuf) vl_free(inputBuf);
      return;
    }

    /* ------------------------------------------------------------
     * Ring buffer: exactly 2W+1 horizontally filtered rows
     * ------------------------------------------------------------ */
    {
      const int R = 2 * W + 1;
      /* Horizontally-filtered ring buffer.  Cache it on the filter and
       * grow on demand (mirrors the gaussFilter caching) so we don't
       * malloc/free (2W+1)*width floats on every smooth call -- this
       * matters for large images with many levels/octaves. */
      size_t rowBufBytes = sizeof(float) * (size_t)R * width;
      if (rowBufBytes > self->smoothRowBufSize) {
        if (self->smoothRowBuf) vl_free(self->smoothRowBuf);
        self->smoothRowBuf = (float*)vl_malloc(rowBufBytes);
        self->smoothRowBufSize = rowBufBytes;
      }
      float* VL_RESTRICT rowBuf = self->smoothRowBuf;

      const int Wplus1 = W + 1;
      int center = W;
      int t;

      GaussianRowFn rowFn = gGaussianRowFn;
      float* VL_RESTRICT rowPtr[64];
      int i;
      for (i = 0; i < R; ++i)
        rowPtr[i] = rowBuf + (size_t)i * width;

      /* ----------------------------------------------------------
       * Initialize window for y = 0 (rows [-W .. +W] clamped)
       * ---------------------------------------------------------- */
      for (t = -W; t <= W; ++t) {
        vl_size srcY;
        if (t < 0) {
          srcY = 0;
        }
        else {
          srcY = (vl_size)t;
          if (srcY >= height) srcY = height - 1;
        }

        rowFn(
          rowBuf + (size_t)(t + W) * width,
          inputImage + srcY * width,
          width,
          k,
          W);
      }

      /* Helper to slide the vertical window by one row. */
#define SLIDE_RING() do {                                \
      int add = center + Wplus1;                         \
      if (add >= R) add -= R;                            \
                                                         \
      {                                                  \
        vl_size srcRow = (vl_size)(y + Wplus1);          \
        if (srcRow >= height) srcRow = height - 1;       \
      rowFn(rowPtr[add],                                 \
            inputImage + srcRow * width,                  \
            width, k, W);                                \
      }                                                  \
                                                         \
      center++;                                          \
      if (center == R) center = 0;                       \
    } while(0)

    /* ----------------------------------------------------------
     * Main loop (W-dispatch hoisted, MSVC optimized)
     * ---------------------------------------------------------- */
      {
        vl_size y, x;
        int i;

        /* Hoist kernel coefficients ONCE */
        float k0 = k[0];
        float kw[64];
        for (i = 1; i <= W; ++i)
          kw[i] = k[i];

        float* VL_RESTRICT rowUp[64];
        float* VL_RESTRICT rowDn[64];

        switch (W) {

          /* (all existing case blocks remain unchanged) */
          /* ======================================================
           * W == 5
           * ====================================================== */
        case 5:
          for (y = 0; y < height; ++y) {

            float* VL_RESTRICT out = outputImage + y * width;
            float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

            for (i = 1; i <= 5; ++i) {
              int up = center - i;
              int dn = center + i;
              if (up < 0) up += R;
              if (dn >= R) dn -= R;
              rowUp[i] = rowBuf + (size_t)up * width;
              rowDn[i] = rowBuf + (size_t)dn * width;
            }

#pragma loop(ivdep)
            for (x = 0; x + 3 < width; x += 4) {

              float a0 =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]);

              float a1 =
                k0 * rowCenter[x + 1] +
                kw[1] * (rowUp[1][x + 1] + rowDn[1][x + 1]) +
                kw[2] * (rowUp[2][x + 1] + rowDn[2][x + 1]) +
                kw[3] * (rowUp[3][x + 1] + rowDn[3][x + 1]) +
                kw[4] * (rowUp[4][x + 1] + rowDn[4][x + 1]) +
                kw[5] * (rowUp[5][x + 1] + rowDn[5][x + 1]);

              float a2 =
                k0 * rowCenter[x + 2] +
                kw[1] * (rowUp[1][x + 2] + rowDn[1][x + 2]) +
                kw[2] * (rowUp[2][x + 2] + rowDn[2][x + 2]) +
                kw[3] * (rowUp[3][x + 2] + rowDn[3][x + 2]) +
                kw[4] * (rowUp[4][x + 2] + rowDn[4][x + 2]) +
                kw[5] * (rowUp[5][x + 2] + rowDn[5][x + 2]);

              float a3 =
                k0 * rowCenter[x + 3] +
                kw[1] * (rowUp[1][x + 3] + rowDn[1][x + 3]) +
                kw[2] * (rowUp[2][x + 3] + rowDn[2][x + 3]) +
                kw[3] * (rowUp[3][x + 3] + rowDn[3][x + 3]) +
                kw[4] * (rowUp[4][x + 3] + rowDn[4][x + 3]) +
                kw[5] * (rowUp[5][x + 3] + rowDn[5][x + 3]);

              out[x] = a0;
              out[x + 1] = a1;
              out[x + 2] = a2;
              out[x + 3] = a3;
            }

            for (; x < width; ++x) {
              out[x] =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]);
            }

            SLIDE_RING();
          }
          break;

          /* ======================================================
           * W == 7
           * ====================================================== */
        case 7:
          for (y = 0; y < height; ++y) {

            float* VL_RESTRICT out = outputImage + y * width;
            float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

            for (i = 1; i <= 7; ++i) {
              int up = center - i;
              int dn = center + i;
              if (up < 0) up += R;
              if (dn >= R) dn -= R;
              rowUp[i] = rowBuf + (size_t)up * width;
              rowDn[i] = rowBuf + (size_t)dn * width;
            }

#pragma loop(ivdep)
            for (x = 0; x + 3 < width; x += 4) {
              float a0 =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]) +
                kw[6] * (rowUp[6][x] + rowDn[6][x]) +
                kw[7] * (rowUp[7][x] + rowDn[7][x]);

              float a1 =
                k0 * rowCenter[x + 1] +
                kw[1] * (rowUp[1][x + 1] + rowDn[1][x + 1]) +
                kw[2] * (rowUp[2][x + 1] + rowDn[2][x + 1]) +
                kw[3] * (rowUp[3][x + 1] + rowDn[3][x + 1]) +
                kw[4] * (rowUp[4][x + 1] + rowDn[4][x + 1]) +
                kw[5] * (rowUp[5][x + 1] + rowDn[5][x + 1]) +
                kw[6] * (rowUp[6][x + 1] + rowDn[6][x + 1]) +
                kw[7] * (rowUp[7][x + 1] + rowDn[7][x + 1]);

              float a2 =
                k0 * rowCenter[x + 2] +
                kw[1] * (rowUp[1][x + 2] + rowDn[1][x + 2]) +
                kw[2] * (rowUp[2][x + 2] + rowDn[2][x + 2]) +
                kw[3] * (rowUp[3][x + 2] + rowDn[3][x + 2]) +
                kw[4] * (rowUp[4][x + 2] + rowDn[4][x + 2]) +
                kw[5] * (rowUp[5][x + 2] + rowDn[5][x + 2]) +
                kw[6] * (rowUp[6][x + 2] + rowDn[6][x + 2]) +
                kw[7] * (rowUp[7][x + 2] + rowDn[7][x + 2]);

              float a3 =
                k0 * rowCenter[x + 3] +
                kw[1] * (rowUp[1][x + 3] + rowDn[1][x + 3]) +
                kw[2] * (rowUp[2][x + 3] + rowDn[2][x + 3]) +
                kw[3] * (rowUp[3][x + 3] + rowDn[3][x + 3]) +
                kw[4] * (rowUp[4][x + 3] + rowDn[4][x + 3]) +
                kw[5] * (rowUp[5][x + 3] + rowDn[5][x + 3]) +
                kw[6] * (rowUp[6][x + 3] + rowDn[6][x + 3]) +
                kw[7] * (rowUp[7][x + 3] + rowDn[7][x + 3]);

              out[x] = a0;
              out[x + 1] = a1;
              out[x + 2] = a2;
              out[x + 3] = a3;
            }

            for (; x < width; ++x) {
              out[x] =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]) +
                kw[6] * (rowUp[6][x] + rowDn[6][x]) +
                kw[7] * (rowUp[7][x] + rowDn[7][x]);
            }

            SLIDE_RING();
          }
          break;

        case 8:
          for (y = 0; y < height; ++y) {

            float* VL_RESTRICT out = outputImage + y * width;
            float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

            for (i = 1; i <= 8; ++i) {
              int up = center - i;
              int dn = center + i;
              if (up < 0) up += R;
              if (dn >= R) dn -= R;
              rowUp[i] = rowBuf + (size_t)up * width;
              rowDn[i] = rowBuf + (size_t)dn * width;
            }

#pragma loop(ivdep)
            for (x = 0; x + 3 < width; x += 4) {

              float a0 =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]) +
                kw[6] * (rowUp[6][x] + rowDn[6][x]) +
                kw[7] * (rowUp[7][x] + rowDn[7][x]) +
                kw[8] * (rowUp[8][x] + rowDn[8][x]);

              float a1 =
                k0 * rowCenter[x + 1] +
                kw[1] * (rowUp[1][x + 1] + rowDn[1][x + 1]) +
                kw[2] * (rowUp[2][x + 1] + rowDn[2][x + 1]) +
                kw[3] * (rowUp[3][x + 1] + rowDn[3][x + 1]) +
                kw[4] * (rowUp[4][x + 1] + rowDn[4][x + 1]) +
                kw[5] * (rowUp[5][x + 1] + rowDn[5][x + 1]) +
                kw[6] * (rowUp[6][x + 1] + rowDn[6][x + 1]) +
                kw[7] * (rowUp[7][x + 1] + rowDn[7][x + 1]) +
                kw[8] * (rowUp[8][x + 1] + rowDn[8][x + 1]);

              float a2 =
                k0 * rowCenter[x + 2] +
                kw[1] * (rowUp[1][x + 2] + rowDn[1][x + 2]) +
                kw[2] * (rowUp[2][x + 2] + rowDn[2][x + 2]) +
                kw[3] * (rowUp[3][x + 2] + rowDn[3][x + 2]) +
                kw[4] * (rowUp[4][x + 2] + rowDn[4][x + 2]) +
                kw[5] * (rowUp[5][x + 2] + rowDn[5][x + 2]) +
                kw[6] * (rowUp[6][x + 2] + rowDn[6][x + 2]) +
                kw[7] * (rowUp[7][x + 2] + rowDn[7][x + 2]) +
                kw[8] * (rowUp[8][x + 2] + rowDn[8][x + 2]);

              float a3 =
                k0 * rowCenter[x + 3] +
                kw[1] * (rowUp[1][x + 3] + rowDn[1][x + 3]) +
                kw[2] * (rowUp[2][x + 3] + rowDn[2][x + 3]) +
                kw[3] * (rowUp[3][x + 3] + rowDn[3][x + 3]) +
                kw[4] * (rowUp[4][x + 3] + rowDn[4][x + 3]) +
                kw[5] * (rowUp[5][x + 3] + rowDn[5][x + 3]) +
                kw[6] * (rowUp[6][x + 3] + rowDn[6][x + 3]) +
                kw[7] * (rowUp[7][x + 3] + rowDn[7][x + 3]) +
                kw[8] * (rowUp[8][x + 3] + rowDn[8][x + 3]);

              out[x] = a0;
              out[x + 1] = a1;
              out[x + 2] = a2;
              out[x + 3] = a3;
            }

            for (; x < width; ++x) {
              out[x] =
                k0 * rowCenter[x] +
                kw[1] * (rowUp[1][x] + rowDn[1][x]) +
                kw[2] * (rowUp[2][x] + rowDn[2][x]) +
                kw[3] * (rowUp[3][x] + rowDn[3][x]) +
                kw[4] * (rowUp[4][x] + rowDn[4][x]) +
                kw[5] * (rowUp[5][x] + rowDn[5][x]) +
                kw[6] * (rowUp[6][x] + rowDn[6][x]) +
                kw[7] * (rowUp[7][x] + rowDn[7][x]) +
                kw[8] * (rowUp[8][x] + rowDn[8][x]);
            }

            SLIDE_RING();
          }
          break;

          /* ======================================================
           * W == 10 (2-accumulator scalar)
           * ====================================================== */
        case 10:
          for (y = 0; y < height; ++y) {
            float* VL_RESTRICT out = outputImage + y * width;
            float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

            for (i = 1; i <= 10; ++i) {
              int up = center - i;
              int dn = center + i;
              if (up < 0) up += R;
              if (dn >= R) dn -= R;
              rowUp[i] = rowBuf + (size_t)up * width;
              rowDn[i] = rowBuf + (size_t)dn * width;
            }

#pragma loop(ivdep)
            for (x = 0; x < width; ++x) {

              float acc0 = 0.0f;
              float acc1 = 0.0f;

              acc0 += kw[1] * (rowUp[1][x] + rowDn[1][x]);
              acc1 += kw[2] * (rowUp[2][x] + rowDn[2][x]);
              acc0 += kw[3] * (rowUp[3][x] + rowDn[3][x]);
              acc1 += kw[4] * (rowUp[4][x] + rowDn[4][x]);
              acc0 += kw[5] * (rowUp[5][x] + rowDn[5][x]);
              acc1 += kw[6] * (rowUp[6][x] + rowDn[6][x]);
              acc0 += kw[7] * (rowUp[7][x] + rowDn[7][x]);
              acc1 += kw[8] * (rowUp[8][x] + rowDn[8][x]);
              acc0 += kw[9] * (rowUp[9][x] + rowDn[9][x]);
              acc1 += kw[10] * (rowUp[10][x] + rowDn[10][x]);

              out[x] = k0 * rowCenter[x] + acc0 + acc1;
            }

            SLIDE_RING();
          }
          break;

          /* ======================================================
           * W == 13 (AVX2 fast path)
           * ====================================================== */
        case 13:
          if (hasAVX2) {
            for (y = 0; y < height; ++y) {

              float* VL_RESTRICT out = outputImage + y * width;
              float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

              for (i = 1; i <= 13; ++i) {
                int up = center - i;
                int dn = center + i;
                if (up < 0) up += R;
                if (dn >= R) dn -= R;
                rowUp[i] = rowBuf + (size_t)up * width;
                rowDn[i] = rowBuf + (size_t)dn * width;
              }

              AccumulateW13_AVX2_Block(k0, rowCenter, rowUp, rowDn, kw, out, width);

              SLIDE_RING();
            }
          }
          else {
            for (y = 0; y < height; ++y) {
              /* scalar fallback */

              float* VL_RESTRICT out = outputImage + y * width;
              float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

              for (i = 1; i <= 13; ++i) {
                int up = center - i;
                int dn = center + i;
                if (up < 0) up += R;
                if (dn >= R) dn -= R;
                rowUp[i] = rowBuf + (size_t)up * width;
                rowDn[i] = rowBuf + (size_t)dn * width;
              }
              for (x = 0; x < width; ++x) {
                float acc = k0 * rowCenter[x];
                for (i = 1; i <= 13; ++i)
                  acc += kw[i] * (rowUp[i][x] + rowDn[i][x]);
                out[x] = acc;
              }

              SLIDE_RING();
            }
          }
          break;

          /* ======================================================
           * Fallback
           * ====================================================== */
        default:
          for (y = 0; y < height; ++y) {

            float* VL_RESTRICT out = outputImage + y * width;
            float* VL_RESTRICT rowCenter = rowBuf + (size_t)center * width;

            for (i = 1; i <= W; ++i) {
              int up = center - i;
              int dn = center + i;
              if (up < 0) up += R;
              if (dn >= R) dn -= R;
              rowUp[i] = rowBuf + (size_t)up * width;
              rowDn[i] = rowBuf + (size_t)dn * width;
            }

#pragma loop(ivdep)
            for (x = 0; x + 1 < width; x += 2) {

              float a0 = k0 * rowCenter[x];
              float a1 = k0 * rowCenter[x + 1];

              for (i = 1; i <= W; ++i) {
                a0 += kw[i] * (rowUp[i][x] + rowDn[i][x]);
                a1 += kw[i] * (rowUp[i][x + 1] + rowDn[i][x + 1]);
              }

              out[x] = a0;
              out[x + 1] = a1;
            }

            for (; x < width; ++x) {
              float acc = k0 * rowCenter[x];
              for (i = 1; i <= W; ++i)
                acc += kw[i] * (rowUp[i][x] + rowDn[i][x]);
              out[x] = acc;
            }

            SLIDE_RING();
          }
          break;
        }
      }

      /* rowBuf is owned by the filter (self->smoothRowBuf); not freed here. */
    }
  }

  if (inputBuf) vl_free(inputBuf);
}

#else
static void
_vl_sift_smooth (VlSiftFilt * self,
                 vl_sift_pix * outputImage,
                 vl_sift_pix * tempImage,
                 vl_sift_pix const * inputImage,
                 vl_size width,
                 vl_size height,
                 double sigma)
{
  /* prepare Gaussian filter */
  if (self->gaussFilterSigma != sigma) {
    vl_uindex j ;
    vl_sift_pix acc = 0 ;

    self->gaussFilterWidth = VL_MAX(ceil(4.0 * sigma), 1) ;
    size_t const gaussFilterSize = sizeof(vl_sift_pix) * (2 * self->gaussFilterWidth + 1);
    if (gaussFilterSize > self->gaussFilterSize) {
    if (self->gaussFilter) vl_free (self->gaussFilter) ;
        self->gaussFilter = vl_malloc (gaussFilterSize) ;
        self->gaussFilterSize = gaussFilterSize;
    }

    self->gaussFilterSigma = sigma ;

    for (j = 0 ; j < 2 * self->gaussFilterWidth + 1 ; ++j) {
      vl_sift_pix d = ((vl_sift_pix)((signed)j - (signed)self->gaussFilterWidth)) / ((vl_sift_pix)sigma) ;
      self->gaussFilter[j] = (vl_sift_pix) exp (- 0.5 * (d*d)) ;
      acc += self->gaussFilter[j] ;
    }
    for (j = 0 ; j < 2 * self->gaussFilterWidth + 1 ; ++j) {
      self->gaussFilter[j] /= acc ;
    }
  }

  if (self->gaussFilterWidth == 0) {
    memcpy (outputImage, inputImage, sizeof(vl_sift_pix) * width * height) ;
    return ;
  }

  vl_imconvcol_vf (tempImage, height,
                   inputImage, width, height, width,
                   self->gaussFilter,
                   - self->gaussFilterWidth, self->gaussFilterWidth,
                   1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;

  vl_imconvcol_vf (outputImage, width,
                   tempImage, height, width, height,
                   self->gaussFilter,
                   - self->gaussFilterWidth, self->gaussFilterWidth,
                   1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;
}
#endif

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Copy and downsample an image
 **
 ** @param dst    output imgae buffer.
 ** @param src    input  image buffer.
 ** @param width  input  image width.
 ** @param height input  image height.
 ** @param d      octaves (non negative).
 **
 ** The function downsamples the image @a d times, reducing it to @c
 ** 1/2^d of its original size. The parameters @a width and @a height
 ** are the size of the input image. The destination image @a dst is
 ** assumed to be <code>floor(width/2^d)</code> pixels wide and
 ** <code>floor(height/2^d)</code> pixels high.
 **/

static void
copy_and_downsample
(vl_sift_pix       *dst,
 vl_sift_pix const *src,
 int width, int height, int d)
{
  int x, y ;

  d = 1 << d ; /* d = 2^d */
  for(y = 0 ; y < height ; y+=d) {
    vl_sift_pix const * srcrowp = src + y * width ;
    for(x = 0 ; x < width - (d-1) ; x+=d) {
      *dst++ = *srcrowp ;
      srcrowp += d ;
    }
  }
}

/** ------------------------------------------------------------------
 ** @brief Create a new SIFT filter
 **
 ** @param width    image width.
 ** @param height   image height.
 ** @param noctaves number of octaves.
 ** @param nlevels  number of levels per octave.
 ** @param o_min    first octave index.
 **
 ** The function allocates and returns a new SIFT filter for the
 ** specified image and scale space geometry.
 **
 ** Setting @a O to a negative value sets the number of octaves to the
 ** maximum possible value depending on the size of the image.
 **
 ** @return the new SIFT filter.
 ** @sa ::vl_sift_delete().
 **/

VL_EXPORT
VlSiftFilt *
vl_sift_new (int width, int height,
             int noctaves, int nlevels,
             int o_min)
{
  VlSiftFilt *f = vl_malloc (sizeof(VlSiftFilt)) ;

  int w   = VL_SHIFT_LEFT (width,  -o_min) ;
  int h   = VL_SHIFT_LEFT (height, -o_min) ;
  int nel = w * h ;

  /* negative value O => calculate max. value */
  if (noctaves < 0) {
    noctaves = VL_MAX (floor (log2 (VL_MIN(width, height))) - o_min - 3, 1) ;
  }

  f-> width   = width ;
  f-> height  = height ;
  f-> O       = noctaves ;
  f-> S       = nlevels ;
  f-> o_min   = o_min ;
  f-> s_min   = -1 ;
  f-> s_max   = nlevels + 1 ;
  f-> o_cur   = o_min ;

#ifdef REDUCE_MEMORY
  /* grad is no longer needed � orientations and descriptors compute
   * gradients on-the-fly from the octave data.
   * dog and temp share a single allocation; temp lives just past dog. */
  {
    const size_t dogSize = (size_t)nel * (size_t)(f->s_max - f->s_min);
    const size_t tempSize = (size_t)nel;
    f->dog = vl_malloc(sizeof(vl_sift_pix) * (dogSize + tempSize));
    f->temp = f->dog + dogSize;
  }

  f->octave = vl_malloc(sizeof(vl_sift_pix) * nel
    * (f->s_max - f->s_min + 1));
#else
  f-> temp    = vl_malloc (sizeof(vl_sift_pix) * nel    ) ;
  f-> octave  = vl_malloc (sizeof(vl_sift_pix) * nel
                        * (f->s_max - f->s_min + 1)  ) ;
  f-> dog     = vl_malloc (sizeof(vl_sift_pix) * nel
                        * (f->s_max - f->s_min    )  ) ;
  f-> grad    = vl_malloc (sizeof(vl_sift_pix) * nel * 2
                        * (f->s_max - f->s_min    )  ) ;
#endif

  f-> sigman  = 0.5 ;
  f-> sigmak  = pow (2.0, 1.0 / nlevels) ;
  f-> sigma0  = 1.6 * f->sigmak ;
  f-> dsigma0 = f->sigma0 * sqrt (1.0 - 1.0 / (f->sigmak*f->sigmak)) ;

  f-> gaussFilter = NULL ;
  f-> gaussFilterSize = 0;
  f-> gaussFilterSigma = 0 ;
  f-> gaussFilterWidth = 0 ;

  f-> smoothRowBuf = NULL ;
  f-> smoothRowBufSize = 0 ;

  f-> octave_width  = 0 ;
  f-> octave_height = 0 ;

  f-> keys     = 0 ;
  f-> nkeys    = 0 ;
  f-> keys_res = 0 ;

  f-> peak_thresh = 0.0 ;
  f-> edge_thresh = 10.0 ;
  f-> norm_thresh = 0.0 ;
  f-> magnif      = 3.0 ;
  f-> windowSize  = NBP / 2 ;

  /* initialize fast_expn stuff */
  fast_expn_init (f) ;

  return f ;
}

/** -------------------------------------------------------------------
 ** @brief Delete SIFT filter
 **
 ** @param f SIFT filter to delete.
 **
 ** The function frees the resources allocated by ::vl_sift_new().
 **/

VL_EXPORT
void
vl_sift_delete (VlSiftFilt* f)
{
  if (f) {
    if (f->keys) vl_free(f->keys);
    if (f->dog) vl_free(f->dog);
    if (f->octave) vl_free(f->octave);
#ifndef REDUCE_MEMORY
    if (f->temp) vl_free(f->temp);
#endif
    if (f->gaussFilter) vl_free(f->gaussFilter);
    if (f->smoothRowBuf) vl_free(f->smoothRowBuf);
    vl_free(f);
  }
}

/** ------------------------------------------------------------------
 ** @brief Start processing a new image
 **
 ** @param f  SIFT filter.
 ** @param im image data.
 **
 ** The function starts processing a new image by computing its
 ** Gaussian scale space at the lower octave. It also empties the
 ** internal keypoint buffer.
 **
 ** @return error code. The function returns ::VL_ERR_EOF if there are
 ** no more octaves to process.
 **
 ** @sa ::vl_sift_process_next_octave().
 **/

VL_EXPORT
int
vl_sift_process_first_octave (VlSiftFilt *f, vl_sift_pix const *im)
{
  int o, s, h, w ;
  double sa, sb ;
  vl_sift_pix *octave ;

  /* shortcuts */
  vl_sift_pix *temp   = f-> temp ;
  int width           = f-> width ;
  int height          = f-> height ;
  int o_min           = f-> o_min ;
  int s_min           = f-> s_min ;
  int s_max           = f-> s_max ;
  double sigma0       = f-> sigma0 ;
  double sigmak       = f-> sigmak ;
  double sigman       = f-> sigman ;
  double dsigma0      = f-> dsigma0 ;

  /* restart from the first */
  f->o_cur = o_min ;
  f->nkeys = 0 ;
  w = f-> octave_width  = VL_SHIFT_LEFT(f->width,  - f->o_cur) ;
  h = f-> octave_height = VL_SHIFT_LEFT(f->height, - f->o_cur) ;

  /* is there at least one octave? */
  if (f->O == 0)
    return VL_ERR_EOF ;

  /* ------------------------------------------------------------------
   *                     Compute the first sublevel of the first octave
   * --------------------------------------------------------------- */

  /*
   * If the first octave has negative index, we upscale the image; if
   * the first octave has positive index, we downscale the image; if
   * the first octave has index zero, we just copy the image.
   */

  octave = vl_sift_get_octave (f, s_min) ;

  if (o_min < 0) {
    /* double once */
    copy_and_upsample_rows (temp,   im,   width,      height) ;
    copy_and_upsample_rows (octave, temp, height, 2 * width ) ;

    /* double more */
    for(o = -1 ; o > o_min ; --o) {
      copy_and_upsample_rows (temp, octave,
                              width << -o,      height << -o ) ;
      copy_and_upsample_rows (octave, temp,
                              width << -o, 2 * (height << -o)) ;
    }
  }
  else if (o_min > 0) {
    /* downsample */
    copy_and_downsample (octave, im, width, height, o_min) ;
  }
  else {
    /* direct copy */
    memcpy(octave, im, sizeof(vl_sift_pix) * width * height) ;
  }

  /*
   * Here we adjust the smoothing of the first level of the octave.
   * The input image is assumed to have nominal smoothing equal to
   * f->simgan.
   */

  sa = sigma0 * pow (sigmak,   s_min) ;
  sb = sigman * pow (2.0,    - o_min) ;

  if (sa > sb) {
    double sd = sqrt (sa*sa - sb*sb) ;
    _vl_sift_smooth (f, octave, temp, octave, w, h, sd) ;
  }

  /* -----------------------------------------------------------------
   *                                          Compute the first octave
   * -------------------------------------------------------------- */

  for(s = s_min + 1 ; s <= s_max ; ++s) {
    double sd = dsigma0 * pow (sigmak, s) ;
    _vl_sift_smooth (f, vl_sift_get_octave(f, s), temp,
                     vl_sift_get_octave(f, s - 1), w, h, sd) ;
  }

  return VL_ERR_OK ;
}

/** ------------------------------------------------------------------
 ** @brief Process next octave
 **
 ** @param f SIFT filter.
 **
 ** The function computes the next octave of the Gaussian scale space.
 ** Notice that this clears the record of any feature detected in the
 ** previous octave.
 **
 ** @return error code. The function returns the error
 ** ::VL_ERR_EOF when there are no more octaves to process.
 **
 ** @sa ::vl_sift_process_first_octave().
 **/

VL_EXPORT
int
vl_sift_process_next_octave (VlSiftFilt *f)
{

  int s, h, w, s_best ;
  double sa, sb ;
  vl_sift_pix *octave, *pt ;

  /* shortcuts */
  vl_sift_pix *temp   = f-> temp ;
  int O               = f-> O ;
  int S               = f-> S ;
  int o_min           = f-> o_min ;
  int s_min           = f-> s_min ;
  int s_max           = f-> s_max ;
  double sigma0       = f-> sigma0 ;
  double sigmak       = f-> sigmak ;
  double dsigma0      = f-> dsigma0 ;

  /* is there another octave ? */
  if (f->o_cur == o_min + O - 1)
    return VL_ERR_EOF ;

  /* retrieve base */
  s_best = VL_MIN(s_min + S, s_max) ;
  w      = vl_sift_get_octave_width  (f) ;
  h      = vl_sift_get_octave_height (f) ;
  pt     = vl_sift_get_octave        (f, s_best) ;
  octave = vl_sift_get_octave        (f, s_min) ;

  /* next octave */
  copy_and_downsample (octave, pt, w, h, 1) ;

  f-> o_cur            += 1 ;
  f-> nkeys             = 0 ;
  w = f-> octave_width  = VL_SHIFT_LEFT(f->width,  - f->o_cur) ;
  h = f-> octave_height = VL_SHIFT_LEFT(f->height, - f->o_cur) ;

  sa = sigma0 * powf (sigmak, s_min     ) ;
  sb = sigma0 * powf (sigmak, s_best - S) ;

  if (sa > sb) {
    double sd = sqrt (sa*sa - sb*sb) ;
    _vl_sift_smooth (f, octave, temp, octave, w, h, sd) ;
  }

  /* ------------------------------------------------------------------
   *                                                        Fill octave
   * --------------------------------------------------------------- */

  for(s = s_min + 1 ; s <= s_max ; ++s) {
    double sd = dsigma0 * pow (sigmak, s) ;
    _vl_sift_smooth (f, vl_sift_get_octave(f, s), temp,
                     vl_sift_get_octave(f, s - 1), w, h, sd) ;
  }

  return VL_ERR_OK ;
}

static inline void DogSubtractSse2(
  vl_sift_pix* dst,
  const vl_sift_pix* a,
  const vl_sift_pix* b,
  int count)
{
  int i = 0;
  int simdEnd = count & ~3;

  for (; i < simdEnd; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 vd = _mm_sub_ps(vb, va);
    _mm_storeu_ps(dst + i, vd);
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
  int stride);

static __forceinline void DogSubtract(
  vl_sift_pix* VL_RESTRICT dst,
  const vl_sift_pix* VL_RESTRICT a,
  const vl_sift_pix* VL_RESTRICT b,
  int width,
  int height,
  int stride)
{
  // Fallbacks: row-based
  for (int y = 0; y < height; ++y) {
    if (hasAVX2) {
      DogSubtractAvx2(dst, a, b, width);
    }
    else {
      DogSubtractSse2(dst, a, b, width);
    }

    dst += stride;
    a += stride;
    b += stride;
  }
}


/** ------------------------------------------------------------------
 ** @brief Detect keypoints
 **
 ** The function detect keypoints in the current octave filling the
 ** internal keypoint buffer. Keypoints can be retrieved by
 ** ::vl_sift_get_keypoints().
 **
 ** @param f SIFT filter.
 **/
#if 1 // tiled
VL_EXPORT
void
vl_sift_detect(VlSiftFilt* f)
{
  vl_sift_pix* __restrict dog = f->dog;
  int          s_min = f->s_min;
  int          s_max = f->s_max;
  int          w = f->octave_width;
  int          h = f->octave_height;
  double       te = f->edge_thresh;
  double       tp = f->peak_thresh;

  int const    xo = 1;      /* x-stride */
  int const    yo = w;      /* y-stride */
  int const    so = w * h;  /* s-stride */

  double       xper = pow(2.0, f->o_cur);

  int x, y, s, i, ii, jj;
  vl_sift_pix const* __restrict pt;
  vl_sift_pix v;
  VlSiftKeypoint* k;

  /* clear current list */
  f->nkeys = 0;

  float const tolerance = (float)(0.8 * tp);

  /* Tile height: chosen so 3 DoG planes � tileH rows � w floats fits in L2.
   * For w=2048, tileH=32 ? 3�34�2048�4 = 835 KB (fits comfortably in 1MB L2).
   * The +2 accounts for the 1-pixel border needed by the extrema scan. */
  int const tileH = (w > 512) ? 32 : h; /* only tile large octaves */

  for (int tyStart = 0; tyStart < h; tyStart += tileH) {
    int const tyEnd = min(tyStart + tileH, h);

    /* DoG border: need 1 row above and below for extrema scan */
    int const dogYStart = max(tyStart - 1, 0);
    int const dogYEnd = min(tyEnd + 1, h);
    int const dogRowCount = dogYEnd - dogYStart;

    /* Compute DoG for this horizontal strip, all scale levels */
    for (s = s_min; s <= s_max - 1; ++s) {
      vl_sift_pix* src_a = vl_sift_get_octave(f, s) + dogYStart * w;
      vl_sift_pix* src_b = vl_sift_get_octave(f, s + 1) + dogYStart * w;
      vl_sift_pix* dst = dog + (s - s_min) * so + dogYStart * w;
      DogSubtract(dst, src_a, src_b, w, dogRowCount, w);
    }

    /* Scan for extrema in this strip � only interior rows [1, h-2] */
    int const scanYStart = max(tyStart, 1);
    int const scanYEnd = min(tyEnd, h - 1);

    for (s = s_min + 1; s <= s_max - 2; ++s) {
      pt = dog + xo + scanYStart * yo + (s - s_min) * so;

      __m128 const vPosTol = _mm_set1_ps(+tolerance);
      __m128 const vNegTol = _mm_set1_ps(-tolerance);

      for (y = scanYStart; y < scanYEnd; ++y) {
        /* SSE2: process 4 pixels at a time for threshold pre-screening */
        int const innerCount = w - 2; /* x in [1, w-2] */
        int xIdx = 0;

        for (; xIdx + 3 < innerCount; xIdx += 4) {
          __m128 vV = _mm_loadu_ps((float const*)pt);

          /* Any pixel >= +tolerance? */
          int maskPos = _mm_movemask_ps(_mm_cmpge_ps(vV, vPosTol));
          /* Any pixel <= -tolerance? */
          int maskNeg = _mm_movemask_ps(_mm_cmple_ps(vV, vNegTol));

          if (maskPos | maskNeg) {
            /* At least one pixel passed threshold � check each scalar */
            int lane;
            for (lane = 0; lane < 4; ++lane) {
              if (maskPos & (1 << lane)) {
                v = *(pt + lane);
                if (v > *(pt + lane + xo) &&
                  v > *(pt + lane - xo) &&
                  v > *(pt + lane + yo) &&
                  v > *(pt + lane - yo) &&
                  v > *(pt + lane + yo + xo) &&
                  v > *(pt + lane + yo - xo) &&
                  v > *(pt + lane - yo + xo) &&
                  v > *(pt + lane - yo - xo) &&
                  v > *(pt + lane + so) &&
                  v > *(pt + lane + xo + so) &&
                  v > *(pt + lane - xo + so) &&
                  v > *(pt + lane + yo + so) &&
                  v > *(pt + lane - yo + so) &&
                  v > *(pt + lane + yo + xo + so) &&
                  v > *(pt + lane + yo - xo + so) &&
                  v > *(pt + lane - yo + xo + so) &&
                  v > *(pt + lane - yo - xo + so) &&
                  v > *(pt + lane - so) &&
                  v > *(pt + lane + xo - so) &&
                  v > *(pt + lane - xo - so) &&
                  v > *(pt + lane + yo - so) &&
                  v > *(pt + lane - yo - so) &&
                  v > *(pt + lane + yo + xo - so) &&
                  v > *(pt + lane + yo - xo - so) &&
                  v > *(pt + lane - yo + xo - so) &&
                  v > *(pt + lane - yo - xo - so)) {
                  goto found4;
                }
              }
              else if (maskNeg & (1 << lane)) {
                v = *(pt + lane);
                if (v < *(pt + lane + xo) &&
                  v < *(pt + lane - xo) &&
                  v < *(pt + lane + yo) &&
                  v < *(pt + lane - yo) &&
                  v < *(pt + lane + yo + xo) &&
                  v < *(pt + lane + yo - xo) &&
                  v < *(pt + lane - yo + xo) &&
                  v < *(pt + lane - yo - xo) &&
                  v < *(pt + lane + so) &&
                  v < *(pt + lane + xo + so) &&
                  v < *(pt + lane - xo + so) &&
                  v < *(pt + lane + yo + so) &&
                  v < *(pt + lane - yo + so) &&
                  v < *(pt + lane + yo + xo + so) &&
                  v < *(pt + lane + yo - xo + so) &&
                  v < *(pt + lane - yo + xo + so) &&
                  v < *(pt + lane - yo - xo + so) &&
                  v < *(pt + lane - so) &&
                  v < *(pt + lane + xo - so) &&
                  v < *(pt + lane - xo - so) &&
                  v < *(pt + lane + yo - so) &&
                  v < *(pt + lane - yo - so) &&
                  v < *(pt + lane + yo + xo - so) &&
                  v < *(pt + lane + yo - xo - so) &&
                  v < *(pt + lane - yo + xo - so) &&
                  v < *(pt + lane - yo - xo - so)) {
                found4:
                  x = (xIdx + lane) + 1; /* +1 because x starts at 1 */
                  if (f->nkeys >= f->keys_res) {
                    f->keys_res += 32768;
                    if (f->keys) {
                      f->keys = vl_realloc(f->keys,
                        f->keys_res * sizeof(VlSiftKeypoint));
                    }
                    else {
                      f->keys = vl_malloc(f->keys_res *
                        sizeof(VlSiftKeypoint));
                    }
                  }
                  k = f->keys + (f->nkeys++);
                  k->ix = x;
                  k->iy = y;
                  k->is = s;
                }
              }
            } /* for lane */
          } /* if any passed */
          pt += 4;
        } /* for xIdx SSE */

        /* Scalar tail for remaining 1�3 pixels */
        for (; xIdx < innerCount; ++xIdx) {
          v = *pt;

          if (v >= +tolerance) {
            if (v > *(pt + xo) &&
              v > *(pt - xo) &&
              v > *(pt + yo) &&
              v > *(pt - yo) &&
              v > *(pt + yo + xo) &&
              v > *(pt + yo - xo) &&
              v > *(pt - yo + xo) &&
              v > *(pt - yo - xo) &&
              v > *(pt + so) &&
              v > *(pt + xo + so) &&
              v > *(pt - xo + so) &&
              v > *(pt + yo + so) &&
              v > *(pt - yo + so) &&
              v > *(pt + yo + xo + so) &&
              v > *(pt + yo - xo + so) &&
              v > *(pt - yo + xo + so) &&
              v > *(pt - yo - xo + so) &&
              v > *(pt - so) &&
              v > *(pt + xo - so) &&
              v > *(pt - xo - so) &&
              v > *(pt + yo - so) &&
              v > *(pt - yo - so) &&
              v > *(pt + yo + xo - so) &&
              v > *(pt + yo - xo - so) &&
              v > *(pt - yo + xo - so) &&
              v > *(pt - yo - xo - so))
              goto foundTail;
          }
          else if (v <= -tolerance) {
            if (v < *(pt + xo) &&
              v < *(pt - xo) &&
              v < *(pt + yo) &&
              v < *(pt - yo) &&
              v < *(pt + yo + xo) &&
              v < *(pt + yo - xo) &&
              v < *(pt - yo + xo) &&
              v < *(pt - yo - xo) &&
              v < *(pt + so) &&
              v < *(pt + xo + so) &&
              v < *(pt - xo + so) &&
              v < *(pt + yo + so) &&
              v < *(pt - yo + so) &&
              v < *(pt + yo + xo + so) &&
              v < *(pt + yo - xo + so) &&
              v < *(pt - yo + xo + so) &&
              v < *(pt - yo - xo + so) &&
              v < *(pt - so) &&
              v < *(pt + xo - so) &&
              v < *(pt - xo - so) &&
              v < *(pt + yo - so) &&
              v < *(pt - yo - so) &&
              v < *(pt + yo + xo - so) &&
              v < *(pt + yo - xo - so) &&
              v < *(pt - yo + xo - so) &&
              v < *(pt - yo - xo - so)) {
            foundTail:
              x = xIdx + 1;
              if (f->nkeys >= f->keys_res) {
                f->keys_res += 32768;
                if (f->keys) {
                  f->keys = vl_realloc(f->keys,
                    f->keys_res * sizeof(VlSiftKeypoint));
                }
                else {
                  f->keys = vl_malloc(f->keys_res *
                    sizeof(VlSiftKeypoint));
                }
              }
              k = f->keys + (f->nkeys++);
              k->ix = x;
              k->iy = y;
              k->is = s;
            }
          }
          pt += 1;
        } /* scalar tail */

        pt += 2; /* skip border pixels */
      } /* for y */
    } /* for s */
  } /* for tyStart */

  /* -----------------------------------------------------------------
   *                                               Refine local maxima
   * -------------------------------------------------------------- */

   /* this pointer is used to write the keypoints back */
  k = f->keys;

  for (i = 0; i < f->nkeys; ++i) {
    int x = f->keys[i].ix;
    int y = f->keys[i].iy;
    int s = f->keys[i].is;

    double Dx = 0, Dy = 0, Ds = 0, Dxx = 0, Dyy = 0, Dss = 0, Dxy = 0, Dxs = 0, Dys = 0;
    double A[3 * 3], b[3];

    int dx = 0;
    int dy = 0;

    int iter, i, j;

    for (iter = 0; iter < 5; ++iter) {
      x += dx;
      y += dy;

      pt = dog
        + xo * x
        + yo * y
        + so * (s - s_min);

#define at(dx,dy,ds) (*( pt + (dx)*xo + (dy)*yo + (ds)*so))
#define Aat(i,j)     (A[(i)+(j)*3])

      /* compute the gradient */
      Dx = 0.5 * (at(+1, 0, 0) - at(-1, 0, 0));
      Dy = 0.5 * (at(0, +1, 0) - at(0, -1, 0));
      Ds = 0.5 * (at(0, 0, +1) - at(0, 0, -1));

      /* compute the Hessian */
      Dxx = (at(+1, 0, 0) + at(-1, 0, 0) - 2.0 * at(0, 0, 0));
      Dyy = (at(0, +1, 0) + at(0, -1, 0) - 2.0 * at(0, 0, 0));
      Dss = (at(0, 0, +1) + at(0, 0, -1) - 2.0 * at(0, 0, 0));

      Dxy = 0.25 * (at(+1, +1, 0) + at(-1, -1, 0) - at(-1, +1, 0) - at(+1, -1, 0));
      Dxs = 0.25 * (at(+1, 0, +1) + at(-1, 0, -1) - at(-1, 0, +1) - at(+1, 0, -1));
      Dys = 0.25 * (at(0, +1, +1) + at(0, -1, -1) - at(0, -1, +1) - at(0, +1, -1));

      Aat(0, 0) = Dxx;
      Aat(1, 1) = Dyy;
      Aat(2, 2) = Dss;
      Aat(0, 1) = Aat(1, 0) = Dxy;
      Aat(0, 2) = Aat(2, 0) = Dxs;
      Aat(1, 2) = Aat(2, 1) = Dys;

      b[0] = -Dx;
      b[1] = -Dy;
      b[2] = -Ds;

      /* Gauss elimination */
      for (j = 0; j < 3; ++j) {
        double maxa = 0;
        double maxabsa = 0;
        int    maxi = -1;
        double tmp;

        for (i = j; i < 3; ++i) {
          double a = Aat(i, j);
          double absa = vl_abs_d(a);
          if (absa > maxabsa) {
            maxa = a;
            maxabsa = absa;
            maxi = i;
          }
        }

        if (maxabsa < 1e-10f) {
          b[0] = 0;
          b[1] = 0;
          b[2] = 0;
          break;
        }

        i = maxi;

        for (jj = j; jj < 3; ++jj) {
          tmp = Aat(i, jj); Aat(i, jj) = Aat(j, jj); Aat(j, jj) = tmp;
          Aat(j, jj) /= maxa;
        }
        tmp = b[j]; b[j] = b[i]; b[i] = tmp;
        b[j] /= maxa;

        for (ii = j + 1; ii < 3; ++ii) {
          double x = Aat(ii, j);
          for (jj = j; jj < 3; ++jj) {
            Aat(ii, jj) -= x * Aat(j, jj);
          }
          b[ii] -= x * b[j];
        }
      }

      for (i = 2; i > 0; --i) {
        double x = b[i];
        for (ii = i - 1; ii >= 0; --ii) {
          b[ii] -= x * Aat(ii, i);
        }
      }

      dx = ((b[0] > 0.6 && x < w - 2) ? 1 : 0)
        + ((b[0] < -0.6 && x > 1) ? -1 : 0);

      dy = ((b[1] > 0.6 && y < h - 2) ? 1 : 0)
        + ((b[1] < -0.6 && y > 1) ? -1 : 0);

      if (dx == 0 && dy == 0) break;
    }

    /* check threshold and other conditions */
    {
      double val = at(0, 0, 0)
        + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]);
      double score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy);
      double xn = x + b[0];
      double yn = y + b[1];
      double sn = s + b[2];

      vl_bool good =
        vl_abs_d(val) > tp &&
        score < (te + 1) * (te + 1) / te &&
        score >= 0 &&
        vl_abs_d(b[0]) < 1.5 &&
        vl_abs_d(b[1]) < 1.5 &&
        vl_abs_d(b[2]) < 1.5 &&
        xn >= 0 &&
        xn <= w - 1 &&
        yn >= 0 &&
        yn <= h - 1 &&
        sn >= s_min &&
        sn <= s_max;

      if (good) {
        k->o = f->o_cur;
        k->ix = x;
        k->iy = y;
        k->is = s;
        k->s = sn;
        k->x = xn * xper;
        k->y = yn * xper;
        k->sigma = f->sigma0 * pow(2.0, sn / f->S) * xper;
        ++k;
      }
    } /* done checking */
  } /* next keypoint to refine */

  /* update keypoint count */
  f->nkeys = (int)(k - f->keys);
}
#else // working no tile
VL_EXPORT
void
vl_sift_detect(VlSiftFilt* f)
{
  vl_sift_pix* __restrict dog = f->dog;  int          s_min = f->s_min;
  int          s_max = f->s_max;
  int          w = f->octave_width;
  int          h = f->octave_height;
  double       te = f->edge_thresh;
  double       tp = f->peak_thresh;

  int const    xo = 1;      /* x-stride */
  int const    yo = w;      /* y-stride */
  int const    so = w * h;  /* s-stride */

  double       xper = pow(2.0, f->o_cur);

  int x, y, s, i, ii, jj;
  vl_sift_pix const* __restrict pt;
  vl_sift_pix v;
  VlSiftKeypoint* k;

  /* clear current list */
  f->nkeys = 0;

  /* compute difference of gaussian (DoG) */
  pt = f->dog;
  for (s = s_min; s <= s_max - 1; ++s) {
    vl_sift_pix* src_a = vl_sift_get_octave(f, s);
    vl_sift_pix* src_b = vl_sift_get_octave(f, s + 1);
    vl_sift_pix* dst = dog + (s - s_min) * so;
    DogSubtract(dst, src_a, src_b, w, h, w);
  }

  /* -----------------------------------------------------------------
   *                                          Find local maxima of DoG
   * -------------------------------------------------------------- */
  float const tolerance = (float)(0.8 * tp);

   /* start from dog [1,1,s_min+1] */
  pt = dog + xo + yo + so;

  for (s = s_min + 1; s <= s_max - 2; ++s) {
    for (y = 1; y < h - 1; ++y) {
      for (x = 1; x < w - 1; ++x) {
        v = *pt;

#define CHECK_NEIGHBORS(CMP,SGN)                    \
        ( v CMP ## = SGN tolerance &&               \
          v CMP *(pt + xo) &&                       \
          v CMP *(pt - xo) &&                       \
          v CMP *(pt + so) &&                       \
          v CMP *(pt - so) &&                       \
          v CMP *(pt + yo) &&                       \
          v CMP *(pt - yo) &&                       \
                                                    \
          v CMP *(pt + yo + xo) &&                  \
          v CMP *(pt + yo - xo) &&                  \
          v CMP *(pt - yo + xo) &&                  \
          v CMP *(pt - yo - xo) &&                  \
                                                    \
          v CMP *(pt + xo      + so) &&             \
          v CMP *(pt - xo      + so) &&             \
          v CMP *(pt + yo      + so) &&             \
          v CMP *(pt - yo      + so) &&             \
          v CMP *(pt + yo + xo + so) &&             \
          v CMP *(pt + yo - xo + so) &&             \
          v CMP *(pt - yo + xo + so) &&             \
          v CMP *(pt - yo - xo + so) &&             \
                                                    \
          v CMP *(pt + xo      - so) &&             \
          v CMP *(pt - xo      - so) &&             \
          v CMP *(pt + yo      - so) &&             \
          v CMP *(pt - yo      - so) &&             \
          v CMP *(pt + yo + xo - so) &&             \
          v CMP *(pt + yo - xo - so) &&             \
          v CMP *(pt - yo + xo - so) &&             \
          v CMP *(pt - yo - xo - so) )

        if (CHECK_NEIGHBORS(> , +) ||
          CHECK_NEIGHBORS(< , -)) {

          /* make room for more keypoints */
          if (f->nkeys >= f->keys_res) {
            f->keys_res += 32768;
            if (f->keys) {
              f->keys = vl_realloc(f->keys,
                f->keys_res *
                sizeof(VlSiftKeypoint));
            }
            else {
              f->keys = vl_malloc(f->keys_res *
                sizeof(VlSiftKeypoint));
            }
          }

          k = f->keys + (f->nkeys++);

          k->ix = x;
          k->iy = y;
          k->is = s;
        }
        pt += 1;
      }
      pt += 2;
    }
    pt += 2 * yo;
  }

  /* -----------------------------------------------------------------
   *                                               Refine local maxima
   * -------------------------------------------------------------- */

   /* this pointer is used to write the keypoints back */
  k = f->keys;

  for (i = 0; i < f->nkeys; ++i) {
    int x = f->keys[i].ix;
    int y = f->keys[i].iy;
    int s = f->keys[i].is;

    double Dx = 0, Dy = 0, Ds = 0, Dxx = 0, Dyy = 0, Dss = 0, Dxy = 0, Dxs = 0, Dys = 0;
    double A[3 * 3], b[3];

    int dx = 0;
    int dy = 0;

    int iter, i, j;

    for (iter = 0; iter < 5; ++iter) {
      x += dx;
      y += dy;

      pt = dog
        + xo * x
        + yo * y
        + so * (s - s_min);

      /** @brief Index GSS @internal */
#define at(dx,dy,ds) (*( pt + (dx)*xo + (dy)*yo + (ds)*so))

      /** @brief Index matrix A @internal */
#define Aat(i,j)     (A[(i)+(j)*3])

      /* compute the gradient */
      Dx = 0.5 * (at(+1, 0, 0) - at(-1, 0, 0));
      Dy = 0.5 * (at(0, +1, 0) - at(0, -1, 0));
      Ds = 0.5 * (at(0, 0, +1) - at(0, 0, -1));

      /* compute the Hessian */
      Dxx = (at(+1, 0, 0) + at(-1, 0, 0) - 2.0 * at(0, 0, 0));
      Dyy = (at(0, +1, 0) + at(0, -1, 0) - 2.0 * at(0, 0, 0));
      Dss = (at(0, 0, +1) + at(0, 0, -1) - 2.0 * at(0, 0, 0));

      Dxy = 0.25 * (at(+1, +1, 0) + at(-1, -1, 0) - at(-1, +1, 0) - at(+1, -1, 0));
      Dxs = 0.25 * (at(+1, 0, +1) + at(-1, 0, -1) - at(-1, 0, +1) - at(+1, 0, -1));
      Dys = 0.25 * (at(0, +1, +1) + at(0, -1, -1) - at(0, -1, +1) - at(0, +1, -1));

      /* solve linear system ....................................... */
      Aat(0, 0) = Dxx;
      Aat(1, 1) = Dyy;
      Aat(2, 2) = Dss;
      Aat(0, 1) = Aat(1, 0) = Dxy;
      Aat(0, 2) = Aat(2, 0) = Dxs;
      Aat(1, 2) = Aat(2, 1) = Dys;

      b[0] = -Dx;
      b[1] = -Dy;
      b[2] = -Ds;

      /* Gauss elimination */
      for (j = 0; j < 3; ++j) {
        double maxa = 0;
        double maxabsa = 0;
        int    maxi = -1;
        double tmp;

        /* look for the maximally stable pivot */
        for (i = j; i < 3; ++i) {
          double a = Aat(i, j);
          double absa = vl_abs_d(a);
          if (absa > maxabsa) {
            maxa = a;
            maxabsa = absa;
            maxi = i;
          }
        }

        /* if singular give up */
        if (maxabsa < 1e-10f) {
          b[0] = 0;
          b[1] = 0;
          b[2] = 0;
          break;
        }

        i = maxi;

        /* swap j-th row with i-th row and normalize j-th row */
        for (jj = j; jj < 3; ++jj) {
          tmp = Aat(i, jj); Aat(i, jj) = Aat(j, jj); Aat(j, jj) = tmp;
          Aat(j, jj) /= maxa;
        }
        tmp = b[j]; b[j] = b[i]; b[i] = tmp;
        b[j] /= maxa;

        /* elimination */
        for (ii = j + 1; ii < 3; ++ii) {
          double x = Aat(ii, j);
          for (jj = j; jj < 3; ++jj) {
            Aat(ii, jj) -= x * Aat(j, jj);
          }
          b[ii] -= x * b[j];
        }
      }

      /* backward substitution */
      for (i = 2; i > 0; --i) {
        double x = b[i];
        for (ii = i - 1; ii >= 0; --ii) {
          b[ii] -= x * Aat(ii, i);
        }
      }

      /* .......................................................... */
      /* If the translation of the keypoint is big, move the keypoint
       * and re-iterate the computation. Otherwise we are all set.
       */

      dx = ((b[0] > 0.6 && x < w - 2) ? 1 : 0)
        + ((b[0] < -0.6 && x > 1) ? -1 : 0);

      dy = ((b[1] > 0.6 && y < h - 2) ? 1 : 0)
        + ((b[1] < -0.6 && y > 1) ? -1 : 0);

      if (dx == 0 && dy == 0) break;
    }

    /* check threshold and other conditions */
    {
      double val = at(0, 0, 0)
        + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]);
      double score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy);
      double xn = x + b[0];
      double yn = y + b[1];
      double sn = s + b[2];

      vl_bool good =
        vl_abs_d(val) > tp &&
        score < (te + 1) * (te + 1) / te &&
        score >= 0 &&
        vl_abs_d(b[0]) < 1.5 &&
        vl_abs_d(b[1]) < 1.5 &&
        vl_abs_d(b[2]) < 1.5 &&
        xn >= 0 &&
        xn <= w - 1 &&
        yn >= 0 &&
        yn <= h - 1 &&
        sn >= s_min &&
        sn <= s_max;

      if (good) {
        k->o = f->o_cur;
        k->ix = x;
        k->iy = y;
        k->is = s;
        k->s = sn;
        k->x = xn * xper;
        k->y = yn * xper;
        k->sigma = f->sigma0 * pow(2.0, sn / f->S) * xper;
        ++k;
      }
    } /* done checking */
  } /* next keypoint to refine */

  /* update keypoint count */
  f->nkeys = (int)(k - f->keys);
}
#endif

#if 0 // Now unused
/** ------------------------------------------------------------------
 ** @brief Update gradients to current GSS octave
 **
 ** @param f SIFT filter.
 **
 ** The function makes sure that the gradient buffer is up-to-date
 ** with the current GSS data.
 **
 ** @remark The minimum octave size is 2x2xS.
 **/
void
vl_sift_update_gradient (VlSiftFilt *f)
{
  int       s_min = f->s_min ;
  int       s_max = f->s_max ;
  int       w     = vl_sift_get_octave_width  (f) ;
  int       h     = vl_sift_get_octave_height (f) ;
  int const xo    = 1 ;
  int const yo    = w ;
  int const so    = h * w ;
  int y, s ;

  if (f->grad_o == f->o_cur) return ;

  for (s  = s_min + 1 ;
       s <= s_max - 2 ; ++ s) {

    vl_sift_pix *src, *end, *grad, gx, gy ;

    /* Store raw (gx, gy) � no sqrt, no atan2 */
#define SAVE_BACK                                                       \
    *grad++ = gx ;                                                      \
    *grad++ = gy ;                                                      \
    ++src ;                                                                \

    src  = vl_sift_get_octave (f,s) ;
    grad = f->grad + 2 * so * (s - s_min -1) ;

    /* first pixel of the first row */
    gx = src[+xo] - src[0] ;
    gy = src[+yo] - src[0] ;
    SAVE_BACK ;

    /* middle pixels of the  first row */
    end = (src - 1) + w - 1 ;
    while (src < end) {
      gx = 0.5 * (src[+xo] - src[-xo]) ;
      gy =        src[+yo] - src[0] ;
      SAVE_BACK ;
    }

    /* last pixel of the first row */
    gx = src[0]   - src[-xo] ;
    gy = src[+yo] - src[0] ;
    SAVE_BACK ;

    for (y = 1 ; y < h -1 ; ++y) {
      /* first pixel of the middle rows */
      gx =        src[+xo] - src[0] ;
      gy = 0.5 * (src[+yo] - src[-yo]) ;
      SAVE_BACK ;

      /* middle pixels of the middle rows */
      end = (src - 1) + w - 1;
      while (src < end) {
        gx = 0.5f * (src[+xo] - src[-xo]);
        gy = 0.5f * (src[+yo] - src[-yo]);
        SAVE_BACK;
      }

      /* last pixel of the middle row */
      gx =        src[0]   - src[-xo] ;
      gy = 0.5 * (src[+yo] - src[-yo]) ;
      SAVE_BACK ;
    }

    /* first pixel of the last row */
    gx = src[+xo] - src[0] ;
    gy = src[  0] - src[-yo] ;
    SAVE_BACK ;

    /* middle pixels of the last row */
    end = (src - 1) + w - 1 ;
    while (src < end) {
      gx = 0.5 * (src[+xo] - src[-xo]) ;
      gy =        src[0]   - src[-yo] ;
      SAVE_BACK ;
    }

    /* last pixel of the last row */
    gx = src[0]   - src[-xo] ;
    gy = src[0]   - src[-yo] ;
    SAVE_BACK ;
  }
  f->grad_o = f->o_cur ;
}
#endif

/** ------------------------------------------------------------------
 ** @brief Calculate the keypoint orientation(s)
 **
 ** @param f        SIFT filter.
 ** @param angles   orientations (output).
 ** @param k        keypoint.
 **
 ** The function computes the orientation(s) of the keypoint @a k.
 ** The function returns the number of orientations found (up to
 ** four). The orientations themselves are written to the vector @a
 ** angles.
 **
 ** @remark The function requires the keypoint octave @a k->o to be
 ** equal to the filter current octave ::vl_sift_get_octave. If this
 ** is not the case, the function returns zero orientations.
 **
 ** @remark The function requires the keypoint scale level @c k->s to
 ** be in the range @c s_min+1 and @c s_max-2 (where usually @c
 ** s_min=0 and @c s_max=S+2). If this is not the case, the function
 ** returns zero orientations.
 **
 ** @return number of orientations found.
 **/

VL_EXPORT
int
vl_sift_calc_keypoint_orientations(VlSiftFilt* f,
  double angles[4],
  VlSiftKeypoint const* k)
{
  double const winf = 1.5;
  double       xper = pow(2.0, f->o_cur);

  int          w = f->octave_width;
  int          h = f->octave_height;
  int const    xo = 1;         /* x-stride in octave */
  int const    yo = w;         /* y-stride in octave */
  double       x = k->x / xper;
  double       y = k->y / xper;
  double       sigma = k->sigma / xper;

  int          xi = (int)(x + 0.5);
  int          yi = (int)(y + 0.5);
  int          si = k->is;

  double const sigmaw = winf * sigma;
  int          W = VL_MAX((int)floor(3.0 * sigmaw), 1);

  int          nangles = 0;

  enum { nbins = 36 };

  double hist[nbins], maxh;
  vl_sift_pix const* src;
  int xs, ys, iter, i;

  /* skip if the keypoint octave is not current */
  if (k->o != f->o_cur)
    return 0;

  /* skip the keypoint if it is out of bounds */
  if (xi < 0 ||
    xi > w - 1 ||
    yi < 0 ||
    yi > h - 1 ||
    si < f->s_min + 1 ||
    si > f->s_max - 2) {
    return 0;
  }

  /* clear histogram */
  memset(hist, 0, sizeof(double) * nbins);

  /* Point to the octave plane for this scale level */
  src = vl_sift_get_octave(f, si);

  /* Precompute constants for SSE path */
  float const invSigmaw2 = (float)(1.0 / (2.0 * sigmaw * sigmaw));
  float const W2_thresh = (float)(W * W + 0.6);
  float const binScale = (float)(nbins / (2.0 * VL_PI));

  __m128 const vXf = _mm_set1_ps((float)x);
  __m128 const vYf = _mm_set1_ps((float)y);
  __m128 const vHalf = _mm_set1_ps(0.5f);
  __m128 const vZero = _mm_setzero_ps();
  __m128 const vOne = _mm_set1_ps(1.0f);
  __m128 const vFour = _mm_set1_ps(4.0f);
  __m128 const vInvSigmaw2 = _mm_set1_ps(invSigmaw2);
  __m128 const vW2thresh = _mm_set1_ps(W2_thresh);
  __m128 const vBinScale = _mm_set1_ps(binScale);
  __m128 const vNbins = _mm_set1_ps((float)nbins);
  __m128 const vStep = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);

  /* FastExp constants */
  __m128 const vExpL2e = _mm_set1_ps(1.442695041f);
  __m128 const vExpC0 = _mm_set1_ps(0.3371894346f);
  __m128 const vExpC1 = _mm_set1_ps(0.657636276f);
  __m128 const vExpC2 = _mm_set1_ps(1.00172476f);

  /* atan2 constants */
  __m128 const vC3 = _mm_set1_ps(0.1821f);
  __m128 const vC1 = _mm_set1_ps(0.9675f);
  __m128 const vHighBit = _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000));
  __m128 const vAbsMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  __m128 const vTwoPi = _mm_set1_ps((float)(2.0 * VL_PI));
  __m128 const vEps = _mm_set1_ps(1.19209290E-07F);
  __m128 const vPi4 = _mm_set1_ps((float)(VL_PI / 4.0));
  __m128 const v3Pi4 = _mm_set1_ps((float)(3.0 * VL_PI / 4.0));

  for (ys = VL_MAX(-W, 1 - yi);
    ys <= VL_MIN(+W, h - 2 - yi); ++ys) {

    float const dyF = (float)(yi + ys) - (float)y;
    __m128 const vDy = _mm_set1_ps(dyF);
    __m128 const vDy2 = _mm_mul_ps(vDy, vDy);

    int const xsMin = VL_MAX(-W, 1 - xi);
    int const xsMax = VL_MIN(+W, w - 2 - xi);
    int const count = xsMax - xsMin + 1;

    vl_sift_pix const* rowSrc = src + (yi + ys) * yo + (xi + xsMin);

    /* Base dx for first 4 pixels */
    __m128 vDxBase = _mm_add_ps(
      _mm_set1_ps((float)(xi + xsMin)), vStep);
    vDxBase = _mm_sub_ps(vDxBase, vXf);

    int remaining = count;
    vl_sift_pix const* pSrc = rowSrc;

    while (remaining >= 4) {
      /* Gradient on-the-fly */
      __m128 vGx = _mm_mul_ps(vHalf,
        _mm_sub_ps(_mm_loadu_ps(pSrc + 1), _mm_loadu_ps(pSrc - 1)));
      __m128 vGy = _mm_mul_ps(vHalf,
        _mm_sub_ps(_mm_loadu_ps(pSrc + yo), _mm_loadu_ps(pSrc - yo)));

      /* r2 = dx*dx + dy*dy */
      __m128 vR2 = _mm_add_ps(_mm_mul_ps(vDxBase, vDxBase), vDy2);

      /* Circular window mask: zero weight where r2 >= threshold */
      __m128 vMask = _mm_cmplt_ps(vR2, vW2thresh);

      /* mod = sqrt(gx*gx + gy*gy) � issue early */
      __m128 vModSq = _mm_add_ps(_mm_mul_ps(vGx, vGx), _mm_mul_ps(vGy, vGy));
      __m128 vMod = _mm_sqrt_ps(vModSq);

      /* FastExp inlined: exp(-r2 * invSigmaw2) */
      __m128 vExpArg = _mm_sub_ps(vZero, _mm_mul_ps(vR2, vInvSigmaw2));
      __m128 vT = _mm_mul_ps(vExpArg, vExpL2e);
      __m128i vI = _mm_cvttps_epi32(vT);
      __m128i vJ = _mm_srli_epi32(_mm_castps_si128(vExpArg), 31);
      vI = _mm_sub_epi32(vI, vJ);
      __m128 vE = _mm_cvtepi32_ps(vI);
      __m128 vF = _mm_sub_ps(vT, vE);
      __m128 vP = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(vExpC0, vF), vExpC1), vF), vExpC2);
      __m128 vWgt = _mm_castsi128_ps(_mm_add_epi32(_mm_slli_epi32(vI, 23), _mm_castps_si128(vP)));

      /* atan2 inlined � drop Newton refinement (matches descriptor) */
      __m128 vAbsY = _mm_add_ps(_mm_and_ps(vGy, vAbsMask), vEps);
      __m128 vAbsX = _mm_and_ps(vGx, vAbsMask);
      __m128 vNum = _mm_sub_ps(vGx, _mm_or_ps(vAbsY, _mm_and_ps(vGx, vHighBit)));
      __m128 vDen = _mm_add_ps(vAbsY, vAbsX);
      __m128 vR_at = _mm_mul_ps(vNum, _mm_rcp_ps(vDen));
      __m128 vXge0 = _mm_cmpge_ps(vGx, vZero);
      __m128 vBase = _mm_or_ps(_mm_and_ps(vXge0, vPi4), _mm_andnot_ps(vXge0, v3Pi4));
      __m128 vRR = _mm_mul_ps(vR_at, vR_at);
      __m128 vAngle = _mm_add_ps(vBase,
        _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(vC3, vRR), vC1), vR_at));
      __m128 vAtan2 = _mm_xor_ps(vAngle, _mm_and_ps(vGy, vHighBit));
      __m128 vShifted = _mm_add_ps(vAtan2, vTwoPi);
      __m128 vAng = _mm_sub_ps(vShifted,
        _mm_and_ps(_mm_cmpge_ps(vShifted, vTwoPi), vTwoPi));

      /* fbin = nbins * ang / (2*pi) */
      __m128 vFbin = _mm_mul_ps(vAng, vBinScale);

      /* weighted mod: mod * wgt, masked by circular window */
      __m128 vWmod = _mm_and_ps(_mm_mul_ps(vMod, vWgt), vMask);

      /* Bilinear scatter into histogram (scalar extract) */
      /* Spill to arrays � avoids 8 shuffle instructions */
      __declspec(align(16)) float fbinArr[4], wmodArr[4];
      _mm_store_ps(fbinArr, vFbin);
      _mm_store_ps(wmodArr, vWmod);

#define ORIENT_SCATTER(lane) {                                          \
        float wmod_s = wmodArr[lane];                                   \
        if (wmod_s > 0.0f) {                                            \
          float fbin_s = fbinArr[lane];                                 \
          int bin = (int)vl_floor_f(fbin_s - 0.5f);                     \
          float rbin = fbin_s - bin - 0.5f;                             \
          int b0 = bin + nbins;                                         \
          if (b0 >= nbins) b0 -= nbins;                                 \
          int b1 = bin + 1;                                             \
          if (b1 >= nbins) b1 -= nbins;                                 \
          hist[b0] += (double)((1.0f - rbin) * wmod_s);                 \
          hist[b1] += (double)(rbin * wmod_s);                          \
        }                                                               \
      }

      ORIENT_SCATTER(0)
      ORIENT_SCATTER(1)
      ORIENT_SCATTER(2)
      ORIENT_SCATTER(3)

#undef ORIENT_SCATTER

      pSrc += 4;
      vDxBase = _mm_add_ps(vDxBase, vFour);
      remaining -= 4;
    }

    /* Scalar tail for remaining 1�3 pixels */
    for (; remaining > 0; --remaining, ++pSrc) {
      /* dx from the maintained SSE register */
      float dxVal = _mm_cvtss_f32(vDxBase);
      double r2 = (double)(dxVal * dxVal + dyF * dyF);

      vDxBase = _mm_add_ps(vDxBase, vOne);

      if (r2 >= W * W + 0.6) {
        continue;
      }
      vl_sift_pix gx = 0.5f * (pSrc[+xo] - pSrc[-xo]);
      vl_sift_pix gy = 0.5f * (pSrc[+yo] - pSrc[-yo]);
      double mod = sqrt((double)(gx * gx + gy * gy));
      double ang = vl_mod_2pi_f(vl_fast_atan2_f(gy, gx) + (float)(2 * VL_PI));
      double wgt = fast_expn(f, r2 / (2 * sigmaw * sigmaw));
      double fbin = nbins * ang / (2 * VL_PI);
      int bin = (int)vl_floor_d(fbin - 0.5);
      double rbin = fbin - bin - 0.5;
      hist[(bin + nbins) % nbins] += (1 - rbin) * mod * wgt;
      hist[(bin + 1) % nbins] += rbin * mod * wgt;
    }
  } /* for ys */

  /* smooth histogram */
  for (iter = 0; iter < 6; iter++) {
    double prev = hist[nbins - 1];
    double first = hist[0];
    int i;
    for (i = 0; i < nbins - 1; i++) {
      double newh = (prev + hist[i] + hist[(i + 1) % nbins]) / 3.0;
      prev = hist[i];
      hist[i] = newh;
    }
    hist[i] = (prev + hist[i] + first) / 3.0;
  }

  /* find the histogram maximum */
  maxh = 0;
  for (i = 0; i < nbins; ++i)
    maxh = VL_MAX(maxh, hist[i]);

  /* find peaks within 80% from max */
  nangles = 0;
  for (i = 0; i < nbins; ++i) {
    double h0 = hist[i];
    double hm = hist[(i - 1 + nbins) % nbins];
    double hp = hist[(i + 1 + nbins) % nbins];

    /* is this a peak? */
    if (h0 > 0.8 * maxh && h0 > hm && h0 > hp) {
      /* quadratic interpolation */
      double di = -0.5 * (hp - hm) / (hp + hm - 2 * h0);
      double th = 2 * VL_PI * (i + di + 0.5) / nbins;
      angles[nangles++] = th;
      if (nangles == 4)
        goto enough_angles;
    }
  }
enough_angles:
  return nangles;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Normalizes in norm L_2 a descriptor
 ** @param begin begin of histogram.
 ** @param end   end of histogram.
 **/

VL_INLINE vl_sift_pix
normalize_histogram
(vl_sift_pix *begin, vl_sift_pix *end)
{
  vl_sift_pix* iter;
  vl_sift_pix  norm = 0.0;

  for (iter = begin; iter != end; ++ iter)
    norm += (*iter) * (*iter);

  norm = vl_fast_sqrt_f (norm) + VL_EPSILON_F;

  for (iter = begin; iter != end; ++ iter)
    *iter /= norm;

  return norm;
}

/** ------------------------------------------------------------------
 ** @brief Run the SIFT descriptor on raw data
 **
 ** @param f        SIFT filter.
 ** @param grad     image gradients.
 ** @param descr    SIFT descriptor (output).
 ** @param width    image width.
 ** @param height   image height.
 ** @param x        keypoint x coordinate.
 ** @param y        keypoint y coordinate.
 ** @param sigma    keypoint scale.
 ** @param angle0   keypoint orientation.
 **
 ** The function runs the SIFT descriptor on raw data. Here @a image
 ** is a 2 x @a width x @a height array (by convention, the memory
 ** layout is a s such the first index is the fastest varying
 ** one). The first @a width x @a height layer of the array contains
 ** the gradient magnitude and the second the gradient angle (in
 ** radians, between 0 and @f$ 2\pi @f$). @a x, @a y and @a sigma give
 ** the keypoint center and scale respectively.
 **
 ** In order to be equivalent to a standard SIFT descriptor the image
 ** gradient must be computed at a smoothing level equal to the scale
 ** of the keypoint. In practice, the actual SIFT algorithm makes the
 ** following additional approximation, which influence the result:
 **
 ** - Scale is discretized in @c S levels.
 ** - The image is downsampled once for each octave (if you do this,
 **   the parameters @a x, @a y and @a sigma must be
 **   scaled too).
 **/

VL_EXPORT
void
vl_sift_calc_raw_descriptor (VlSiftFilt const *f,
                             vl_sift_pix const* grad,
                             vl_sift_pix *descr,
                             int width, int height,
                             double x, double y,
                             double sigma,
                             double angle0)
{
#if 0 // JPB WIP BUG Unused
  double const magnif = f-> magnif;

  int          w      = width;
  int          h      = height;
  int const    xo     = 2;         /* x-stride */
  int const    yo     = 2 * w;     /* y-stride */

  int          xi     = (int) (x + 0.5);
  int          yi     = (int) (y + 0.5);

  double const st0    = sin (angle0);
  double const ct0    = cos (angle0);
  double const SBP    = magnif * sigma + VL_EPSILON_D;
  int    const W      = floor
    (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5);

  int const binto = 1;          /* bin theta-stride */
  int const binyo = NBO * NBP;  /* bin y-stride */
  int const binxo = NBO;        /* bin x-stride */

  int bin, dxi, dyi;
  vl_sift_pix const *pt;
  vl_sift_pix       *dpt;

  /* check bounds */
  if(xi    <  0               ||
     xi    >= w               ||
     yi    <  0               ||
     yi    >= h -    1        )
    return;

  /* clear descriptor */
  memset (descr, 0, sizeof(vl_sift_pix) * NBO*NBP*NBP);

  /* Center the scale space and the descriptor on the current keypoint.
   * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
   */
  pt  = grad + xi*xo + yi*yo;
  dpt = descr + (NBP/2) * binyo + (NBP/2) * binxo;

#undef atd
#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

  /*
   * Process pixels in the intersection of the image rectangle
   * (1,1)-(M-1,N-1) and the keypoint bounding box.
   */
  for(dyi =  VL_MAX(- W,   - yi   );
      dyi <= VL_MIN(+ W, h - yi -1); ++ dyi) {

    for(dxi =  VL_MAX(- W,   - xi   );
        dxi <= VL_MIN(+ W, w - xi -1); ++ dxi) {

      /* retrieve */
      vl_sift_pix mod   = *( pt + dxi*xo + dyi*yo + 0 );
      vl_sift_pix angle = *( pt + dxi*xo + dyi*yo + 1 );
      vl_sift_pix theta = vl_mod_2pi_f (angle - angle0);

      /* fractional displacement */
      vl_sift_pix dx = xi + dxi - x;
      vl_sift_pix dy = yi + dyi - y;

      /* get the displacement normalized w.r.t. the keypoint
         orientation and extension */
      vl_sift_pix nx = ( ct0 * dx + st0 * dy) / SBP;
      vl_sift_pix ny = (-st0 * dx + ct0 * dy) / SBP;
      vl_sift_pix nt = NBO * theta / (2 * VL_PI);

      /* Get the Gaussian weight of the sample. The Gaussian window
       * has a standard deviation equal to NBP/2. Note that dx and dy
       * are in the normalized frame, so that -NBP/2 <= dx <=
       * NBP/2. */
      vl_sift_pix const wsigma = f->windowSize;

      vl_sift_pix win = fast_expn
        (f, (nx*nx + ny*ny)/(2.0 * wsigma * wsigma));

      /* The sample will be distributed in 8 adjacent bins.
         We start from the ``lower-left'' bin. */
      int         binx = (int)vl_floor_f (nx - 0.5);
      int         biny = (int)vl_floor_f (ny - 0.5);
      int         bint = (int)vl_floor_f (nt);
      vl_sift_pix rbinx = nx - (binx + 0.5);
      vl_sift_pix rbiny = ny - (biny + 0.5);
      vl_sift_pix rbint = nt - bint;
      int         dbinx;
      int         dbiny;
      int         dbint;

      /* Distribute the current sample into the 8 adjacent bins*/
      for(dbinx = 0; dbinx < 2; ++dbinx) {
        for(dbiny = 0; dbiny < 2; ++dbiny) {
          for(dbint = 0; dbint < 2; ++dbint) {

            if (binx + dbinx >= - (NBP/2) &&
                binx + dbinx <    (NBP/2) &&
                biny + dbiny >= - (NBP/2) &&
                biny + dbiny <    (NBP/2) ) {
              vl_sift_pix weight = win
                * mod
                * vl_abs_f (1 - dbinx - rbinx)
                * vl_abs_f (1 - dbiny - rbiny)
                * vl_abs_f (1 - dbint - rbint);

              atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight;
            }
          }
        }
      }
    }
  }

  /* Standard SIFT descriptors are normalized, truncated and normalized again */
  if(1) {

    /* normalize L2 norm */
    vl_sift_pix norm = normalize_histogram (descr, descr + NBO*NBP*NBP);

    /*
       Set the descriptor to zero if it is lower than our
       norm_threshold.  We divide by the number of samples in the
       descriptor region because the Gaussian window used in the
       calculation of the descriptor is not normalized.
     */
    int numSamples =
      (VL_MIN(W, w - xi -1) - VL_MAX(-W, - xi) + 1) *
      (VL_MIN(W, h - yi -1) - VL_MAX(-W, - yi) + 1);

    if(f-> norm_thresh && norm < f-> norm_thresh * numSamples) {
        for(bin = 0; bin < NBO*NBP*NBP; ++ bin)
            descr [bin] = 0;
    }
    else {
      /* truncate at 0.2. */
      for(bin = 0; bin < NBO*NBP*NBP; ++ bin) {
        if (descr [bin] > 0.2) descr [bin] = 0.2;
      }

      /* normalize again. */
      normalize_histogram (descr, descr + NBO*NBP*NBP);
    }
  }
#endif
}

/** ------------------------------------------------------------------
 ** @brief Compute the descriptor of a keypoint
 **
 ** @param f        SIFT filter.
 ** @param descr    SIFT descriptor (output)
 ** @param k        keypoint.
 ** @param angle0   keypoint direction.
 **
 ** The function computes the SIFT descriptor of the keypoint @a k of
 ** orientation @a angle0. The function fills the buffer @a descr
 ** which must be large enough to hold the descriptor.
 **
 ** The function assumes that the keypoint is on the current octave.
 ** If not, it does not do anything.
 **/

// AVX2 will not help here.
VL_EXPORT
void
vl_sift_calc_keypoint_descriptor(VlSiftFilt* __restrict f,
  vl_sift_pix* __restrict descr,
  VlSiftKeypoint const* __restrict k,
  double angle0)
{
  /*
     The SIFT descriptor is a three dimensional histogram of the
     position and orientation of the gradient.  There are NBP bins for
     each spatial dimension and NBO bins for the orientation dimension,
     for a total of NBP x NBP x NBO bins.

     The support of each spatial bin has an extension of SBP = 3sigma
     pixels, where sigma is the scale of the keypoint.  Thus all the
     bins together have a support SBP x NBP pixels wide. Since
     weighting and interpolation of pixel is used, the support extends
     by another half bin. Therefore, the support is a square window of
     SBP x (NBP + 1) pixels. Finally, since the patch can be
     arbitrarily rotated, we need to consider a window 2W += sqrt(2) x
     SBP x (NBP + 1) pixels wide.
  */
  double const magnif = f->magnif;

  double       xper = pow(2.0, f->o_cur);

  int          w = f->octave_width;
  int          h = f->octave_height;
  int const    xo = 1;         /* x-stride in octave */
  int const    yo = w;         /* y-stride in octave */
  double       x = k->x / xper;
  double       y = k->y / xper;
  double       sigma = k->sigma / xper;

  int          xi = (int)(x + 0.5);
  int          yi = (int)(y + 0.5);
  int          si = k->is;

  float  const st0 = (float)sin(angle0);
  float  const ct0 = (float)cos(angle0);
  double const SBP = magnif * sigma + VL_EPSILON_D;
  float  const invSBP = (float)(1.0 / SBP);
  int    const W = (int)floor
  (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5);

  float  const wsigma = (float)f->windowSize;
  float  const negInvSigma2 = -1.0f / (2.0f * wsigma * wsigma);
  float  const ntFactor = (float)(NBO / (2.0 * VL_PI));

  /* SSE2 constants */
  __m128 const vCt0 = _mm_set1_ps(ct0);
  __m128 const vSt0 = _mm_set1_ps(st0);
  __m128 const vNegSt0 = _mm_set1_ps(-st0);
  __m128 const vInvSBP = _mm_set1_ps(invSBP);
  __m128 const vNegInvSig2 = _mm_set1_ps(negInvSigma2);
  __m128 const vNtFactor = _mm_set1_ps(ntFactor);
  __m128 const vZero = _mm_setzero_ps();
  __m128 const vHalf = _mm_set1_ps(0.5f);
  __m128 const vOne = _mm_set1_ps(1.0f);
  __m128 const vFour = _mm_set1_ps(4.0f);
  __m128 const vEight = _mm_set1_ps(8.0f);
  __m128 const vXf = _mm_set1_ps((float)x);
  __m128 const vStep = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);

  /* atan2 constants */
  __m128 const vC3 = _mm_set1_ps(0.1821f);
  __m128 const vC1 = _mm_set1_ps(0.9675f);
  __m128 const vHighBit = _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000));
  __m128 const vAbsMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  __m128 const vTwoPiA = _mm_set1_ps((float)(2.0 * VL_PI));
  __m128 const vEpsA = _mm_set1_ps(1.19209290E-07F);
  __m128 const vPi4 = _mm_set1_ps((float)(VL_PI / 4.0));
  __m128 const v3Pi4 = _mm_set1_ps((float)(3.0 * VL_PI / 4.0));

  /* FastExp constants */
  __m128 const vExpL2e = _mm_set1_ps(1.442695041f);
  __m128 const vExpC0 = _mm_set1_ps(0.3371894346f);
  __m128 const vExpC1 = _mm_set1_ps(0.657636276f);
  __m128 const vExpC2 = _mm_set1_ps(1.00172476f);

  /* Bias-trick floor constants */
  __m128 const vBiasXY = _mm_set1_ps(255.5f);
  __m128 const vBiasT = _mm_set1_ps(256.0f);
  __m128i const vBiasI = _mm_set1_epi32(256);

  int bin, dxi, dyi;
  vl_sift_pix const* src;

  /* check bounds */
  if (k->o != f->o_cur ||
    xi < 0 ||
    xi >= w ||
    yi < 0 ||
    yi >= h - 1 ||
    si < f->s_min + 1 ||
    si > f->s_max - 2)
    return;

  /* Allocate descriptor with 1-float wraparound per spatial cell.
   * Layout: 9 floats per cell instead of 8.  bin[8] aliases bin[0].
   * This eliminates the & 7 mask and the split store when _t0=7. */
  float descrPad[NBP * NBP * (NBO + 1)]; /* 4*4*9 = 144 floats */
  memset(descrPad, 0, sizeof(descrPad));

  /* Point directly at the octave plane for this scale level */
  src = vl_sift_get_octave(f, si);

  for (dyi = VL_MAX(-W, 1 - yi);
    dyi <= VL_MIN(+W, h - yi - 2); ++dyi) {

    /* Precompute row-constant terms */
    float const dy = (float)(yi + dyi) - (float)y;
    __m128 const vSt0_dy = _mm_set1_ps(st0 * dy);
    __m128 const vCt0_dy = _mm_set1_ps(ct0 * dy);

    int const dxiMin = VL_MAX(-W, 1 - xi);
    int const dxiMax = VL_MIN(+W, w - xi - 2);
    int const count = dxiMax - dxiMin + 1;

    /* Row pointer into octave for this dy */
    vl_sift_pix const* rowSrc = src + (yi + dyi) * yo + (xi + dxiMin);

    /* ---- SSE41 path: 4 pixels at a time ---- */
    dxi = dxiMin;
    {
      /* Base dx values for the first 4 pixels */
      __m128 vDxBase = _mm_add_ps(
        _mm_set1_ps((float)(xi + dxi)),
        vStep);
      vDxBase = _mm_sub_ps(vDxBase, vXf);

      int remaining = count;
      vl_sift_pix const* pSrc = rowSrc;

      // Doing two groups of 4 at once improves the function by about 5%
      while (remaining >= 8) {
        /* ============ GROUP A: pixels 0�3 ============ */
        __m128 vLeftA = _mm_loadu_ps(pSrc - 1);
        __m128 vRightA = _mm_loadu_ps(pSrc + 1);
        __m128 vGxA = _mm_mul_ps(vHalf, _mm_sub_ps(vRightA, vLeftA));
        __m128 vUpA = _mm_loadu_ps(pSrc - yo);
        __m128 vDownA = _mm_loadu_ps(pSrc + yo);
        __m128 vGyA = _mm_mul_ps(vHalf, _mm_sub_ps(vDownA, vUpA));

        __m128 vRgxA = _mm_add_ps(_mm_mul_ps(vCt0, vGxA), _mm_mul_ps(vSt0, vGyA));
        __m128 vRgyA = _mm_add_ps(_mm_mul_ps(vNegSt0, vGxA), _mm_mul_ps(vCt0, vGyA));

        __m128 vModSqA = _mm_add_ps(_mm_mul_ps(vRgxA, vRgxA), _mm_mul_ps(vRgyA, vRgyA));
        __m128 vModA = _mm_sqrt_ps(vModSqA); /* 11-cyc latency � fill below */

        /* ============ GROUP B: pixels 4�7 (loads during A's sqrt) ============ */
        __m128 vLeftB = _mm_loadu_ps(pSrc + 4 - 1);
        __m128 vRightB = _mm_loadu_ps(pSrc + 4 + 1);
        __m128 vGxB = _mm_mul_ps(vHalf, _mm_sub_ps(vRightB, vLeftB));
        __m128 vUpB = _mm_loadu_ps(pSrc + 4 - yo);
        __m128 vDownB = _mm_loadu_ps(pSrc + 4 + yo);
        __m128 vGyB = _mm_mul_ps(vHalf, _mm_sub_ps(vDownB, vUpB));

        __m128 vRgxB = _mm_add_ps(_mm_mul_ps(vCt0, vGxB), _mm_mul_ps(vSt0, vGyB));
        __m128 vRgyB = _mm_add_ps(_mm_mul_ps(vNegSt0, vGxB), _mm_mul_ps(vCt0, vGyB));

        __m128 vModSqB = _mm_add_ps(_mm_mul_ps(vRgxB, vRgxB), _mm_mul_ps(vRgyB, vRgyB));
        __m128 vModB = _mm_sqrt_ps(vModSqB); /* 11-cyc latency � fill below */

        /* ============ GROUP A: spatial + atan2 (fills A's sqrt window) ============ */
        __m128 vNxA = _mm_mul_ps(
          _mm_add_ps(_mm_mul_ps(vCt0, vDxBase), vSt0_dy), vInvSBP);
        __m128 vNyA = _mm_mul_ps(
          _mm_add_ps(_mm_mul_ps(vNegSt0, vDxBase), vCt0_dy), vInvSBP);
        __m128 vR2A = _mm_add_ps(_mm_mul_ps(vNxA, vNxA), _mm_mul_ps(vNyA, vNyA));

        /* FastExp inlined for A: exp(-r2 * invSig2) */
        __m128 vExpArgA = _mm_mul_ps(vR2A, vNegInvSig2); /* negative arg ? exp(-r�/2s�) */
        __m128 vT_A = _mm_mul_ps(vExpArgA, vExpL2e);
        __m128i vI_A = _mm_cvttps_epi32(vT_A);
        __m128i vJ_A = _mm_srli_epi32(_mm_castps_si128(vExpArgA), 31);
        vI_A = _mm_sub_epi32(vI_A, vJ_A);
        __m128 vE_A = _mm_cvtepi32_ps(vI_A);
        __m128 vF_A = _mm_sub_ps(vT_A, vE_A);
        __m128 vP_A = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(vExpC0, vF_A), vExpC1), vF_A), vExpC2);
        __m128 vWinA = _mm_castsi128_ps(_mm_add_epi32(_mm_slli_epi32(vI_A, 23), _mm_castps_si128(vP_A)));

        /* atan2 for A � no Newton refinement on rcp */
        __m128 vAbsYA = _mm_add_ps(_mm_and_ps(vRgyA, vAbsMask), vEpsA);
        __m128 vAbsXA = _mm_and_ps(vRgxA, vAbsMask);
        __m128 vNumA = _mm_sub_ps(vRgxA, _mm_or_ps(vAbsYA, _mm_and_ps(vRgxA, vHighBit)));
        __m128 vDenA = _mm_add_ps(vAbsYA, vAbsXA);
        __m128 vR_atA = _mm_mul_ps(vNumA, _mm_rcp_ps(vDenA));
        __m128 vXge0A = _mm_cmpge_ps(vRgxA, vZero);
        __m128 vBaseA = _mm_or_ps(_mm_and_ps(vXge0A, vPi4), _mm_andnot_ps(vXge0A, v3Pi4));
        __m128 vRRA = _mm_mul_ps(vR_atA, vR_atA);
        __m128 vAngleA = _mm_add_ps(vBaseA,
          _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(vC3, vRRA), vC1), vR_atA));
        __m128 vAtan2A = _mm_xor_ps(vAngleA, _mm_and_ps(vRgyA, vHighBit));
        __m128 vShiftedA = _mm_add_ps(vAtan2A, vTwoPiA);
        __m128 vThetaA = _mm_sub_ps(vShiftedA,
          _mm_and_ps(_mm_cmpge_ps(vShiftedA, vTwoPiA), vTwoPiA));

        /* A: finalize (vModA ready by now) */
        __m128 vNtA = _mm_mul_ps(vThetaA, vNtFactor);
        __m128 vWmodA = _mm_mul_ps(vWinA, vModA);

        /* ============ GROUP B: spatial + atan2 (fills B's sqrt window) ============ */
        __m128 vDxBaseB = _mm_add_ps(vDxBase, vFour);
        __m128 vNxB = _mm_mul_ps(
          _mm_add_ps(_mm_mul_ps(vCt0, vDxBaseB), vSt0_dy), vInvSBP);
        __m128 vNyB = _mm_mul_ps(
          _mm_add_ps(_mm_mul_ps(vNegSt0, vDxBaseB), vCt0_dy), vInvSBP);
        __m128 vR2B = _mm_add_ps(_mm_mul_ps(vNxB, vNxB), _mm_mul_ps(vNyB, vNyB));

        /* FastExp inlined for B: exp(-r2 * invSig2) */
        __m128 vExpArgB = _mm_mul_ps(vR2B, vNegInvSig2);
        __m128 vT_B = _mm_mul_ps(vExpArgB, vExpL2e);
        __m128i vI_B = _mm_cvttps_epi32(vT_B);
        __m128i vJ_B = _mm_srli_epi32(_mm_castps_si128(vExpArgB), 31);
        vI_B = _mm_sub_epi32(vI_B, vJ_B);
        __m128 vE_B = _mm_cvtepi32_ps(vI_B);
        __m128 vF_B = _mm_sub_ps(vT_B, vE_B);
        __m128 vP_B = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(vExpC0, vF_B), vExpC1), vF_B), vExpC2);
        __m128 vWinB = _mm_castsi128_ps(_mm_add_epi32(_mm_slli_epi32(vI_B, 23), _mm_castps_si128(vP_B)));

        /* atan2 for B � no Newton refinement on rcp */
        __m128 vAbsYB = _mm_add_ps(_mm_and_ps(vRgyB, vAbsMask), vEpsA);
        __m128 vAbsXB = _mm_and_ps(vRgxB, vAbsMask);
        __m128 vNumB = _mm_sub_ps(vRgxB, _mm_or_ps(vAbsYB, _mm_and_ps(vRgxB, vHighBit)));
        __m128 vDenB = _mm_add_ps(vAbsYB, vAbsXB);
        __m128 vR_atB = _mm_mul_ps(vNumB, _mm_rcp_ps(vDenB));
        __m128 vXge0B = _mm_cmpge_ps(vRgxB, vZero);
        __m128 vBaseB = _mm_or_ps(_mm_and_ps(vXge0B, vPi4), _mm_andnot_ps(vXge0B, v3Pi4));
        __m128 vRRB = _mm_mul_ps(vR_atB, vR_atB);
        __m128 vAngleB = _mm_add_ps(vBaseB,
          _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(vC3, vRRB), vC1), vR_atB));
        __m128 vAtan2B = _mm_xor_ps(vAngleB, _mm_and_ps(vRgyB, vHighBit));
        __m128 vShiftedB = _mm_add_ps(vAtan2B, vTwoPiA);
        __m128 vThetaB = _mm_sub_ps(vShiftedB,
          _mm_and_ps(_mm_cmpge_ps(vShiftedB, vTwoPiA), vTwoPiA));

        __m128 vNtB = _mm_mul_ps(vThetaB, vNtFactor);
        __m128 vWmodB = _mm_mul_ps(vWinB, vModB);

#define DESCR_STRIDE (NBO + 1)  /* 9 floats per orientation cell */

#define EXTRACT_F(vec, lane) \
  ((lane)==0 ? _mm_cvtss_f32(vec) : \
   _mm_cvtss_f32(_mm_shuffle_ps((vec),(vec),_MM_SHUFFLE((lane),(lane),(lane),(lane)))))

#define EXTRACT_I(vec, lane) \
  ((lane)==0 ? _mm_cvtsi128_si32(vec) : \
   _mm_cvtsi128_si32(_mm_shuffle_epi32((vec),_MM_SHUFFLE((lane),(lane),(lane),(lane)))))

#define SCATTER_PIXEL(lane) {                                                  \
            float const _wt0 = wt0Arr[lane];                                  \
            float const _wt1 = wt1Arr[lane];                                  \
            int   const _bxi = bxiArr[lane];                                   \
            int   const _byi = byiArr[lane];                                   \
            int   const _bti = btiArr[lane];                                   \
            int   const _t0  = _bti & 7;                                      \
            int   const _t1  = _t0 + 1;                                       \
            int   const _x0  = _bxi + (NBP / 2);                              \
            int   const _x1  = _x0 + 1;                                       \
            int   const _y0  = _byi + (NBP / 2);                              \
            int   const _y1  = _y0 + 1;                                       \
            float const _wx0 = wx0Arr[lane];                                   \
            float const _wx1 = wx1Arr[lane];                                   \
            float const _wy0 = wy0Arr[lane];                                   \
            float const _wy1 = wy1Arr[lane];                                   \
                                                                               \
            if ((unsigned)_x0 <= 2u && (unsigned)_y0 <= 2u) {                 \
              float* _h00 = descrPad + (_y0 * NBP + _x0) * DESCR_STRIDE;      \
              float* _h10 = _h00 + DESCR_STRIDE;                              \
              float* _h01 = _h00 + NBP * DESCR_STRIDE;                        \
              float* _h11 = _h01 + DESCR_STRIDE;                              \
              float _w00 = _wx0 * _wy0;                                       \
              float _w01 = _wx0 * _wy1;                                       \
              float _w10 = _wx1 * _wy0;                                       \
              float _w11 = _wx1 * _wy1;                                       \
              _h00[_t0] += _w00 * _wt0;  _h00[_t1] += _w00 * _wt1;           \
              _h01[_t0] += _w01 * _wt0;  _h01[_t1] += _w01 * _wt1;           \
              _h10[_t0] += _w10 * _wt0;  _h10[_t1] += _w10 * _wt1;           \
              _h11[_t0] += _w11 * _wt0;  _h11[_t1] += _w11 * _wt1;           \
            } else {                                                           \
              if ((unsigned)_y0 < (unsigned)NBP) {                             \
                if ((unsigned)_x0 < (unsigned)NBP) {                           \
                  float _ww = _wx0 * _wy0;                                    \
                  float* _hh = descrPad + (_y0 * NBP + _x0) * DESCR_STRIDE;   \
                  _hh[_t0] += _ww * _wt0;  _hh[_t1] += _ww * _wt1;           \
                }                                                              \
                if ((unsigned)_x1 < (unsigned)NBP) {                           \
                  float _ww = _wx1 * _wy0;                                    \
                  float* _hh = descrPad + (_y0 * NBP + _x1) * DESCR_STRIDE;   \
                  _hh[_t0] += _ww * _wt0;  _hh[_t1] += _ww * _wt1;           \
                }                                                              \
              }                                                                \
              if ((unsigned)_y1 < (unsigned)NBP) {                             \
                if ((unsigned)_x0 < (unsigned)NBP) {                           \
                  float _ww = _wx0 * _wy1;                                    \
                  float* _hh = descrPad + (_y1 * NBP + _x0) * DESCR_STRIDE;   \
                  _hh[_t0] += _ww * _wt0;  _hh[_t1] += _ww * _wt1;           \
                }                                                              \
                if ((unsigned)_x1 < (unsigned)NBP) {                           \
                  float _ww = _wx1 * _wy1;                                    \
                  float* _hh = descrPad + (_y1 * NBP + _x1) * DESCR_STRIDE;   \
                  _hh[_t0] += _ww * _wt0;  _hh[_t1] += _ww * _wt1;           \
                }                                                              \
              }                                                                \
            }                                                                  \
          }

        /* ============ GROUP A: bin + scatter ============ */
        {
          __m128i vBinxi = _mm_sub_epi32(_mm_cvttps_epi32(_mm_add_ps(vNxA, vBiasXY)), vBiasI);
          __m128i vBinyi = _mm_sub_epi32(_mm_cvttps_epi32(_mm_add_ps(vNyA, vBiasXY)), vBiasI);
          __m128i vBinti = _mm_sub_epi32(_mm_cvttps_epi32(_mm_add_ps(vNtA, vBiasT)), vBiasI);
          __m128 vBinxf = _mm_cvtepi32_ps(vBinxi);
          __m128 vBinyf = _mm_cvtepi32_ps(vBinyi);
          __m128 vBintf = _mm_cvtepi32_ps(vBinti);

          __m128 vRbinx = _mm_sub_ps(vNxA, _mm_add_ps(vBinxf, vHalf));
          __m128 vRbiny = _mm_sub_ps(vNyA, _mm_add_ps(vBinyf, vHalf));
          __m128 vRbint = _mm_sub_ps(vNtA, vBintf);
          __m128 vWx0 = _mm_sub_ps(vOne, vRbinx); __m128 vWx1 = vRbinx;
          __m128 vWy0 = _mm_sub_ps(vOne, vRbiny); __m128 vWy1 = vRbiny;
          __m128 vWt0 = _mm_mul_ps(vWmodA, _mm_sub_ps(vOne, vRbint));
          __m128 vWt1 = _mm_mul_ps(vWmodA, vRbint);

          /* Spill to aligned stack arrays � avoids 24 shuffle instructions */
          __declspec(align(16)) float wx0Arr[4], wx1Arr[4], wy0Arr[4], wy1Arr[4];
          __declspec(align(16)) float wt0Arr[4], wt1Arr[4];
          __declspec(align(16)) int   bxiArr[4], byiArr[4], btiArr[4];
          _mm_store_ps(wx0Arr, vWx0);  _mm_store_ps(wx1Arr, vWx1);
          _mm_store_ps(wy0Arr, vWy0);  _mm_store_ps(wy1Arr, vWy1);
          _mm_store_ps(wt0Arr, vWt0);  _mm_store_ps(wt1Arr, vWt1);
          _mm_store_si128((__m128i*)bxiArr, vBinxi);
          _mm_store_si128((__m128i*)byiArr, vBinyi);
          _mm_store_si128((__m128i*)btiArr, vBinti);

          SCATTER_PIXEL(0) SCATTER_PIXEL(1) SCATTER_PIXEL(2) SCATTER_PIXEL(3)
        }

        /* ============ GROUP B: bin + scatter ============ */
        {
          __m128i vBinxi = _mm_sub_epi32(_mm_cvttps_epi32(_mm_add_ps(vNxB, vBiasXY)), vBiasI);
          __m128i vBinyi = _mm_sub_epi32(_mm_cvttps_epi32(_mm_add_ps(vNyB, vBiasXY)), vBiasI);
          __m128i vBinti = _mm_sub_epi32(_mm_cvttps_epi32(_mm_add_ps(vNtB, vBiasT)), vBiasI);
          __m128 vBinxf = _mm_cvtepi32_ps(vBinxi);
          __m128 vBinyf = _mm_cvtepi32_ps(vBinyi);
          __m128 vBintf = _mm_cvtepi32_ps(vBinti);

          __m128 vRbinx = _mm_sub_ps(vNxB, _mm_add_ps(vBinxf, vHalf));
          __m128 vRbiny = _mm_sub_ps(vNyB, _mm_add_ps(vBinyf, vHalf));
          __m128 vRbint = _mm_sub_ps(vNtB, vBintf);
          __m128 vWx0 = _mm_sub_ps(vOne, vRbinx); __m128 vWx1 = vRbinx;
          __m128 vWy0 = _mm_sub_ps(vOne, vRbiny); __m128 vWy1 = vRbiny;
          __m128 vWt0 = _mm_mul_ps(vWmodB, _mm_sub_ps(vOne, vRbint));
          __m128 vWt1 = _mm_mul_ps(vWmodB, vRbint);

          __declspec(align(16)) float wx0Arr[4], wx1Arr[4], wy0Arr[4], wy1Arr[4];
          __declspec(align(16)) float wt0Arr[4], wt1Arr[4];
          __declspec(align(16)) int   bxiArr[4], byiArr[4], btiArr[4];
          _mm_store_ps(wx0Arr, vWx0);  _mm_store_ps(wx1Arr, vWx1);
          _mm_store_ps(wy0Arr, vWy0);  _mm_store_ps(wy1Arr, vWy1);
          _mm_store_ps(wt0Arr, vWt0);  _mm_store_ps(wt1Arr, vWt1);
          _mm_store_si128((__m128i*)bxiArr, vBinxi);
          _mm_store_si128((__m128i*)byiArr, vBinyi);
          _mm_store_si128((__m128i*)btiArr, vBinti);

          SCATTER_PIXEL(0) SCATTER_PIXEL(1) SCATTER_PIXEL(2) SCATTER_PIXEL(3)
        }

#undef SCATTER_PIXEL
#undef EXTRACT_F
#undef EXTRACT_I

        pSrc += 8;
        vDxBase = _mm_add_ps(vDxBase, vEight);
        remaining -= 8;
        dxi += 8;
      } /* while remaining >= 8 */

      /* ---- Masked SSE tail for remaining 1-7 pixels ---- */
      while (remaining > 0) {
        __m128 vLeft = _mm_loadu_ps(pSrc - 1);
        __m128 vRight = _mm_loadu_ps(pSrc + 1);
        __m128 vGx = _mm_mul_ps(vHalf, _mm_sub_ps(vRight, vLeft));

        __m128 vUpR = _mm_loadu_ps(pSrc - yo);
        __m128 vDownR = _mm_loadu_ps(pSrc + yo);
        __m128 vGy = _mm_mul_ps(vHalf, _mm_sub_ps(vDownR, vUpR));

        __m128 vRgx = _mm_add_ps(_mm_mul_ps(vCt0, vGx), _mm_mul_ps(vSt0, vGy));
        __m128 vRgy = _mm_add_ps(_mm_mul_ps(vNegSt0, vGx), _mm_mul_ps(vCt0, vGy));

        __m128 vModSq = _mm_add_ps(_mm_mul_ps(vRgx, vRgx), _mm_mul_ps(vRgy, vRgy));
        __m128 vMod = _mm_sqrt_ps(vModSq);

        /* atan2 inlined � no Newton refinement on rcp (matches main loop) */
        __m128 vAbsY = _mm_add_ps(_mm_and_ps(vRgy, vAbsMask), vEpsA);
        __m128 vAbsX = _mm_and_ps(vRgx, vAbsMask);
        __m128 vNum = _mm_sub_ps(vRgx, _mm_or_ps(vAbsY, _mm_and_ps(vRgx, vHighBit)));
        __m128 vDen = _mm_add_ps(vAbsY, vAbsX);
        __m128 vR_at = _mm_mul_ps(vNum, _mm_rcp_ps(vDen));
        __m128 vXge0 = _mm_cmpge_ps(vRgx, vZero);
        __m128 vBase = _mm_or_ps(_mm_and_ps(vXge0, vPi4), _mm_andnot_ps(vXge0, v3Pi4));
        __m128 vRR = _mm_mul_ps(vR_at, vR_at);
        __m128 vAngle = _mm_add_ps(vBase,
          _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(vC3, vRR), vC1), vR_at));
        __m128 vAtan2 = _mm_xor_ps(vAngle, _mm_and_ps(vRgy, vHighBit));
        __m128 vShifted = _mm_add_ps(vAtan2, vTwoPiA);
        __m128 vTheta = _mm_sub_ps(vShifted,
          _mm_and_ps(_mm_cmpge_ps(vShifted, vTwoPiA), vTwoPiA));

        __m128 vNt = _mm_mul_ps(vTheta, vNtFactor);

        __m128 vNx = _mm_mul_ps(
          _mm_add_ps(_mm_mul_ps(vCt0, vDxBase), vSt0_dy), vInvSBP);
        __m128 vNy = _mm_mul_ps(
          _mm_add_ps(_mm_mul_ps(vNegSt0, vDxBase), vCt0_dy), vInvSBP);

        __m128 vR2 = _mm_add_ps(_mm_mul_ps(vNx, vNx), _mm_mul_ps(vNy, vNy));

        /* FastExp inlined */
        __m128 vExpArg = _mm_mul_ps(vR2, vNegInvSig2);
        __m128 vT = _mm_mul_ps(vExpArg, vExpL2e);
        __m128i vI = _mm_cvttps_epi32(vT);
        __m128i vJ = _mm_srli_epi32(_mm_castps_si128(vExpArg), 31);
        vI = _mm_sub_epi32(vI, vJ);
        __m128 vE = _mm_cvtepi32_ps(vI);
        __m128 vF = _mm_sub_ps(vT, vE);
        __m128 vP = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(vExpC0, vF), vExpC1), vF), vExpC2);
        __m128 vWin = _mm_castsi128_ps(_mm_add_epi32(_mm_slli_epi32(vI, 23), _mm_castps_si128(vP)));

        __m128 vWmod = _mm_mul_ps(vWin, vMod);

        /* Bias-trick floor */
        __m128i vBinxi = _mm_sub_epi32(_mm_cvttps_epi32(_mm_add_ps(vNx, vBiasXY)), vBiasI);
        __m128i vBinyi = _mm_sub_epi32(_mm_cvttps_epi32(_mm_add_ps(vNy, vBiasXY)), vBiasI);
        __m128i vBinti = _mm_sub_epi32(_mm_cvttps_epi32(_mm_add_ps(vNt, vBiasT)), vBiasI);
        __m128 vBinxf = _mm_cvtepi32_ps(vBinxi);
        __m128 vBinyf = _mm_cvtepi32_ps(vBinyi);
        __m128 vBintf = _mm_cvtepi32_ps(vBinti);

        __m128 vRbinx = _mm_sub_ps(vNx, _mm_add_ps(vBinxf, vHalf));
        __m128 vRbiny = _mm_sub_ps(vNy, _mm_add_ps(vBinyf, vHalf));
        __m128 vRbint = _mm_sub_ps(vNt, vBintf);

        __m128 vWx0 = _mm_sub_ps(vOne, vRbinx); __m128 vWx1 = vRbinx;
        __m128 vWy0 = _mm_sub_ps(vOne, vRbiny); __m128 vWy1 = vRbiny;
        __m128 vWt0 = _mm_mul_ps(vWmod, _mm_sub_ps(vOne, vRbint));
        __m128 vWt1 = _mm_mul_ps(vWmod, vRbint);

        /* Spill to aligned stack arrays � matches main loop approach */
        __declspec(align(16)) float wx0Arr[4], wx1Arr[4], wy0Arr[4], wy1Arr[4];
        __declspec(align(16)) float wt0Arr[4], wt1Arr[4];
        __declspec(align(16)) int   bxiArr[4], byiArr[4], btiArr[4];
        _mm_store_ps(wx0Arr, vWx0);  _mm_store_ps(wx1Arr, vWx1);
        _mm_store_ps(wy0Arr, vWy0);  _mm_store_ps(wy1Arr, vWy1);
        _mm_store_ps(wt0Arr, vWt0);  _mm_store_ps(wt1Arr, vWt1);
        _mm_store_si128((__m128i*)bxiArr, vBinxi);
        _mm_store_si128((__m128i*)byiArr, vBinyi);
        _mm_store_si128((__m128i*)btiArr, vBinti);

        {
          int const tailCount = (remaining < 4) ? remaining : 4;

#define SCATTER_PIXEL_TAIL(lane)                                               \
          if (tailCount > (lane)) {                                            \
            float const _wt0 = wt0Arr[lane];                                  \
            float const _wt1 = wt1Arr[lane];                                  \
            int   const _bxi = bxiArr[lane];                                   \
            int   const _byi = byiArr[lane];                                   \
            int   const _bti = btiArr[lane];                                   \
            int   const _t0  = _bti & 7;                                      \
            int   const _t1  = _t0 + 1;                                       \
            int   const _x0  = _bxi + (NBP / 2);                              \
            int   const _x1  = _x0 + 1;                                       \
            int   const _y0  = _byi + (NBP / 2);                              \
            int   const _y1  = _y0 + 1;                                       \
            float const _wx0 = wx0Arr[lane];                                   \
            float const _wx1 = wx1Arr[lane];                                   \
            float const _wy0 = wy0Arr[lane];                                   \
            float const _wy1 = wy1Arr[lane];                                   \
                                                                               \
            if ((unsigned)_x0 <= 2u && (unsigned)_y0 <= 2u) {                 \
              float* _h00 = descrPad + (_y0 * NBP + _x0) * DESCR_STRIDE;      \
              float* _h10 = _h00 + DESCR_STRIDE;                              \
              float* _h01 = _h00 + NBP * DESCR_STRIDE;                        \
              float* _h11 = _h01 + DESCR_STRIDE;                              \
              float _w00 = _wx0 * _wy0;                                       \
              float _w01 = _wx0 * _wy1;                                       \
              float _w10 = _wx1 * _wy0;                                       \
              float _w11 = _wx1 * _wy1;                                       \
              _h00[_t0] += _w00 * _wt0; _h00[_t1] += _w00 * _wt1;            \
              _h01[_t0] += _w01 * _wt0; _h01[_t1] += _w01 * _wt1;            \
              _h10[_t0] += _w10 * _wt0; _h10[_t1] += _w10 * _wt1;            \
              _h11[_t0] += _w11 * _wt0; _h11[_t1] += _w11 * _wt1;            \
            } else {                                                           \
              if ((unsigned)_y0 < (unsigned)NBP) {                             \
                if ((unsigned)_x0 < (unsigned)NBP) {                           \
                  float _ww = _wx0 * _wy0;                                    \
                  float* _hh = descrPad + (_y0 * NBP + _x0) * DESCR_STRIDE;   \
                  _hh[_t0] += _ww * _wt0; _hh[_t1] += _ww * _wt1;            \
                }                                                              \
                if ((unsigned)_x1 < (unsigned)NBP) {                           \
                  float _ww = _wx1 * _wy0;                                    \
                  float* _hh = descrPad + (_y0 * NBP + _x1) * DESCR_STRIDE;   \
                  _hh[_t0] += _ww * _wt0; _hh[_t1] += _ww * _wt1;            \
                }                                                              \
              }                                                                \
              if ((unsigned)_y1 < (unsigned)NBP) {                             \
                if ((unsigned)_x0 < (unsigned)NBP) {                           \
                  float _ww = _wx0 * _wy1;                                    \
                  float* _hh = descrPad + (_y1 * NBP + _x0) * DESCR_STRIDE;   \
                  _hh[_t0] += _ww * _wt0; _hh[_t1] += _ww * _wt1;            \
                }                                                              \
                if ((unsigned)_x1 < (unsigned)NBP) {                           \
                  float _ww = _wx1 * _wy1;                                    \
                  float* _hh = descrPad + (_y1 * NBP + _x1) * DESCR_STRIDE;   \
                  _hh[_t0] += _ww * _wt0; _hh[_t1] += _ww * _wt1;            \
                }                                                              \
              }                                                                \
            }                                                                  \
          }

          SCATTER_PIXEL_TAIL(0)
          SCATTER_PIXEL_TAIL(1)
          SCATTER_PIXEL_TAIL(2)
          SCATTER_PIXEL_TAIL(3)

#undef SCATTER_PIXEL_TAIL

          pSrc += tailCount;
          vDxBase = _mm_add_ps(vDxBase, _mm_set1_ps((float)tailCount));
          remaining -= tailCount;
        }
      } /* masked SSE tail */    }
  } /* dyi */

  /* Fold wraparound bin[8] back into bin[0], then compact to descr[] */
  {
    int c;
    for (c = 0; c < NBP * NBP; ++c) {
      float* cell = descrPad + c * DESCR_STRIDE;
      cell[0] += cell[NBO]; /* fold bin[8] ? bin[0] */
      /* Copy 8 floats to output (skip the padding slot) */
      descr[c * NBO + 0] = cell[0];
      descr[c * NBO + 1] = cell[1];
      descr[c * NBO + 2] = cell[2];
      descr[c * NBO + 3] = cell[3];
      descr[c * NBO + 4] = cell[4];
      descr[c * NBO + 5] = cell[5];
      descr[c * NBO + 6] = cell[6];
      descr[c * NBO + 7] = cell[7];
    }
  }

#undef DESCR_STRIDE

  /* Standard SIFT descriptors are normalized, truncated and normalized again */
  if (1) {

    /* Normalize the histogram to L2 unit length. */
    vl_sift_pix norm = normalize_histogram(descr, descr + NBO * NBP * NBP);

    /* Set the descriptor to zero if it is lower than our norm_threshold */
    if (f->norm_thresh && norm < f->norm_thresh) {
      for (bin = 0; bin < NBO * NBP * NBP; ++bin)
        descr[bin] = 0;
    }
    else {

      /* Truncate at 0.2. */
      for (bin = 0; bin < NBO * NBP * NBP; ++bin) {
        if (descr[bin] > 0.2) descr[bin] = 0.2;
      }

      /* Normalize again. */
      normalize_histogram(descr, descr + NBO * NBP * NBP);
    }
  }
}

/** ------------------------------------------------------------------
 ** @brief Initialize a keypoint from its position and scale
 **
 ** @param f     SIFT filter.
 ** @param k     SIFT keypoint (output).
 ** @param x     x coordinate of the keypoint center.
 ** @param y     y coordinate of the keypoint center.
 ** @param sigma keypoint scale.
 **
 ** The function initializes a keypoint structure @a k from
 ** the location @a x
 ** and @a y and the scale @a sigma of the keypoint. The keypoint structure
 ** maps the keypoint to an octave and scale level of the discretized
 ** Gaussian scale space, which is required for instance to compute the
 ** keypoint SIFT descriptor.
 **
 ** @par Algorithm
 **
 ** The formula linking the keypoint scale sigma to the octave and
 ** scale indexes is
 **
 ** @f[ \sigma(o,s) = \sigma_0 2^{o+s/S} @f]
 **
 ** In addition to the scale index @e s (which can be fractional due
 ** to scale interpolation) a keypoint has an integer scale index @e
 ** is too (which is the index of the scale level where it was
 ** detected in the DoG scale space). We have the constraints (@ref
 ** sift-tech-detector see also the "SIFT detector"):
 **
 ** - @e o is integer in the range @f$ [o_\mathrm{min},
 **   o_{\mathrm{min}}+O-1] @f$.
 ** - @e is is integer in the range @f$ [s_\mathrm{min}+1,
 **   s_\mathrm{max}-2] @f$.  This depends on how the scale is
 **   determined during detection, and must be so here because
 **   gradients are computed only for this range of scale levels
 **   and are required for the calculation of the SIFT descriptor.
 ** - @f$ |s - is| < 0.5 @f$ for detected keypoints in most cases due
 **   to the interpolation technique used during detection. However
 **   this is not necessary.
 **
 ** Thus octave o represents scales @f$ \{ \sigma(o, s) : s \in
 ** [s_\mathrm{min}+1-.5, s_\mathrm{max}-2+.5] \} @f$. Note that some
 ** scales may be represented more than once. For each scale, we
 ** select the largest possible octave that contains it, i.e.
 **
 ** @f[
 **  o(\sigma)
 **  = \max \{ o \in \mathbb{Z} :
 **    \sigma_0 2^{\frac{s_\mathrm{min}+1-.5}{S}} \leq \sigma \}
 **  = \mathrm{floor}\,\left[
 **    \log_2(\sigma / \sigma_0) - \frac{s_\mathrm{min}+1-.5}{S}\right]
 ** @f]
 **
 ** and then
 **
 ** @f[
 ** s(\sigma) = S  \left[\log_2(\sigma / \sigma_0) - o(\sigma)\right],
 ** \quad
 ** is(\sigma) = \mathrm{round}\,(s(\sigma))
 ** @f]
 **
 ** In practice, both @f$ o(\sigma) @f$ and @f$ is(\sigma) @f$ are
 ** clamped to their feasible range as determined by the SIFT filter
 ** parameters.
 **/

VL_EXPORT
void
vl_sift_keypoint_init (VlSiftFilt const *f,
                       VlSiftKeypoint *k,
                       double x,
                       double y,
                       double sigma)
{
  int    o, ix, iy, is;
  double s, phi, xper;

  phi = log2 ((sigma + VL_EPSILON_D) / f->sigma0);
  o   = (int)vl_floor_d (phi -  ((double) f->s_min + 0.5) / f->S);
  o   = VL_MIN (o, f->o_min + f->O - 1);
  o   = VL_MAX (o, f->o_min           );
  s   = f->S * (phi - o);

  is  = (int)(s + 0.5);
  is  = VL_MIN(is, f->s_max - 2);
  is  = VL_MAX(is, f->s_min + 1);

  xper = pow (2.0, o);
  ix   = (int)(x / xper + 0.5);
  iy   = (int)(y / xper + 0.5);

  k -> o  = o;

  k -> ix = ix;
  k -> iy = iy;
  k -> is = is;

  k -> x = x;
  k -> y = y;
  k -> s = s;

  k->sigma = sigma;
}
