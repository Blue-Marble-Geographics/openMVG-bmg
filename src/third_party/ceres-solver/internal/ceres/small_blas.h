// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// Simple blas functions for use in the Schur Eliminator. These are
// fairly basic implementations which already yield a significant
// speedup in the eliminator performance.

#ifndef CERES_INTERNAL_SMALL_BLAS_H_
#define CERES_INTERNAL_SMALL_BLAS_H_

#include "ceres/internal/port.h"
#include "ceres/internal/eigen.h"
#include "glog/logging.h"

// Selects the inner loop order for the four custom (non-Eigen) small
// BLAS kernels below: MatrixMatrixMultiplyNaive,
// MatrixTransposeMatrixMultiplyNaive, and the custom branches of
// MatrixVectorMultiply / MatrixTransposeVectorMultiply.
//
// When CERES_SMALL_BLAS_AXPY = 1 (the default) three additional
// optimizations are also enabled:
//   1. CERES_SMALL_BLAS_FORCE_INLINE on every kernel, eliminating call
//      frames in the inner-most Schur eliminator loops where the
//      kernels are invoked once per (chunk row, cell).
//   2. AXPY-form inner loops (described below).
//   3. MatrixTransposeMatrixMultiplySelf, a specialised C op A'A kernel
//      that computes only the upper triangle and mirrors -- halving
//      FLOPs for the diagonal-block updates in the Schur eliminator
//      (E'E in ChunkDiagonalBlockAndGradient/BackSubstitute and the
//      F'F diagonal in [No]EBlockRowOuterProduct).
//
//   CERES_SMALL_BLAS_AXPY = 1  (default): AXPY-form loops.  Reads from
//     A in unit-stride row order and broadcasts the other operand.
//     Inner loops vectorize cleanly and are typically 2-4x faster on
//     the dynamic-shape paths used by the 2_3_d Schur eliminator.
//     NOT bit-identical to the legacy loops -- the floating-point
//     accumulation order changes (k-outer instead of k-inner), so
//     the last few bits of each result may differ.
//
//   CERES_SMALL_BLAS_AXPY = 0: legacy dot-product loops, bit-identical
//     to upstream Ceres.  Define CERES_SMALL_BLAS_NO_AXPY before
//     including this header (or via -D) to force this mode.  Force
//     inlining and the symmetric self-multiply specialisation are
//     disabled in this mode as well, so the build matches upstream.
#if !defined(CERES_SMALL_BLAS_AXPY)
# if defined(CERES_SMALL_BLAS_NO_AXPY)
#  define CERES_SMALL_BLAS_AXPY 0
# else
#  define CERES_SMALL_BLAS_AXPY 1
# endif
#endif

// Force inlining attribute, conditional on the AXPY mode so the
// legacy bit-identical build stays as close to upstream as possible.
#if CERES_SMALL_BLAS_AXPY
# if defined(_MSC_VER)
#  define CERES_SMALL_BLAS_FORCE_INLINE __forceinline
# elif defined(__GNUC__) || defined(__clang__)
#  define CERES_SMALL_BLAS_FORCE_INLINE inline __attribute__((always_inline))
# else
#  define CERES_SMALL_BLAS_FORCE_INLINE inline
# endif
#else
# define CERES_SMALL_BLAS_FORCE_INLINE inline
#endif

namespace ceres {
namespace internal {

// The following three macros are used to share code and reduce
// template junk across the various GEMM variants.
#define CERES_GEMM_BEGIN(name)                                          \
  template<int kRowA, int kColA, int kRowB, int kColB, int kOperation>  \
  CERES_SMALL_BLAS_FORCE_INLINE void name(const double* A,              \
                   const int num_row_a,                                 \
                   const int num_col_a,                                 \
                   const double* B,                                     \
                   const int num_row_b,                                 \
                   const int num_col_b,                                 \
                   double* C,                                           \
                   const int start_row_c,                               \
                   const int start_col_c,                               \
                   const int row_stride_c,                              \
                   const int col_stride_c)

#define CERES_GEMM_NAIVE_HEADER                                         \
  DCHECK_GT(num_row_a, 0);                                              \
  DCHECK_GT(num_col_a, 0);                                              \
  DCHECK_GT(num_row_b, 0);                                              \
  DCHECK_GT(num_col_b, 0);                                              \
  DCHECK_GE(start_row_c, 0);                                            \
  DCHECK_GE(start_col_c, 0);                                            \
  DCHECK_GT(row_stride_c, 0);                                           \
  DCHECK_GT(col_stride_c, 0);                                           \
  DCHECK((kRowA == Eigen::Dynamic) || (kRowA == num_row_a));            \
  DCHECK((kColA == Eigen::Dynamic) || (kColA == num_col_a));            \
  DCHECK((kRowB == Eigen::Dynamic) || (kRowB == num_row_b));            \
  DCHECK((kColB == Eigen::Dynamic) || (kColB == num_col_b));            \
  const int NUM_ROW_A = (kRowA != Eigen::Dynamic ? kRowA : num_row_a);  \
  const int NUM_COL_A = (kColA != Eigen::Dynamic ? kColA : num_col_a);  \
  const int NUM_ROW_B = (kRowB != Eigen::Dynamic ? kRowB : num_row_b);  \
  const int NUM_COL_B = (kColB != Eigen::Dynamic ? kColB : num_col_b);

#define CERES_GEMM_EIGEN_HEADER                                         \
  const typename EigenTypes<kRowA, kColA>::ConstMatrixRef               \
  Aref(A, num_row_a, num_col_a);                                        \
  const typename EigenTypes<kRowB, kColB>::ConstMatrixRef               \
  Bref(B, num_row_b, num_col_b);                                        \
  MatrixRef Cref(C, row_stride_c, col_stride_c);                        \

#define CERES_CALL_GEMM(name)                                           \
  name<kRowA, kColA, kRowB, kColB, kOperation>(                         \
      A, num_row_a, num_col_a,                                          \
      B, num_row_b, num_col_b,                                          \
      C, start_row_c, start_col_c, row_stride_c, col_stride_c);


// For the matrix-matrix functions below, there are three variants for
// each functionality. Foo, FooNaive and FooEigen. Foo is the one to
// be called by the user. FooNaive is a basic loop based
// implementation and FooEigen uses Eigen's implementation. Foo
// chooses between FooNaive and FooEigen depending on how many of the
// template arguments are fixed at compile time. Currently, FooEigen
// is called if all matrix dimensions are compile time
// constants. FooNaive is called otherwise. This leads to the best
// performance currently.
//
// The MatrixMatrixMultiply variants compute:
//
//   C op A * B;
//
// The MatrixTransposeMatrixMultiply variants compute:
//
//   C op A' * B
//
// where op can be +=, -=, or =.
//
// The template parameters (kRowA, kColA, kRowB, kColB) allow
// specialization of the loop at compile time. If this information is
// not available, then Eigen::Dynamic should be used as the template
// argument.
//
//   kOperation =  1  -> C += A * B
//   kOperation = -1  -> C -= A * B
//   kOperation =  0  -> C  = A * B
//
// The functions can write into matrices C which are larger than the
// matrix A * B. This is done by specifying the true size of C via
// row_stride_c and col_stride_c, and then indicating where A * B
// should be written into by start_row_c and start_col_c.
//
// Graphically if row_stride_c = 10, col_stride_c = 12, start_row_c =
// 4 and start_col_c = 5, then if A = 3x2 and B = 2x4, we get
//
//   ------------
//   ------------
//   ------------
//   ------------
//   -----xxxx---
//   -----xxxx---
//   -----xxxx---
//   ------------
//   ------------
//   ------------
//
CERES_GEMM_BEGIN(MatrixMatrixMultiplyEigen) {
  CERES_GEMM_EIGEN_HEADER
  Eigen::Block<MatrixRef, kRowA, kColB>
    block(Cref, start_row_c, start_col_c, num_row_a, num_col_b);

  if (kOperation > 0) {
    block.noalias() += Aref * Bref;
  } else if (kOperation < 0) {
    block.noalias() -= Aref * Bref;
  } else {
    block.noalias() = Aref * Bref;
  }
}

CERES_GEMM_BEGIN(MatrixMatrixMultiplyNaive) {
  CERES_GEMM_NAIVE_HEADER
  DCHECK_EQ(NUM_COL_A, NUM_ROW_B);

  const int NUM_ROW_C = NUM_ROW_A;
  const int NUM_COL_C = NUM_COL_B;
  DCHECK_LE(start_row_c + NUM_ROW_C, row_stride_c);
  DCHECK_LE(start_col_c + NUM_COL_C, col_stride_c);

  // Caller contract: A, B, C never alias.  Aliasing as __restrict
  // lets the compiler vectorize the inner reduction without proving
  // non-aliasing across translation units.
  const double* __restrict A_r = A;
  const double* __restrict B_r = B;
  double* __restrict C_r = C;

#if CERES_SMALL_BLAS_AXPY
  // AXPY form: walk B by rows (k-outer, col-inner).  Inner loop reads
  // a row of B at unit stride and writes C at unit stride; the scalar
  // a = A[row,k] is broadcast.  This vectorises into
  // vbroadcastsd + vfmadd231pd / vmulpd+vaddpd on AVX2/FMA.
  for (int row = 0; row < NUM_ROW_C; ++row) {
    const double* __restrict A_row = A_r + row * NUM_COL_A;
    double* __restrict C_row =
        C_r + (row + start_row_c) * col_stride_c + start_col_c;

    if (kOperation == 0) {
      // Zero-init the destination row before accumulating; converts
      // the kernel to pure C += A * B for the rest of the inner work.
      for (int col = 0; col < NUM_COL_C; ++col) {
        C_row[col] = 0.0;
      }
    }

    for (int k = 0; k < NUM_COL_A; ++k) {
      const double a = A_row[k];
      const double* __restrict B_k = B_r + k * NUM_COL_B;
      if (kOperation >= 0) {
        for (int col = 0; col < NUM_COL_C; ++col) {
          C_row[col] += a * B_k[col];
        }
      } else {
        for (int col = 0; col < NUM_COL_C; ++col) {
          C_row[col] -= a * B_k[col];
        }
      }
    }
  }
#else
  // Legacy dot-product form (bit-identical to upstream Ceres).
  for (int row = 0; row < NUM_ROW_C; ++row) {
    const double* __restrict A_row = A_r + row * NUM_COL_A;
    const int    C_row_base       = (row + start_row_c) * col_stride_c
                                  + start_col_c;
    for (int col = 0; col < NUM_COL_C; ++col) {
      double tmp = 0.0;
      for (int k = 0; k < NUM_COL_A; ++k) {
        tmp += A_row[k] * B_r[k * NUM_COL_B + col];
      }

      const int index = C_row_base + col;
      if (kOperation > 0) {
        C_r[index] += tmp;
      } else if (kOperation < 0) {
        C_r[index] -= tmp;
      } else {
        C_r[index] = tmp;
      }
    }
  }
#endif  // CERES_SMALL_BLAS_AXPY
}

CERES_GEMM_BEGIN(MatrixMatrixMultiply) {
#ifdef CERES_NO_CUSTOM_BLAS

  CERES_CALL_GEMM(MatrixMatrixMultiplyEigen)
  return;

#else

  if (kRowA != Eigen::Dynamic && kColA != Eigen::Dynamic &&
      kRowB != Eigen::Dynamic && kColB != Eigen::Dynamic) {
    CERES_CALL_GEMM(MatrixMatrixMultiplyEigen)
  } else {
    CERES_CALL_GEMM(MatrixMatrixMultiplyNaive)
  }

#endif
}

CERES_GEMM_BEGIN(MatrixTransposeMatrixMultiplyEigen) {
  CERES_GEMM_EIGEN_HEADER
  Eigen::Block<MatrixRef, kColA, kColB> block(Cref,
                                              start_row_c, start_col_c,
                                              num_col_a, num_col_b);
  if (kOperation > 0) {
    block.noalias() += Aref.transpose() * Bref;
  } else if (kOperation < 0) {
    block.noalias() -= Aref.transpose() * Bref;
  } else {
    block.noalias() = Aref.transpose() * Bref;
  }
}

CERES_GEMM_BEGIN(MatrixTransposeMatrixMultiplyNaive) {
  CERES_GEMM_NAIVE_HEADER
  DCHECK_EQ(NUM_ROW_A, NUM_ROW_B);

  const int NUM_ROW_C = NUM_COL_A;
  const int NUM_COL_C = NUM_COL_B;
  DCHECK_LE(start_row_c + NUM_ROW_C, row_stride_c);
  DCHECK_LE(start_col_c + NUM_COL_C, col_stride_c);

  // See note on MatrixMatrixMultiplyNaive: __restrict locals.
  const double* __restrict A_r = A;
  const double* __restrict B_r = B;
  double* __restrict C_r = C;

#if CERES_SMALL_BLAS_AXPY
  // AXPY form: outer over k (the contracted dim), then over output
  // rows, with the unit-stride col-loop innermost.  This reads A and
  // B by rows (unit stride) and broadcasts a = A[k,row] across the
  // inner loop -- the same vectorisable shape as MatrixMatrixMultiply.
  // The previous form read A and B both column-strided, which prevents
  // SIMD entirely.
  if (kOperation == 0) {
    // Zero the (NUM_ROW_C x NUM_COL_C) destination sub-block once;
    // every (k, row, col) update from here on is +=.
    for (int row = 0; row < NUM_ROW_C; ++row) {
      double* __restrict C_row =
          C_r + (row + start_row_c) * col_stride_c + start_col_c;
      for (int col = 0; col < NUM_COL_C; ++col) {
        C_row[col] = 0.0;
      }
    }
  }

  for (int k = 0; k < NUM_ROW_A; ++k) {
    const double* __restrict A_k = A_r + k * NUM_COL_A;  // row k of A
    const double* __restrict B_k = B_r + k * NUM_COL_B;  // row k of B
    for (int row = 0; row < NUM_ROW_C; ++row) {
      const double a = A_k[row];                          // = A[k,row]
      double* __restrict C_row =
          C_r + (row + start_row_c) * col_stride_c + start_col_c;
      if (kOperation >= 0) {
        for (int col = 0; col < NUM_COL_C; ++col) {
          C_row[col] += a * B_k[col];
        }
      } else {
        for (int col = 0; col < NUM_COL_C; ++col) {
          C_row[col] -= a * B_k[col];
        }
      }
    }
  }
#else
  // Legacy dot-product form (bit-identical to upstream Ceres).
  for (int row = 0; row < NUM_ROW_C; ++row) {
    const int C_row_base = (row + start_row_c) * col_stride_c + start_col_c;
    for (int col = 0; col < NUM_COL_C; ++col) {
      double tmp = 0.0;
      for (int k = 0; k < NUM_ROW_A; ++k) {
        tmp += A_r[k * NUM_COL_A + row] * B_r[k * NUM_COL_B + col];
      }

      const int index = C_row_base + col;
      if (kOperation > 0) {
        C_r[index]+= tmp;
      } else if (kOperation < 0) {
        C_r[index]-= tmp;
      } else {
        C_r[index]= tmp;
      }
    }
  }
#endif  // CERES_SMALL_BLAS_AXPY
}

CERES_GEMM_BEGIN(MatrixTransposeMatrixMultiply) {
#ifdef CERES_NO_CUSTOM_BLAS

  CERES_CALL_GEMM(MatrixTransposeMatrixMultiplyEigen)
  return;

#else

  if (kRowA != Eigen::Dynamic && kColA != Eigen::Dynamic &&
      kRowB != Eigen::Dynamic && kColB != Eigen::Dynamic) {
    CERES_CALL_GEMM(MatrixTransposeMatrixMultiplyEigen)
  } else {
    CERES_CALL_GEMM(MatrixTransposeMatrixMultiplyNaive)
  }

#endif
}

// Matrix-Vector multiplication
//
// c op A * b;
//
// where op can be +=, -=, or =.
//
// The template parameters (kRowA, kColA) allow specialization of the
// loop at compile time. If this information is not available, then
// Eigen::Dynamic should be used as the template argument.
//
// kOperation =  1  -> c += A' * b
// kOperation = -1  -> c -= A' * b
// kOperation =  0  -> c  = A' * b
template<int kRowA, int kColA, int kOperation>
CERES_SMALL_BLAS_FORCE_INLINE void MatrixVectorMultiply(const double* A,
                                 const int num_row_a,
                                 const int num_col_a,
                                 const double* b,
                                 double* c) {
#ifdef CERES_NO_CUSTOM_BLAS
  const typename EigenTypes<kRowA, kColA>::ConstMatrixRef
      Aref(A, num_row_a, num_col_a);
  const typename EigenTypes<kColA>::ConstVectorRef bref(b, num_col_a);
  typename EigenTypes<kRowA>::VectorRef cref(c, num_row_a);

  // lazyProduct works better than .noalias() for matrix-vector
  // products.
  if (kOperation > 0) {
    cref += Aref.lazyProduct(bref);
  } else if (kOperation < 0) {
    cref -= Aref.lazyProduct(bref);
  } else {
    cref = Aref.lazyProduct(bref);
  }
#else

  DCHECK_GT(num_row_a, 0);
  DCHECK_GT(num_col_a, 0);
  DCHECK((kRowA == Eigen::Dynamic) || (kRowA == num_row_a));
  DCHECK((kColA == Eigen::Dynamic) || (kColA == num_col_a));

  const int NUM_ROW_A = (kRowA != Eigen::Dynamic ? kRowA : num_row_a);
  const int NUM_COL_A = (kColA != Eigen::Dynamic ? kColA : num_col_a);

  // Caller contract: A, b, c never alias.
  const double* __restrict A_r = A;
  const double* __restrict b_r = b;
  double* __restrict c_r = c;

  for (int row = 0; row < NUM_ROW_A; ++row) {
    // Hoist the row base of A out of the inner column reduction.  On
    // the dynamic-kColA path this saves one imul per (row, col) and
    // turns the inner loop into a clean unit-stride dot product.
    const double* __restrict A_row = A_r + row * NUM_COL_A;
    double tmp = 0.0;
    for (int col = 0; col < NUM_COL_A; ++col) {
      tmp += A_row[col] * b_r[col];
    }

    if (kOperation > 0) {
      c_r[row] += tmp;
    } else if (kOperation < 0) {
      c_r[row] -= tmp;
    } else {
      c_r[row] = tmp;
    }
  }
#endif  // CERES_NO_CUSTOM_BLAS
}

// Similar to MatrixVectorMultiply, except that A is transposed, i.e.,
//
// c op A' * b;
template<int kRowA, int kColA, int kOperation>
CERES_SMALL_BLAS_FORCE_INLINE void MatrixTransposeVectorMultiply(const double* A,
                                          const int num_row_a,
                                          const int num_col_a,
                                          const double* b,
                                          double* c) {
#ifdef CERES_NO_CUSTOM_BLAS
  const typename EigenTypes<kRowA, kColA>::ConstMatrixRef
      Aref(A, num_row_a, num_col_a);
  const typename EigenTypes<kRowA>::ConstVectorRef bref(b, num_row_a);
  typename EigenTypes<kColA>::VectorRef cref(c, num_col_a);

  // lazyProduct works better than .noalias() for matrix-vector
  // products.
  if (kOperation > 0) {
    cref += Aref.transpose().lazyProduct(bref);
  } else if (kOperation < 0) {
    cref -= Aref.transpose().lazyProduct(bref);
  } else {
    cref = Aref.transpose().lazyProduct(bref);
  }
#else

  DCHECK_GT(num_row_a, 0);
  DCHECK_GT(num_col_a, 0);
  DCHECK((kRowA == Eigen::Dynamic) || (kRowA == num_row_a));
  DCHECK((kColA == Eigen::Dynamic) || (kColA == num_col_a));

  const int NUM_ROW_A = (kRowA != Eigen::Dynamic ? kRowA : num_row_a);
  const int NUM_COL_A = (kColA != Eigen::Dynamic ? kColA : num_col_a);

  // Caller contract: A, b, c never alias.
  const double* __restrict A_r = A;
  const double* __restrict b_r = b;
  double* __restrict c_r = c;

#if CERES_SMALL_BLAS_AXPY
  // AXPY form: outer over k (rows of A), inner over j (cols of A,
  // also length of c).  Inner loop reads A row k at unit stride and
  // writes c at unit stride; bk = b[k] is broadcast.  Vectorises into
  // a tight vfmadd / vfnmadd loop.  The previous form read a column
  // of A at stride NUM_COL_A doubles, defeating SIMD.
  if (kOperation == 0) {
    for (int j = 0; j < NUM_COL_A; ++j) {
      c_r[j] = 0.0;
    }
  }

  for (int k = 0; k < NUM_ROW_A; ++k) {
    const double bk = b_r[k];
    const double* __restrict A_k = A_r + k * NUM_COL_A;  // row k of A
    if (kOperation >= 0) {
      for (int j = 0; j < NUM_COL_A; ++j) {
        c_r[j] += A_k[j] * bk;
      }
    } else {
      for (int j = 0; j < NUM_COL_A; ++j) {
        c_r[j] -= A_k[j] * bk;
      }
    }
  }
#else
  // Legacy dot-product form (bit-identical to upstream Ceres):
  // outer over output entries, inner reduction with strided A reads.
  for (int row = 0; row < NUM_COL_A; ++row) {
    const double* __restrict A_col = A_r + row;
    double tmp = 0.0;
    for (int col = 0; col < NUM_ROW_A; ++col) {
      tmp += A_col[col * NUM_COL_A] * b_r[col];
    }

    if (kOperation > 0) {
      c_r[row] += tmp;
    } else if (kOperation < 0) {
      c_r[row] -= tmp;
    } else {
      c_r[row] = tmp;
    }
  }
#endif  // CERES_SMALL_BLAS_AXPY
#endif  // CERES_NO_CUSTOM_BLAS
}

#undef CERES_GEMM_BEGIN
#undef CERES_GEMM_EIGEN_HEADER
#undef CERES_GEMM_NAIVE_HEADER
#undef CERES_CALL_GEMM

// Symmetric self-multiply: C op A' * A.
//
// The result is symmetric (NUM_COL_A x NUM_COL_A).  Computes the
// upper triangle only (i <= j) and mirrors to the lower triangle,
// halving the FLOP count compared with a general MTMM call.
//
// Correctness contract for kOperation == 1 (+=) and -1 (-=): the
// caller guarantees that the destination sub-block of C is already
// symmetric.  All Schur eliminator diagonal-block accumulators
// satisfy this -- they are zero-initialised by lhs->SetZero() (or
// start as a diagonal D*D matrix) and only ever receive symmetric
// updates, so symmetry is preserved across calls.
//
// Falls back to a general MatrixTransposeMatrixMultiply call when the
// AXPY family of optimisations is disabled, so the bit-identical
// legacy build path stays untouched.
template<int kRowA, int kColA, int kOperation>
CERES_SMALL_BLAS_FORCE_INLINE void MatrixTransposeMatrixMultiplySelf(
    const double* A,
    const int num_row_a,
    const int num_col_a,
    double* C,
    const int start_row_c,
    const int start_col_c,
    const int row_stride_c,
    const int col_stride_c) {
#if !CERES_SMALL_BLAS_AXPY
  // Bit-identical fallback: dispatch to the legacy MTMM with B == A.
  MatrixTransposeMatrixMultiply<kRowA, kColA, kRowA, kColA, kOperation>(
      A, num_row_a, num_col_a,
      A, num_row_a, num_col_a,
      C, start_row_c, start_col_c, row_stride_c, col_stride_c);
#else
  DCHECK_GT(num_row_a, 0);
  DCHECK_GT(num_col_a, 0);
  DCHECK_GE(start_row_c, 0);
  DCHECK_GE(start_col_c, 0);
  DCHECK_GT(row_stride_c, 0);
  DCHECK_GT(col_stride_c, 0);
  DCHECK((kRowA == Eigen::Dynamic) || (kRowA == num_row_a));
  DCHECK((kColA == Eigen::Dynamic) || (kColA == num_col_a));

  const int NUM_ROW_A = (kRowA != Eigen::Dynamic ? kRowA : num_row_a);
  const int NUM_COL_A = (kColA != Eigen::Dynamic ? kColA : num_col_a);
  const int N = NUM_COL_A;

  DCHECK_LE(start_row_c + N, row_stride_c);
  DCHECK_LE(start_col_c + N, col_stride_c);

  const double* __restrict A_r = A;
  double* __restrict C_r = C;

  // Step 1: optionally zero the upper triangle (only for kOperation==0).
  if (kOperation == 0) {
    for (int i = 0; i < N; ++i) {
      double* __restrict C_row =
          C_r + (start_row_c + i) * col_stride_c + start_col_c;
      for (int j = i; j < N; ++j) {
        C_row[j] = 0.0;
      }
    }
  }

  // Step 2: AXPY accumulation over k.  For each row k of A, broadcast
  // a_ki = A[k,i] and update the upper triangle of row i:
  //   C[i,j] += a_ki * A[k,j]   for j >= i
  // The inner j-loop reads A[k,j..] and writes C_row[j..] at unit
  // stride; both vectorise into vfmadd231pd / vfnmadd231pd packs.
  for (int k = 0; k < NUM_ROW_A; ++k) {
    const double* __restrict A_k = A_r + k * NUM_COL_A;  // row k of A
    for (int i = 0; i < N; ++i) {
      const double a_ki = A_k[i];
      double* __restrict C_row =
          C_r + (start_row_c + i) * col_stride_c + start_col_c;
      if (kOperation >= 0) {
        for (int j = i; j < N; ++j) {
          C_row[j] += a_ki * A_k[j];
        }
      } else {
        for (int j = i; j < N; ++j) {
          C_row[j] -= a_ki * A_k[j];
        }
      }
    }
  }

  // Step 3: mirror upper triangle into lower triangle.  After the
  // accumulation above the upper triangle (j >= i) holds the correct
  // C[i,j].  Because the input C was symmetric and A'A is symmetric,
  // the lower triangle's correct value equals the upper's value at
  // the transposed index.
  for (int i = 0; i < N; ++i) {
    const int upper_row_base =
        (start_row_c + i) * col_stride_c + start_col_c;
    for (int j = i + 1; j < N; ++j) {
      const int lower_idx =
          (start_row_c + j) * col_stride_c + start_col_c + i;
      C_r[lower_idx] = C_r[upper_row_base + j];
    }
  }
#endif  // CERES_SMALL_BLAS_AXPY
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SMALL_BLAS_H_
