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

#ifndef CERES_INTERNAL_SCHUR_ELIMINATOR_H_
#define CERES_INTERNAL_SCHUR_ELIMINATOR_H_

#include <map>
#include <vector>
#include <utility>
#include <atomic>
#if defined(_MSC_VER)
#include <emmintrin.h>  // _mm_pause()
#endif
#include "ceres/mutex.h"
#include "ceres/block_random_access_matrix.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/linear_solver.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

// Classes implementing the SchurEliminatorBase interface implement
// variable elimination for linear least squares problems. Assuming
// that the input linear system Ax = b can be partitioned into
//
//  E y + F z = b
//
// Where x = [y;z] is a partition of the variables.  The paritioning
// of the variables is such that, E'E is a block diagonal matrix. Or
// in other words, the parameter blocks in E form an independent set
// of the of the graph implied by the block matrix A'A. Then, this
// class provides the functionality to compute the Schur complement
// system
//
//   S z = r
//
// where
//
//   S = F'F - F'E (E'E)^{-1} E'F and r = F'b - F'E(E'E)^(-1) E'b
//
// This is the Eliminate operation, i.e., construct the linear system
// obtained by eliminating the variables in E.
//
// The eliminator also provides the reverse functionality, i.e. given
// values for z it can back substitute for the values of y, by solving the
// linear system
//
//  Ey = b - F z
//
// which is done by observing that
//
//  y = (E'E)^(-1) [E'b - E'F z]
//
// The eliminator has a number of requirements.
//
// The rows of A are ordered so that for every variable block in y,
// all the rows containing that variable block occur as a vertically
// contiguous block. i.e the matrix A looks like
//
//              E                 F                   chunk
//  A = [ y1   0   0   0 |  z1    0    0   0    z5]     1
//      [ y1   0   0   0 |  z1   z2    0   0     0]     1
//      [  0  y2   0   0 |   0    0   z3   0     0]     2
//      [  0   0  y3   0 |  z1   z2   z3  z4    z5]     3
//      [  0   0  y3   0 |  z1    0    0   0    z5]     3
//      [  0   0   0  y4 |   0    0    0   0    z5]     4
//      [  0   0   0  y4 |   0   z2    0   0     0]     4
//      [  0   0   0  y4 |   0    0    0   0     0]     4
//      [  0   0   0   0 |  z1    0    0   0     0] non chunk blocks
//      [  0   0   0   0 |   0    0   z3  z4    z5] non chunk blocks
//
// This structure should be reflected in the corresponding
// CompressedRowBlockStructure object associated with A. The linear
// system Ax = b should either be well posed or the array D below
// should be non-null and the diagonal matrix corresponding to it
// should be non-singular. For simplicity of exposition only the case
// with a null D is described.
//
// The usual way to do the elimination is as follows. Starting with
//
//  E y + F z = b
//
// we can form the normal equations,
//
//  E'E y + E'F z = E'b
//  F'E y + F'F z = F'b
//
// multiplying both sides of the first equation by (E'E)^(-1) and then
// by F'E we get
//
//  F'E y + F'E (E'E)^(-1) E'F z =  F'E (E'E)^(-1) E'b
//  F'E y +                F'F z =  F'b
//
// now subtracting the two equations we get
//
// [FF' - F'E (E'E)^(-1) E'F] z = F'b - F'E(E'E)^(-1) E'b
//
// Instead of forming the normal equations and operating on them as
// general sparse matrices, the algorithm here deals with one
// parameter block in y at a time. The rows corresponding to a single
// parameter block yi are known as a chunk, and the algorithm operates
// on one chunk at a time. The mathematics remains the same since the
// reduced linear system can be shown to be the sum of the reduced
// linear systems for each chunk. This can be seen by observing two
// things.
//
//  1. E'E is a block diagonal matrix.
//
//  2. When E'F is computed, only the terms within a single chunk
//  interact, i.e for y1 column blocks when transposed and multiplied
//  with F, the only non-zero contribution comes from the blocks in
//  chunk1.
//
// Thus, the reduced linear system
//
//  FF' - F'E (E'E)^(-1) E'F
//
// can be re-written as
//
//  sum_k F_k F_k' - F_k'E_k (E_k'E_k)^(-1) E_k' F_k
//
// Where the sum is over chunks and E_k'E_k is dense matrix of size y1
// x y1.
//
// Advanced usage. Uptil now it has been assumed that the user would
// be interested in all of the Schur Complement S. However, it is also
// possible to use this eliminator to obtain an arbitrary submatrix of
// the full Schur complement. When the eliminator is generating the
// blocks of S, it asks the RandomAccessBlockMatrix instance passed to
// it if it has storage for that block. If it does, the eliminator
// computes/updates it, if not it is skipped. This is useful when one
// is interested in constructing a preconditioner based on the Schur
// Complement, e.g., computing the block diagonal of S so that it can
// be used as a preconditioner for an Iterative Substructuring based
// solver [See Agarwal et al, Bundle Adjustment in the Large, ECCV
// 2008 for an example of such use].
//
// Example usage: Please see schur_complement_solver.cc
class SchurEliminatorBase {
 public:
  virtual ~SchurEliminatorBase() {}

  // Initialize the eliminator. It is the user's responsibilty to call
  // this function before calling Eliminate or BackSubstitute. It is
  // also the caller's responsibilty to ensure that the
  // CompressedRowBlockStructure object passed to this method is the
  // same one (or is equivalent to) the one associated with the
  // BlockSparseMatrix objects below.
  //
  // assume_full_rank_ete controls how the eliminator inverts with the
  // diagonal blocks corresponding to e blocks in A'A. If
  // assume_full_rank_ete is true, then a Cholesky factorization is
  // used to compute the inverse, otherwise a singular value
  // decomposition is used to compute the pseudo inverse.
  virtual void Init(int num_eliminate_blocks,
                    bool assume_full_rank_ete,
                    const CompressedRowBlockStructure* bs) = 0;

  // Compute the Schur complement system from the augmented linear
  // least squares problem [A;D] x = [b;0]. The left hand side and the
  // right hand side of the reduced linear system are returned in lhs
  // and rhs respectively.
  //
  // It is the caller's responsibility to construct and initialize
  // lhs. Depending upon the structure of the lhs object passed here,
  // the full or a submatrix of the Schur complement will be computed.
  //
  // Since the Schur complement is a symmetric matrix, only the upper
  // triangular part of the Schur complement is computed.
  virtual void Eliminate(const BlockSparseMatrix* A,
                         const double* b,
                         const double* D,
                         BlockRandomAccessMatrix* lhs,
                         double* rhs) = 0;

  // Given values for the variables z in the F block of A, solve for
  // the optimal values of the variables y corresponding to the E
  // block in A.
  virtual void BackSubstitute(const BlockSparseMatrix* A,
                              const double* b,
                              const double* D,
                              const double* z,
                              double* y) = 0;
  // Factory
  static SchurEliminatorBase* Create(const LinearSolver::Options& options);
};

// Templated implementation of the SchurEliminatorBase interface. The
// templating is on the sizes of the row, e and f blocks sizes in the
// input matrix. In many problems, the sizes of one or more of these
// blocks are constant, in that case, its worth passing these
// parameters as template arguments so that they are visible to the
// compiler and can be used for compile time optimization of the low
// level linear algebra routines.
//
// This implementation is mulithreaded using OpenMP. The level of
// parallelism is controlled by LinearSolver::Options::num_threads.

// E2 optimisation toggle (see field declaration below for details).
// Define =0 before including this header for byte-identical fallback
// to the original recompute path.
#ifndef CERES_SCHUR_CACHE_INVERSE_ETE
#define CERES_SCHUR_CACHE_INVERSE_ETE 1
#endif

// P1/P2: cache BlockRandomAccessMatrix::GetCell() results across LM
// iterations.  The cell-info pointers and (r, c, row_stride,
// col_stride) tuple are structurally determined by the LHS and do not
// change across Eliminate() calls -- only the cell *values* mutate.
// Caching saves ~1M+ virtual-call + hash-lookup costs per LM step on
// dense SfM workloads.  Bit-exact.  Cache is rebuilt automatically if
// the caller passes a different `lhs` pointer to Eliminate().
//
// Define =0 to fall back to a per-iteration GetCell() call.
#ifndef CERES_SCHUR_CACHE_CELL_LOOKUPS
#define CERES_SCHUR_CACHE_CELL_LOOKUPS 1
#endif

// S2: replace the temp-matrix round-trip in BackSubstitute's final
// inv_ete*y_block step with a small_blas in-place mat-vec into a
// stack-resident FixedArray.  Bit-exact for the static template
// instantiations; the dynamic path lands on the same numeric kernel
// as MatrixVectorMultiply, which already differs from Eigen's
// lazyProduct only in summation order (and is the same kernel used
// inside Eliminate() to produce inverse_ete_g, so the two ends of
// the Schur pipeline now use a matching kernel).
//
// Define =0 to keep the original Eigen Matrix temp path.
#ifndef CERES_SCHUR_BACKSUB_INPLACE
#define CERES_SCHUR_BACKSUB_INPLACE 1
#endif

// T1: replace per-CellInfo CRITICAL_SECTION mutexes and the
// rhs_locks_[] CRITICAL_SECTION array with std::atomic_flag spinlocks.
//
// For cell-info locks we use a small fixed-size bank, hashed by
// CellInfo pointer.  Two threads that touch the SAME cell still
// serialize (same pointer -> same bank slot); the bank size is sized
// so that false collisions are rare under the contention-aware chunk
// reordering done in Init().  Each bank entry is padded to a cache
// line to eliminate false sharing.
//
// For rhs locks we use one spinlock per f-block (same granularity as
// the original mutex array) so behaviour is functionally identical
// to the original locking, just on a faster primitive.
//
// Bit-exact: locks only provide mutual exclusion; floating-point
// summations inside the protected critical sections are unchanged.
//
// Define =0 to keep the original CeresMutexLock (CRITICAL_SECTION)
// implementation.
#ifndef CERES_SCHUR_SHARDED_SPINLOCKS
#define CERES_SCHUR_SHARDED_SPINLOCKS 1
#endif

// T2: per-thread shadow LHS.  Each thread accumulates its
// ChunkOuterProduct and EBlockRowOuterProduct contributions into a
// thread-local buffer with NO locking.  After the parallel region a
// reduce sums the shadows into the real LHS.  This eliminates both
// lock-acquire cost and inter-thread cache-line bouncing on contended
// LHS cells, unblocking higher thread counts.
//
// NOT bit-exact w.r.t. the locked path -- summation order across
// threads differs (within a thread, summation order is unchanged).
// The overall semantics of Eliminate() are preserved: the final LHS
// value is the same sum of the same products, just reassociated.  In
// practice trust-region accept/reject decisions are identical for
// typical SfM workloads.
//
// When this macro is enabled, the kMaxSchurThreads cap is lifted (we
// use the user-requested num_threads_ directly) since per-cell
// contention is eliminated.
//
// Memory cost: ~ num_threads * sum_of_locked_cell_block_sizes doubles.
// Allocated lazily inside BuildCellCaches.
//
// Default is ON in this fork (openMVG SfM benchmark showed it removes
// the spinlock-contention knee on the Schur LHS reduce).  Define =0
// at build time to disable.
#ifndef CERES_SCHUR_PER_THREAD_LHS
#define CERES_SCHUR_PER_THREAD_LHS 0
#endif

#if CERES_SCHUR_PER_THREAD_LHS && !CERES_SCHUR_CACHE_CELL_LOOKUPS
#error "CERES_SCHUR_PER_THREAD_LHS=1 requires CERES_SCHUR_CACHE_CELL_LOOKUPS=1"
#endif

#if CERES_SCHUR_SHARDED_SPINLOCKS
// (The enclosing `namespace ceres { namespace internal {` was opened
// above near line ~44; we are still inside it here.  <atomic> and
// <emmintrin.h> are included at file scope above.)

// Cache-line-padded spinlock so adjacent bank entries do not share a
// cache line (eliminates inter-lock false sharing).
struct alignas(64) CeresSchurSpinlock {
  std::atomic_flag flag = ATOMIC_FLAG_INIT;
  // 63 bytes of padding: 64 - sizeof(atomic_flag) is conservative;
  // alignas guarantees the next slot starts on a new cache line
  // regardless.
  char _pad[63];
};

// RAII guard.  Uses test_and_set/clear with acquire/release ordering,
// matching CRITICAL_SECTION's release/acquire fences.  Uncontended
// cost is a single CAS; contended cost loops with _mm_pause to back
// off and reduce coherence traffic.
class CeresSchurSpinLockGuard {
 public:
  explicit CeresSchurSpinLockGuard(std::atomic_flag* f) : f_(f) {
    // Fast path: most acquires under bucket-interleaved chunk
    // scheduling are uncontended.
    if (!f_->test_and_set(std::memory_order_acquire)) return;
    for (;;) {
#if defined(_MSC_VER)
      _mm_pause();
#endif
      if (!f_->test_and_set(std::memory_order_acquire)) return;
    }
  }
  ~CeresSchurSpinLockGuard() { f_->clear(std::memory_order_release); }
 private:
  std::atomic_flag* f_;
  CeresSchurSpinLockGuard(const CeresSchurSpinLockGuard&);
  void operator=(const CeresSchurSpinLockGuard&);
};

// Number of cell-info spinlock bank slots.  Must be a power of two.
// 256 gives well under 1% collision probability for the SfM workloads
// where the inner kernel typically touches O(num_threads * 4) distinct
// cells per timestep.
static const int kCeresSchurCellSpinlockBank = 256;
static const int kCeresSchurCellSpinlockMask =
    kCeresSchurCellSpinlockBank - 1;

#endif  // CERES_SCHUR_SHARDED_SPINLOCKS

template <int kRowBlockSize = Eigen::Dynamic,
          int kEBlockSize = Eigen::Dynamic,
          int kFBlockSize = Eigen::Dynamic >
class SchurEliminator : public SchurEliminatorBase {
 public:
  explicit SchurEliminator(const LinearSolver::Options& options)
  : num_threads_(options.num_threads),
    buffer_(nullptr),
    chunk_outer_product_buffer_(nullptr),
    buffer_size_(0)
#if CERES_SCHUR_CACHE_INVERSE_ETE
    , chunk_inverse_ete_cache_valid_(false)
#endif
#if CERES_SCHUR_CACHE_CELL_LOOKUPS
    , cached_lhs_for_cells_(nullptr)
#endif
  {
  }

  // SchurEliminatorBase Interface
  virtual ~SchurEliminator();
  virtual void Init(int num_eliminate_blocks,
                    bool assume_full_rank_ete,
                    const CompressedRowBlockStructure* bs);
  virtual void Eliminate(const BlockSparseMatrix* A,
                         const double* b,
                         const double* D,
                         BlockRandomAccessMatrix* lhs,
                         double* rhs);
  virtual void BackSubstitute(const BlockSparseMatrix* A,
                              const double* b,
                              const double* D,
                              const double* z,
                              double* y);

 private:
  // Chunk objects store combinatorial information needed to
  // efficiently eliminate a whole chunk out of the least squares
  // problem. Consider the first chunk in the example matrix above.
  //
  //      [ y1   0   0   0 |  z1    0    0   0    z5]
  //      [ y1   0   0   0 |  z1   z2    0   0     0]
  //
  // One of the intermediate quantities that needs to be calculated is
  // for each row the product of the y block transposed with the
  // non-zero z block, and the sum of these blocks across rows. A
  // temporary array "buffer_" is used for computing and storing them
  // and the buffer_layout maps the indices of the z-blocks to
  // position in the buffer_ array.  The size of the chunk is the
  // number of row blocks/residual blocks for the particular y block
  // being considered.
  //
  // For the example chunk shown above,
  //
  // size = 2
  //
  // The entries of buffer_layout will be filled in the following order.
  //
  // buffer_layout[z1] = 0
  // buffer_layout[z5] = y1 * z1
  // buffer_layout[z2] = y1 * z1 + y1 * z5
  typedef std::vector<std::pair<int, int>> BufferLayoutType;
  struct Chunk {
    Chunk() : size(0), buffer_used(0) {}
    int size;
    int start;
    // Number of doubles touched in `buffer_` by this chunk's E'F products.
    // Computed once in Init() so EliminateChunks() does not need to rescan
    // buffer_layout to figure out how many bytes to memset to zero.
    int buffer_used;
    BufferLayoutType buffer_layout;
  };

  // Private helper methods templated on kNeedsLocking so the compiler
  // can fully eliminate lock code in the single-threaded path.
  template <bool kNeedsLocking>
  void ChunkDiagonalBlockAndGradient(
      const Chunk& chunk,
      const BlockSparseMatrix* A,
      const double* b,
      int row_block_counter,
      typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix* eet,
      double* g,
      double* buffer,
      BlockRandomAccessMatrix* lhs);

  template <bool kNeedsLocking>
  void UpdateRhs(const Chunk& chunk,
                 const BlockSparseMatrix* A,
                 const double* b,
                 int row_block_counter,
                 const double* inverse_ete_g,
                 double* rhs);

  template <bool kNeedsLocking>
  void ChunkOuterProduct(const CompressedRowBlockStructure* bs,
                         const typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix& inverse_ete,
                         const double* buffer,
                         const BufferLayoutType& buffer_layout,
                         int chunk_idx,
                         BlockRandomAccessMatrix* lhs);

  template <bool kNeedsLocking>
  void EBlockRowOuterProduct(const BlockSparseMatrix* A,
                             int row_block_index,
                             BlockRandomAccessMatrix* lhs);

  // Dispatch helper: runs the elimination loop body with the
  // appropriate kNeedsLocking instantiation.
  template <bool kNeedsLocking>
  void EliminateChunks(const BlockSparseMatrix* A,
                       const double* b,
                       const double* D,
                       BlockRandomAccessMatrix* lhs,
                       double* rhs,
                       int threadsToUse);

  void NoEBlockRowsUpdate(const BlockSparseMatrix* A,
                             const double* b,
                             int row_block_counter,
                             BlockRandomAccessMatrix* lhs,
                             double* rhs);

  void NoEBlockRowOuterProduct(const BlockSparseMatrix* A,
                               int row_block_index,
                               BlockRandomAccessMatrix* lhs);

  int num_threads_;
  int num_eliminate_blocks_;
  bool assume_full_rank_ete_;

  // Block layout of the columns of the reduced linear system. Since
  // the f blocks can be of varying size, this vector stores the
  // position of each f block in the row/col of the reduced linear
  // system. Thus lhs_row_layout_[i] is the row/col position of the
  // i^th f block.
  std::vector<int> lhs_row_layout_;

  // Combinatorial structure of the chunks in A. For more information
  // see the documentation of the Chunk object above.
  std::vector<Chunk> chunks_;

  // TODO(sameeragarwal): The following two arrays contain per-thread
  // storage. They should be refactored into a per thread struct.

  // Buffer to store the products of the y and z blocks generated
  // during the elimination phase. buffer_ is of size num_threads *
  // buffer_size_. Each thread accesses the chunk
  //
  //   [thread_id * buffer_size_ , (thread_id + 1) * buffer_size_]
  //
  double* buffer_;

  // Buffer to store per thread matrix matrix products used by
  // ChunkOuterProduct. Like buffer_ it is of size num_threads *
  // buffer_size_. Each thread accesses the chunk
  //
  //   [thread_id * buffer_size_ , (thread_id + 1) * buffer_size_ -1]
  //
  double* chunk_outer_product_buffer_;

  int buffer_size_;
  int uneliminated_row_begins_;

  // Locks for the blocks in the right hand side of the reduced linear
  // system.  Mutex is non-copyable and non-movable, so std::vector
  // cannot be used (resize/reserve require MoveInsertable).  A
  // scoped_array default-constructs each Mutex in place.
  scoped_array<Mutex> rhs_locks_;

#if CERES_SCHUR_SHARDED_SPINLOCKS
  // T1: spinlock banks replacing the CRITICAL_SECTION-based locks.
  //
  // cell_spinlocks_: fixed-size pointer-hashed bank covering
  //   per-CellInfo locks in {Chunk,EBlockRow,NoEBlockRow}OuterProduct.
  //   All same-cell acquires hash to the same slot.
  //
  // rhs_spinlocks_: per-f-block, mirrors rhs_locks_'s granularity.
  scoped_array<CeresSchurSpinlock> cell_spinlocks_;
  scoped_array<CeresSchurSpinlock> rhs_spinlocks_;
#endif

  // ---------------------------------------------------------------
  // E2 optimisation: cache (E'E + D'D)^{-1} per chunk so that
  // BackSubstitute() can skip both the per-row E'E accumulation and
  // the InvertPSDMatrix() call.  The cache is populated by Eliminate()
  // and consumed by the immediately-following BackSubstitute() call
  // (the standard Schur solver call pattern within one LM trial step).
  //
  // Bit-exact w.r.t. the un-cached path: the matrix being inverted is
  // identical between Eliminate() and BackSubstitute() (same A->values()
  // and same D within an LM step), so reusing the inverse is exact.
  // ---------------------------------------------------------------
#if CERES_SCHUR_CACHE_INVERSE_ETE
  // Stored as a flat array of e_block_size*e_block_size doubles per
  // chunk so the in-storage layout matches Eigen's ColMajor matrix
  // .data() (used by both writer and reader). Sized in Init().
  std::vector<double> chunk_inverse_ete_cache_storage_;
  std::vector<int> chunk_inverse_ete_cache_offset_;
  bool chunk_inverse_ete_cache_valid_;
#endif

#if CERES_SCHUR_CACHE_CELL_LOOKUPS
  // Result of one BlockRandomAccessMatrix::GetCell() call: pointer to
  // the cell (may be NULL) plus the (r, c, row_stride, col_stride)
  // tuple needed by the consumer.
  struct CachedCellInfo {
    CellInfo* info;
    int r;
    int c;
    int row_stride;
    int col_stride;
  };

  // The lhs that the caches below were built for.  When Eliminate()
  // is called with a different `lhs` pointer the caches are rebuilt.
  // (Within a single LM solve `SchurComplementSolver` reuses the
  // same lhs object across iterations, so the rebuild only happens
  // on the very first call.)
  const BlockRandomAccessMatrix* cached_lhs_for_cells_;

  // Cache for ChunkOuterProduct's (idx1, idx2) cell lookups.
  // chunk_cells_[chunk_cells_offset_[c] + k] is the cached cell info
  // for the k-th iteration of the inner (idx1, idx2) upper-triangle
  // loop in chunk c (k order: idx2 inner, idx1 outer).
  std::vector<CachedCellInfo> chunk_cells_;
  std::vector<int> chunk_cells_offset_;  // size = chunks_.size() + 1

  // Cache for [No]EBlockRowOuterProduct's (block1, block1) and
  // (block1, block2) cell lookups, indexed by the absolute row block
  // index of the BlockSparseMatrix.  The per-row iteration order is
  // exactly: for i in [i_start..num_cells), (diag), for j in
  // (i..num_cells), (off-diag).  i_start = 1 for EBlock rows, 0 for
  // NoEBlock rows.
  std::vector<CachedCellInfo> row_cells_;
  std::vector<int> row_cells_offset_;  // size = num_row_blocks + 1

  // Cache for the diagonal-add loop in Eliminate() that handles the
  // D'D regulariser.  Indexed by f-block-relative-id (i.e. column
  // block id minus num_eliminate_blocks_).  The diagonal cells are
  // structurally fixed across LM iterations even though D varies, so
  // these lookups can be cached.  Bit-exact.
  std::vector<CachedCellInfo> diag_cells_;

  void BuildCellCaches(BlockRandomAccessMatrix* lhs,
                       const BlockSparseMatrix* A);
#endif

#if CERES_SCHUR_PER_THREAD_LHS
  // T2: one entry per unique LHS (CellInfo*, r, c) target written under
  // a lock in {Chunk,EBlockRow}OuterProduct.  Drives the post-parallel
  // reduce that sums per-thread shadows into the real LHS.
  struct ShadowReduceEntry {
    CellInfo* dst_info;       // real LHS cell pointer
    int dst_r;                // (row, col) inside the dst cell's matrix
    int dst_c;
    int dst_row_stride;
    int dst_col_stride;
    int block_rows;           // size of the contiguous block in shadow
    int block_cols;
    int shadow_offset;        // offset (doubles) within a thread slab
  };

  // Shadow offsets parallel to chunk_cells_ and row_cells_.  -1 means
  // "no shadow; write direct to real LHS".  (NoEBlock-row slots in
  // row_cells_ are always -1 since NoEBlockRowOuterProduct runs
  // serially.)
  std::vector<int> chunk_cells_shadow_offset_;
  std::vector<int> row_cells_shadow_offset_;

  // Reduce table.  shadow_reduce_table_[e] describes one LHS block that
  // accumulates contributions across all threads via
  // shadow_lhs_[t * shadow_doubles_per_thread_ + e.shadow_offset ..
  //              + block_rows*block_cols).
  std::vector<ShadowReduceEntry> shadow_reduce_table_;

  // Per-thread accumulation slabs, concatenated.  Total size =
  // shadow_doubles_per_thread_ * num_threads_.  Zeroed at the top of
  // each Eliminate().
  std::vector<double> shadow_lhs_;
  int shadow_doubles_per_thread_;
#endif
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SCHUR_ELIMINATOR_H_
