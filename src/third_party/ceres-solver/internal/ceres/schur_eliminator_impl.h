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
// TODO(sameeragarwal): row_block_counter can perhaps be replaced by
// Chunk::start ?

#ifndef CERES_INTERNAL_SCHUR_ELIMINATOR_IMPL_H_
#define CERES_INTERNAL_SCHUR_ELIMINATOR_IMPL_H_

// Eigen has an internal threshold switching between different matrix
// multiplication algorithms. In particular for matrices larger than
// EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD it uses a cache friendly
// matrix matrix product algorithm that has a higher setup cost. For
// matrix sizes close to this threshold, especially when the matrices
// are thin and long, the default choice may not be optimal. This is
// the case for us, as the default choice causes a 30% performance
// regression when we moved from Eigen2 to Eigen3.

#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 10

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/port.h"

#ifdef CERES_USE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <map>
#include "ceres/block_random_access_matrix.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/fixed_array.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/invert_psd_matrix.h"
#include "ceres/map_util.h"
#include "ceres/schur_eliminator.h"
#include "ceres/small_blas.h"
#include "ceres/stl_util.h"
#include "Eigen/Dense"
#include "glog/logging.h"

namespace ceres {
namespace internal {

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::~SchurEliminator() {
  _aligned_free(buffer_);
  _aligned_free(chunk_outer_product_buffer_);
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::Init(
    int num_eliminate_blocks,
    bool assume_full_rank_ete,
    const CompressedRowBlockStructure* bs) {
  CHECK_GT(num_eliminate_blocks, 0)
      << "SchurComplementSolver cannot be initialized with "
      << "num_eliminate_blocks = 0.";

  num_eliminate_blocks_ = num_eliminate_blocks;
  assume_full_rank_ete_ = assume_full_rank_ete;

  const int num_col_blocks = bs->cols.size();
  const int num_row_blocks = bs->rows.size();

  buffer_size_ = 1;
  chunks_.clear();
  // Upper bound: every row could start its own chunk.  Avoids the
  // log2(N) reallocations that push_back would otherwise incur as
  // chunks_ grows.
  chunks_.reserve(num_row_blocks);
  lhs_row_layout_.clear();

  int lhs_num_rows = 0;
  // Add a map object for each block in the reduced linear system
  // and build the row/column block structure of the reduced linear
  // system.
  lhs_row_layout_.resize(num_col_blocks - num_eliminate_blocks_);
  for (int i = num_eliminate_blocks_; i < num_col_blocks; ++i) {
    lhs_row_layout_[i - num_eliminate_blocks_] = lhs_num_rows;
    lhs_num_rows += bs->cols[i].size;
  }

  int r = 0;
  // Iterate over the row blocks of A, and detect the chunks. The
  // matrix should already have been ordered so that all rows
  // containing the same y block are vertically contiguous. Along
  // the way also compute the amount of space each chunk will need
  // to perform the elimination.
  while (r < num_row_blocks) {
    const int chunk_block_id = bs->rows[r].cells.front().block_id;
    if (chunk_block_id >= num_eliminate_blocks_) {
      break;
    }

    chunks_.push_back(Chunk());
    Chunk& chunk = chunks_.back();
    chunk.size = 0;
    chunk.start = r;
    // Typical SfM chunks have 2-3 distinct f-blocks. Reserving 4
    // dodges the 0->1 / 1->2 / 2->4 grow sequence on every chunk.
    chunk.buffer_layout.reserve(4);
    int buffer_size = 0;
    const int e_block_size = bs->cols[chunk_block_id].size;

    // Add to the chunk until the first block in the row is
    // different than the one in the first row for the chunk.
    while (r + chunk.size < num_row_blocks) {
      const CompressedRow& row = bs->rows[r + chunk.size];
      if (row.cells.front().block_id != chunk_block_id) {
        break;
      }

      // Iterate over the blocks in the row, ignoring the first
      // block since it is the one to be eliminated.
      const int num_cells = static_cast<int>(row.cells.size());
      for (int c = 1; c < num_cells; ++c) {
        const Cell& cell = row.cells[c];
        // Linear scan insert � buffer_layout is tiny (typically 2-3 entries per chunk in SfM)
        bool found = false;
        for (const auto& entry : chunk.buffer_layout) {
          if (entry.first == cell.block_id) {
            found = true;
            break;
          }
        }
        if (!found) {
          chunk.buffer_layout.push_back({cell.block_id, buffer_size});
          buffer_size += e_block_size * bs->cols[cell.block_id].size;
        }
      }

      buffer_size_ = std::max(buffer_size, buffer_size_);
      ++chunk.size;
    }

    CHECK_GT(chunk.size, 0);
    // Sort by block_id to maintain the sorted iteration order that
    // ChunkOuterProduct requires (block1 <= block2 for upper triangular access).
    std::sort(chunk.buffer_layout.begin(), chunk.buffer_layout.end());
    // Precompute the high-water mark of `buffer_` actually touched by
    // this chunk so EliminateChunks can memset just that prefix.
    int chunk_buffer_used = 0;
    for (const auto& entry : chunk.buffer_layout) {
      const int end = entry.second
          + e_block_size * bs->cols[entry.first].size;
      chunk_buffer_used = std::max(chunk_buffer_used, end);
    }
    chunk.buffer_used = chunk_buffer_used;
    r += chunk.size;
  }
  const Chunk& chunk = chunks_.back();

  uneliminated_row_begins_ = chunk.start + chunk.size;

  // Contention-aware chunk reordering for multi-threaded elimination.
  //
  // Two chunks contend when they share f-blocks (cameras): their
  // EBlockRowOuterProduct / ChunkOuterProduct / UpdateRhs calls will
  // compete for the same LHS cells and RHS locks.  In the natural
  // (row) order, spatially nearby 3D points observe the same cameras,
  // so consecutive chunks have high f-block overlap.  With
  // schedule(dynamic, 256) consecutive batches go to different
  // threads, maximising contention.
  //
  // Instead, we bucket chunks by their primary f-block (the first
  // entry in buffer_layout � the lowest-numbered camera), then
  // interleave the buckets so that consecutive chunks in the
  // iteration order are likely to touch *different* LHS cells.
  // This is O(N) and preserves spatial locality within each bucket.
  if (num_threads_ > 1 && chunks_.size() > 1) {
    const int num_f_blocks = num_col_blocks - num_eliminate_blocks_;
    // Bucket by primary f-block id (relative to eliminate blocks).
    std::vector<std::vector<int>> buckets(num_f_blocks);
    // Chunks with no f-blocks have nothing to contend on; park them
    // at the tail of the reordered list rather than biasing bucket 0.
    std::vector<int> empty_chunks;
    const int num_chunks = static_cast<int>(chunks_.size());
    for (int i = 0; i < num_chunks; ++i) {
      if (!chunks_[i].buffer_layout.empty()) {
        const int primary = chunks_[i].buffer_layout.front().first
                            - num_eliminate_blocks_;
        buckets[primary].push_back(i);
      } else {
        empty_chunks.push_back(i);
      }
    }

    // Round-robin interleave: pick one chunk from each non-empty
    // bucket in turn, so adjacent chunks in the final order are
    // unlikely to share their primary camera.
    std::vector<Chunk> reordered;
    reordered.reserve(chunks_.size());
    std::vector<int> bucket_pos(num_f_blocks, 0);
    bool progress = true;
    while (progress) {
      progress = false;
      for (int b = 0; b < num_f_blocks; ++b) {
        if (bucket_pos[b] < static_cast<int>(buckets[b].size())) {
          reordered.push_back(std::move(chunks_[buckets[b][bucket_pos[b]]]));
          ++bucket_pos[b];
          progress = true;
        }
      }
    }
    for (int i : empty_chunks) {
      reordered.push_back(std::move(chunks_[i]));
    }
    chunks_ = std::move(reordered);
  }

  _aligned_free(buffer_);
  buffer_ = (double*) _aligned_malloc(buffer_size_ * sizeof(double) * num_threads_, 64);

  // chunk_outer_product_buffer_ only needs to store e_block_size *
  // f_block_size, which is always less than buffer_size_, so we just
  // allocate buffer_size_ per thread.
  _aligned_free(chunk_outer_product_buffer_);
  chunk_outer_product_buffer_ = (double*) _aligned_malloc(buffer_size_ * sizeof(double) * num_threads_, 64);

  // Mutex is non-copyable/non-movable, so we can't use std::vector here.
  // scoped_array default-constructs each Mutex in place.
  rhs_locks_.reset(new Mutex[num_col_blocks - num_eliminate_blocks_]);

#if CERES_SCHUR_SHARDED_SPINLOCKS
  // T1: allocate the spinlock banks.  Each CeresSchurSpinlock is
  // cache-line aligned, so this gives one full cache line per slot.
  cell_spinlocks_.reset(new CeresSchurSpinlock[kCeresSchurCellSpinlockBank]);
  rhs_spinlocks_.reset(
      new CeresSchurSpinlock[num_col_blocks - num_eliminate_blocks_]);
#endif

#if CERES_SCHUR_CACHE_INVERSE_ETE
  // Size the per-chunk inverse(E'E + D'D) cache.  Each chunk's
  // e_block_size is bs->cols[ bs->rows[chunk.start].cells.front().block_id ].size.
  {
    const int num_chunks = static_cast<int>(chunks_.size());
    chunk_inverse_ete_cache_offset_.assign(num_chunks + 1, 0);
    int total = 0;
    for (int i = 0; i < num_chunks; ++i) {
      const Chunk& chunk = chunks_[i];
      const int e_block_id = bs->rows[chunk.start].cells.front().block_id;
      const int e_block_size = bs->cols[e_block_id].size;
      chunk_inverse_ete_cache_offset_[i] = total;
      total += e_block_size * e_block_size;
    }
    chunk_inverse_ete_cache_offset_[num_chunks] = total;
    chunk_inverse_ete_cache_storage_.assign(total, 0.0);
    chunk_inverse_ete_cache_valid_ = false;
  }
#endif

#if CERES_SCHUR_CACHE_CELL_LOOKUPS
  // Invalidate any prior GetCell cache; the chunk structure may have
  // changed.  The cache will be rebuilt lazily on the next Eliminate()
  // call that has an lhs in hand.
  cached_lhs_for_cells_ = nullptr;
  chunk_cells_.clear();
  chunk_cells_offset_.clear();
  row_cells_.clear();
  row_cells_offset_.clear();
  diag_cells_.clear();
#endif

#if CERES_SCHUR_PER_THREAD_LHS
  // T2: clear shadow state.  Will be populated lazily by BuildCellCaches.
  chunk_cells_shadow_offset_.clear();
  row_cells_shadow_offset_.clear();
  shadow_reduce_table_.clear();
  shadow_lhs_.clear();
  shadow_doubles_per_thread_ = 0;
#endif
}

#if CERES_SCHUR_CACHE_CELL_LOOKUPS
// Walk the structure of A and the chunk layout in the SAME order as
// ChunkOuterProduct + EBlockRowOuterProduct + NoEBlockRowOuterProduct
// do, recording the result of each lhs->GetCell() call into a flat
// per-chunk / per-row cache.  Called from Eliminate() the first time
// (and whenever the caller passes a different lhs pointer).
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
BuildCellCaches(BlockRandomAccessMatrix* lhs, const BlockSparseMatrix* A) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const int num_chunks = static_cast<int>(chunks_.size());
  const int num_row_blocks = static_cast<int>(bs->rows.size());

  // ---- ChunkOuterProduct cache --------------------------------------
  chunk_cells_offset_.assign(num_chunks + 1, 0);
  int total_chunk = 0;
  for (int c = 0; c < num_chunks; ++c) {
    const int bl = static_cast<int>(chunks_[c].buffer_layout.size());
    chunk_cells_offset_[c] = total_chunk;
    total_chunk += bl * (bl + 1) / 2;
  }
  chunk_cells_offset_[num_chunks] = total_chunk;
  chunk_cells_.assign(total_chunk, CachedCellInfo{nullptr, 0, 0, 0, 0});

  for (int c = 0; c < num_chunks; ++c) {
    const Chunk& chunk = chunks_[c];
    const auto* bl_data = chunk.buffer_layout.data();
    const int bl_size = static_cast<int>(chunk.buffer_layout.size());
    int pos = chunk_cells_offset_[c];
    for (int idx1 = 0; idx1 < bl_size; ++idx1) {
      const int block1 = bl_data[idx1].first - num_eliminate_blocks_;
      for (int idx2 = idx1; idx2 < bl_size; ++idx2, ++pos) {
        const int block2 = bl_data[idx2].first - num_eliminate_blocks_;
        CachedCellInfo& slot = chunk_cells_[pos];
        slot.info = lhs->GetCell(block1, block2,
                                 &slot.r, &slot.c,
                                 &slot.row_stride, &slot.col_stride);
      }
    }
  }

  // ---- [No]EBlockRowOuterProduct cache ------------------------------
  // One flat table indexed by absolute row_block_index.  Two regimes:
  //   * For rows in eliminated chunks (row < uneliminated_row_begins_)
  //     the iteration in EBlockRowOuterProduct is i = 1..num_cells-1,
  //     and for each i: (diag i) then j = i+1..num_cells-1 (off-diag).
  //   * For rows >= uneliminated_row_begins_, NoEBlockRowOuterProduct
  //     iterates i = 0..num_cells-1, same inner pattern.
  // Rows that are not visited by either path are left with a zero-
  // length cache slice (their offset == next row's offset).
  row_cells_offset_.assign(num_row_blocks + 1, 0);

  auto row_count = [&](int r) -> int {
    const int num_cells = static_cast<int>(bs->rows[r].cells.size());
    int i_start;
    if (r < uneliminated_row_begins_) {
      i_start = 1;  // EBlock row -- skip the e-block at cells[0]
    } else {
      i_start = 0;  // NoEBlock row -- all cells participate
    }
    int count = 0;
    for (int i = i_start; i < num_cells; ++i) {
      ++count;                    // diagonal lookup
      count += (num_cells - 1 - i);  // off-diagonal lookups
    }
    return count;
  };

  int total_row = 0;
  for (int r = 0; r < num_row_blocks; ++r) {
    row_cells_offset_[r] = total_row;
    total_row += row_count(r);
  }
  row_cells_offset_[num_row_blocks] = total_row;
  row_cells_.assign(total_row, CachedCellInfo{nullptr, 0, 0, 0, 0});

  for (int r = 0; r < num_row_blocks; ++r) {
    const CompressedRow& row = bs->rows[r];
    const int num_cells = static_cast<int>(row.cells.size());
    const int i_start = (r < uneliminated_row_begins_) ? 1 : 0;
    int pos = row_cells_offset_[r];
    for (int i = i_start; i < num_cells; ++i) {
      const int block1 = row.cells[i].block_id - num_eliminate_blocks_;
      {
        CachedCellInfo& slot = row_cells_[pos++];
        slot.info = lhs->GetCell(block1, block1,
                                 &slot.r, &slot.c,
                                 &slot.row_stride, &slot.col_stride);
      }
      for (int j = i + 1; j < num_cells; ++j) {
        const int block2 = row.cells[j].block_id - num_eliminate_blocks_;
        CachedCellInfo& slot = row_cells_[pos++];
        slot.info = lhs->GetCell(block1, block2,
                                 &slot.r, &slot.c,
                                 &slot.row_stride, &slot.col_stride);
      }
    }
  }

  // ---- Diagonal-add cache (Eliminate's `if (D != NULL)` loop) -----
  // One entry per f-block (i.e. one per column block past
  // num_eliminate_blocks_).  The diagonal (b, b) lookup is structurally
  // determined by the lhs sparsity; D's values do not affect which cell
  // we hit, so caching the lookup is bit-exact.
  {
    const int num_col_blocks = static_cast<int>(bs->cols.size());
    const int num_f_blocks = num_col_blocks - num_eliminate_blocks_;
    diag_cells_.assign(num_f_blocks, CachedCellInfo{nullptr, 0, 0, 0, 0});
    for (int b = 0; b < num_f_blocks; ++b) {
      CachedCellInfo& slot = diag_cells_[b];
      slot.info = lhs->GetCell(b, b,
                               &slot.r, &slot.c,
                               &slot.row_stride, &slot.col_stride);
    }
  }

#if CERES_SCHUR_PER_THREAD_LHS
  // ---- T2: build shadow offset tables + reduce table -------------
  // For every cache slot used by a LOCKED write (chunk_cells_ entries
  // and row_cells_ entries from EBlock rows), assign a slot within
  // each thread's shadow slab.  Cells with NULL info, and cache slots
  // for the serial NoEBlockRow path, get -1 (write goes direct to
  // real LHS).
  {
    chunk_cells_shadow_offset_.assign(chunk_cells_.size(), -1);
    row_cells_shadow_offset_.assign(row_cells_.size(), -1);
    shadow_reduce_table_.clear();
    shadow_doubles_per_thread_ = 0;

    // Dedupe key: same (CellInfo*, r, c) -> same shadow slot.  Two
    // cache positions that target the same LHS cell must share an
    // offset, else the reduce would double-count.  std::map of pair
    // works without custom hashing.
    typedef std::pair<CellInfo*, std::pair<int, int> > KeyT;
    std::map<KeyT, int> key_to_entry;

    // Helper: enroll a (cell, r, c, strides, br, bc) target and
    // return its shadow_offset (or -1 if info is NULL).
    auto enroll = [&](CellInfo* info, int r, int c,
                      int row_stride, int col_stride,
                      int br, int bc) -> int {
      if (info == NULL) return -1;
      KeyT key(info, std::make_pair(r, c));
      auto it = key_to_entry.find(key);
      if (it != key_to_entry.end()) {
        return shadow_reduce_table_[it->second].shadow_offset;
      }
      const int offset = shadow_doubles_per_thread_;
      ShadowReduceEntry e;
      e.dst_info = info;
      e.dst_r = r;
      e.dst_c = c;
      e.dst_row_stride = row_stride;
      e.dst_col_stride = col_stride;
      e.block_rows = br;
      e.block_cols = bc;
      e.shadow_offset = offset;
      const int idx = static_cast<int>(shadow_reduce_table_.size());
      shadow_reduce_table_.push_back(e);
      key_to_entry[key] = idx;
      shadow_doubles_per_thread_ += br * bc;
      return offset;
    };

    // Chunk-outer-product slots.  Walk in the same iteration order
    // as BuildCellCaches's chunk loop above, AND as
    // ChunkOuterProduct's runtime loop, so cache_pos lines up.
    for (int c = 0; c < num_chunks; ++c) {
      const Chunk& chunk = chunks_[c];
      const auto* bl_data = chunk.buffer_layout.data();
      const int bl_size = static_cast<int>(chunk.buffer_layout.size());
      int pos = chunk_cells_offset_[c];
      for (int idx1 = 0; idx1 < bl_size; ++idx1) {
        const int block1_id = bl_data[idx1].first;
        const int block1_size = bs->cols[block1_id].size;
        for (int idx2 = idx1; idx2 < bl_size; ++idx2, ++pos) {
          const int block2_id = bl_data[idx2].first;
          const int block2_size = bs->cols[block2_id].size;
          const CachedCellInfo& slot = chunk_cells_[pos];
          chunk_cells_shadow_offset_[pos] =
              enroll(slot.info, slot.r, slot.c,
                     slot.row_stride, slot.col_stride,
                     block1_size, block2_size);
        }
      }
    }

    // EBlock-row slots (rows with r < uneliminated_row_begins_).
    // The corresponding NoEBlock-row slots stay at -1 because
    // NoEBlockRowOuterProduct runs serially outside the parallel
    // region.
    for (int r2 = 0; r2 < num_row_blocks; ++r2) {
      if (r2 >= uneliminated_row_begins_) continue;
      const CompressedRow& row = bs->rows[r2];
      const int num_cells = static_cast<int>(row.cells.size());
      int pos = row_cells_offset_[r2];
      for (int i = 1; i < num_cells; ++i) {  // i_start = 1 for EBlock
        const int b1_size = bs->cols[row.cells[i].block_id].size;
        // diag (block1, block1)
        {
          const CachedCellInfo& slot = row_cells_[pos];
          row_cells_shadow_offset_[pos] =
              enroll(slot.info, slot.r, slot.c,
                     slot.row_stride, slot.col_stride,
                     b1_size, b1_size);
          ++pos;
        }
        // off-diag (block1, block2)
        for (int j = i + 1; j < num_cells; ++j) {
          const int b2_size = bs->cols[row.cells[j].block_id].size;
          const CachedCellInfo& slot = row_cells_[pos];
          row_cells_shadow_offset_[pos] =
              enroll(slot.info, slot.r, slot.c,
                     slot.row_stride, slot.col_stride,
                     b1_size, b2_size);
          ++pos;
        }
      }
    }

    // Allocate the per-thread slabs.  T2 lifts the kMaxSchurThreads
    // cap (per-cell contention is gone), so size for the
    // user-requested num_threads_.
    const int nth = std::max(1, num_threads_);
    shadow_lhs_.assign(
        static_cast<size_t>(shadow_doubles_per_thread_) * nth, 0.0);
  }
#endif  // CERES_SCHUR_PER_THREAD_LHS

  cached_lhs_for_cells_ = lhs;
}
#endif

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
Eliminate(const BlockSparseMatrix* A,
          const double* b,
          const double* D,
          BlockRandomAccessMatrix* lhs,
          double* rhs) {
  if (lhs->num_rows() > 0) {
    lhs->SetZero();
    VectorRef(rhs, lhs->num_rows()).setZero();
  }

#if CERES_SCHUR_CACHE_INVERSE_ETE
  // Invalidate the inverse(E'E) cache; EliminateChunks() will refill it.
  // BackSubstitute() must only consume the cache when this flag is true.
  chunk_inverse_ete_cache_valid_ = false;
#endif

#if CERES_SCHUR_CACHE_CELL_LOOKUPS
  // First call -- or first call with a different lhs object -- builds
  // the GetCell caches consumed by ChunkOuterProduct,
  // EBlockRowOuterProduct and NoEBlockRowOuterProduct.  Subsequent LM
  // iterations reuse the cached cell pointers and (r, c, stride)
  // tuples, eliminating millions of virtual + hash-map lookups per
  // solve.  Bit-exact: GetCell is a pure function of the lhs's
  // structural sparsity, which is fixed across a solve.
  if (cached_lhs_for_cells_ != lhs) {
    BuildCellCaches(lhs, A);
  }
#endif

  const CompressedRowBlockStructure* bs = A->block_structure();
  const int num_col_blocks = bs->cols.size();
  const Block* const __restrict col_blocks = bs->cols.data();
  // (No use of A->values() here -- it is fetched per-callee in
  // EliminateChunks / NoEBlockRowsUpdate where it is actually needed.)

  // Add the diagonal to the schur complement.
  if (D != NULL) {
    const double* const __restrict D_ptr = D;
    // Each iteration writes to a distinct diagonal cell (block_id i ->
    // GetCell(block_id, block_id)) so there is zero contention. The
    // per-iter work is small (one Eigen diagonal += square), so only
    // parallelise when there are enough iters to amortise the OMP
    // fork/join overhead.
    const int diag_iters = num_col_blocks - num_eliminate_blocks_;
    const int diag_threads = num_threads_;
#pragma omp parallel for num_threads(diag_threads) schedule(static) if (diag_threads > 1 && diag_iters >= 64)
    for (int i = num_eliminate_blocks_; i < num_col_blocks; ++i) {
      const int block_id = i - num_eliminate_blocks_;
#if CERES_SCHUR_CACHE_CELL_LOOKUPS
      // Reuse the cached structural lookup (built above in
      // BuildCellCaches when cached_lhs_for_cells_ first turned
      // non-null).  Bit-exact w.r.t. the un-cached path.
      const CachedCellInfo& cached_diag = diag_cells_[block_id];
      CellInfo* cell_info = cached_diag.info;
      const int r          = cached_diag.r;
      const int c          = cached_diag.c;
      const int row_stride = cached_diag.row_stride;
      const int col_stride = cached_diag.col_stride;
#else
      int r, c, row_stride, col_stride;
      CellInfo* cell_info = lhs->GetCell(block_id, block_id,
                                         &r, &c,
                                         &row_stride, &col_stride);
#endif
      if (cell_info != NULL) {
        const int block_size = col_blocks[i].size;
        typename EigenTypes<Eigen::Dynamic>::ConstVectorRef
            diag(D_ptr + col_blocks[i].position, block_size);

        MatrixRef m(cell_info->values, row_stride, col_stride);
        m.block(r, c, block_size, block_size).diagonal()
            += diag.array().square().matrix();
      }
    }
  }

  // Eliminate y blocks one chunk at a time.  For each chunk, compute
  // the entries of the normal equations and the gradient vector block
  // corresponding to the y block and then apply Gaussian elimination
  // to them. The matrix ete stores the normal matrix corresponding to
  // the block being eliminated and array buffer_ contains the
  // non-zero blocks in the row corresponding to this y block in the
  // normal equations. This computation is done in
  // ChunkDiagonalBlockAndGradient. UpdateRhs then applies gaussian
  // elimination to the rhs of the normal equations, updating the rhs
  // of the reduced linear system by modifying rhs blocks for all the
  // z blocks that share a row block/residual term with the y
  // block. EliminateRowOuterProduct does the corresponding operation
  // for the lhs of the reduced linear system.

  // The Schur elimination loop has heavy mutex contention in
  // ChunkOuterProduct, EBlockRowOuterProduct, and UpdateRhs.
  // On high-core-count machines, more threads cause more contention
  // than useful parallelism since the per-lock work is tiny.
  // Cap threads to avoid contention while preserving parallelism
  // on machines with fewer cores.
#if CERES_SCHUR_PER_THREAD_LHS
  // T2 eliminates the per-cell contention that motivated the cap;
  // use the user-requested thread count directly.
  int threadsToUse = std::max(1, num_threads_);
#else
  const int kMaxSchurThreads = 6; // 6 marginally better than 8 or 4 on big data.
  int threadsToUse = std::min(num_threads_, kMaxSchurThreads);
#endif

#if CERES_SCHUR_PER_THREAD_LHS
  // T2: zero the per-thread shadow slabs.  Each thread accumulates
  // its ChunkOuterProduct and EBlockRowOuterProduct contributions
  // here from scratch; the reduce below sums them into the real LHS.
  if (!shadow_lhs_.empty()) {
    memset(shadow_lhs_.data(), 0, shadow_lhs_.size() * sizeof(double));
  }
#endif

  // Dispatch to a template instantiation so the compiler can fully
  // eliminate lock code in the single-threaded path.
  if (threadsToUse > 1) {
    EliminateChunks<true>(A, b, D, lhs, rhs, threadsToUse);
  } else {
    EliminateChunks<false>(A, b, D, lhs, rhs, threadsToUse);
  }

#if CERES_SCHUR_PER_THREAD_LHS
  // T2: reduce per-thread shadow slabs into the real LHS.  Each table
  // entry writes to a distinct LHS region (deduped at build time), so
  // this parallelises without locks.  Reduce runs only when the
  // parallel path was taken; the single-thread EliminateChunks wrote
  // straight to the LHS (see ChunkOuterProduct / EBlockRowOuterProduct
  // dispatch on kNeedsLocking below).
  if (threadsToUse > 1 && !shadow_reduce_table_.empty()) {
    const int nth = threadsToUse;
    const int dpt = shadow_doubles_per_thread_;
    const int num_entries = static_cast<int>(shadow_reduce_table_.size());
    const ShadowReduceEntry* const __restrict tab =
        shadow_reduce_table_.data();
    const double* const __restrict slab = shadow_lhs_.data();
#pragma omp parallel for num_threads(nth) schedule(static) if (nth > 1 && num_entries >= 64)
    for (int e = 0; e < num_entries; ++e) {
      const ShadowReduceEntry& ent = tab[e];
      double* const __restrict dst = ent.dst_info->values;
      const int br = ent.block_rows;
      const int bc = ent.block_cols;
      const int rs = ent.dst_row_stride;
      const int dr = ent.dst_r;
      const int dc = ent.dst_c;
      const int off = ent.shadow_offset;
      for (int t = 0; t < nth; ++t) {
        const double* const __restrict src =
            slab + static_cast<size_t>(t) * dpt + off;
        // Compact br x bc col-major block in src; (dr, dc) block of
        // a (rs x dst_col_stride) col-major matrix at dst.
        for (int j = 0; j < bc; ++j) {
          const int dst_col_off = (dc + j) * rs + dr;
          const int src_col_off = j * br;
          for (int i = 0; i < br; ++i) {
            dst[dst_col_off + i] += src[src_col_off + i];
          }
        }
      }
    }
  }
#endif

#if CERES_SCHUR_CACHE_INVERSE_ETE
  // EliminateChunks() has populated chunk_inverse_ete_cache_storage_ for
  // every chunk; signal BackSubstitute() that the cache is consumable.
  chunk_inverse_ete_cache_valid_ = true;
#endif

  // For rows with no e_blocks, the schur complement update reduces to
  // S += F'F.
  NoEBlockRowsUpdate(A, b,  uneliminated_row_begins_, lhs, rhs);
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
template <bool kNeedsLocking>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
EliminateChunks(const BlockSparseMatrix* A,
                const double* b,
                const double* D,
                BlockRandomAccessMatrix* lhs,
                double* rhs,
                int threadsToUse) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const Block* const __restrict col_blocks = bs->cols.data();

  // Static scheduling, paired with the round-robin chunk reordering done
  // in Init(). The reordering interleaves chunks that share cameras across
  // distinct threads, so a static carve-up already distributes contended
  // LHS cells well -- and avoids the per-iteration scheduling overhead
  // that dynamic would impose. Do NOT switch to schedule(dynamic) without
  // also reverting the chunk reordering: they are a paired design.
#pragma omp parallel for num_threads(threadsToUse) schedule(static) if (kNeedsLocking)
  for (int i = 0; i < chunks_.size(); ++i) {
#ifdef CERES_USE_OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    double* __restrict buffer = buffer_ + thread_id * buffer_size_;
    const Chunk& chunk = chunks_[i];
    const int e_block_id = bs->rows[chunk.start].cells.front().block_id;
    const int e_block_size = col_blocks[e_block_id].size;

    // Zero only the prefix of `buffer` actually used by this chunk
    // (precomputed in Init).  Saves a rescan of buffer_layout per call.
    if (chunk.buffer_used > 0) {
      memset(buffer, 0, chunk.buffer_used * sizeof(double));
    }

    typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix
        ete(e_block_size, e_block_size);

    if (D != NULL) {
      const typename EigenTypes<kEBlockSize>::ConstVectorRef
          diag(D + col_blocks[e_block_id].position, e_block_size);
      ete = diag.array().square().matrix().asDiagonal();
    } else {
      ete.setZero();
    }

    FixedArray<double, 8> g(e_block_size);
    memset(g.get(), 0, e_block_size * sizeof(double));

    // We are going to be computing
    //
    //   S += F'F - F'E(E'E)^{-1}E'F
    //
    // for each Chunk. The computation is broken down into a number of
    // function calls as below.

    // Compute the outer product of the e_blocks with themselves (ete
    // = E'E). Compute the product of the e_blocks with the
    // corresponding f_blocks (buffer = E'F), the gradient of the terms
    // in this chunk (g) and add the outer product of the f_blocks to
    // Schur complement (S += F'F).
    ChunkDiagonalBlockAndGradient<kNeedsLocking>(
        chunk, A, b, chunk.start, &ete, g.get(), buffer, lhs);

    // Normally one wouldn't compute the inverse explicitly, but
    // e_block_size will typically be a small number like 3, in
    // which case its much faster to compute the inverse once and
    // use it to multiply other matrices/vectors instead of doing a
    // Solve call over and over again.
    typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix inverse_ete =
        InvertPSDMatrix<kEBlockSize>(assume_full_rank_ete_, ete);

#if CERES_SCHUR_CACHE_INVERSE_ETE
    // Stash the inverse so BackSubstitute() can skip the per-row E'E
    // accumulation and the InvertPSDMatrix() call entirely.
    {
      const int n2 = e_block_size * e_block_size;
      double* __restrict dst = chunk_inverse_ete_cache_storage_.data()
          + chunk_inverse_ete_cache_offset_[i];
      memcpy(dst, inverse_ete.data(), n2 * sizeof(double));
    }
#endif

    // For the current chunk compute and update the rhs of the reduced
    // linear system.
    //
    //   rhs = F'b - F'E(E'E)^(-1) E'b
    if (rhs) {
      FixedArray<double, 8> inverse_ete_g(e_block_size);
      MatrixVectorMultiply<kEBlockSize, kEBlockSize, 0>(
        inverse_ete.data(),
        e_block_size,
        e_block_size,
        g.get(),
        inverse_ete_g.get());

      UpdateRhs<kNeedsLocking>(chunk, A, b, chunk.start, inverse_ete_g.get(), rhs);
    }

    // S -= F'E(E'E)^{-1}E'F
    ChunkOuterProduct<kNeedsLocking>(bs, inverse_ete, buffer, chunk.buffer_layout, i, lhs);
  }
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
BackSubstitute(const BlockSparseMatrix* A,
               const double* b,
               const double* D,
               const double* z,
               double* y) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const Block* const __restrict col_blocks = bs->cols.data();
  const int* const __restrict lhs_layout = lhs_row_layout_.data();
  const double* const __restrict values = A->values();
  // Alias z/y as __restrict so the compiler may assume they do not
  // alias each other or the matrix values within the inner loops.
  // Caller contract (SchurComplementSolver) always passes distinct
  // allocations for the rhs read (z), the matrix values, and the
  // back-substitution write target (y).
  const double* const __restrict z_ptr = z;

  // BackSubstitute has no mutex contention (each chunk writes to its
  // own y block), so use all available threads.
  int threadsToUse = num_threads_;
#if CERES_SCHUR_CACHE_INVERSE_ETE
  // When the cache is valid, BackSubstitute can completely skip the
  // per-row E'E += accumulation and the InvertPSDMatrix call.  This
  // is bit-exact: the matrix being inverted is identical between
  // Eliminate() and BackSubstitute() within an LM step.
  const bool kUseInverseEteCache = chunk_inverse_ete_cache_valid_;
#else
  const bool kUseInverseEteCache = false;
#endif
#pragma omp parallel for num_threads(threadsToUse) schedule(static) if (threadsToUse > 1)
  for (int i = 0; i < chunks_.size(); ++i) {
    const Chunk& chunk = chunks_[i];
    const int e_block_id = bs->rows[chunk.start].cells.front().block_id;
    const int e_block_size = col_blocks[e_block_id].size;

    double* __restrict y_ptr = y + col_blocks[e_block_id].position;
    typename EigenTypes<kEBlockSize>::VectorRef y_block(y_ptr, e_block_size);

    typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix
        ete(e_block_size, e_block_size);

    if (!kUseInverseEteCache) {
      if (D != NULL) {
        const typename EigenTypes<kEBlockSize>::ConstVectorRef
            diag(D + col_blocks[e_block_id].position, e_block_size);
        ete = diag.array().square().matrix().asDiagonal();
      } else {
        ete.setZero();
      }
    }

    // Allocate sj once outside the loop; all rows in a chunk share
    // the same row block size, so we reuse the buffer across iterations.
    // do_init=0: skip the value-init (zero-fill) loop in the FixedArray
    // ctor and the matching Destroy loop in the dtor. The very first use
    // of `sj` inside the row loop below is a wholesale `memcpy` over the
    // full sj_size range, so the init writes would just be overwritten.
    const int sj_size = bs->rows[chunk.start].block.size;
    FixedArray<double, 8, 0> sj(sj_size);

    for (int j = 0; j < chunk.size; ++j) {
      const CompressedRow& row = bs->rows[chunk.start + j];
      const Cell& e_cell = row.cells.front();
      DCHECK_EQ(e_block_id, e_cell.block_id);
      // Snapshot loop-invariant row.block.size and e_cell.position so
      // they survive across the MatrixVectorMultiply / MTMM calls in
      // the inner loop without MSVC re-issuing the loads.
      const int row_block_size = row.block.size;
      const int e_cell_position = e_cell.position;

      memcpy(sj.get(),
             b + bs->rows[chunk.start + j].block.position,
             sj_size * sizeof(double));

      // Walk cells with pointer arithmetic instead of cells[c] indexing
      const Cell* __restrict cell_ptr = row.cells.data() + 1;
      const Cell* const cell_end = row.cells.data() + row.cells.size();
      for (; cell_ptr < cell_end; ++cell_ptr) {
        const int f_block_id = cell_ptr->block_id;
        const int f_block_size = col_blocks[f_block_id].size;
        const int r_block = f_block_id - num_eliminate_blocks_;

        MatrixVectorMultiply<kRowBlockSize, kFBlockSize, -1>(
            values + cell_ptr->position, row_block_size, f_block_size,
            z_ptr + lhs_layout[r_block],
            sj.get());
      }

      MatrixTransposeVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
          values + e_cell_position, row_block_size, e_block_size,
          sj.get(),
          y_ptr);

      if (!kUseInverseEteCache) {
        // Symmetric A'A: see ChunkDiagonalBlockAndGradient.
        MatrixTransposeMatrixMultiplySelf
            <kRowBlockSize, kEBlockSize, 1>(
                values + e_cell_position, row_block_size, e_block_size,
                ete.data(), 0, 0, e_block_size, e_block_size);
      }
    }

#if CERES_SCHUR_CACHE_INVERSE_ETE
    if (kUseInverseEteCache) {
      // Reuse the inverse computed during Eliminate().  The stored
      // bytes are the .data() of an Eigen ColMajor matrix of size
      // e_block_size x e_block_size.
      const double* const __restrict inv_ete_data =
          chunk_inverse_ete_cache_storage_.data()
              + chunk_inverse_ete_cache_offset_[i];
#if CERES_SCHUR_BACKSUB_INPLACE
      // S2: in-place mat-vec via small_blas.  Read the e_block_size
      // entries of y_block into a stack-resident FixedArray, run the
      // kEBlockSize-templated MatrixVectorMultiply (kOperation = 0
      // i.e. plain assign), and copy the result back.  This avoids
      // both (a) an Eigen heap allocation on the kEBlockSize=Dynamic
      // path and (b) the redundant temp-matrix copy on the static path.
      FixedArray<double, 8> y_in(e_block_size);
      FixedArray<double, 8> y_out(e_block_size);
      memcpy(y_in.get(), y_ptr, e_block_size * sizeof(double));
      MatrixVectorMultiply<kEBlockSize, kEBlockSize, 0>(
          inv_ete_data, e_block_size, e_block_size,
          y_in.get(), y_out.get());
      memcpy(y_ptr, y_out.get(), e_block_size * sizeof(double));
#else
      typedef typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix EteMat;
      Eigen::Map<const EteMat> inv_ete(
          inv_ete_data, e_block_size, e_block_size);
      typename EigenTypes<kEBlockSize>::Matrix y_tmp = inv_ete * y_block;
      y_block = y_tmp;
#endif
    } else {
      y_block = InvertPSDMatrix<kEBlockSize>(assume_full_rank_ete_, ete)
          * y_block;
    }
#else
    y_block = InvertPSDMatrix<kEBlockSize>(assume_full_rank_ete_, ete)
        * y_block;
#endif
  }
}

// Update the rhs of the reduced linear system. Compute
//
//   F'b - F'E(E'E)^(-1) E'b
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
template <bool kNeedsLocking>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
UpdateRhs(const Chunk& chunk,
          const BlockSparseMatrix* A,
          const double* b,
          int row_block_counter,
          const double* inverse_ete_g,
          double* rhs) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const Block* const __restrict col_blocks = bs->cols.data();
  const int* const __restrict lhs_layout = lhs_row_layout_.data();
  const int e_block_id = bs->rows[chunk.start].cells.front().block_id;
  const int e_block_size = col_blocks[e_block_id].size;

  int b_pos = bs->rows[row_block_counter].block.position;
  const double* const __restrict values = A->values();
  // rhs is the only mutable pointer in the inner loop; alias it as
  // __restrict so the compiler can assume it does not overlap with
  // `values` or `inverse_ete_g`.
  double* const __restrict rhs_ptr = rhs;
  for (int j = 0; j < chunk.size; ++j) {
    const CompressedRow& row = bs->rows[row_block_counter + j];
    const Cell& e_cell = row.cells.front();
    // Hoist row.block.size and e_cell.position so MSVC doesn't reload
    // them across the opaque CeresMutexLock ctor/dtor in the inner loop.
    const int row_block_size = row.block.size;
    const int e_cell_position = e_cell.position;

    typename EigenTypes<kRowBlockSize>::Vector sj =
        typename EigenTypes<kRowBlockSize>::ConstVectorRef
        (b + b_pos, row_block_size);

    MatrixVectorMultiply<kRowBlockSize, kEBlockSize, -1>(
        values + e_cell_position, row_block_size, e_block_size,
        inverse_ete_g, sj.data());

    const Cell* __restrict cell_ptr = row.cells.data() + 1;
    const Cell* const cell_end = row.cells.data() + row.cells.size();
    for (; cell_ptr < cell_end; ++cell_ptr) {
      const int block_id = cell_ptr->block_id;
      const int block_size = col_blocks[block_id].size;
      const int block = block_id - num_eliminate_blocks_;
      if (kNeedsLocking) {
#if CERES_SCHUR_SHARDED_SPINLOCKS
        CeresSchurSpinLockGuard l(&rhs_spinlocks_[block].flag);
#else
        CeresMutexLock l(&rhs_locks_[block]);
#endif
        MatrixTransposeVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
            values + cell_ptr->position,
            row_block_size, block_size,
            sj.data(), rhs_ptr + lhs_layout[block]);
      } else {
        MatrixTransposeVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
            values + cell_ptr->position,
            row_block_size, block_size,
            sj.data(), rhs_ptr + lhs_layout[block]);
      }
    }
    b_pos += row_block_size;
  }
}

// Given a Chunk - set of rows with the same e_block, e.g. in the
//
//                E                   F
//      [ y11   0   0   0 |  z11     0    0   0    z51]
//      [ y12   0   0   0 |  z12   z22    0   0      0]
//
// this function computes twp matrices. The diagonal block matrix
//
//   ete = y11 * y11' + y12 * y12'
//
// and the off diagonal blocks in the Guass Newton Hessian.
//
//   buffer = [y11'(z11 + z12), y12' * z22, y11' * z51]
//
// which are zero compressed versions of the block sparse matrices E'E
// and E'F.
//
// and the gradient of the e_block, E'b.
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
template <bool kNeedsLocking>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
ChunkDiagonalBlockAndGradient(
    const Chunk& chunk,
    const BlockSparseMatrix* A,
    const double* b,
    int row_block_counter,
    typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix* ete,
    double* g,
    double* __restrict buffer,
    BlockRandomAccessMatrix* lhs) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const Block* const __restrict col_blocks = bs->cols.data();

  int b_pos = bs->rows[row_block_counter].block.position;
  const int e_block_size = ete->rows();

  // Iterate over the rows in this chunk, for each row, compute the
  // contribution of its F blocks to the Schur supplement, the
  // contribution of its E block to the matrix EE' (ete), and the
  // corresponding block in the gradient vector.
  const double* const __restrict values = A->values();
  const auto* const __restrict bl_data = chunk.buffer_layout.data();
  const int bl_size = static_cast<int>(chunk.buffer_layout.size());

  for (int j = 0; j < chunk.size; ++j) {
    const CompressedRow& row = bs->rows[row_block_counter + j];
    // Snapshot row.block.size once per row -- EBlockRowOuterProduct
    // (called below) does virtual lhs->GetCell() dispatches that
    // would otherwise force MSVC to reload it inside the cell loop.
    const int row_block_size = row.block.size;

    if (row.cells.size() > 1) {
      EBlockRowOuterProduct<kNeedsLocking>(A, row_block_counter + j, lhs);
    }

    // Extract the e_block, ETE += E_i' E_i
    const Cell& e_cell = row.cells.front();
    const double* const __restrict e_values = values + e_cell.position;

    // Symmetric A'A: half the FLOPs of the general MTMM (gated on
    // CERES_SMALL_BLAS_AXPY; falls back to MTMM otherwise).
    MatrixTransposeMatrixMultiplySelf
        <kRowBlockSize, kEBlockSize, 1>(
            e_values, row_block_size, e_block_size,
            ete->data(), 0, 0, e_block_size, e_block_size);

    // g += E_i' b_i
    MatrixTransposeVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
        e_values, row_block_size, e_block_size,
        b + b_pos,
        g);

    // buffer = E'F. Both cells and buffer_layout are sorted by
    // block_id, so we can walk them together (merge-scan) instead
    // of restarting the linear scan for each cell.
    int bl_hint = 0;
    const Cell* __restrict cell_ptr = row.cells.data() + 1;
    const Cell* const cell_end = row.cells.data() + row.cells.size();
    for (; cell_ptr < cell_end; ++cell_ptr) {
      const int f_block_id = cell_ptr->block_id;
      const int f_block_size = col_blocks[f_block_id].size;
      // Advance the hint � since both sequences are sorted, we only
      // need to scan forward from where we left off.
      while (bl_hint < bl_size && bl_data[bl_hint].first < f_block_id) {
        ++bl_hint;
      }
      double* __restrict buffer_ptr = buffer + bl_data[bl_hint].second;
      MatrixTransposeMatrixMultiply
          <kRowBlockSize, kEBlockSize, kRowBlockSize, kFBlockSize, 1>(
          e_values, row_block_size, e_block_size,
          values + cell_ptr->position, row_block_size, f_block_size,
          buffer_ptr, 0, 0, e_block_size, f_block_size);
    }
    b_pos += row_block_size;
  }
}

// Schur complement matrix, i.e
//
//  S -= F'E(E'E)^{-1}E'F.
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
template <bool kNeedsLocking>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
ChunkOuterProduct(const CompressedRowBlockStructure* bs,
                  const typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix& inverse_ete,
                  const double* __restrict buffer,
                  const BufferLayoutType& buffer_layout,
                  int chunk_idx,
                  BlockRandomAccessMatrix* lhs) {
  // This is the most computationally expensive part of this
  // code. Profiling experiments reveal that the bottleneck is not the
  // computation of the right-hand matrix product, but memory
  // references to the left hand side.
  const int e_block_size = inverse_ete.rows();
  const Block* const __restrict col_blocks = bs->cols.data();

#ifdef CERES_USE_OPENMP
  int thread_id = omp_get_thread_num();
#else
  int thread_id = 0;
#endif
  double* __restrict b1_transpose_inverse_ete =
      chunk_outer_product_buffer_ + thread_id * buffer_size_;

  const auto* const __restrict bl_data = buffer_layout.data();
  const size_t bl_size = buffer_layout.size();

#if CERES_SCHUR_CACHE_CELL_LOOKUPS
  // Cached cell-info slice for this chunk (P1).  The iteration order
  // below matches the order used in BuildCellCaches() exactly: idx2
  // inner, idx1 outer, upper triangle inclusive of diagonal.
  const CachedCellInfo* const __restrict cell_cache =
      chunk_cells_.data() + chunk_cells_offset_[chunk_idx];
  int cache_pos = 0;
#endif

#if CERES_SCHUR_PER_THREAD_LHS
  // T2: redirect locked writes to this thread's shadow slab.
  const int* const __restrict cell_shadow_off =
      chunk_cells_shadow_offset_.data() + chunk_cells_offset_[chunk_idx];
  const int shadow_dpt = shadow_doubles_per_thread_;
  double* const __restrict shadow_slab =
      shadow_lhs_.empty()
          ? nullptr
          : shadow_lhs_.data() +
                static_cast<size_t>(thread_id) * shadow_dpt;
#endif

  // S(i,j) -= bi' * ete^{-1} b_j
  for (size_t idx1 = 0; idx1 < bl_size; ++idx1) {
    const int block1_id = bl_data[idx1].first;
    const int block1_size = col_blocks[block1_id].size;
    MatrixTransposeMatrixMultiply
        <kEBlockSize, kFBlockSize, kEBlockSize, kEBlockSize, 0>(
        buffer + bl_data[idx1].second, e_block_size, block1_size,
        inverse_ete.data(), e_block_size, e_block_size,
        b1_transpose_inverse_ete, 0, 0, block1_size, e_block_size);

    for (size_t idx2 = idx1; idx2 < bl_size; ++idx2) {
      // Snapshot bl_data[idx2].first once.  block2_id is needed both
      // before the virtual lhs->GetCell() (to compute block2) and
      // after (to look up block2_size in col_blocks).  Hoisting it
      // saves MSVC from re-loading through bl_data across GetCell.
      const int block2_id = bl_data[idx2].first;

#if CERES_SCHUR_CACHE_CELL_LOOKUPS
      const int slot_idx = cache_pos++;
      const CachedCellInfo& cached = cell_cache[slot_idx];
      CellInfo* cell_info = cached.info;
      const int r          = cached.r;
      const int c          = cached.c;
      const int row_stride = cached.row_stride;
      const int col_stride = cached.col_stride;
#else
      const int block1 = block1_id - num_eliminate_blocks_;
      const int block2 = block2_id - num_eliminate_blocks_;
      int r, c, row_stride, col_stride;
      CellInfo* cell_info = lhs->GetCell(block1, block2,
                                         &r, &c,
                                         &row_stride, &col_stride);
#endif
      if (cell_info != NULL) {
        const int block2_size = col_blocks[block2_id].size;
        if (kNeedsLocking) {
#if CERES_SCHUR_PER_THREAD_LHS
          // Write into this thread's shadow slot for the target cell.
          // No lock needed -- each thread has its own slab.  The
          // reduce in Eliminate() sums all threads' shadows into the
          // real LHS after the parallel region.
          const int shadow_off = cell_shadow_off[slot_idx];
          double* const __restrict tgt = shadow_slab + shadow_off;
          MatrixMatrixMultiply
              <kFBlockSize, kEBlockSize, kEBlockSize, kFBlockSize, -1>(
                  b1_transpose_inverse_ete, block1_size, e_block_size,
                  buffer + bl_data[idx2].second, e_block_size, block2_size,
                  tgt, 0, 0, block1_size, block2_size);
#else
#if CERES_SCHUR_SHARDED_SPINLOCKS
          // Pointer-hashed bank: same cell -> same slot, so writes
          // to the SAME cell still serialize.  Shift by 5 because
          // CellInfo is allocated in arrays of >= 32-byte structs;
          // the low bits carry no entropy.
          const std::size_t h =
              (reinterpret_cast<std::uintptr_t>(cell_info) >> 5)
              & kCeresSchurCellSpinlockMask;
          CeresSchurSpinLockGuard l(&cell_spinlocks_[h].flag);
#else
          CeresMutexLock l(&cell_info->m);
#endif
          MatrixMatrixMultiply
              <kFBlockSize, kEBlockSize, kEBlockSize, kFBlockSize, -1>(
                  b1_transpose_inverse_ete, block1_size, e_block_size,
                  buffer + bl_data[idx2].second, e_block_size, block2_size,
                  cell_info->values, r, c, row_stride, col_stride);
#endif
        } else {
          MatrixMatrixMultiply
              <kFBlockSize, kEBlockSize, kEBlockSize, kFBlockSize, -1>(
                  b1_transpose_inverse_ete, block1_size, e_block_size,
                  buffer + bl_data[idx2].second, e_block_size, block2_size,
                  cell_info->values, r, c, row_stride, col_stride);
        }
      }
    }
  }
}

// For rows with no e_blocks, the schur complement update reduces to S
// += F'F. This function iterates over the rows of A with no e_block,
// and calls NoEBlockRowOuterProduct on each row.
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
NoEBlockRowsUpdate(const BlockSparseMatrix* A,
                   const double* b,
                   int row_block_counter,
                   BlockRandomAccessMatrix* lhs,
                   double* rhs) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const Block* const __restrict col_blocks = bs->cols.data();
  const int* const __restrict lhs_layout = lhs_row_layout_.data();
  const double* const __restrict values = A->values();
  // b/rhs are distinct caller-owned allocations; tell the compiler so.
  const double* const __restrict b_ptr = b;
  double* const __restrict rhs_ptr = rhs;
  const int num_rows = static_cast<int>(bs->rows.size());
  for (; row_block_counter < num_rows; ++row_block_counter) {
    const CompressedRow& row = bs->rows[row_block_counter];
    // Snapshot row.block fields once per row.  NoEBlockRowOuterProduct
    // (called below) does virtual lhs->GetCell() dispatches, so MSVC
    // would otherwise have to re-load row.block.size each time around
    // the cell loop.
    const int row_block_size = row.block.size;
    const int row_block_position = row.block.position;
    const Cell* __restrict cell_ptr = row.cells.data();
    const Cell* const cell_end = cell_ptr + row.cells.size();
    for (; cell_ptr < cell_end; ++cell_ptr) {
      const int block_id = cell_ptr->block_id;
      const int block_size = col_blocks[block_id].size;
      const int block = block_id - num_eliminate_blocks_;
      MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
          values + cell_ptr->position, row_block_size, block_size,
          b_ptr + row_block_position,
          rhs_ptr + lhs_layout[block]);
    }
    NoEBlockRowOuterProduct(A, row_block_counter, lhs);
  }
}

// A row r of A, which has no e_blocks gets added to the Schur
// Complement as S += r r'. This function is responsible for computing
// the contribution of a single row r to the Schur complement. It is
// very similar in structure to EBlockRowOuterProduct except for
// one difference. It does not use any of the template
// parameters. This is because the algorithm used for detecting the
// static structure of the matrix A only pays attention to rows with
// e_blocks. This is becasue rows without e_blocks are rare and
// typically arise from regularization terms in the original
// optimization problem, and have a very different structure than the
// rows with e_blocks. Including them in the static structure
// detection will lead to most template parameters being set to
// dynamic. Since the number of rows without e_blocks is small, the
// lack of templating is not an issue.
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
NoEBlockRowOuterProduct(const BlockSparseMatrix* A,
                        int row_block_index,
                        BlockRandomAccessMatrix* lhs) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const Block* const __restrict col_blocks = bs->cols.data();
  const CompressedRow& row = bs->rows[row_block_index];
  const double* const __restrict values = A->values();
  const int num_cells = static_cast<int>(row.cells.size());
  const Cell* const __restrict cells = row.cells.data();
  // row.block.size is invariant across both loops below; pull it into a
  // local so MSVC does not have to re-prove non-aliasing across the
  // virtual lhs->GetCell() calls between iterations.
  const int row_block_size = row.block.size;
  // NoEBlockRowOuterProduct is called from NoEBlockRowsUpdate which
  // runs outside the parallel region, so locking is never needed.

#if CERES_SCHUR_CACHE_CELL_LOOKUPS
  const CachedCellInfo* const __restrict cell_cache =
      row_cells_.data() + row_cells_offset_[row_block_index];
  int cache_pos = 0;
#endif

  for (int i = 0; i < num_cells; ++i) {
    // Hoist cells[i] into a local reference and snapshot its fields
    // once per outer iteration.  These are loop-invariant for the inner
    // j-loop, but MSVC's alias analysis can't always prove that across
    // the virtual GetCell() call, leading it to re-issue the loads each
    // inner iteration.  This is a pure rewrite, no FP work changes.
    const Cell& cell_i = cells[i];
    const int cell_i_block_id = cell_i.block_id;
    const int cell_i_position = cell_i.position;

    const int block1_size = col_blocks[cell_i_block_id].size;
#if CERES_SCHUR_CACHE_CELL_LOOKUPS
    const CachedCellInfo& cached_diag = cell_cache[cache_pos++];
    CellInfo* cell_info = cached_diag.info;
    const int r          = cached_diag.r;
    const int c          = cached_diag.c;
    const int row_stride = cached_diag.row_stride;
    const int col_stride = cached_diag.col_stride;
#else
    const int block1 = cell_i_block_id - num_eliminate_blocks_;
    DCHECK_GE(block1, 0);
    int r, c, row_stride, col_stride;
    CellInfo* cell_info = lhs->GetCell(block1, block1,
                                       &r, &c,
                                       &row_stride, &col_stride);
#endif
    if (cell_info != NULL) {
      // Diagonal block: cell += F_i' F_i is symmetric.  Use the
      // half-FLOP self-multiply.  No locking needed in this dispatch
      // because NoEBlockRowOuterProduct runs outside the parallel
      // region (see comment above).
      MatrixTransposeMatrixMultiplySelf
          <Eigen::Dynamic, Eigen::Dynamic, 1>(
              values + cell_i_position, row_block_size, block1_size,
              cell_info->values, r, c, row_stride, col_stride);
    }

    for (int j = i + 1; j < num_cells; ++j) {
      const Cell& cell_j = cells[j];
      const int cell_j_block_id = cell_j.block_id;
#if CERES_SCHUR_CACHE_CELL_LOOKUPS
      const CachedCellInfo& cached_off = cell_cache[cache_pos++];
      CellInfo* cell_info_off = cached_off.info;
      const int rr          = cached_off.r;
      const int cc          = cached_off.c;
      const int row_stride2 = cached_off.row_stride;
      const int col_stride2 = cached_off.col_stride;
#else
      const int block1 = cell_i_block_id - num_eliminate_blocks_;
      const int block2 = cell_j_block_id - num_eliminate_blocks_;
      DCHECK_GE(block2, 0);
      DCHECK_LT(block1, block2);
      int rr, cc, row_stride2, col_stride2;
      CellInfo* cell_info_off = lhs->GetCell(block1, block2,
                                             &rr, &cc,
                                             &row_stride2, &col_stride2);
#endif
      if (cell_info_off != NULL) {
        const int block2_size = col_blocks[cell_j_block_id].size;
        MatrixTransposeMatrixMultiply
            <Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, 1>(
                values + cell_i_position, row_block_size, block1_size,
                values + cell_j.position, row_block_size, block2_size,
                cell_info_off->values, rr, cc, row_stride2, col_stride2);
      }
    }
  }
}

// For a row with an e_block, compute the contribition S += F'F. This
// function has the same structure as NoEBlockRowOuterProduct, except
// that this function uses the template parameters.
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
template <bool kNeedsLocking>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
EBlockRowOuterProduct(const BlockSparseMatrix* A,
                      int row_block_index,
                      BlockRandomAccessMatrix* lhs) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const Block* const __restrict col_blocks = bs->cols.data();
  const CompressedRow& row = bs->rows[row_block_index];
  const double* const __restrict values = A->values();
  const int num_cells = static_cast<int>(row.cells.size());
  const Cell* const __restrict cells = row.cells.data();
  // row.block.size is invariant across both loops below; same MSVC
  // alias-analysis reasoning as in NoEBlockRowOuterProduct.
  const int row_block_size = row.block.size;

#if CERES_SCHUR_CACHE_CELL_LOOKUPS
  const CachedCellInfo* const __restrict cell_cache =
      row_cells_.data() + row_cells_offset_[row_block_index];
  int cache_pos = 0;
#endif

#if CERES_SCHUR_PER_THREAD_LHS
  // T2: per-thread shadow redirect for locked writes.
  const int* const __restrict cell_shadow_off =
      row_cells_shadow_offset_.data() + row_cells_offset_[row_block_index];
#ifdef CERES_USE_OPENMP
  const int eb_thread_id = omp_get_thread_num();
#else
  const int eb_thread_id = 0;
#endif
  const int eb_shadow_dpt = shadow_doubles_per_thread_;
  double* const __restrict eb_shadow_slab =
      shadow_lhs_.empty()
          ? nullptr
          : shadow_lhs_.data() +
                static_cast<size_t>(eb_thread_id) * eb_shadow_dpt;
#endif

  for (int i = 1; i < num_cells; ++i) {
    // Snapshot cells[i] once per outer iteration so MSVC doesn't
    // re-issue the loads across the virtual GetCell()/CeresMutexLock
    // calls.  Pure rewrite, no FP work changes.
    const Cell& cell_i = cells[i];
    const int cell_i_block_id = cell_i.block_id;
    const int cell_i_position = cell_i.position;

    const int block1_size = col_blocks[cell_i_block_id].size;
#if CERES_SCHUR_CACHE_CELL_LOOKUPS
    const int slot_idx_diag = cache_pos++;
    const CachedCellInfo& cached_diag = cell_cache[slot_idx_diag];
    CellInfo* cell_info = cached_diag.info;
    const int r          = cached_diag.r;
    const int c          = cached_diag.c;
    const int row_stride = cached_diag.row_stride;
    const int col_stride = cached_diag.col_stride;
#else
    const int block1 = cell_i_block_id - num_eliminate_blocks_;
    DCHECK_GE(block1, 0);
    int r, c, row_stride, col_stride;
    CellInfo* cell_info = lhs->GetCell(block1, block1,
                                       &r, &c,
                                       &row_stride, &col_stride);
#endif
    if (cell_info != NULL) {
      if (kNeedsLocking) {
#if CERES_SCHUR_PER_THREAD_LHS
        const int shadow_off_diag = cell_shadow_off[slot_idx_diag];
        double* const __restrict tgt_diag =
            eb_shadow_slab + shadow_off_diag;
        // Compact block1_size x block1_size diagonal block in shadow.
        MatrixTransposeMatrixMultiplySelf
            <kRowBlockSize, kFBlockSize, 1>(
                values + cell_i_position, row_block_size, block1_size,
                tgt_diag, 0, 0, block1_size, block1_size);
#else
#if CERES_SCHUR_SHARDED_SPINLOCKS
        const std::size_t h =
            (reinterpret_cast<std::uintptr_t>(cell_info) >> 5)
            & kCeresSchurCellSpinlockMask;
        CeresSchurSpinLockGuard l(&cell_spinlocks_[h].flag);
#else
        CeresMutexLock l(&cell_info->m);
#endif
        // Diagonal block: cell += F_i' F_i is symmetric.
        MatrixTransposeMatrixMultiplySelf
            <kRowBlockSize, kFBlockSize, 1>(
                values + cell_i_position, row_block_size, block1_size,
                cell_info->values, r, c, row_stride, col_stride);
#endif
      } else {
        MatrixTransposeMatrixMultiplySelf
            <kRowBlockSize, kFBlockSize, 1>(
                values + cell_i_position, row_block_size, block1_size,
                cell_info->values, r, c, row_stride, col_stride);
      }
    }

    for (int j = i + 1; j < num_cells; ++j) {
      const Cell& cell_j = cells[j];
      const int cell_j_block_id = cell_j.block_id;
      const int block2_size = col_blocks[cell_j_block_id].size;
#if CERES_SCHUR_CACHE_CELL_LOOKUPS
      const int slot_idx_off = cache_pos++;
      const CachedCellInfo& cached_off = cell_cache[slot_idx_off];
      CellInfo* cell_info_off = cached_off.info;
      const int rr          = cached_off.r;
      const int cc          = cached_off.c;
      const int row_stride2 = cached_off.row_stride;
      const int col_stride2 = cached_off.col_stride;
#else
      const int block1 = cell_i_block_id - num_eliminate_blocks_;
      const int block2 = cell_j_block_id - num_eliminate_blocks_;
      DCHECK_GE(block2, 0);
      DCHECK_LT(block1, block2);
      int rr, cc, row_stride2, col_stride2;
      CellInfo* cell_info_off = lhs->GetCell(block1, block2,
                                             &rr, &cc,
                                             &row_stride2, &col_stride2);
#endif
      if (cell_info_off != NULL) {
        if (kNeedsLocking) {
#if CERES_SCHUR_PER_THREAD_LHS
          const int shadow_off_off = cell_shadow_off[slot_idx_off];
          double* const __restrict tgt_off =
              eb_shadow_slab + shadow_off_off;
          MatrixTransposeMatrixMultiply
              <kRowBlockSize, kFBlockSize, kRowBlockSize, kFBlockSize, 1>(
                  values + cell_i_position, row_block_size, block1_size,
                  values + cell_j.position, row_block_size, block2_size,
                  tgt_off, 0, 0, block1_size, block2_size);
#else
#if CERES_SCHUR_SHARDED_SPINLOCKS
          const std::size_t h =
              (reinterpret_cast<std::uintptr_t>(cell_info_off) >> 5)
              & kCeresSchurCellSpinlockMask;
          CeresSchurSpinLockGuard l(&cell_spinlocks_[h].flag);
#else
          CeresMutexLock l(&cell_info_off->m);
#endif
          MatrixTransposeMatrixMultiply
              <kRowBlockSize, kFBlockSize, kRowBlockSize, kFBlockSize, 1>(
                  values + cell_i_position, row_block_size, block1_size,
                  values + cell_j.position, row_block_size, block2_size,
                  cell_info_off->values, rr, cc, row_stride2, col_stride2);
#endif
        } else {
          MatrixTransposeMatrixMultiply
              <kRowBlockSize, kFBlockSize, kRowBlockSize, kFBlockSize, 1>(
                  values + cell_i_position, row_block_size, block1_size,
                  values + cell_j.position, row_block_size, block2_size,
                  cell_info_off->values, rr, cc, row_stride2, col_stride2);
        }
      }
    }
  }
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SCHUR_ELIMINATOR_IMPL_H_
