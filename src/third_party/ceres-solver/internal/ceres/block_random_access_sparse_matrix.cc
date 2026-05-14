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

#include "ceres/block_random_access_sparse_matrix.h"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>
#include "ceres/internal/port.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/mutex.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

using std::make_pair;
using std::pair;
using std::set;
using std::vector;

BlockRandomAccessSparseMatrix::BlockRandomAccessSparseMatrix(
    const vector<int>& blocks,
    const set<pair<int, int> >& block_pairs)
    : kMaxRowBlocks(10 * 1000 * 1000),
      blocks_(blocks),
      cell_info_pool_(NULL) {
  CHECK_LT(blocks.size(), kMaxRowBlocks);

  // Build the row/column layout vector and count the number of scalar
  // rows/columns.
  int num_cols = 0;
  block_positions_.reserve(blocks_.size());
  for (int i = 0; i < blocks_.size(); ++i) {
    block_positions_.push_back(num_cols);
    num_cols += blocks_[i];
  }

  // Count the number of scalar non-zero entries and build the layout
  // object for looking into the values array of the
  // TripletSparseMatrix.
  int num_nonzeros = 0;
  for (set<pair<int, int> >::const_iterator it = block_pairs.begin();
       it != block_pairs.end();
       ++it) {
    const int row_block_size = blocks_[it->first];
    const int col_block_size = blocks_[it->second];
    num_nonzeros += row_block_size * col_block_size;
  }

  VLOG(1) << "Matrix Size [" << num_cols
          << "," << num_cols
          << "] " << num_nonzeros;

  tsm_.reset(new TripletSparseMatrix(num_cols, num_cols, num_nonzeros));
  tsm_->set_num_nonzeros(num_nonzeros);
  int* rows = tsm_->mutable_rows();
  int* cols = tsm_->mutable_cols();
  double* values = tsm_->mutable_values();

  // Bulk-initialise the values array to 1.0. Bit-identical to writing 1.0
  // per element inside the layout loop below; here it's a single trivially
  // vectorisable pass instead of being interleaved with rows/cols stores.
  // (These values are placeholders: every meaningful numeric write into
  // the matrix is performed by the Schur eliminator after SetZero().)
  std::fill_n(values, num_nonzeros, 1.0);

  // Final size is known: one cell per block pair.  Reserve up front
  // to avoid log2(N) reallocations as the vector grows.
  cell_values_.reserve(block_pairs.size());

  // Single contiguous allocation for all CellInfo objects, replacing what
  // was an individual `new CellInfo(...)` per block pair (and matching
  // `delete` per pair in the dtor).  block_pairs.size() may be 0 here;
  // `new CellInfo[0]` is well-formed and `delete[]` on it is well-formed.
  cell_info_pool_ = new CellInfo[block_pairs.size()];
  size_t cell_idx = 0;

  // Single pass: build layout, cell_values, AND fill sparsity pattern.
  int pos = 0;
  for (set<pair<int, int> >::const_iterator it = block_pairs.begin();
       it != block_pairs.end();
       ++it) {
    const int row_block_id = it->first;
    const int col_block_id = it->second;
    const int row_block_size = blocks_[row_block_id];
    const int col_block_size = blocks_[col_block_id];

    cell_values_.push_back(make_pair(make_pair(row_block_id, col_block_id),
                                     values + pos));
    CellInfo* const cell_info = &cell_info_pool_[cell_idx++];
    cell_info->values = values + pos;
    layout_[IntPairToLong(row_block_id, col_block_id)] = cell_info;

    // Fill sparsity pattern for this block in-place. (values[] is already
    // initialised to 1.0 by the std::fill_n above.)
    const int row_start = block_positions_[row_block_id];
    const int col_start = block_positions_[col_block_id];
    for (int r = 0; r < row_block_size; ++r) {
      for (int c = 0; c < col_block_size; ++c, ++pos) {
        rows[pos] = row_start + r;
        cols[pos] = col_start + c;
      }
    }
  }
}

// Assume that the user does not hold any locks on any cell blocks
// when they are calling SetZero.
BlockRandomAccessSparseMatrix::~BlockRandomAccessSparseMatrix() {
  // CellInfo objects live in the contiguous cell_info_pool_ array; their
  // destructors (notably ~Mutex) run via delete[] on that array.  Nothing
  // owned individually by layout_ entries.
  delete[] cell_info_pool_;
}

CellInfo* BlockRandomAccessSparseMatrix::GetCell(int row_block_id,
                                                 int col_block_id,
                                                 int* row,
                                                 int* col,
                                                 int* row_stride,
                                                 int* col_stride) {
  const LayoutType::iterator it  =
      layout_.find(IntPairToLong(row_block_id, col_block_id));
  if (it == layout_.end()) {
    return NULL;
  }

  // Each cell is stored contiguously as its own little dense matrix.
  *row = 0;
  *col = 0;
  *row_stride = blocks_[row_block_id];
  *col_stride = blocks_[col_block_id];
  return it->second;
}

// Assume that the user does not hold any locks on any cell blocks
// when they are calling SetZero.
void BlockRandomAccessSparseMatrix::SetZero() {
  if (tsm_->num_nonzeros()) {
    VectorRef(tsm_->mutable_values(),
              tsm_->num_nonzeros()).setZero();
  }
}

void BlockRandomAccessSparseMatrix::SymmetricRightMultiply(const double* x,
                                                           double* y) const {
  vector< pair<pair<int, int>, double*> >::const_iterator it =
      cell_values_.begin();
  for (; it != cell_values_.end(); ++it) {
    const int row = it->first.first;
    const int row_block_size = blocks_[row];
    const int row_block_pos = block_positions_[row];

    const int col = it->first.second;
    const int col_block_size = blocks_[col];
    const int col_block_pos = block_positions_[col];

    MatrixVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
        it->second, row_block_size, col_block_size,
        x + col_block_pos,
        y + row_block_pos);

    // Since the matrix is symmetric, but only the upper triangular
    // part is stored, if the block being accessed is not a diagonal
    // block, then use the same block to do the corresponding lower
    // triangular multiply also.
    if (row != col) {
      MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
          it->second, row_block_size, col_block_size,
          x + row_block_pos,
          y + col_block_pos);
    }
  }
}

}  // namespace internal
}  // namespace ceres
