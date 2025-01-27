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

#include "ceres/block_sparse_matrix.h"

#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "ceres/block_structure.h"
#include "ceres/internal/eigen.h"
#include "ceres/random.h"
#include "ceres/small_blas.h"
#include "ceres/triplet_sparse_matrix.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

using std::vector;

BlockSparseMatrix::~BlockSparseMatrix() {}

BlockSparseMatrix::BlockSparseMatrix(
    CompressedRowBlockStructure* block_structure)
    : num_rows_(0),
      num_cols_(0),
      num_nonzeros_(0),
      values_(NULL),
      block_structure_(block_structure) {
  DCHECK_NOTNULL(block_structure_.get());

  // Count the number of columns in the matrix.
  const auto& col_sizes = block_structure_->col_sizes;
  for (const auto& col_size : col_sizes) {
    num_cols_ += col_size;
  }

  // Count the number of non-zero entries and the number of rows in
  // the matrix.
  int num_rows = block_structure_->rows.size();
  for (int i = 0; i < num_rows; ++i) {
    const CompressedRow& row = block_structure_->rows[i];
    const int row_block_size = row.block.size;
    num_rows_ += row_block_size;

    const Cell* __restrict cells = row.cells;
    const int cnt = row.num_cells;
    for (int j = 0; j < cnt; ++j) {
      const auto& cell = cells[j];
      const int col_block_id = cell.block_id;
      const int col_block_size = col_sizes[col_block_id];
      num_nonzeros_ += col_block_size * row_block_size;
    }
  }

  DCHECK_GE(num_rows_, 0);
  DCHECK_GE(num_cols_, 0);
  DCHECK_GE(num_nonzeros_, 0);
  VLOG(2) << "Allocating values array with "
          << num_nonzeros_ * sizeof(double) << " bytes.";  // NOLINT
  values_.reset(new double[num_nonzeros_]);
  max_num_nonzeros_ = num_nonzeros_;
  DCHECK_NOTNULL(values_.get());
}

void BlockSparseMatrix::SetZero() {
  std::fill(values_.get(), values_.get() + num_nonzeros_, 0.0);
}

void BlockSparseMatrix::RightMultiply(const double* x,  double* y) const {
  DCHECK_NOTNULL(x);
  DCHECK_NOTNULL(y);

  int num_rows = block_structure_->rows.size();
  const auto& col_sizes = block_structure_->col_sizes;
  const auto& col_positions = block_structure_->col_positions;
  auto* __restrict values_ptr = values_.get();
  for (int i = 0; i < num_rows; ++i) {
    const CompressedRow& row = block_structure_->rows[i];
    const int row_block_pos = row.block.position;
    const int row_block_size = row.block.size;
    const Cell* __restrict cells = row.cells;
    const int cnt = row.num_cells;
    for (int j = 0; j < cnt; ++j) {
      const auto& cell = cells[j];
      const int col_block_id = cell.block_id;
      const int col_block_size = col_sizes[col_block_id];
      const int col_block_pos = col_positions[col_block_id];
      MatrixVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
          values_ptr + cell.position, row_block_size, col_block_size,
          x + col_block_pos,
          y + row_block_pos);
    }
  }
}

void BlockSparseMatrix::LeftMultiply(const double* x, double* y) const {
  DCHECK_NOTNULL(x);
  DCHECK_NOTNULL(y);

  int num_rows = block_structure_->rows.size();
  const auto& col_sizes = block_structure_->col_sizes;
  const auto& col_positions = block_structure_->col_positions;
  auto* __restrict values_ptr = values_.get();
  for (int i = 0; i < num_rows; ++i) {
    const CompressedRow& row = block_structure_->rows[i];
    const int row_block_pos = row.block.position;
    const int row_block_size = row.block.size;
    const Cell* __restrict cells = row.cells;
    const int cnt = row.num_cells;
    for (int j = 0; j < cnt; ++j) {
      const auto& cell = cells[j];
      const int col_block_id = cell.block_id;
      const int col_block_size = col_sizes[col_block_id];
      const int col_block_pos = col_positions[col_block_id];
      MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
          values_ptr + cell.position, row_block_size, col_block_size,
          x + row_block_pos,
          y + col_block_pos);
    }
  }
}

void BlockSparseMatrix::SquaredColumnNorm(double* x) const {
  DCHECK_NOTNULL(x);
  VectorRef(x, num_cols_).setZero();
  int num_rows = block_structure_->rows.size();
  const auto& col_sizes = block_structure_->col_sizes;
  const auto& col_positions = block_structure_->col_positions;
  auto* __restrict values_ptr = values_.get();
  for (int i = 0; i < num_rows; ++i) {
    const CompressedRow& row = block_structure_->rows[i];
    const int row_block_size = row.block.size;
    const Cell* __restrict cells = row.cells;
    const int cnt = row.num_cells;
    for (int j = 0; j < cnt; ++j) {
      const auto& cell = cells[j];
      const int col_block_id = cell.block_id;
      const int col_block_size = col_sizes[col_block_id];
      const int col_block_pos = col_positions[col_block_id];
      const MatrixRef m(values_ptr + cell.position,
                        row_block_size, col_block_size);
      VectorRef(x + col_block_pos, col_block_size) += m.colwise().squaredNorm();
    }
  }
}

void BlockSparseMatrix::ScaleColumns(const double* scale) {
  DCHECK_NOTNULL(scale);

  const int num_rows = block_structure_->rows.size();
  const auto& col_sizes = block_structure_->col_sizes;
  const auto& col_positions = block_structure_->col_positions;
  auto* __restrict values_ptr = values_.get();
  for (int i = 0; i < num_rows; ++i) {
    const auto& block = block_structure_->rows[i];
    int row_block_size = block.block.size;
    const Cell* __restrict cells = block.cells;
    const int cnt = block.num_cells;
    for (int j = 0; j < cnt; ++j) {
      const auto& cell = cells[j];
      const int col_block_id = cell.block_id;
      const int col_block_size = col_sizes[col_block_id];
      const int col_block_pos = col_positions[col_block_id];
      MatrixRef m(values_ptr + cell.position,
                        row_block_size, col_block_size);
      m *= ConstVectorRef(scale + col_block_pos, col_block_size).asDiagonal();
    }
  }
}

void BlockSparseMatrix::ToDenseMatrix(Matrix* dense_matrix) const {
  DCHECK_NOTNULL(dense_matrix);

  dense_matrix->resize(num_rows_, num_cols_);
  dense_matrix->setZero();
  Matrix& m = *dense_matrix;

  const auto& col_sizes = block_structure_->col_sizes;
  const auto& col_positions = block_structure_->col_positions;
  auto* __restrict values_ptr = values_.get();
  const int num_rows = block_structure_->rows.size();
  for (int i = 0; i < num_rows; ++i) {
    const auto& block = block_structure_->rows[i];
    const int row_block_pos = block.block.position;
    const int row_block_size = block.block.size;
    const Cell* __restrict cells = block.cells;
    const int cnt = block.num_cells;
    for (int j = 0; j < cnt; ++j) {
      const auto& cell = cells[j];
      const int col_block_id = cell.block_id;
      const int col_block_size = col_sizes[col_block_id];
      const int col_block_pos = col_positions[col_block_id];
      const int jac_pos = cell.position;
      m.block(row_block_pos, col_block_pos, row_block_size, col_block_size)
          += MatrixRef(values_ptr + jac_pos, row_block_size, col_block_size);
    }
  }
}

void BlockSparseMatrix::ToTripletSparseMatrix(
    TripletSparseMatrix* matrix) const {
  DCHECK_NOTNULL(matrix);

  matrix->Reserve(num_nonzeros_);
  matrix->Resize(num_rows_, num_cols_);
  matrix->SetZero();

  for (int i = 0; i < block_structure_->rows.size(); ++i) {
    int row_block_pos = block_structure_->rows[i].block.position;
    int row_block_size = block_structure_->rows[i].block.size;
    const Cell* cells = block_structure_->rows[i].cells;
    const int cnt = block_structure_->rows[i].num_cells;
    for (int j = 0; j < cnt; ++j) {
      int col_block_id = cells[j].block_id;
      int col_block_size = block_structure_->col_sizes[col_block_id];
      int col_block_pos = block_structure_->col_positions[col_block_id];
      int jac_pos = cells[j].position;
       for (int r = 0; r < row_block_size; ++r) {
        for (int c = 0; c < col_block_size; ++c, ++jac_pos) {
          matrix->mutable_rows()[jac_pos] = row_block_pos + r;
          matrix->mutable_cols()[jac_pos] = col_block_pos + c;
          matrix->mutable_values()[jac_pos] = values_[jac_pos];
        }
      }
    }
  }
  matrix->set_num_nonzeros(num_nonzeros_);
}

// Return a pointer to the block structure. We continue to hold
// ownership of the object though.
const CompressedRowBlockStructure* BlockSparseMatrix::block_structure()
    const {
  return block_structure_.get();
}

CompressedRowBlockStructure* BlockSparseMatrix::block_structure()
{
  return block_structure_.get();
}

void BlockSparseMatrix::ToTextFile(FILE* file) const {
  CHECK_NOTNULL(file);
  for (int i = 0; i < block_structure_->rows.size(); ++i) {
    const int row_block_pos = block_structure_->rows[i].block.position;
    const int row_block_size = block_structure_->rows[i].block.size;
    const Cell* cells = block_structure_->rows[i].cells;
    const int cnt = block_structure_->rows[i].num_cells;
    for (int j = 0; j < cnt; ++j) {
      const int col_block_id = cells[j].block_id;
      const int col_block_size = block_structure_->col_sizes[col_block_id];
      const int col_block_pos = block_structure_->col_positions[col_block_id];
      int jac_pos = cells[j].position;
      for (int r = 0; r < row_block_size; ++r) {
        for (int c = 0; c < col_block_size; ++c) {
          fprintf(file, "% 10d % 10d %17f\n",
                  row_block_pos + r,
                  col_block_pos + c,
                  values_[jac_pos++]);
        }
      }
    }
  }
}

BlockSparseMatrix* BlockSparseMatrix::CreateDiagonalMatrix(
    const double* diagonal, const std::vector<int>& col_sizes, const std::vector<int>& col_positions) {
  // Create the block structure for the diagonal matrix.
  CompressedRowBlockStructure* bs = new CompressedRowBlockStructure();
  bs->all_cells.resize(1);

  bs->col_sizes = col_sizes;
  bs->col_positions = col_positions;
  int position = 0;
  bs->rows.resize(col_sizes.size(), CompressedRow(bs->all_cells));
  for (int i = 0; i < col_sizes.size(); ++i) {
    CompressedRow& row = bs->rows[i];
    row.block = Block(col_sizes[i], col_positions[i]);
    Cell& cell = row.cells[0];
    cell.block_id = i;
    cell.position = position;
    position += row.block.size * row.block.size;
  }

  // Create the BlockSparseMatrix with the given block structure.
  BlockSparseMatrix* matrix = new BlockSparseMatrix(bs);
  matrix->SetZero();

  // Fill the values array of the block sparse matrix.
  double* values = matrix->mutable_values();
  for (int i = 0; i < col_sizes.size(); ++i) {
    const int size = col_sizes[i];
    for (int j = 0; j < size; ++j) {
      // (j + 1) * size is compact way of accessing the (j,j) entry.
      values[j * (size + 1)] = diagonal[j];
    }
    diagonal += size;
    values += size * size;
  }

  return matrix;
}

void BlockSparseMatrix::AppendRows(const BlockSparseMatrix& m) {
  throw std::runtime_error( "Unsupported" );
#if 0 // JPB WIP BUG
  const int old_num_nonzeros = num_nonzeros_;
  const int old_num_row_blocks = block_structure_->rows.size();
  const CompressedRowBlockStructure* m_bs = m.block_structure();
  block_structure_->rows.resize(old_num_row_blocks + m_bs->rows.size());

  for (int i = 0; i < m_bs->rows.size(); ++i) {
    const CompressedRow& m_row = m_bs->rows[i];
    CompressedRow& row = block_structure_->rows[old_num_row_blocks + i];
    row.block.size = m_row.block.size;
    row.block.position = num_rows_;
    num_rows_ += m_row.block.size;
    row.cells.resize(m_row.cells.size());
    for (int c = 0; c < m_row.cells.size(); ++c) {
      const int block_id = m_row.cells[c].block_id;
      row.cells[c].block_id = block_id;
      row.cells[c].position = num_nonzeros_;
      num_nonzeros_ += m_row.block.size * m_bs->col_sizes[block_id];
    }
  }

  if (num_nonzeros_ > max_num_nonzeros_) {
    double* new_values = new double[num_nonzeros_];
    std::copy(values_.get(), values_.get() + old_num_nonzeros, new_values);
    values_.reset(new_values);
    max_num_nonzeros_ = num_nonzeros_;
  }

  std::copy(m.values(),
            m.values() + m.num_nonzeros(),
            values_.get() + old_num_nonzeros);
#endif
}

void BlockSparseMatrix::DeleteRowBlocks(const int delta_row_blocks) {
  throw std::runtime_error( "Unsupported" );
#if 0
  const int num_row_blocks = block_structure_->rows.size();
  int delta_num_nonzeros = 0;
  int delta_num_rows = 0;
  const auto& column_blocks_sizes = block_structure_->col_sizes;
  for (int i = 0; i < delta_row_blocks; ++i) {
    const CompressedRow& row = block_structure_->rows[num_row_blocks - i - 1];
    delta_num_rows += row.block.size;
    for (int c = 0; c < row.cells.size(); ++c) {
      const Cell& cell = row.cells[c];
      delta_num_nonzeros += row.block.size * column_blocks_sizes[cell.block_id];
    }
  }
  num_nonzeros_ -= delta_num_nonzeros;
  num_rows_ -= delta_num_rows;
  block_structure_->rows.resize(num_row_blocks - delta_row_blocks);
#endif
}

BlockSparseMatrix* BlockSparseMatrix::CreateRandomMatrix(
    const BlockSparseMatrix::RandomMatrixOptions& options) {
  throw std::runtime_error( "Unsupported" );
#if 0
  CHECK_GT(options.num_row_blocks, 0);
  CHECK_GT(options.min_row_block_size, 0);
  CHECK_GT(options.max_row_block_size, 0);
  CHECK_LE(options.min_row_block_size, options.max_row_block_size);
  CHECK_GT(options.num_col_blocks, 0);
  CHECK_GT(options.min_col_block_size, 0);
  CHECK_GT(options.max_col_block_size, 0);
  CHECK_LE(options.min_col_block_size, options.max_col_block_size);
  CHECK_GT(options.block_density, 0.0);
  CHECK_LE(options.block_density, 1.0);

  CompressedRowBlockStructure* bs = new CompressedRowBlockStructure();
    // Generate the col block structure.
  int col_block_position = 0;
  for (int i = 0; i < options.num_col_blocks; ++i) {
    // Generate a random integer in [min_col_block_size, max_col_block_size]
    const int delta_block_size =
        Uniform(options.max_col_block_size - options.min_col_block_size);
    const int col_block_size = options.min_col_block_size + delta_block_size;
    bs->col_sizes.push_back(col_block_size);
    bs->col_positions.push_back(col_block_position);
    col_block_position += col_block_size;
  }


  bool matrix_has_blocks = false;
  while (!matrix_has_blocks) {
    LOG(INFO) << "clearing";
    bs->rows.clear();
    int row_block_position = 0;
    int value_position = 0;
    for (int r = 0; r < options.num_row_blocks; ++r) {

      const int delta_block_size =
          Uniform(options.max_row_block_size - options.min_row_block_size);
      const int row_block_size = options.min_row_block_size + delta_block_size;
      bs->rows.push_back(CompressedRow());
      CompressedRow& row = bs->rows.back();
      row.block.size = row_block_size;
      row.block.position = row_block_position;
      row_block_position += row_block_size;
      for (int c = 0; c < options.num_col_blocks; ++c) {
        if (RandDouble() > options.block_density) continue;

        row.cells.push_back(Cell());
        Cell& cell = row.cells.back();
        cell.block_id = c;
        cell.position = value_position;
        value_position += row_block_size * bs->col_sizes[c];
        matrix_has_blocks = true;
      }
    }
  }

  BlockSparseMatrix* matrix = new BlockSparseMatrix(bs);
  double* values = matrix->mutable_values();
  for (int i = 0; i < matrix->num_nonzeros(); ++i) {
    values[i] = RandNormal();
  }

  return matrix;
#else
  return nullptr;
#endif
}

}  // namespace internal
}  // namespace ceres
