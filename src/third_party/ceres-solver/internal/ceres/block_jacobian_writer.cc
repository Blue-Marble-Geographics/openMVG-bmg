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
// Author: keir@google.com (Keir Mierle)

#include "ceres/block_jacobian_writer.h"

#include "ceres/block_evaluate_preparer.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/port.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

using std::vector;

namespace {

// Given the residual block ordering, build a lookup table to determine which
// per-parameter jacobian goes where in the overall program jacobian.
//
// Since we expect to use a Schur type linear solver to solve the LM step, take
// extra care to place the E blocks and the F blocks contiguously. E blocks are
// the first num_eliminate_blocks parameter blocks as indicated by the parameter
// block ordering. The remaining parameter blocks are the F blocks.
//
// TODO(keir): Consider if we should use a boolean for each parameter block
// instead of num_eliminate_blocks.
void BuildJacobianLayout(const Program& program,
                         int num_eliminate_blocks,
                         vector<int*>* jacobian_layout,
                         vector<int>* jacobian_layout_storage) {
  const vector<ResidualBlock*>& residual_blocks = program.residual_blocks();

  // Iterate over all the active residual blocks and determine how many E blocks
  // are there. This will determine where the F blocks start in the jacobian
  // matrix. Also compute the number of jacobian blocks.
  int f_block_pos = 0;
  int num_jacobian_blocks = 0;
  for (int i = 0; i < residual_blocks.size(); ++i) {
    ResidualBlock* residual_block = residual_blocks[i];
    const int num_residuals = residual_block->NumResiduals();
    const int num_parameter_blocks = residual_block->NumParameterBlocks();

    // Advance f_block_pos over each E block for this residual.
    for (int j = 0; j < num_parameter_blocks; ++j) {
      ParameterBlock* parameter_block = residual_block->parameter_blocks()[j];
      if (!parameter_block->IsConstant()) {
        // Only count blocks for active parameters.
        num_jacobian_blocks++;
        if (parameter_block->index() < num_eliminate_blocks) {
          f_block_pos += num_residuals * parameter_block->LocalSize();
        }
      }
    }
  }

  // We now know that the E blocks are laid out starting at zero, and the F
  // blocks are laid out starting at f_block_pos. Iterate over the residual
  // blocks again, and this time fill the jacobian_layout array with the
  // position information.

  jacobian_layout->resize(program.NumResidualBlocks());
  jacobian_layout_storage->resize(num_jacobian_blocks);

  int e_block_pos = 0;
  int* jacobian_pos = &(*jacobian_layout_storage)[0];
  for (int i = 0; i < residual_blocks.size(); ++i) {
    const ResidualBlock* residual_block = residual_blocks[i];
    const int num_residuals = residual_block->NumResiduals();
    const int num_parameter_blocks = residual_block->NumParameterBlocks();

    (*jacobian_layout)[i] = jacobian_pos;
    for (int j = 0; j < num_parameter_blocks; ++j) {
      ParameterBlock* parameter_block = residual_block->parameter_blocks()[j];
      const int parameter_block_index = parameter_block->index();
      if (parameter_block->IsConstant()) {
        continue;
      }
      const int jacobian_block_size =
          num_residuals * parameter_block->LocalSize();
      if (parameter_block_index < num_eliminate_blocks) {
        *jacobian_pos = e_block_pos;
        e_block_pos += jacobian_block_size;
      } else {
        *jacobian_pos = f_block_pos;
        f_block_pos += jacobian_block_size;
      }
      jacobian_pos++;
    }
  }
}

// Order a row's cells by block_id.
//
// Rows carry a handful of cells (2-4 for a typical bundle adjustment residual),
// and there is one row per residual block, so this runs hundreds of thousands of
// times on a large problem. std::sort pays introsort's fixed setup on every one
// of those calls; a straight insertion sort over a few elements sitting in L1 is
// strictly less work at these sizes.
inline void SortCellsByBlockId(Cell* cells, int num_cells) {
  for (int i = 1; i < num_cells; ++i) {
    const Cell cell = cells[i];
    int j = i - 1;
    for (; j >= 0 && cells[j].block_id > cell.block_id; --j) {
      cells[j + 1] = cells[j];
    }
    cells[j + 1] = cell;
  }
}

}  // namespace

BlockJacobianWriter::BlockJacobianWriter(const Evaluator::Options& options,
                                         Program* program)
    : program_(program) {
  CHECK_GE(options.num_eliminate_blocks, 0)
      << "num_eliminate_blocks must be greater than 0.";

  BuildJacobianLayout(*program,
                      options.num_eliminate_blocks,
                      &jacobian_layout_,
                      &jacobian_layout_storage_);
}

// Create evaluate prepareres that point directly into the final jacobian. This
// makes the final Write() a nop.
BlockEvaluatePreparer* BlockJacobianWriter::CreateEvaluatePreparers(
    int num_threads) {
  int max_derivatives_per_residual_block =
      program_->MaxDerivativesPerResidualBlock();

  BlockEvaluatePreparer* preparers = new BlockEvaluatePreparer[num_threads];
  for (int i = 0; i < num_threads; i++) {
    preparers[i].Init(&jacobian_layout_[0], max_derivatives_per_residual_block);
  }
  return preparers;
}

SparseMatrix* BlockJacobianWriter::CreateJacobian() const {
  CompressedRowBlockStructure* bs = new CompressedRowBlockStructure;

  const vector<ParameterBlock*>& parameter_blocks =
      program_->parameter_blocks();

  // Construct the column blocks.
  bs->cols.resize(parameter_blocks.size());
  for (int i = 0, cursor = 0; i < parameter_blocks.size(); ++i) {
    CHECK_NE(parameter_blocks[i]->index(), -1);
    CHECK(!parameter_blocks[i]->IsConstant());
    bs->cols[i].size = parameter_blocks[i]->LocalSize();
    bs->cols[i].position = cursor;
    cursor += bs->cols[i].size;
  }

  // Construct the cells in each row.
  const vector<ResidualBlock*>& residual_blocks = program_->residual_blocks();
  const int num_rows = residual_blocks.size();
  int row_block_position = 0;
  bs->rows.resize(num_rows);

  // Scratch space for assembling a row before it is copied into its final home.
  // Building and sorting here keeps the working set in L1 and lets the row's
  // vector be sized exactly once. Residuals with more parameter blocks than fit
  // on the stack fall back to a heap buffer that is reused across rows.
  const int kMaxStackCells = 8;
  Cell stack_cells[kMaxStackCells];
  vector<Cell> heap_cells;

  for (int i = 0; i < num_rows; ++i) {
    const ResidualBlock* residual_block = residual_blocks[i];
    CompressedRow* row = &bs->rows[i];

    row->block.size = residual_block->NumResiduals();
    row->block.position = row_block_position;
    row_block_position += row->block.size;

    const int num_parameter_blocks = residual_block->NumParameterBlocks();
    ParameterBlock* const* row_parameter_blocks =
        residual_block->parameter_blocks();
    const int* row_jacobian_layout = jacobian_layout_[i];

    Cell* cells = stack_cells;
    if (num_parameter_blocks > kMaxStackCells) {
      heap_cells.resize(num_parameter_blocks);
      cells = &heap_cells[0];
    }

    // Gather the active parameters of this row in a single pass. The previous
    // version walked the parameter blocks twice -- once to count them and once
    // to fill them in -- which doubled the pointer chasing through
    // ParameterBlock. That chasing is the part that actually misses cache here,
    // since a bundle adjustment problem has one distinct point block per
    // residual and those blocks are scattered across the heap.
    //
    // Note that the counting pass tested index() != -1 while the filling pass
    // tested IsConstant(). Those agree for a program that has been through
    // Program::RemoveFixedBlocks, but only IsConstant() matches the predicate
    // BuildJacobianLayout used when it laid out jacobian_layout_, so that is the
    // one the layout index must be advanced on. Using a single predicate for
    // both sizing and filling removes the discrepancy.
    int num_active_parameter_blocks = 0;
    for (int j = 0; j < num_parameter_blocks; ++j) {
      const ParameterBlock* parameter_block = row_parameter_blocks[j];
      if (parameter_block->IsConstant()) {
        continue;
      }
      Cell& cell = cells[num_active_parameter_blocks];
      cell.block_id = parameter_block->index();
      // There is only layout information for active parameters, so this indexes
      // by the active count rather than by j.
      cell.position = row_jacobian_layout[num_active_parameter_blocks];
      num_active_parameter_blocks++;
    }

    SortCellsByBlockId(cells, num_active_parameter_blocks);

    // assign() sizes the vector exactly and copy-constructs into it. resize()
    // would default-construct every Cell only to have it overwritten.
    row->cells.assign(cells, cells + num_active_parameter_blocks);
  }

  BlockSparseMatrix* jacobian = new BlockSparseMatrix(bs);
  CHECK_NOTNULL(jacobian);
  return jacobian;
}

}  // namespace internal
}  // namespace ceres
