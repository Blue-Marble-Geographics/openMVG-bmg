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

#include "ceres/parameter_block_ordering.h"

#include "ceres/graph.h"
#include "ceres/graph_algorithms.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/map_util.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/wall_time.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

using std::map;
using std::set;
using std::vector;

// Optimized drop-in replacement for the reference implementation that
// builds an explicit Graph<ParameterBlock*> and then calls
// StableIndependentSetOrdering.  Produces the identical ordering but
// avoids:
//   - Building HashSet/HashMap-based Graph (per-vertex hash-table allocs).
//   - Calling graph.Neighbors(v).size() twice per stable_sort compare
//     (hash-map lookup with pointer hashing).
//   - HashMap<Vertex, char> vertex_color lookups in the greedy loops.
//
// The algorithm is semantically identical to
//   scoped_ptr<Graph<ParameterBlock*>> g(CreateHessianGraph(program));
//   ordering = {non-const params in program order};
//   StableIndependentSetOrdering(*g, ordering);
//   append constants in program order;
//
// Correctness invariants preserved:
//   1. "degree" = number of distinct non-constant parameter blocks that
//      share at least one residual with this vertex.  Matches
//      graph.Neighbors(v).size() exactly.
//   2. stable_sort by ascending degree; ties broken by insertion order
//      (= program order of non-constant blocks), matching the reference.
//   3. Greedy pass and remainder pass walk vertexQueue in the same
//      sorted order as the reference's vertex_queue.
//   4. Constants appended in program order at the end.
int ComputeStableSchurOrdering(const Program& program,
                               vector<ParameterBlock*>* ordering) {
  CHECK_NOTNULL(ordering)->clear();
  EventLogger eventLogger("ComputeStableSchurOrdering");

  const vector<ParameterBlock*>& parameterBlocks = program.parameter_blocks();
  const vector<ResidualBlock*>& residualBlocks = program.residual_blocks();

  const int numParameterBlocks = static_cast<int>(parameterBlocks.size());
  const int numResidualBlocks = static_cast<int>(residualBlocks.size());

  // Count non-constant parameter blocks for precise reservation.
  int numNonConstant = 0;
  for (int i = 0; i < numParameterBlocks; ++i) {
    if (!parameterBlocks[i]->IsConstant()) {
      ++numNonConstant;
    }
  }

  // Degenerate case: nothing to order; just append constants.
  if (numNonConstant == 0) {
    for (int i = 0; i < numParameterBlocks; ++i) {
      if (parameterBlocks[i]->IsConstant()) {
        ordering->push_back(parameterBlocks[i]);
      }
    }
    return 0;
  }

  // Map each non-constant ParameterBlock* to a dense [0, numNonConstant)
  // index so the rest of the computation uses small integers.
  HashMap<ParameterBlock*, int> blockToIndex;
  blockToIndex.reserve(numNonConstant);

  vector<ParameterBlock*> nonConstantBlocks;
  nonConstantBlocks.reserve(numNonConstant);

  for (int i = 0; i < numParameterBlocks; ++i) {
    ParameterBlock* const pb = parameterBlocks[i];
    if (pb->IsConstant()) {
      continue;
    }
    const int idx = static_cast<int>(nonConstantBlocks.size());
    nonConstantBlocks.push_back(pb);
    blockToIndex.emplace(pb, idx);
  }

  // Packed CSR-like adjacency:
  //   residualAdj[r] = {offset, count} into residualBlockIds giving
  //     the indices of the non-constant parameter blocks referenced
  //     by residual r (residuals with < 2 non-constant blocks are
  //     skipped entirely since they contribute no edges).
  //   residualsForBlock[v] = list of residual ids incident on vertex v.
  struct ResidualAdj {
    int offset;
    int count;
  };
  vector<ResidualAdj> residualAdj;
  residualAdj.reserve(numResidualBlocks);

  vector<int> residualBlockIds;
  residualBlockIds.reserve(numResidualBlocks * 2);

  vector<vector<int> > residualsForBlock(numNonConstant);

  for (int r = 0; r < numResidualBlocks; ++r) {
    const ResidualBlock* residualBlock = residualBlocks[r];
    ParameterBlock* const* pb = residualBlock->parameter_blocks();
    const int numPb = residualBlock->NumParameterBlocks();

    const int start = static_cast<int>(residualBlockIds.size());
    int count = 0;

    if (numPb == 2) {
      // Fast path: almost all SfM reprojection residuals.
      if (!pb[0]->IsConstant() && !pb[1]->IsConstant()) {
        residualBlockIds.push_back(blockToIndex.find(pb[0])->second);
        residualBlockIds.push_back(blockToIndex.find(pb[1])->second);
        count = 2;
      }
    } else {
      for (int j = 0; j < numPb; ++j) {
        if (pb[j]->IsConstant()) {
          continue;
        }
        residualBlockIds.push_back(blockToIndex.find(pb[j])->second);
        ++count;
      }
    }

    // Residuals touching fewer than 2 non-constant blocks contribute
    // no edges; drop them so they do not inflate degrees.
    if (count < 2) {
      residualBlockIds.resize(start);
      continue;
    }

    const int residualIndex = static_cast<int>(residualAdj.size());
    ResidualAdj adj;
    adj.offset = start;
    adj.count = count;
    residualAdj.push_back(adj);

    for (int j = 0; j < count; ++j) {
      residualsForBlock[residualBlockIds[start + j]].push_back(residualIndex);
    }
  }
  eventLogger.AddEvent("BuildInverseIncidence");

  // Compute each vertex's degree = number of DISTINCT non-self neighbors
  // across all incident residuals.  The mark[] array implements the
  // classic O(sum degrees) unique-count pattern.
  vector<int> degree(numNonConstant, 0);
  vector<int> mark(numNonConstant, -1);

  for (int i = 0; i < numNonConstant; ++i) {
    int deg = 0;
    const vector<int>& incidentResiduals = residualsForBlock[i];

    for (int rr = 0; rr < static_cast<int>(incidentResiduals.size()); ++rr) {
      const ResidualAdj& adj = residualAdj[incidentResiduals[rr]];
      const int begin = adj.offset;
      const int end = begin + adj.count;

      for (int p = begin; p < end; ++p) {
        const int nbr = residualBlockIds[p];
        if (nbr == i) {
          continue;
        }
        if (mark[nbr] == i) {
          continue;
        }
        mark[nbr] = i;
        ++deg;
      }
    }
    degree[i] = deg;
  }
  eventLogger.AddEvent("ComputeDegrees");

  // Stable sort by ascending degree; ties break by original (program)
  // order since stable_sort preserves relative order of equal elements.
  vector<int> vertexQueue(numNonConstant);
  for (int i = 0; i < numNonConstant; ++i) {
    vertexQueue[i] = i;
  }
  std::stable_sort(vertexQueue.begin(), vertexQueue.end(),
                   [&](int lhs, int rhs) {
                     return degree[lhs] < degree[rhs];
                   });
  eventLogger.AddEvent("DegreeSort");

  // Classic greedy independent-set pass with white/grey/black colors.
  // Identical in structure to StableIndependentSetOrdering.
  const char kWhite = 0;
  const char kGrey = 1;
  const char kBlack = 2;

  vector<char> color(numNonConstant, kWhite);

  ordering->reserve(numParameterBlocks);

  for (int q = 0; q < numNonConstant; ++q) {
    const int v = vertexQueue[q];
    if (color[v] != kWhite) {
      continue;
    }

    ordering->push_back(nonConstantBlocks[v]);
    color[v] = kBlack;

    // Mark every non-self neighbor grey.  Walking residualsForBlock
    // and the packed adjacency is the flat-array equivalent of the
    // reference's graph.Neighbors(vertex) HashSet iteration.
    const vector<int>& incidentResiduals = residualsForBlock[v];
    for (int rr = 0; rr < static_cast<int>(incidentResiduals.size()); ++rr) {
      const ResidualAdj& adj = residualAdj[incidentResiduals[rr]];
      const int begin = adj.offset;
      const int end = begin + adj.count;

      for (int p = begin; p < end; ++p) {
        const int nbr = residualBlockIds[p];
        if (nbr != v) {
          color[nbr] = kGrey;
        }
      }
    }
  }

  const int independentSetSize = static_cast<int>(ordering->size());
  eventLogger.AddEvent("StableIndependentSet");

  // Append remaining (non-independent-set) non-constant blocks in
  // sorted-queue order.  Matches the reference's final vertex_queue
  // walk that pushes kGrey vertices.
  for (int q = 0; q < numNonConstant; ++q) {
    const int v = vertexQueue[q];
    DCHECK(color[v] != kWhite);
    if (color[v] != kBlack) {
      ordering->push_back(nonConstantBlocks[v]);
    }
  }
  eventLogger.AddEvent("RemainingNonConstantParameterBlocks");

  // Finally, append constants in program order.
  for (int i = 0; i < numParameterBlocks; ++i) {
    ParameterBlock* const pb = parameterBlocks[i];
    if (pb->IsConstant()) {
      ordering->push_back(pb);
    }
  }
  eventLogger.AddEvent("ConstantParameterBlocks");

  return independentSetSize;
}

int ComputeSchurOrdering(const Program& program,
                         vector<ParameterBlock*>* ordering) {
  CHECK_NOTNULL(ordering)->clear();

  scoped_ptr<Graph< ParameterBlock*> > graph(CreateHessianGraph(program));
  int independent_set_size = IndependentSetOrdering(*graph, ordering);
  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();

  // Add the excluded blocks to back of the ordering vector.
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    ParameterBlock* parameter_block = parameter_blocks[i];
    if (parameter_block->IsConstant()) {
      ordering->push_back(parameter_block);
    }
  }

  return independent_set_size;
}

void ComputeRecursiveIndependentSetOrdering(const Program& program,
                                            ParameterBlockOrdering* ordering) {
  CHECK_NOTNULL(ordering)->Clear();
  const vector<ParameterBlock*> parameter_blocks = program.parameter_blocks();
  scoped_ptr<Graph< ParameterBlock*> > graph(CreateHessianGraph(program));

  int num_covered = 0;
  int round = 0;
  while (num_covered < parameter_blocks.size()) {
    vector<ParameterBlock*> independent_set_ordering;
    const int independent_set_size =
        IndependentSetOrdering(*graph, &independent_set_ordering);
    for (int i = 0; i < independent_set_size; ++i) {
      ParameterBlock* parameter_block = independent_set_ordering[i];
      ordering->AddElementToGroup(parameter_block->mutable_user_state(), round);
      graph->RemoveVertex(parameter_block);
    }
    num_covered += independent_set_size;
    ++round;
  }
}

Graph<ParameterBlock*>* CreateHessianGraph(const Program& program) {
  Graph<ParameterBlock*>* graph = CHECK_NOTNULL(new Graph<ParameterBlock*>);
  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    ParameterBlock* parameter_block = parameter_blocks[i];
    if (!parameter_block->IsConstant()) {
      graph->AddVertex(parameter_block);
    }
  }

  const vector<ResidualBlock*>& residual_blocks = program.residual_blocks();
  for (int i = 0; i < residual_blocks.size(); ++i) {
    const ResidualBlock* residual_block = residual_blocks[i];
    const int num_parameter_blocks = residual_block->NumParameterBlocks();
    ParameterBlock* const* parameter_blocks =
        residual_block->parameter_blocks();
    for (int j = 0; j < num_parameter_blocks; ++j) {
      if (parameter_blocks[j]->IsConstant()) {
        continue;
      }

      for (int k = j + 1; k < num_parameter_blocks; ++k) {
        if (parameter_blocks[k]->IsConstant()) {
          continue;
        }

        graph->AddEdge(parameter_blocks[j], parameter_blocks[k]);
      }
    }
  }

  return graph;
}

void OrderingToGroupSizes(const ParameterBlockOrdering* ordering,
                          vector<int>* group_sizes) {
  CHECK_NOTNULL(group_sizes)->clear();
  if (ordering == NULL) {
    return;
  }

  const map<int, set<double*> >& group_to_elements =
      ordering->group_to_elements();
  for (map<int, set<double*> >::const_iterator it = group_to_elements.begin();
       it != group_to_elements.end();
       ++it) {
    group_sizes->push_back(it->second.size());
  }
}

}  // namespace internal
}  // namespace ceres
