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

int ComputeStableSchurOrdering(const Program& program,
                         vector<ParameterBlock*>* ordering) {
  CHECK_NOTNULL(ordering)->clear();
  EventLogger event_logger("ComputeStableSchurOrdering");
  scoped_ptr<Graph< ParameterBlock*> > graph(CreateHessianGraph(program));
  event_logger.AddEvent("CreateHessianGraph");

  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();
  const HashSet<ParameterBlock*>& vertices = graph->vertices();
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    if (vertices.count(parameter_blocks[i]) > 0) {
      ordering->push_back(parameter_blocks[i]);
    }
  }
  event_logger.AddEvent("Preordering");

  int independent_set_size = StableIndependentSetOrdering(*graph, ordering);
  event_logger.AddEvent("StableIndependentSet");

  // Add the excluded blocks to back of the ordering vector.
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    ParameterBlock* parameter_block = parameter_blocks[i];
    if (parameter_block->IsConstant()) {
      ordering->push_back(parameter_block);
    }
  }
  event_logger.AddEvent("ConstantParameterBlocks");

  return independent_set_size;
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
  const vector<ResidualBlock*>& residual_blocks = program.residual_blocks();

  // Count non-constant vertices for precise reservation.
  int num_vertices = 0;
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    if (!parameter_blocks[i]->IsConstant()) {
      ++num_vertices;
    }
  }

  // Count edges (upper bound: one per pair of non-constant params per residual).
  int num_edges = 0;
  for (int i = 0; i < residual_blocks.size(); ++i) {
    const int npb = residual_blocks[i]->NumParameterBlocks();
    // For typical SfM residuals npb=2, so edges_per_residual=1
    int non_const = 0;
    ParameterBlock* const* pb = residual_blocks[i]->parameter_blocks();
    for (int j = 0; j < npb; ++j) {
      if (!pb[j]->IsConstant()) ++non_const;
    }
    num_edges += non_const * (non_const - 1) / 2;
  }

  graph->Reserve(num_vertices, num_edges);

  for (int i = 0; i < parameter_blocks.size(); ++i) {
    ParameterBlock* parameter_block = parameter_blocks[i];
    if (!parameter_block->IsConstant()) {
      graph->AddVertex(parameter_block);
    }
  }

  const int num_residual_blocks = static_cast<int>(residual_blocks.size());
  for (int i = 0; i < num_residual_blocks; ++i) {
    const ResidualBlock* residual_block = residual_blocks[i];
    const int num_pb = residual_block->NumParameterBlocks();
    ParameterBlock* const* pb = residual_block->parameter_blocks();

    // Fast path: most SfM residuals have exactly 2 non-constant parameter blocks.
    if (num_pb == 2) {
      if (!pb[0]->IsConstant() && !pb[1]->IsConstant()) {
        graph->AddEdge(pb[0], pb[1]);
      }
      continue;
    }

    for (int j = 0; j < num_pb; ++j) {
      if (pb[j]->IsConstant()) {
        continue;
      }
      for (int k = j + 1; k < num_pb; ++k) {
        if (pb[k]->IsConstant()) {
          continue;
        }
        graph->AddEdge(pb[j], pb[k]);
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
