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
//
// A convenience class for cost functions which are statically sized.
// Compared to the dynamically-sized base class, this reduces boilerplate.
//
// The kNumResiduals template parameter can be a constant such as 2 or 5, or it
// can be ceres::DYNAMIC. If kNumResiduals is ceres::DYNAMIC, then subclasses
// are responsible for calling set_num_residuals() at runtime.

#ifndef CERES_PUBLIC_SIZED_COST_FUNCTION_H_
#define CERES_PUBLIC_SIZED_COST_FUNCTION_H_

#include "ceres/types.h"
#include "ceres/cost_function.h"
#include "glog/logging.h"

namespace ceres {

template<int kNumResiduals,
         int N0 = 0, int N1 = 0, int N2 = 0, int N3 = 0, int N4 = 0,
         int N5 = 0, int N6 = 0, int N7 = 0, int N8 = 0, int N9 = 0>
class SizedCostFunction : public CostFunction {
 public:
  SizedCostFunction() {
    DCHECK(kNumResiduals > 0 || kNumResiduals == DYNAMIC)
        << "Cost functions must have at least one residual block.";

    // This block breaks the 80 column rule to keep it somewhat readable.
    DCHECK((!N1 && !N2 && !N3 && !N4 && !N5 && !N6 && !N7 && !N8 && !N9) ||
          ((N1 > 0) && !N2 && !N3 && !N4 && !N5 && !N6 && !N7 && !N8 && !N9) ||
          ((N1 > 0) && (N2 > 0) && !N3 && !N4 && !N5 && !N6 && !N7 && !N8 && !N9) ||                                   // NOLINT
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && !N4 && !N5 && !N6 && !N7 && !N8 && !N9) ||                              // NOLINT
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && !N5 && !N6 && !N7 && !N8 && !N9) ||                         // NOLINT
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && (N5 > 0) && !N6 && !N7 && !N8 && !N9) ||                    // NOLINT
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && (N5 > 0) && (N6 > 0) && !N7 && !N8 && !N9) ||               // NOLINT
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && (N5 > 0) && (N6 > 0) && (N7 > 0) && !N8 && !N9) ||          // NOLINT
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && (N5 > 0) && (N6 > 0) && (N7 > 0) && (N8 > 0) && !N9) ||     // NOLINT
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && (N5 > 0) && (N6 > 0) && (N7 > 0) && (N8 > 0) && (N9 > 0)))  // NOLINT
        << "Zero block cannot precede a non-zero block. Block sizes are "
        << "(ignore trailing 0s): " << N0 << ", " << N1 << ", " << N2 << ", "
        << N3 << ", " << N4 << ", " << N5 << ", " << N6 << ", " << N7 << ", "
        << N8 << ", " << N9;

    set_num_residuals(kNumResiduals);

    if (N0) add_parameter_block_sizes(N0);
    if (N1) add_parameter_block_sizes(N1);
    if (N2) add_parameter_block_sizes(N2);
    if (N3) add_parameter_block_sizes(N3);
    if (N4) add_parameter_block_sizes(N4);
    if (N5) add_parameter_block_sizes(N5);
    if (N6) add_parameter_block_sizes(N6);
    if (N7) add_parameter_block_sizes(N7);
    if (N8) add_parameter_block_sizes(N8);
    if (N9) add_parameter_block_sizes(N9);
  }

  virtual ~SizedCostFunction() { }

  // Subclasses must implement Evaluate().
};

}  // namespace ceres

#endif  // CERES_PUBLIC_SIZED_COST_FUNCTION_H_
