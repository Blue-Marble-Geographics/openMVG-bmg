// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_DATA_BA_CERES_HPP
#define OPENMVG_SFM_SFM_DATA_BA_CERES_HPP

#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/sfm/sfm_data_BA.hpp"

namespace ceres { class CostFunction; }
namespace openMVG { namespace cameras { struct IntrinsicBase; } }
namespace openMVG { namespace sfm { struct SfM_Data; } }

namespace openMVG {
namespace sfm {

/// Validate the analytic Pinhole+Radial3 reprojection cost-function Jacobian
/// against AutoDiff over `trials` random parameter sets. Returns the maximum
/// |residual or Jacobian disagreement| observed. Pass criterion: < 1e-7.
/// Implemented in sfm_data_BA_ceres.cpp; declared here so callers can run it
/// as a one-shot validation step before flipping the analytic toggle.
double RunSelfTest_AnalyticReprojectionCost_Radial3(int trials = 200);

/// Create the appropriate cost functor according the provided input camera intrinsic model
/// Can be residual cost functor can be weighetd if desired (default 0.0 means no weight).
ceres::CostFunction * IntrinsicsToCostFunction
(
  cameras::IntrinsicBase * intrinsic,
  const Vec2 & observation,
  const double weight = 0.0
);

class Bundle_Adjustment_Ceres : public Bundle_Adjustment
{
  public:
  struct BA_Ceres_options
  {
    bool bVerbose_;
    unsigned int nb_threads_;
    bool bCeres_summary_;
    int linear_solver_type_;
    int preconditioner_type_;
    int sparse_linear_algebra_library_type_;
    double parameter_tolerance_;
    double gradient_tolerance_;
    double function_tolerance_;
    bool bUse_loss_function_;
    int max_num_iterations_;
    int max_linear_solver_iterations_;
    int max_num_consecutive_invalid_steps_;
    /// If true, Ceres re-optimizes structure parameters cheaply between LM
    /// outer steps. Typically cuts outer iteration count 30-50% with no
    /// quality difference; the inner sub-problem is independent per point
    /// so it parallelizes well.
    bool use_inner_iterations_;

    /// Adaptive plateau-based early termination for intermediate BAs.
    ///
    /// When > 0, an IterationCallback monitors the *relative* cost change
    /// (|cost_change| / cost) across successful LM steps. If the relative
    /// improvement falls below `plateau_relative_tolerance_` for
    /// `plateau_patience_` consecutive successful iterations -- and we have
    /// completed at least `plateau_min_iterations_` iterations overall --
    /// the solver returns SOLVER_TERMINATE_SUCCESSFULLY.
    ///
    /// This is a *complement* to Ceres' built-in absolute-tolerance checks:
    /// it lets easy scenes (where geometry settles in a few iters) exit
    /// fast, while hard scenes (residuals still moving each step) continue
    /// up to `max_num_iterations_`. Disabled by default (0.0) so the
    /// reference STRICT preset is unaffected.
    double plateau_relative_tolerance_;
    int    plateau_min_iterations_;
    int    plateau_patience_;

    BA_Ceres_options(const bool bVerbose = false, bool bmultithreaded = true);
  };
  private:
    BA_Ceres_options ceres_options_;

  public:
  explicit Bundle_Adjustment_Ceres
  (
    const Bundle_Adjustment_Ceres::BA_Ceres_options & options =
    std::move(BA_Ceres_options())
  );

  BA_Ceres_options & ceres_options();

  bool Adjust
  (
    // the SfM scene to refine
    sfm::SfM_Data & sfm_data,
    // tell which parameter needs to be adjusted
    const Optimize_Options & options
  ) override;
};

} // namespace sfm
} // namespace openMVG

#endif // OPENMVG_SFM_SFM_DATA_BA_CERES_HPP
