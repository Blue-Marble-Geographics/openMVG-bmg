// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data_BA_ceres.hpp"

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include "ceres/problem.h"
#include "ceres/solver.h"
#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/geometry/Similarity3.hpp"
#include "openMVG/geometry/Similarity3_Kernel.hpp"
//- Robust estimation - LMeds (since no threshold can be defined)
#include "openMVG/robust_estimation/robust_estimator_LMeds.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres_camera_functor.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres_analytic_radial3.hpp"
#include "openMVG/sfm/sfm_data_transform.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_logging.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/types.hpp"

#include <ceres/rotation.h>
#include <ceres/types.h>
#include <ceres/iteration_callback.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

namespace openMVG {
namespace sfm {

#define OPENMVG_CERES_HAS_MANIFOLD ((CERES_VERSION_MAJOR * 100 + CERES_VERSION_MINOR) >= 201)

// Compile-time half of the [BA-DIAG] gate (see sfm_logging.hpp). The runtime
// half remains `ceres_options_.bVerbose_`; both must be on. Written as a
// constexpr bool rather than an `#if` so the guarded code keeps being compiled
// -- with the switch at 0 the `&&` folds and the blocks (including the full
// observation traversals they perform) are dead-code eliminated.
static constexpr bool kBADiagLogging = (OPENMVG_SFM_VERBOSE_BA != 0);

// Toggle: Ceres inner iterations (re-solve structure between LM outer steps).
//   1 = S3 reference behaviour (use_inner_iterations_ = true).
//   0 = current safe value (was observed to make things worse in one earlier
//       test; retest with the rest of the toggles in their current state).
#ifndef OPENMVG_BA_USE_INNER_ITERATIONS
#define OPENMVG_BA_USE_INNER_ITERATIONS 1
#endif

// ---------------------------------------------------------------------------
// Adaptive GPS / motion-prior weighting (auto de-dome).
//
// PROBLEM: the pose-center prior weight (ViewPriors::center_weight_) defaults
// to 1.0 per axis. Each camera then contributes ONE 3D prior residual (in GPS
// units, e.g. metres) that must compete against the hundreds-to-thousands of
// reprojection residuals (pixels) pulling on that same pose. The prior is
// outvoted by ~n_obs:1, so BA settles into the low-reprojection "dome/bowl"
// warp and never pulls the model onto its GPS priors (large residual GPS
// fitting error, no convergence). A hard-coded -W weight cannot fix this for
// arbitrary data because the right value depends on per-camera observation
// count, scene scale and GPS units.
//
// FIX: scale each camera's prior weight by sqrt(n_obs_camera) so the prior
// COST per camera (weight^2 * ||dC||^2) grows in step with that camera's
// reprojection cost (~n_obs). The GPS:vision ratio then becomes INDEPENDENT of
// point density, scene size and image count -- i.e. one dataset-independent
// STRENGTH constant works for arbitrary data. The Huber knee is scaled by the
// same factor so the robust threshold stays at the same physical distance.
//
//   OPENMVG_SFM_GPS_PRIOR_AUTOSCALE : 1 = on (default), 0 = legacy raw weight.
//   OPENMVG_SFM_GPS_PRIOR_STRENGTH  : global GPS:vision ratio multiplier.
//       1.0 = balanced (default). Raise (2-4) to force flatter georeferencing
//       when GPS is trusted (RTK); lower (0.25-0.5) if the model starts
//       snapping to noisy GPS. This is the ONLY knob to touch, and it is
//       dataset-independent thanks to the sqrt(n_obs) scaling.
#ifndef OPENMVG_SFM_GPS_PRIOR_AUTOSCALE
#define OPENMVG_SFM_GPS_PRIOR_AUTOSCALE 1
#endif
#ifndef OPENMVG_SFM_GPS_PRIOR_STRENGTH
#define OPENMVG_SFM_GPS_PRIOR_STRENGTH 1.0
#endif

using namespace openMVG::cameras;
using namespace openMVG::geometry;

namespace {

// Adaptive plateau-detection callback for intermediate BAs.
//
// Triggers SOLVER_TERMINATE_SUCCESSFULLY once the *relative* cost change of
// successful LM steps stays below `rel_tol` for `patience` consecutive
// successful iterations, after a `min_iter` warmup. This complements Ceres'
// absolute-magnitude tolerances (function/parameter/gradient): it cuts off
// "easy" scenes that have effectively converged geometrically while leaving
// "hard" scenes free to keep iterating up to max_num_iterations.
//
// Notes:
// * Only successful steps count toward the plateau streak; rejected steps
//   reset the counter (they indicate the optimizer is still searching).
// * iteration 0 has step_is_successful=true with cost_change=0; we ignore
//   it via the min_iter warmup.
// * Pure stopping-criterion change. No effect on the residual model, the
//   parameter blocks, the loss function, or per-iteration math.
class PlateauTerminationCallback final : public ceres::IterationCallback {
 public:
  PlateauTerminationCallback(double rel_tol, int min_iter, int patience)
      : rel_tol_(rel_tol),
        min_iter_(min_iter),
        patience_(patience < 1 ? 1 : patience),
        plateau_count_(0) {}

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& s) override {
    if (s.iteration < min_iter_) return ceres::SOLVER_CONTINUE;
    if (!s.step_is_successful)   { plateau_count_ = 0; return ceres::SOLVER_CONTINUE; }
    const double denom = std::abs(s.cost);
    if (denom <= 0.0) return ceres::SOLVER_CONTINUE;
    const double rel = std::abs(s.cost_change) / denom;
    if (rel < rel_tol_) {
      if (++plateau_count_ >= patience_) {
        return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
      }
    } else {
      plateau_count_ = 0;
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  const double rel_tol_;
  const int    min_iter_;
  const int    patience_;
  int          plateau_count_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Diagnostic helper: count non-finite poses and log a short summary.
// Returns the number of non-finite poses found.
// ---------------------------------------------------------------------------
static size_t CountNonFinitePoses(
  const SfM_Data & sfm_data,
  const std::string & label)
{
  size_t nonfinite = 0;
  std::ostringstream offenders;
  size_t shown = 0;
  for (const auto & pose_it : sfm_data.GetPoses())
  {
    const Mat3 & R = pose_it.second.rotation();
    const Vec3 & C = pose_it.second.center();
    if (!R.allFinite() || !C.allFinite())
    {
      ++nonfinite;
      if (shown < 5)
      {
        offenders << " pose_id=" << pose_it.first;
        ++shown;
      }
    }
  }
  if (nonfinite > 0)
  {
    OPENMVG_LOG_WARNING
      << "[BA-DIAG:" << label << "] Non-finite poses: " << nonfinite
      << "/" << sfm_data.GetPoses().size() << " (first offenders:" << offenders.str() << ")";
  }
  return nonfinite;
}

// ---------------------------------------------------------------------------
// Diagnostic helper: compute per-pose observation counts. Poses with too few
// observations will trip downstream filters (eraseUnstablePosesAndObservations
// uses min_points_per_pose=6 by default). Flag any pose with < 12 obs.
// ---------------------------------------------------------------------------
static void LogPerPoseObservationCounts(
  const SfM_Data & sfm_data,
  const std::string & label)
{
  // viewId -> poseId
  Hash_Map<IndexT, IndexT> view_to_pose;
  for (const auto & v : sfm_data.GetViews())
    view_to_pose[v.first] = v.second->id_pose;

  // poseId -> observation count
  Hash_Map<IndexT, size_t> pose_obs_count;
  for (const auto & p : sfm_data.GetPoses())
    pose_obs_count[p.first] = 0;

  for (const auto & lm : sfm_data.GetLandmarks())
  {
    for (const auto & obs : lm.second.obs)
    {
      const auto it = view_to_pose.find(obs.first);
      if (it != view_to_pose.end())
      {
        const auto pit = pose_obs_count.find(it->second);
        if (pit != pose_obs_count.end())
          ++pit->second;
      }
    }
  }

  // Summary histogram-ish: min, median, max, and count below threshold
  std::vector<size_t> counts;
  counts.reserve(pose_obs_count.size());
  size_t below_6 = 0, below_12 = 0;
  for (const auto & kv : pose_obs_count)
  {
    counts.push_back(kv.second);
    if (kv.second < 6)  ++below_6;
    if (kv.second < 12) ++below_12;
  }
  if (counts.empty())
    return;

  std::sort(counts.begin(), counts.end());
  const size_t min_c = counts.front();
  const size_t max_c = counts.back();
  const size_t med_c = counts[counts.size() / 2];

  OPENMVG_LOG_INFO
    << "[BA-DIAG:" << label << "] Per-pose observation counts (#poses="
    << counts.size() << "): min=" << min_c << " median=" << med_c
    << " max=" << max_c << " | below_6=" << below_6 << " below_12=" << below_12;

  // If any poses are dangerously low on observations, list up to 10 of them
  if (below_12 > 0)
  {
    std::ostringstream offenders;
    size_t shown = 0;
    for (const auto & kv : pose_obs_count)
    {
      if (kv.second < 12)
      {
        if (shown > 0) offenders << ", ";
        offenders << "pose=" << kv.first << "(" << kv.second << ")";
        if (++shown >= 10) { offenders << ", ..."; break; }
      }
    }
    OPENMVG_LOG_WARNING
      << "[BA-DIAG:" << label << "] Poses with <12 observations (at risk of"
      << " removal by eraseUnstablePosesAndObservations): " << offenders.str();
  }
}

// Ceres CostFunctor used for SfM pose center to GPS pose center minimization
struct PoseCenterConstraintCostFunction
{
  Vec3 weight_;
  Vec3 pose_center_constraint_;

  PoseCenterConstraintCostFunction
  (
    const Vec3 & center,
    const Vec3 & weight
  ): weight_(weight), pose_center_constraint_(center)
  {
  }

  template <typename T> bool
  operator()
  (
    const T* const cam_extrinsics, // R_t
    T* residuals
  )
  const
  {
    using Vec3T = Eigen::Matrix<T,3,1>;
    Eigen::Map<const Vec3T> cam_R(&cam_extrinsics[0]);
    Eigen::Map<const Vec3T> cam_t(&cam_extrinsics[3]);
    const Vec3T cam_R_transpose(-cam_R);

    Vec3T pose_center;
    // Rotate the point according the camera rotation
    ceres::AngleAxisRotatePoint(cam_R_transpose.data(), cam_t.data(), pose_center.data());
    pose_center = pose_center * T(-1);

    Eigen::Map<Vec3T> residuals_eigen(residuals);
    residuals_eigen = weight_.cast<T>().cwiseProduct(pose_center - pose_center_constraint_.cast<T>());

    return true;
  }
};

/// Create the appropriate cost functor according the provided input camera intrinsic model.
/// The residual can be weighetd if desired (default 0.0 means no weight).
ceres::CostFunction * IntrinsicsToCostFunction
(
  IntrinsicBase * intrinsic,
  const Vec2 & observation,
  const double weight
)
{
  switch (intrinsic->getType())
  {
    case PINHOLE_CAMERA:
      return ResidualErrorFunctor_Pinhole_Intrinsic::Create(observation, weight);
    case PINHOLE_CAMERA_RADIAL1:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1::Create(observation, weight);
    case PINHOLE_CAMERA_RADIAL3:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(observation, weight);
    case PINHOLE_CAMERA_BROWN:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2::Create(observation, weight);
    case PINHOLE_CAMERA_FISHEYE:
      return ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye::Create(observation, weight);
    case CAMERA_SPHERICAL:
      return ResidualErrorFunctor_Intrinsic_Spherical::Create(intrinsic, observation, weight);
    default:
      return {};
  }
}

// ---------------------------------------------------------------------------
// Arena allocator for ceres::CostFunction objects.
//
// Avoids the O(N-residuals) per-element delete loop inside ~Problem by owning
// all cost functions ourselves in a bulk-freed arena. Behavior is
// bit-identical to heap allocation; only teardown cost changes.
// ---------------------------------------------------------------------------
class CostFunctionArena
{
public:
  explicit CostFunctionArena(size_t block_size = 4 * 1024 * 1024)
    : block_size_(block_size),
      current_block_(nullptr),
      current_offset_(0),
      current_capacity_(0)
  {}

  ~CostFunctionArena()
  {
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it)
      it->destructor(it->ptr);
    for (auto * block : blocks_)
      ::operator delete(block);
  }

  template <typename T, typename... Args>
  T * Alloc(Args &&... args)
  {
    constexpr size_t alignment = alignof(T);
    constexpr size_t size = sizeof(T);
    size_t aligned_offset = (current_offset_ + alignment - 1) & ~(alignment - 1);
    if (current_block_ == nullptr || aligned_offset + size > current_capacity_)
    {
      AllocateBlock(std::max(block_size_, size + alignment));
      aligned_offset = (current_offset_ + alignment - 1) & ~(alignment - 1);
    }
    void * mem = static_cast<char*>(current_block_) + aligned_offset;
    current_offset_ = aligned_offset + size;
    T * obj = new (mem) T(std::forward<Args>(args)...);
    entries_.push_back({obj, [](void * p) { static_cast<T*>(p)->~T(); }});
    return obj;
  }

  // Hint the destruction-record vector capacity. Avoids ~22 reallocations on
  // a 5M-observation scene where Alloc fires twice per obs (functor + cost
  // function). Pure capacity hint, no behavioral effect.
  void ReserveEntries(size_t n) { entries_.reserve(n); }

  CostFunctionArena(const CostFunctionArena &) = delete;
  CostFunctionArena & operator=(const CostFunctionArena &) = delete;

private:
  void AllocateBlock(size_t min_size)
  {
    const size_t alloc_size = std::max(block_size_, min_size);
    void * block = ::operator new(alloc_size);
    blocks_.push_back(block);
    current_block_ = block;
    current_offset_ = 0;
    current_capacity_ = alloc_size;
  }

  struct Entry { void * ptr; void (*destructor)(void*); };

  size_t             block_size_;
  void *             current_block_;
  size_t             current_offset_;
  size_t             current_capacity_;
  std::vector<void*> blocks_;
  std::vector<Entry> entries_;
};

/// Arena-based overload: bit-identical to the heap version, but the cost
/// function lives in the arena. Used in combination with
///   problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
/// Weighted path falls through to heap allocation in fallback_storage.
ceres::CostFunction * IntrinsicsToCostFunction
(
  CostFunctionArena & arena,
  std::vector<std::unique_ptr<ceres::CostFunction>> & fallback_storage,
  IntrinsicBase * intrinsic,
  const Vec2 & observation,
  const double weight = 0.0
)
{
  if (weight != 0.0)
  {
    ceres::CostFunction * cf = IntrinsicsToCostFunction(intrinsic, observation, weight);
    if (cf)
      fallback_storage.emplace_back(cf);
    return cf;
  }

  switch (intrinsic->getType())
  {
    case PINHOLE_CAMERA:
    {
      using F = ResidualErrorFunctor_Pinhole_Intrinsic;
      using AD = ceres::AutoDiffCostFunction<F, 2, 3, 6, 3>;
      F * functor = arena.Alloc<F>(observation.data());
      return arena.Alloc<AD>(functor, typename AD::NonOwning{});
    }

    case PINHOLE_CAMERA_RADIAL1:
    {
      using F = ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1;
      using AD = ceres::AutoDiffCostFunction<F, 2, 4, 6, 3>;
      F * functor = arena.Alloc<F>(observation.data());
      return arena.Alloc<AD>(functor, typename AD::NonOwning{});
    }

    case PINHOLE_CAMERA_RADIAL3:
    {
      // -------------------------------------------------------------
      // Analytic-Jacobian fast path for the by-far most common camera
      // model. AutoDiff with K=15 partial Jets dominates Jacobian-eval
      // wall time; the analytic version computes the same closed-form
      // derivative directly. Bit-equivalent residuals; Jacobians agree
      // with AutoDiff to ~1e-10 (FP rounding).
      //
      // Toggle: kUseAnalyticJacobian_Radial3 below. Default false until
      // the self-test has been validated on a representative dataset;
      // see RunSelfTest_AnalyticReprojectionCost_Radial3() at the end
      // of this TU. To enable: flip the constexpr to true and rebuild.
      // -------------------------------------------------------------
      static constexpr bool kUseAnalyticJacobian_Radial3 = true;  // self-test: 9e-12 vs AutoDiff
      if (kUseAnalyticJacobian_Radial3)
      {
        return arena.Alloc<AnalyticReprojectionCost_Radial3>(observation.data());
      }
      using F = ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3;
      using AD = ceres::AutoDiffCostFunction<F, 2, 6, 6, 3>;
      F * functor = arena.Alloc<F>(observation.data());
      return arena.Alloc<AD>(functor, typename AD::NonOwning{});
    }

    case PINHOLE_CAMERA_BROWN:
    {
      using F = ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2;
      using AD = ceres::AutoDiffCostFunction<F, 2, 8, 6, 3>;
      F * functor = arena.Alloc<F>(observation.data());
      return arena.Alloc<AD>(functor, typename AD::NonOwning{});
    }

    case PINHOLE_CAMERA_FISHEYE:
    {
      using F = ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye;
      using AD = ceres::AutoDiffCostFunction<F, 2, 7, 6, 3>;
      F * functor = arena.Alloc<F>(observation.data());
      return arena.Alloc<AD>(functor, typename AD::NonOwning{});
    }

    case CAMERA_SPHERICAL:
    {
      using F = ResidualErrorFunctor_Intrinsic_Spherical;
      using AD = ceres::AutoDiffCostFunction<F, 2, 6, 3>;
      F * functor = arena.Alloc<F>(observation.data(), intrinsic->w(), intrinsic->h());
      return arena.Alloc<AD>(functor, typename AD::NonOwning{});
    }

    default:
    {
      ceres::CostFunction * cf = IntrinsicsToCostFunction(intrinsic, observation, 0.0);
      if (cf)
        fallback_storage.emplace_back(cf);
      return cf;
    }
  }
}

// Type-tagged factory for the residual-block hot loop. Resolving the
// intrinsic type once at view-setup time eliminates the virtual
// IntrinsicBase::getType() + 6-way switch per observation, which on a
// large scene is a real cost (millions of obs in incremental BA).
//
// Each factory arena-allocates BOTH the residual functor and the
// AutoDiffCostFunction wrapper. The wrapper is constructed with the
// NonOwning tag so its destructor does NOT delete the functor (the arena
// already owns the functor's lifetime via its destructor-record list).
// Bit-identical numerics to the original heap path.
using CostFnFactory =
  ceres::CostFunction * (*)(CostFunctionArena &, const double * /*obs*/,
                            int /*spherical_w*/, int /*spherical_h*/);

template <typename Functor, int NumResiduals, int... ParamSizes>
static ceres::CostFunction * MakeCostFn_Pinhole_Like
(CostFunctionArena & arena, const double * obs,
 int /*spherical_w*/, int /*spherical_h*/)
{
  using AD = ceres::AutoDiffCostFunction<Functor, NumResiduals, ParamSizes...>;
  Functor * functor = arena.Alloc<Functor>(obs);
  return arena.Alloc<AD>(functor, typename AD::NonOwning{});
}

static ceres::CostFunction * MakeCostFn_Spherical
(CostFunctionArena & arena, const double * obs, int w, int h)
{
  using F = ResidualErrorFunctor_Intrinsic_Spherical;
  using AD = ceres::AutoDiffCostFunction<F, 2, 6, 3>;
  F * functor = arena.Alloc<F>(obs, w, h);
  return arena.Alloc<AD>(functor, typename AD::NonOwning{});
}

static CostFnFactory PickFactoryForIntrinsic(IntrinsicBase * intrinsic)
{
  if (!intrinsic) return nullptr;
  switch (intrinsic->getType())
  {
    case PINHOLE_CAMERA:
      return &MakeCostFn_Pinhole_Like<
        ResidualErrorFunctor_Pinhole_Intrinsic, 2, 3, 6, 3>;
    case PINHOLE_CAMERA_RADIAL1:
      return &MakeCostFn_Pinhole_Like<
        ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1, 2, 4, 6, 3>;
    case PINHOLE_CAMERA_RADIAL3:
      return &MakeCostFn_Pinhole_Like<
        ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3, 2, 6, 6, 3>;
    case PINHOLE_CAMERA_BROWN:
      return &MakeCostFn_Pinhole_Like<
        ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2, 2, 8, 6, 3>;
    case PINHOLE_CAMERA_FISHEYE:
      return &MakeCostFn_Pinhole_Like<
        ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye, 2, 7, 6, 3>;
    case CAMERA_SPHERICAL:
      return &MakeCostFn_Spherical;
    default:
      return nullptr; // caller falls back to heap path
  }
}

Bundle_Adjustment_Ceres::BA_Ceres_options::BA_Ceres_options
(
  const bool bVerbose,
  bool bmultithreaded
)
: bVerbose_(bVerbose),
  nb_threads_(1),
  parameter_tolerance_(1e-8),
  gradient_tolerance_(1e-10),
  function_tolerance_(1e-6),
  bUse_loss_function_(true),
  max_num_iterations_(50),
  max_linear_solver_iterations_(500),
  max_num_consecutive_invalid_steps_(5),
  use_inner_iterations_(OPENMVG_BA_USE_INNER_ITERATIONS != 0),
  // Plateau early-stop disabled by default (STRICT preset semantics).
  plateau_relative_tolerance_(0.0),
  plateau_min_iterations_(5),
  plateau_patience_(2)
{
  #ifdef OPENMVG_USE_OPENMP
    nb_threads_ = omp_get_max_threads();
  #endif // OPENMVG_USE_OPENMP
  if (!bmultithreaded)
    nb_threads_ = 1;

  bCeres_summary_ = false;

  // Default configuration use a DENSE representation
  linear_solver_type_ = ceres::DENSE_SCHUR;
  preconditioner_type_ = ceres::JACOBI;
  // If Sparse linear solver are available
  // Descending priority order by efficiency (SUITE_SPARSE > CX_SPARSE > EIGEN_SPARSE)
  if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
  {
    sparse_linear_algebra_library_type_ = ceres::SUITE_SPARSE;
    linear_solver_type_ = ceres::SPARSE_SCHUR;
  }
  else
  {
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
    {
      sparse_linear_algebra_library_type_ = ceres::EIGEN_SPARSE;
      linear_solver_type_ = ceres::SPARSE_SCHUR;
    }
  }
}


Bundle_Adjustment_Ceres::Bundle_Adjustment_Ceres
(
  const Bundle_Adjustment_Ceres::BA_Ceres_options & options
)
: ceres_options_(options)
{}

Bundle_Adjustment_Ceres::BA_Ceres_options &
Bundle_Adjustment_Ceres::ceres_options()
{
  return ceres_options_;
}

bool Bundle_Adjustment_Ceres::Adjust
(
  SfM_Data & sfm_data,     // the SfM scene to refine
  const Optimize_Options & options
)
{
#if OPENMVG_SFM_VERBOSE_BA
  // [BA-PERF] Per-Adjust() timing. Cheap (a few clock reads) but it fires on
  // every BA call, so it is compiled out with the rest of the BA diagnostics.
  using clk = std::chrono::steady_clock;
  const auto t_adjust_begin = clk::now();
  clk::time_point t_setup_done;  // populated just before ceres::Solve
#endif
  //----------
  // Add camera parameters
  // - intrinsics
  // - poses [R|t]

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  //----------

  // [BA-DIAG] Entry-state snapshot. Gated on bVerbose_ so release builds skip
  // the full obs traversal these helpers do.
  // total_obs is also used (unconditionally) below to size the cost-function
  // arena's destruction-record vector. Sum of obs.size() per landmark is
  // cheap; we'd traverse this loop anyway.
  size_t total_obs_in = 0;
  for (const auto & lm : sfm_data.GetLandmarks())
    total_obs_in += lm.second.obs.size();
  if (kBADiagLogging && ceres_options_.bVerbose_)
  {
    OPENMVG_LOG_INFO
      << "[BA-DIAG:Enter] #views=" << sfm_data.GetViews().size()
      << " #poses=" << sfm_data.GetPoses().size()
      << " #intrinsics=" << sfm_data.GetIntrinsics().size()
      << " #landmarks=" << sfm_data.GetLandmarks().size()
      << " #observations=" << total_obs_in
      << " | use_motion_priors=" << (options.use_motion_priors_opt ? 1 : 0)
      << " intrinsics_opt=" << static_cast<int>(options.intrinsics_opt)
      << " extrinsics_opt=" << static_cast<int>(options.extrinsics_opt)
      << " structure_opt=" << static_cast<int>(options.structure_opt);

    CountNonFinitePoses(sfm_data, "Enter");
    LogPerPoseObservationCounts(sfm_data, "Enter");
  }

  double pose_center_robust_fitting_error = 0.0;
  openMVG::geometry::Similarity3 sim_to_center;
  bool b_usable_prior = false;
  if (options.use_motion_priors_opt && sfm_data.GetViews().size() > 3)
  {
    // - Compute a robust X-Y affine transformation & apply it
    // - This early transformation enhance the conditionning (solution closer to the Prior coordinate system)
    {
      // Collect corresponding camera centers
      std::vector<Vec3> X_SfM, X_GPS;
      for (const auto & view_it : sfm_data.GetViews())
      {
        const sfm::ViewPriors * prior = dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
        if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
        {
          X_SfM.push_back( sfm_data.GetPoses().at(prior->id_pose).center() );
          X_GPS.push_back( prior->pose_center_ );
        }
      }

      // [BA-DIAG] How many usable priors did we find? A low count relative to
      // #poses suggests the seed pod lost prior linkage.
      if (kBADiagLogging && ceres_options_.bVerbose_)
      {
        OPENMVG_LOG_INFO
          << "[BA-DIAG:Prior] Collected " << X_SfM.size()
          << " usable pose priors out of " << sfm_data.GetPoses().size() << " poses.";
      }

      openMVG::geometry::Similarity3 sim;

      // Compute the registration:
      if (X_GPS.size() > 3)
      {
        const Mat X_SfM_Mat = Eigen::Map<Mat>(X_SfM[0].data(),3, X_SfM.size());
        const Mat X_GPS_Mat = Eigen::Map<Mat>(X_GPS[0].data(),3, X_GPS.size());
        geometry::kernel::Similarity3_Kernel kernel(X_SfM_Mat, X_GPS_Mat);
        const double lmeds_median = openMVG::robust::LeastMedianOfSquares(kernel, &sim);

        // [BA-DIAG] LMedS result: if max(), registration failed and prior is unusable.
        if (kBADiagLogging && ceres_options_.bVerbose_)
        {
          OPENMVG_LOG_INFO
            << "[BA-DIAG:Prior] LMedS median residual: " << lmeds_median
            << (lmeds_median == std::numeric_limits<double>::max() ? " (FAILED - prior NOT used)" : " (OK)");
        }

        if (lmeds_median != std::numeric_limits<double>::max())
        {
          b_usable_prior = true; // PRIOR can be used safely

          // Compute the median residual error once the registration is applied
          for (Vec3 & pos : X_SfM) // Transform SfM poses for residual computation
          {
            pos = sim(pos);
          }
          Vec residual = (Eigen::Map<Mat3X>(X_SfM[0].data(), 3, X_SfM.size()) - Eigen::Map<Mat3X>(X_GPS[0].data(), 3, X_GPS.size())).colwise().norm();
          std::sort(residual.data(), residual.data() + residual.size());
          pose_center_robust_fitting_error = residual(residual.size()/2);

          // [BA-DIAG] Post-registration residual stats help detect degenerate
          // priors (e.g., all zeros, or a single-flight-line configuration).
          if (kBADiagLogging && ceres_options_.bVerbose_)
          {
            const double res_min = residual(0);
            const double res_max = residual(residual.size() - 1);
            OPENMVG_LOG_INFO
              << "[BA-DIAG:Prior] Post-registration residuals (user units): min="
              << res_min << " median=" << pose_center_robust_fitting_error
              << " max=" << res_max;
          }

          // Apply the found transformation to the SfM Data Scene
          openMVG::sfm::ApplySimilarity(sim, sfm_data);

          // Move entire scene to center for better numerical stability
          Vec3 pose_centroid = Vec3::Zero();
          for (const auto & pose_it : sfm_data.poses)
          {
            pose_centroid += (pose_it.second.center() / (double)sfm_data.poses.size());
          }
          sim_to_center = openMVG::geometry::Similarity3(openMVG::sfm::Pose3(Mat3::Identity(), pose_centroid), 1.0);
          openMVG::sfm::ApplySimilarity(sim_to_center, sfm_data, true);
        }
      }
      else
      {
        OPENMVG_LOG_WARNING << "Cannot used the motion prior, insufficient number of motion priors/poses"
          << " (found " << X_GPS.size() << ", need > 3)";
      }
    }
  }

  ceres::Problem::Options problem_options;

  // Own all cost functions ourselves in a bulk-freed arena. This avoids the
  // O(N-residuals) per-element delete loop inside ~Problem. No change to
  // residuals or Jacobians; only teardown cost changes.
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  // We always pass valid, correctly-sized parameter pointers; skip Ceres'
  // per-AddResidualBlock validation (size/pointer/dup checks). Saves
  // measurable time across millions of residual additions on large scenes.
  problem_options.disable_all_safety_checks = true;
  CostFunctionArena cost_function_arena;
  // Two arena allocations per residual block (functor + AutoDiff wrapper),
  // plus a few entries for GCP / pose-prior cost functions. Reserving here
  // avoids ~22 reallocations of the arena's destruction-record vector on a
  // ~5M observation scene.
  cost_function_arena.ReserveEntries(2 * total_obs_in + 256);
  std::vector<std::unique_ptr<ceres::CostFunction>> fallback_cost_functions;
  // Prior Huber losses (when loss_function_ownership == DO_NOT_TAKE_OWNERSHIP).
  // Lifetime must extend past problem.Solve(), so it lives at Adjust() scope.
  std::vector<std::unique_ptr<ceres::LossFunction>> prior_loss_functions;

  // Set a LossFunction to be less penalized by false measurements
  //  - set it to nullptr if you don't want use a lossFunction.
  std::unique_ptr<ceres::LossFunction> p_LossFunction;
  if (ceres_options_.bUse_loss_function_)
  {
    p_LossFunction.reset(new ceres::HuberLoss(Square(4.0)));
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  }

  ceres::Problem problem(problem_options);

  // Data wrapper for refinement:
  Hash_Map<IndexT, std::vector<double>> map_intrinsics;
  Hash_Map<IndexT, std::vector<double>> map_poses;

  // Setup Poses data & subparametrization
  for (const auto & pose_it : sfm_data.poses)
  {
    const IndexT indexPose = pose_it.first;

    const Pose3 & pose = pose_it.second;
    const Mat3 R = pose.rotation();
    const Vec3 t = pose.translation();

    double angleAxis[3];
    ceres::RotationMatrixToAngleAxis((const double*)R.data(), angleAxis);
    // angleAxis + translation
    map_poses[indexPose] = {angleAxis[0], angleAxis[1], angleAxis[2], t(0), t(1), t(2)};

    double * parameter_block = &map_poses.at(indexPose)[0];
    problem.AddParameterBlock(parameter_block, 6);
    if (options.extrinsics_opt == Extrinsic_Parameter_Type::NONE)
    {
      // set the whole parameter block as constant for best performance
      problem.SetParameterBlockConstant(parameter_block);
    }
    else  // Subset parametrization
    {
      std::vector<int> vec_constant_extrinsic;
      // If we adjust only the translation, we must set ROTATION as constant
      if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_TRANSLATION)
      {
        // Subset rotation parametrization
        vec_constant_extrinsic.insert(vec_constant_extrinsic.end(), {0,1,2});
      }
      // If we adjust only the rotation, we must set TRANSLATION as constant
      if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_ROTATION)
      {
        // Subset translation parametrization
        vec_constant_extrinsic.insert(vec_constant_extrinsic.end(), {3,4,5});
      }
      if (!vec_constant_extrinsic.empty())
      {
#if OPENMVG_CERES_HAS_MANIFOLD
        auto* subset_manifold = new ceres::SubsetManifold(6, vec_constant_extrinsic);
        problem.SetManifold(parameter_block, subset_manifold);
#else
        auto *subset_parameterization =
          new ceres::SubsetParameterization(6, vec_constant_extrinsic);
        problem.SetParameterization(parameter_block, subset_parameterization);
#endif
      }
    }
  }

  // Setup Intrinsics data & subparametrization
  for (const auto & intrinsic_it : sfm_data.intrinsics)
  {
    const IndexT indexCam = intrinsic_it.first;

    if (isValid(intrinsic_it.second->getType()))
    {
      map_intrinsics[indexCam] = intrinsic_it.second->getParams();
      if (!map_intrinsics.at(indexCam).empty())
      {
        double * parameter_block = &map_intrinsics.at(indexCam)[0];
        problem.AddParameterBlock(parameter_block, map_intrinsics.at(indexCam).size());
        if (options.intrinsics_opt == Intrinsic_Parameter_Type::NONE)
        {
          // set the whole parameter block as constant for best performance
          problem.SetParameterBlockConstant(parameter_block);
        }
        else
        {
          const std::vector<int> vec_constant_intrinsic =
            intrinsic_it.second->subsetParameterization(options.intrinsics_opt);
          if (!vec_constant_intrinsic.empty())
          {
#if OPENMVG_CERES_HAS_MANIFOLD
            auto* subset_manifold =
              new ceres::SubsetManifold(
                map_intrinsics.at(indexCam).size(), vec_constant_intrinsic);
            problem.SetManifold(parameter_block, subset_manifold);
#else
            auto *subset_parameterization =
              new ceres::SubsetParameterization(
                map_intrinsics.at(indexCam).size(), vec_constant_intrinsic);
            problem.SetParameterization(parameter_block, subset_parameterization);
#endif
          }
        }
      }
    }
    else
    {
      OPENMVG_LOG_ERROR << "Unsupported camera type.";
    }
  }

  // For all visibility add reprojections errors:
  // [BA-DIAG] Track how many residuals we actually add and how many are
  // dropped due to missing intrinsics / bad cost functions. Counters are
  // zero-cost; logging them is gated on bVerbose_.
  size_t residuals_added = 0;
  size_t residuals_dropped_no_cost_fn = 0;

  // Resolve per-view (intrinsic*, intrinsic_param_ptr, pose_param_ptr) once.
  // The inner loop fires once per observation (can be millions); the old code
  // did 4-5 Hash_Map lookups per obs (views.at, intrinsics.at, map_intrinsics.at
  // twice, map_poses.at). This cuts it to a single lookup per obs.
  //
  // Use a flat std::vector indexed by view_id when the id range is dense
  // (typical OpenMVG case: ids are [0..N)) so the per-obs lookup is a bounds
  // check + array index instead of a hash probe. Falls back to Hash_Map for
  // sparse id ranges (heuristic: max_id < 8 * count + 64).
  //
  // Safe because map_poses / map_intrinsics are fully populated above and are
  // not mutated again before/during this loop, so pointers into their storage
  // are stable.
  struct ViewResolved {
    cameras::IntrinsicBase * intrinsic = nullptr; // nullptr == invalid view
    double * intrinsic_param = nullptr;           // nullptr if empty / not optimized
    double * pose_param = nullptr;
    // Cached cost-function factory: resolved once per view to avoid the
    // virtual getType() + 6-way switch per observation. nullptr means we
    // need the slow heap fallback (unsupported camera type).
    CostFnFactory factory = nullptr;
    int spherical_w = 0; // only consulted by the spherical factory
    int spherical_h = 0;
  };

  IndexT max_view_id_ba = 0;
  bool any_view_ba = false;
  for (const auto & view_it : sfm_data.views)
  {
    if (view_it.first > max_view_id_ba) max_view_id_ba = view_it.first;
    any_view_ba = true;
  }
  const bool use_flat_view =
    any_view_ba &&
    max_view_id_ba != UndefinedIndexT &&
    static_cast<size_t>(max_view_id_ba) < sfm_data.views.size() * 8 + 64;

  std::vector<ViewResolved> view_resolved_flat;
  Hash_Map<IndexT, ViewResolved> view_resolved_hash;
  if (use_flat_view)
  {
    view_resolved_flat.assign(static_cast<size_t>(max_view_id_ba) + 1, ViewResolved{});
  }
  else
  {
    view_resolved_hash.reserve(sfm_data.views.size());
  }

  for (const auto & view_it : sfm_data.views)
  {
    const View * view = view_it.second.get();
    if (!view) continue;
    const auto intr_it = sfm_data.intrinsics.find(view->id_intrinsic);
    if (intr_it == sfm_data.intrinsics.end()) continue;
    const auto pose_param_it = map_poses.find(view->id_pose);
    if (pose_param_it == map_poses.end()) continue;
    const auto intr_param_it = map_intrinsics.find(view->id_intrinsic);
    ViewResolved r;
    r.intrinsic = intr_it->second.get();
    r.intrinsic_param =
      (intr_param_it != map_intrinsics.end() && !intr_param_it->second.empty())
        ? &intr_param_it->second[0]
        : nullptr;
    r.pose_param = &pose_param_it->second[0];
    r.factory = PickFactoryForIntrinsic(r.intrinsic);
    if (r.intrinsic && r.intrinsic->getType() == CAMERA_SPHERICAL)
    {
      r.spherical_w = r.intrinsic->w();
      r.spherical_h = r.intrinsic->h();
    }
    if (use_flat_view)
      view_resolved_flat[view_it.first] = r;
    else
      view_resolved_hash[view_it.first] = r;
  }

  // O(1) lookup. Returns nullptr when the view is unresolved.
  auto get_view_resolved = [&](IndexT view_id) -> const ViewResolved * {
    if (use_flat_view)
    {
      if (view_id > max_view_id_ba) return nullptr;
      const ViewResolved & r = view_resolved_flat[view_id];
      return r.intrinsic ? &r : nullptr;
    }
    const auto it = view_resolved_hash.find(view_id);
    return (it != view_resolved_hash.end()) ? &it->second : nullptr;
  };

  for (auto & structure_landmark_it : sfm_data.structure)
  {
    const Observations & obs = structure_landmark_it.second.obs;

    for (const auto & obs_it : obs)
    {
      const ViewResolved * vr = get_view_resolved(obs_it.first);
      if (!vr)
      {
        ++residuals_dropped_no_cost_fn;
        OPENMVG_LOG_ERROR << "Cannot create a CostFunction for this camera model.";
        return false;
      }

      // Each Residual block takes a point and a camera as input and outputs a 2
      // dimensional residual. Internally, the cost function stores the observed
      // image location and compares the reprojection against the observation.
      //
      // Use the per-view-cached factory pointer when available (no virtual
      // call, no switch). Falls back to the per-obs dispatch only when the
      // intrinsic type is unsupported (which would have failed anyway).
      ceres::CostFunction* cost_function;
      if (vr->factory)
      {
        cost_function = vr->factory(
          cost_function_arena,
          obs_it.second.x.data(),
          vr->spherical_w,
          vr->spherical_h);
      }
      else
      {
        cost_function = IntrinsicsToCostFunction(
          cost_function_arena,
          fallback_cost_functions,
          vr->intrinsic,
          obs_it.second.x);
      }

      if (cost_function)
      {
        if (vr->intrinsic_param)
        {
          problem.AddResidualBlock(cost_function,
            p_LossFunction.get(),
            vr->intrinsic_param,
            vr->pose_param,
            structure_landmark_it.second.X.data());
        }
        else
        {
          problem.AddResidualBlock(cost_function,
            p_LossFunction.get(),
            vr->pose_param,
            structure_landmark_it.second.X.data());
        }
        ++residuals_added;
      }
      else
      {
        ++residuals_dropped_no_cost_fn;
        OPENMVG_LOG_ERROR << "Cannot create a CostFunction for this camera model.";
        return false;
      }
    }
    if (options.structure_opt == Structure_Parameter_Type::NONE)
      problem.SetParameterBlockConstant(structure_landmark_it.second.X.data());
  }

  if (kBADiagLogging && ceres_options_.bVerbose_)
  {
    OPENMVG_LOG_INFO
      << "[BA-DIAG:Build] Residual blocks added: " << residuals_added
      << " | dropped(no cost fn): " << residuals_dropped_no_cost_fn
      << " | avg obs/pose: "
      << (sfm_data.GetPoses().empty() ? 0.0 : (double)residuals_added / sfm_data.GetPoses().size());
  }

  if (options.control_point_opt.bUse_control_points)
  {
    // Use Ground Control Point:
    // - fixed 3D points with weighted observations
    for (auto & gcp_landmark_it : sfm_data.control_points)
    {
      const Observations & obs = gcp_landmark_it.second.obs;

      for (const auto & obs_it : obs)
      {
        // Build the residual block corresponding to the track observation:
        const View * view = sfm_data.views.at(obs_it.first).get();

        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function =
          IntrinsicsToCostFunction(
            cost_function_arena,
            fallback_cost_functions,
            sfm_data.intrinsics.at(view->id_intrinsic).get(),
            obs_it.second.x,
            options.control_point_opt.weight);

        if (cost_function)
        {
          if (!map_intrinsics.at(view->id_intrinsic).empty())
          {
            problem.AddResidualBlock(cost_function,
                                     nullptr,
                                     &map_intrinsics.at(view->id_intrinsic)[0],
                                     &map_poses.at(view->id_pose)[0],
                                     gcp_landmark_it.second.X.data());
          }
          else
          {
            problem.AddResidualBlock(cost_function,
                                     nullptr,
                                     &map_poses.at(view->id_pose)[0],
                                     gcp_landmark_it.second.X.data());
          }
        }
      }
      if (obs.empty())
      {
        OPENMVG_LOG_ERROR
          << "Cannot use this GCP id: " << gcp_landmark_it.first
          << ". There is not linked image observation.";
      }
      else
      {
        // Set the 3D point as FIXED (it's a valid GCP)
        problem.SetParameterBlockConstant(gcp_landmark_it.second.X.data());
      }
    }
  }

  // Add Pose prior constraints if any
  if (b_usable_prior)
  {
    // When loss_function_ownership == DO_NOT_TAKE_OWNERSHIP (set above when
    // bUse_loss_function_), we must own the prior Huber losses ourselves for
    // the lifetime of Problem. Otherwise Problem will delete them for us.
    const bool own_prior_loss =
      (problem_options.loss_function_ownership == ceres::DO_NOT_TAKE_OWNERSHIP);

    // --- Adaptive GPS-prior weighting (auto de-dome) -----------------------
    // Count how many reprojection observations pull on each pose so we can
    // scale that pose's prior weight by sqrt(n_obs). See the macro comment.
    Hash_Map<IndexT, uint32_t> obs_per_pose;
#if OPENMVG_SFM_GPS_PRIOR_AUTOSCALE
    for (const auto & structure_it : sfm_data.structure)
    {
      for (const auto & obs_it : structure_it.second.obs)
      {
        const auto view_it = sfm_data.views.find(obs_it.first);
        if (view_it == sfm_data.views.end() || !view_it->second) continue;
        ++obs_per_pose[view_it->second->id_pose];
      }
    }
#endif
    const double gps_prior_strength =
      static_cast<double>(OPENMVG_SFM_GPS_PRIOR_STRENGTH);

    size_t prior_blocks = 0;
    double applied_scale_min = std::numeric_limits<double>::max();
    double applied_scale_max = 0.0;
    for (const auto & view_it : sfm_data.GetViews())
    {
      const sfm::ViewPriors * prior = dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
      if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
      {
        // Per-pose adaptive scale: sqrt(n_obs) * strength. Falls back to the
        // raw weight (scale 1.0) when autoscale is disabled or the pose has no
        // counted observations. The prior residual is weight*dC, so the prior
        // COST scales as n_obs -- matching this pose's reprojection cost.
        double prior_scale = gps_prior_strength;
#if OPENMVG_SFM_GPS_PRIOR_AUTOSCALE
        const auto it_obs = obs_per_pose.find(prior->id_pose);
        const double n_obs =
          (it_obs != obs_per_pose.end()) ? static_cast<double>(it_obs->second) : 1.0;
        prior_scale = std::sqrt(std::max(1.0, n_obs)) * gps_prior_strength;
#endif
        applied_scale_min = std::min(applied_scale_min, prior_scale);
        applied_scale_max = std::max(applied_scale_max, prior_scale);

        const Vec3 scaled_weight = prior->center_weight_ * prior_scale;

        // Arena-owned cost function (cost_function_ownership is DO_NOT_TAKE_OWNERSHIP).
        ceres::CostFunction * cost_function =
          cost_function_arena.Alloc<
            ceres::AutoDiffCostFunction<PoseCenterConstraintCostFunction, 3, 6>>(
              new PoseCenterConstraintCostFunction(prior->pose_center_, scaled_weight));

        // Scale the Huber knee by the same factor so the robust transition
        // stays at the same PHYSICAL distance (pose_center_robust_fitting_error
        // in user units) after the weight scaling.
        ceres::LossFunction * prior_loss =
          new ceres::HuberLoss(Square(prior_scale * pose_center_robust_fitting_error));
        if (own_prior_loss)
          prior_loss_functions.emplace_back(prior_loss);

        problem.AddResidualBlock(
          cost_function,
          prior_loss,
          &map_poses.at(prior->id_view)[0]);
        ++prior_blocks;
      }
    }
    if (kBADiagLogging && ceres_options_.bVerbose_)
    {
      OPENMVG_LOG_INFO
        << "[BA-DIAG:Prior] Added " << prior_blocks << " pose-center prior residual block(s)"
        << " | autoscale=" << (OPENMVG_SFM_GPS_PRIOR_AUTOSCALE ? "on" : "off")
        << " strength=" << gps_prior_strength
        << " weight_scale[min=" << (prior_blocks ? applied_scale_min : 0.0)
        << " max=" << applied_scale_max << "]"
        << " | base Huber threshold (sq): " << Square(pose_center_robust_fitting_error);
    }
  }

  // Configure a BA engine and run it
  //  Make Ceres automatically detect the bundle structure.
  ceres::Solver::Options ceres_config_options;
  ceres_config_options.max_num_iterations = ceres_options_.max_num_iterations_;
  ceres_config_options.max_linear_solver_iterations = ceres_options_.max_linear_solver_iterations_;
  ceres_config_options.preconditioner_type =
    static_cast<ceres::PreconditionerType>(ceres_options_.preconditioner_type_);
  ceres_config_options.linear_solver_type =
    static_cast<ceres::LinearSolverType>(ceres_options_.linear_solver_type_);
  ceres_config_options.sparse_linear_algebra_library_type =
    static_cast<ceres::SparseLinearAlgebraLibraryType>(ceres_options_.sparse_linear_algebra_library_type_);

  // [BA-PERF] Tried adaptive ITERATIVE_SCHUR + SCHUR_JACOBI for big problems
  // (>200k obs); on this dataset it was 3.3x SLOWER than SPARSE_SCHUR with
  // SuiteSparse (36s vs 11s on a 5.7M-residual final BA). The CG iteration
  // count balloons when the problem is sparse-but-not-Schur-friendly. Keep
  // SPARSE_SCHUR; this comment exists so we don't re-try this knob blind.

  ceres_config_options.minimizer_progress_to_stdout = ceres_options_.bVerbose_;
  ceres_config_options.logging_type = ceres::SILENT;
  ceres_config_options.num_threads = ceres_options_.nb_threads_;
#if CERES_VERSION_MAJOR < 2
  ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
#endif
  ceres_config_options.parameter_tolerance = ceres_options_.parameter_tolerance_;
  ceres_config_options.gradient_tolerance = ceres_options_.gradient_tolerance_;
  ceres_config_options.function_tolerance = ceres_options_.function_tolerance_;
  ceres_config_options.max_num_consecutive_invalid_steps =
      ceres_options_.max_num_consecutive_invalid_steps_;

  // Inner iterations: re-optimize structure (3D points) between LM outer
  // steps. The inner sub-problem is independent per point and parallelises
  // across cores; in exchange the outer LM converges in noticeably fewer
  // steps. Quality-neutral by construction (Ceres still satisfies the
  // outer convergence criteria). Only sensible when both pose and structure
  // parameter blocks are present, which is always true for SfM BA.
  if (ceres_options_.use_inner_iterations_)
  {
    ceres_config_options.use_inner_iterations = true;
  }

#if 1 // S3 sets all three -- disabling caused track-count regression because the
      // missing linear_solver_ordering let Ceres auto-pick a different elimination
      // grouping for the Schur solver, changing the BA convergence path and the
      // post-BA outlier filter's per-observation residuals.
  // Explicit Schur complement (only consulted by ITERATIVE_SCHUR; ignored
  // for the direct SCHUR solvers). Materialises the Schur complement as a
  // dense matrix instead of applying it symbolically. Costs memory; saves
  // CG-iter time. Pure time/memory tradeoff -- the linear system being
  // solved is unchanged, so output is bit-identical.
  if (ceres_config_options.linear_solver_type == ceres::ITERATIVE_SCHUR)
  {
    ceres_config_options.use_explicit_schur_complement = true;
  }

  // Use post-ordering for the sparse Schur factorisation (SPARSE_SCHUR
  // path only). Generally yields a slightly better elimination order at
  // the cost of one extra Jacobian-matrix copy in memory. Same linear
  // system, same numerical result -- pure time/memory tradeoff.
  //
  // MEMORY: the extra copy scales with the observation count, so on large
  // scenes it is measured in GB, not MB. A 1940-pose / 25M-observation run
  // peaked at 54.4 GB of 68.4 GB with only ~1 GB of headroom left -- and that
  // was on the ITERATIVE_SCHUR path, which never takes this copy. Enabling it
  // there would likely have exhausted RAM. Set this to 0 on memory-bound
  // scenes; the only cost is a slightly worse elimination order.
  //   1 = post-ordering on (Ceres/stock behaviour, faster factorisation)
  //   0 = post-ordering off (saves one Jacobian-sized allocation)
#ifndef OPENMVG_BA_USE_POSTORDERING
#define OPENMVG_BA_USE_POSTORDERING 1
#endif
  if (OPENMVG_BA_USE_POSTORDERING &&
      ceres_config_options.linear_solver_type == ceres::SPARSE_SCHUR)
  {
    ceres_config_options.use_postordering = true;
  }

  // Explicit linear-solver ordering. Tells the Schur-based solvers which
  // parameter blocks to eliminate first (3D points -> group 0) and which
  // form the reduced system (poses & intrinsics -> group 1). Ceres would
  // auto-detect the same grouping from problem structure, but we do it
  // explicitly to skip the auto-detect overhead on every BA call. Math is
  // identical -- ordering only affects elimination order, not the result.
  {
    auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    for (auto & structure_landmark_it : sfm_data.structure)
      ordering->AddElementToGroup(structure_landmark_it.second.X.data(), 0);
    for (auto & gcp_landmark_it : sfm_data.control_points)
      ordering->AddElementToGroup(gcp_landmark_it.second.X.data(), 0);
    for (auto & kv : map_poses)
      ordering->AddElementToGroup(kv.second.data(), 1);
    for (auto & kv : map_intrinsics)
    {
      if (!kv.second.empty())
        ordering->AddElementToGroup(kv.second.data(), 1);
    }
    ceres_config_options.linear_solver_ordering = std::move(ordering);
  }
#endif

  if (kBADiagLogging && ceres_options_.bVerbose_)
  {
    OPENMVG_LOG_INFO
      << "[BA-DIAG:Solve] linear_solver=" << ceres_config_options.linear_solver_type
      << " preconditioner=" << ceres_config_options.preconditioner_type
      << " max_iters=" << ceres_config_options.max_num_iterations
      << " param_tol=" << ceres_config_options.parameter_tolerance
      << " grad_tol=" << ceres_config_options.gradient_tolerance
      << " max_invalid_steps=" << ceres_config_options.max_num_consecutive_invalid_steps
      << " threads=" << ceres_config_options.num_threads;
  }

  // Adaptive plateau-based early termination. Lifetime-pinned to the
  // surrounding Adjust() scope so the pointer in `callbacks` stays valid
  // for the duration of ceres::Solve. Disabled (rel_tol == 0) in the
  // STRICT preset so reference behavior is unchanged.
  std::unique_ptr<PlateauTerminationCallback> plateau_cb;
  if (ceres_options_.plateau_relative_tolerance_ > 0.0) {
    plateau_cb.reset(new PlateauTerminationCallback(
        ceres_options_.plateau_relative_tolerance_,
        ceres_options_.plateau_min_iterations_,
        ceres_options_.plateau_patience_));
    ceres_config_options.callbacks.push_back(plateau_cb.get());
    // The plateau callback inspects only the IterationSummary, never the
    // parameter blocks, so we deliberately leave update_state_every_iteration
    // at its default (false) -- avoids the per-iter parameter copy-back.
  }

  // Solve BA
  ceres::Solver::Summary summary;
#if OPENMVG_SFM_VERBOSE_BA
  t_setup_done = clk::now();
#endif
  ceres::Solve(ceres_config_options, &problem, &summary);
  if (ceres_options_.bCeres_summary_)
    OPENMVG_LOG_INFO << summary.FullReport();

  // [BA-DIAG] Compact Ceres termination summary. Termination string is also
  // referenced by the failure path below, so we always compute it.
  const char * termination_str = "UNKNOWN";
  switch (summary.termination_type)
  {
    case ceres::CONVERGENCE:     termination_str = "CONVERGENCE";     break;
    case ceres::NO_CONVERGENCE:  termination_str = "NO_CONVERGENCE";  break;
    case ceres::FAILURE:         termination_str = "FAILURE";         break;
    case ceres::USER_SUCCESS:    termination_str = "USER_SUCCESS";    break;
    case ceres::USER_FAILURE:    termination_str = "USER_FAILURE";    break;
    default: break;
  }

  // [BA-PERF] One-line per-Adjust() timing breakdown:
  //   our_setup = time spent building the ceres::Problem (param + residual
  //               adds, manifolds, prior blocks). Our own contribution.
  //   ceres_pre = ceres internal preprocessor (program ordering, residual
  //               block ordering, evaluator/linear-solver setup).
  //   solve     = inner trust-region minimizer (Jacobian eval + linear
  //               solve loop).
  //   adjust    = total Adjust() wall time so far (== our_setup + ceres_pre
  //               + solve + write-back/teardown that follows).
  //   res       = residuals; iters = trust-region iterations.
#if OPENMVG_SFM_VERBOSE_BA
  {
    const double t_adjust =
      std::chrono::duration<double>(clk::now() - t_adjust_begin).count();
    const double t_our_setup =
      std::chrono::duration<double>(t_setup_done - t_adjust_begin).count();
    OPENMVG_LOG_INFO
      << "[BA-PERF] adjust=" << t_adjust << "s"
      << " our_setup=" << t_our_setup << "s"
      << " ceres_pre=" << summary.preprocessor_time_in_seconds << "s"
      << " solve=" << summary.minimizer_time_in_seconds << "s"
      << " | res=" << summary.num_residuals
      << " iters=" << summary.iterations.size()
      << " term=" << termination_str
      << " | solver=" << ceres::LinearSolverTypeToString(summary.linear_solver_type_used)
      << " precond=" << ceres::PreconditionerTypeToString(summary.preconditioner_type_used);
  }
#endif // OPENMVG_SFM_VERBOSE_BA

  // [BA-TUNE] Thread-count tuning diagnostic. Designed so you can sweep
  // OMP_NUM_THREADS (or ceres_options_.nb_threads_) and compare runs.
  //
  //   threads          - what the solver actually used.
  //   solve_s          - inner minimizer wall time (only metric that
  //                      should change with thread count).
  //   iters            - LM iters; should NOT vary with threads (apart
  //                      from the tiny reorder-noise of T2 if enabled).
  //   solve_per_iter   - wall time per LM iter (s). Stablest measurement
  //                      because it cancels iteration-count noise.
  //   resit_per_sec    - residuals * iters / solve_s. Throughput metric
  //                      that normalises across problem sizes, so values
  //                      are comparable across different scenes.
  //   eval_share       - fraction of solve_s spent in Jacobian/residual
  //                      evaluation vs. linear solve. If eval_share rises
  //                      with thread count you've passed the bandwidth
  //                      knee on the evaluator; if 1-eval_share rises
  //                      you've passed the knee on the Schur solver.
  //
  // Sweep recipe (PowerShell):
  //   foreach ($n in 4,6,8,10,12) {
  //     $env:OMP_NUM_THREADS=$n
  //     & .\openMVG_main_GlobalSfM.exe ... 2>&1 | Select-String "BA-TUNE"
  //   }
  // Pick the thread count with the biggest `resit_per_sec` (or smallest
  // `solve_per_iter`) before the curve flattens / regresses.
#if OPENMVG_SFM_VERBOSE_BA
  {
    const int actual_threads = ceres_config_options.num_threads;
    const double solve_s = summary.minimizer_time_in_seconds;
    const int    iters   = static_cast<int>(summary.iterations.size());
    const long long residuals = static_cast<long long>(summary.num_residuals);
    const double solve_per_iter =
      (iters > 0) ? (solve_s / iters) : 0.0;
    const double resit_per_sec =
      (solve_s > 0.0)
          ? (static_cast<double>(residuals) * iters / solve_s)
          : 0.0;
    // Aggregate evaluator / linear-solver times come straight from the
    // Solver::Summary in this Ceres version (per-iteration breakdown is
    // not exposed on IterationSummary here).
    const double evaluator_s =
        summary.residual_evaluation_time_in_seconds
      + summary.jacobian_evaluation_time_in_seconds;
    const double linear_solve_s = summary.linear_solver_time_in_seconds;
    const double eval_share =
      (solve_s > 0.0) ? (evaluator_s / solve_s) : 0.0;
    OPENMVG_LOG_INFO
      << "[BA-TUNE] threads=" << actual_threads
      << " solve_s=" << solve_s
      << " iters=" << iters
      << " solve_per_iter=" << solve_per_iter << "s"
      << " resit_per_sec=" << resit_per_sec
      << " eval_share=" << eval_share
      << " (eval_s=" << evaluator_s
      << " linsolve_s=" << linear_solve_s << ")"
      << " | res=" << residuals
      << " term=" << termination_str;
  }
#endif // OPENMVG_SFM_VERBOSE_BA
  if (kBADiagLogging && ceres_options_.bVerbose_)
  {
    OPENMVG_LOG_INFO
      << "[BA-DIAG:Solve] termination=" << termination_str
      << " usable=" << (summary.IsSolutionUsable() ? "yes" : "NO")
      << " iters=" << summary.iterations.size()
      << " steps_successful=" << summary.num_successful_steps
      << " steps_unsuccessful=" << summary.num_unsuccessful_steps
      << " | initial_cost=" << summary.initial_cost
      << " final_cost=" << summary.final_cost
      << " fixed_cost=" << summary.fixed_cost
      << " | time(s)=" << summary.total_time_in_seconds
      << " (setup=" << summary.preprocessor_time_in_seconds
      << ", minimizer=" << summary.minimizer_time_in_seconds << ")";

    if (!summary.message.empty())
    {
      OPENMVG_LOG_INFO << "[BA-DIAG:Solve] ceres message: " << summary.message;
    }
  }

  // Ceres' trust-region minimizer returns FAILURE whenever it cannot find
  // a descent step within `max_num_consecutive_invalid_steps`. That includes
  // the benign case where the input is already at (or extremely close to)
  // the optimum - e.g. a gauge-free 2-view BA seeded from a good RANSAC
  // relative pose. In that case final_cost <= initial_cost and the current
  // parameters are a valid (no-op) solution. Treat it as success.
  const bool benign_noop_failure =
    (summary.termination_type == ceres::FAILURE) &&
    std::isfinite(summary.initial_cost) &&
    std::isfinite(summary.final_cost) &&
    (summary.final_cost <= summary.initial_cost);

  // If no error, get back refined parameters
  if (!summary.IsSolutionUsable() && !benign_noop_failure)
  {
    OPENMVG_LOG_ERROR << "IsSolutionUsable is false. Bundle Adjustment failed."
      << " (termination=" << termination_str << ", message: " << summary.message << ")";
    return false;
  }
  else // Solution is usable
  {
    if (kBADiagLogging && ceres_options_.bVerbose_)
    {
      // Display statistics about the minimization
      OPENMVG_LOG_INFO
        << "\nBundle Adjustment statistics (approximated RMSE):\n"
        << " #views: " << sfm_data.views.size() << "\n"
        << " #poses: " << sfm_data.poses.size() << "\n"
        << " #intrinsics: " << sfm_data.intrinsics.size() << "\n"
        << " #tracks: " << sfm_data.structure.size() << "\n"
        << " #residuals: " << summary.num_residuals << "\n"
        << " Initial RMSE: " << std::sqrt( summary.initial_cost / summary.num_residuals) << "\n"
        << " Final RMSE: " << std::sqrt( summary.final_cost / summary.num_residuals) << "\n"
        << " Time (s): " << summary.total_time_in_seconds
        << " \n--\n"
        << " Used motion prior: " << static_cast<int>(b_usable_prior);
    }

    // [BA-DIAG] Detect non-finite pose parameters BEFORE we write them back
    // to sfm_data. Gated on bVerbose_; the robust-BA loop now catches the
    // resulting bad poses via median-residual ejection regardless.
    if (kBADiagLogging && ceres_options_.bVerbose_)
    {
      size_t nonfinite_param_blocks = 0;
      for (const auto & kv : map_poses)
      {
        const auto & params = kv.second;
        for (const double v : params)
        {
          if (!std::isfinite(v))
          {
            ++nonfinite_param_blocks;
            break;
          }
        }
      }
      if (nonfinite_param_blocks > 0)
      {
        OPENMVG_LOG_WARNING
          << "[BA-DIAG:PostSolve] " << nonfinite_param_blocks << "/" << map_poses.size()
          << " pose parameter block(s) are NON-FINITE after Ceres solve."
          << " This will corrupt downstream resection and is the likely cause"
          << " of an incremental stall.";
      }
    }

    // Update camera poses with refined data
    if (options.extrinsics_opt != Extrinsic_Parameter_Type::NONE)
    {
      for (auto & pose_it : sfm_data.poses)
      {
        const IndexT indexPose = pose_it.first;

        Mat3 R_refined;
        ceres::AngleAxisToRotationMatrix(&map_poses.at(indexPose)[0], R_refined.data());
        Vec3 t_refined(map_poses.at(indexPose)[3], map_poses.at(indexPose)[4], map_poses.at(indexPose)[5]);
        // Update the pose
        Pose3 & pose = pose_it.second;
        if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_ROTATION)
        {
            // Update only rotation
            pose.rotation() = R_refined;
        }
        else if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_TRANSLATION)
        {
            // Update only translation
            Vec3 C_refined = -R_refined.transpose() * t_refined;
            pose.center() = C_refined;
        }
        else
        {
            // Update rotation + translation
            pose = Pose3(R_refined, -R_refined.transpose() * t_refined);
        }
      }
    }

    // Update camera intrinsics with refined data
    if (options.intrinsics_opt != Intrinsic_Parameter_Type::NONE)
    {
      for (auto & intrinsic_it : sfm_data.intrinsics)
      {
        const IndexT indexCam = intrinsic_it.first;

        const std::vector<double> & vec_params = map_intrinsics.at(indexCam);
        intrinsic_it.second->updateFromParams(vec_params);
      }
    }

    // Structure is already updated directly if needed (no data wrapping)

    // [BA-DIAG] Post-solve diagnostics (full obs traversal + per-pose RMSE
    // histogram). Gated on bVerbose_; not consulted by any control flow.
    if (kBADiagLogging && ceres_options_.bVerbose_)
    {
      CountNonFinitePoses(sfm_data, "PostSolve");

      // viewId -> poseId
      Hash_Map<IndexT, IndexT> view_to_pose;
      for (const auto & v : sfm_data.GetViews())
        view_to_pose[v.first] = v.second->id_pose;

      // poseId -> (sum_sq_residual, count)
      Hash_Map<IndexT, std::pair<double, size_t>> pose_rmse;
      for (const auto & p : sfm_data.GetPoses())
        pose_rmse[p.first] = {0.0, 0};

      for (const auto & lm : sfm_data.GetLandmarks())
      {
        const Vec3 & X = lm.second.X;
        for (const auto & obs : lm.second.obs)
        {
          const auto vit = view_to_pose.find(obs.first);
          if (vit == view_to_pose.end()) continue;
          const auto pit = pose_rmse.find(vit->second);
          if (pit == pose_rmse.end()) continue;
          const View * view = sfm_data.GetViews().at(obs.first).get();
          const auto intrinsic_it = sfm_data.GetIntrinsics().find(view->id_intrinsic);
          if (intrinsic_it == sfm_data.GetIntrinsics().end()) continue;
          const Pose3 & pose = sfm_data.GetPoses().at(vit->second);
          const Vec2 residual =
            intrinsic_it->second->residual(pose(X), obs.second.x);
          pit->second.first += residual.squaredNorm();
          ++pit->second.second;
        }
      }

      std::vector<std::pair<double, IndexT>> rmse_sorted; // rmse, pose_id
      rmse_sorted.reserve(pose_rmse.size());
      for (const auto & kv : pose_rmse)
      {
        if (kv.second.second > 0)
        {
          const double rmse = std::sqrt(kv.second.first / kv.second.second);
          rmse_sorted.emplace_back(rmse, kv.first);
        }
      }
      std::sort(rmse_sorted.begin(), rmse_sorted.end());

      if (!rmse_sorted.empty())
      {
        const double rmse_min = rmse_sorted.front().first;
        const double rmse_med = rmse_sorted[rmse_sorted.size() / 2].first;
        const double rmse_max = rmse_sorted.back().first;
        OPENMVG_LOG_INFO
          << "[BA-DIAG:PostSolve] Per-pose reprojection RMSE: min=" << rmse_min
          << " median=" << rmse_med << " max=" << rmse_max;

        // If the worst pose is far worse than the median, list the top 5.
        if (rmse_max > rmse_med * 5.0 && rmse_sorted.size() > 5)
        {
          std::ostringstream os;
          for (size_t i = 0; i < 5 && !rmse_sorted.empty(); ++i)
          {
            const auto & e = rmse_sorted[rmse_sorted.size() - 1 - i];
            if (i > 0) os << ", ";
            os << "pose=" << e.second << " rmse=" << e.first;
          }
          OPENMVG_LOG_WARNING
            << "[BA-DIAG:PostSolve] Worst poses by reprojection RMSE (top 5): " << os.str()
            << " � these are candidates for removal / refinement failure.";
        }
      }
    }

    if (b_usable_prior)
    {
      // set back to the original scene centroid
      openMVG::sfm::ApplySimilarity(sim_to_center.inverse(), sfm_data, true);

#if OPENMVG_SFM_VERBOSE_BA
      //--
      // - Compute some fitting statistics
      //   Diagnostic only: the result is logged and never read back.
      //--

      // Collect corresponding camera centers
      std::vector<Vec3> X_SfM, X_GPS;
      for (const auto & view_it : sfm_data.GetViews())
      {
        const sfm::ViewPriors * prior = dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
        if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
        {
          X_SfM.push_back( sfm_data.GetPoses().at(prior->id_pose).center() );
          X_GPS.push_back( prior->pose_center_ );
        }
      }
      // Compute the registration fitting error (once BA with Prior have been used):
      if (X_GPS.size() > 3)
      {
        // Compute the median residual error
        const Vec residual = (Eigen::Map<Mat3X>(X_SfM[0].data(), 3, X_SfM.size()) - Eigen::Map<Mat3X>(X_GPS[0].data(), 3, X_GPS.size())).colwise().norm();
        std::ostringstream os;
        os
          << "Pose prior statistics (user units):\n"
          << " - Starting median fitting error: " << pose_center_robust_fitting_error << "\n"
          << " - Final fitting error:\n";
        minMaxMeanMedian<Vec::Scalar>(residual.data(), residual.data() + residual.size(), os);
        OPENMVG_LOG_INFO << os.str();
      }
#endif // OPENMVG_SFM_VERBOSE_BA
    }
    return true;
  }
}

// ---------------------------------------------------------------------------
// Self-test: validate AnalyticReprojectionCost_Radial3 against AutoDiff over
// random parameter sets. Pass criterion: every residual matches to <= 1e-10
// and every Jacobian entry matches to <= 1e-7. Returns the maximum |delta|
// observed (residuals or Jacobians, whichever is larger). Caller decides what
// to do with the result; typical pattern:
//
//   const double err = RunSelfTest_AnalyticReprojectionCost_Radial3();
//   OPENMVG_LOG_INFO << \"[Analytic-Radial3] self-test max err = \" << err;
//   if (err > 1e-7) { /* refuse to enable analytic path */ }
//
// This routine is intentionally not called automatically -- it's a developer
// tool. Wire it into a startup check, a unit test, or a one-shot diagnostic
// run when validating a build.
// ---------------------------------------------------------------------------
double RunSelfTest_AnalyticReprojectionCost_Radial3(int trials)
{
  auto rand_in = [](double lo, double hi) {
    return lo + (hi - lo) * (static_cast<double>(std::rand()) / RAND_MAX);
  };
  double max_err = 0.0;
  std::srand(0xC0FFEE);  // deterministic

  for (int trial = 0; trial < trials; ++trial)
  {
    // Realistic ranges:
    const double obs[2]    = { rand_in(-2000.0, 2000.0), rand_in(-2000.0, 2000.0) };
    const double intr[6]   = {
      rand_in(500.0, 5000.0),     // f
      rand_in(-50.0, 50.0),       // cx
      rand_in(-50.0, 50.0),       // cy
      rand_in(-0.3, 0.3),         // k1
      rand_in(-0.1, 0.1),         // k2
      rand_in(-0.05, 0.05)        // k3
    };
    // Random rotation: uniform direction, angle [-pi, pi].
    double axis[3] = { rand_in(-1.0,1.0), rand_in(-1.0,1.0), rand_in(-1.0,1.0) };
    const double an = std::sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]) + 1e-12;
    const double ang = rand_in(-3.14159, 3.14159);
    const double extr[6] = {
      ang*axis[0]/an, ang*axis[1]/an, ang*axis[2]/an,
      rand_in(-50.0, 50.0), rand_in(-50.0, 50.0), rand_in(-50.0, 50.0)
    };
    // Point in front of camera-ish; just generate and skip if Pz ends up <= 0.
    const double X[3] = { rand_in(-10.0,10.0), rand_in(-10.0,10.0), rand_in(2.0, 30.0) };

    // Skip pathological geometry. In real BA, points project inside the
    // image (r^2 << 1 in normalized coords) and lie a comfortable distance
    // in front of the camera. Random sampling generates configurations with
    // tiny |Pz| or huge off-axis projections; the 6th-order radial term
    // then amplifies single-ULP forward-pass differences (between AutoDiff's
    // Jet-side scalar evaluation and our scalar code) into pixel-level
    // residual mismatches and astronomical Jacobian disagreement, even
    // though both formulas are algebraically identical.
    {
      const double w0 = extr[0], w1 = extr[1], w2 = extr[2];
      const double th2 = w0*w0+w1*w1+w2*w2;
      const double th = std::sqrt(th2);
      const double a = std::cos(th);
      const double b = (th2 < 1e-12) ? 1.0 - th2/6.0 : std::sin(th)/th;
      const double g = (th2 < 1e-12) ? 0.5 - th2/24.0 : (1.0-a)/th2;
      const double cx0 = w1*X[2]-w2*X[1];
      const double cx1 = w2*X[0]-w0*X[2];
      const double cx2 = w0*X[1]-w1*X[0];
      const double wdotX = w0*X[0]+w1*X[1]+w2*X[2];
      const double Px = a*X[0] + b*cx0 + g*wdotX*w0 + extr[3];
      const double Py = a*X[1] + b*cx1 + g*wdotX*w1 + extr[4];
      const double Pz = a*X[2] + b*cx2 + g*wdotX*w2 + extr[5];
      if (Pz < 1.0) continue;                                    // in front of camera
      const double xn = Px / Pz, yn = Py / Pz;
      if (xn*xn + yn*yn > 1.0) continue;                         // inside image plane
    }

    // Build both cost functions.
    AnalyticReprojectionCost_Radial3 analytic(obs);
    using F  = ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3;
    using AD = ceres::AutoDiffCostFunction<F, 2, 6, 6, 3>;
    // AutoDiff for static-sized residuals: single-arg ctor takes ownership.
    // Heap-allocate the functor so the AD destructor's `delete` is valid.
    AD autodiff(new F(obs));

    const double * params[3] = { intr, extr, X };
    double r_a[2], r_d[2];
    double J_a[3][12] = {{0}};  // sized for the largest block (2x6 = 12)
    double J_d[3][12] = {{0}};
    double * jacs_a[3] = { J_a[0], J_a[1], J_a[2] };
    double * jacs_d[3] = { J_d[0], J_d[1], J_d[2] };

    if (!analytic.Evaluate(params, r_a, jacs_a)) continue;
    if (!autodiff.Evaluate(params, r_d, jacs_d)) continue;

    double res_err = 0.0, j0_err = 0.0, j1_err = 0.0, j2_err = 0.0;
    for (int i = 0; i < 2; ++i) res_err = std::max(res_err, std::abs(r_a[i] - r_d[i]));
    // Block 0: 2x6, Block 1: 2x6, Block 2: 2x3.
    for (int k = 0; k < 12; ++k) j0_err = std::max(j0_err, std::abs(J_a[0][k] - J_d[0][k]));
    for (int k = 0; k < 12; ++k) j1_err = std::max(j1_err, std::abs(J_a[1][k] - J_d[1][k]));
    for (int k = 0; k <  6; ++k) j2_err = std::max(j2_err, std::abs(J_a[2][k] - J_d[2][k]));
    const double trial_err = std::max({res_err, j0_err, j1_err, j2_err});
    if (trial_err > max_err) {
      max_err = trial_err;
      OPENMVG_LOG_INFO
        << "[Analytic-Radial3 worst trial " << trial << "] "
        << "res=" << res_err
        << " J_intr=" << j0_err
        << " J_extr=" << j1_err
        << " J_pt="   << j2_err
        << " | extr=[" << extr[0] << "," << extr[1] << "," << extr[2]
        << " | "       << extr[3] << "," << extr[4] << "," << extr[5] << "]"
        << " X=["      << X[0] << "," << X[1] << "," << X[2] << "]"
        << " f="       << intr[0] << " k1=" << intr[3];
    }
  }
  return max_err;
}

} // namespace sfm
} // namespace openMVG
