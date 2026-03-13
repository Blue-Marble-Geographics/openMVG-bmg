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
#include "openMVG/sfm/sfm_data_transform.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/types.hpp"

#include <ceres/rotation.h>
#include <ceres/types.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace openMVG {
namespace sfm {

#define OPENMVG_CERES_HAS_MANIFOLD ((CERES_VERSION_MAJOR * 100 + CERES_VERSION_MINOR) >= 201)

using namespace openMVG::cameras;
using namespace openMVG::geometry;

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

/// Arena allocator for ceres::CostFunction objects.
/// Allocates cost functions from large contiguous memory blocks to avoid
/// per-observation heap allocation overhead. When the arena is destroyed,
/// all cost functions are destructed in bulk — far cheaper than having
/// ceres::Problem individually delete each one via STLDeleteUniqueContainerPointers.
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
    // Destroy all cost functions in reverse order, then free blocks
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it)
    {
      it->destructor(it->ptr);
    }
    for (auto* block : blocks_)
    {
      ::operator delete(block);
    }
  }

  /// Allocate and construct any type T with the given constructor args.
  /// Returns a raw pointer (lifetime managed by the arena).
  /// The destructor will be called when the arena is destroyed.
  template <typename T, typename... Args>
  T* Alloc(Args&&... args)
  {
    constexpr size_t alignment = alignof(T);
    constexpr size_t size = sizeof(T);

    size_t aligned_offset = (current_offset_ + alignment - 1) & ~(alignment - 1);
    if (current_block_ == nullptr || aligned_offset + size > current_capacity_)
    {
      AllocateBlock(std::max(block_size_, size + alignment));
      aligned_offset = (current_offset_ + alignment - 1) & ~(alignment - 1);
    }

    void* mem = static_cast<char*>(current_block_) + aligned_offset;
    current_offset_ = aligned_offset + size;

    T* obj = new (mem) T(std::forward<Args>(args)...);
    entries_.push_back({obj, [](void* p) { static_cast<T*>(p)->~T(); }});
    return obj;
  }

  /// Allocate and construct a CostFunction of type T with the given constructor args.
  /// Returns a raw pointer (lifetime managed by the arena).
  template <typename T, typename... Args>
  T* Create(Args&&... args)
  {
    static_assert(std::is_base_of<ceres::CostFunction, T>::value,
                  "T must derive from ceres::CostFunction");
    return Alloc<T>(std::forward<Args>(args)...);
  }

  /// Pre-reserve the entries vector to avoid reallocation during construction.
  void Reserve(size_t count)
  {
    entries_.reserve(count);
  }

  CostFunctionArena(const CostFunctionArena&) = delete;
  CostFunctionArena& operator=(const CostFunctionArena&) = delete;

private:
  void AllocateBlock(size_t min_size)
  {
    size_t alloc_size = std::max(block_size_, min_size);
    void* block = ::operator new(alloc_size);
    blocks_.push_back(block);
    current_block_ = block;
    current_offset_ = 0;
    current_capacity_ = alloc_size;
  }

  struct Entry
  {
    void* ptr;
    void (*destructor)(void*);
  };

  size_t block_size_;
  void* current_block_;
  size_t current_offset_;
  size_t current_capacity_;
  std::vector<void*> blocks_;
  std::vector<Entry> entries_;
};

/// Create the appropriate cost functor according the provided input camera intrinsic model.
/// The residual can be weighted if desired (default 0.0 means no weight).
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

/// Arena-based version: constructs cost functions in the provided arena.
/// Returns a pointer whose lifetime is managed by the arena (no individual delete).
/// For non-pinhole camera types (fisheye, spherical), falls back to heap allocation
/// and stores the pointer in fallback_storage for manual cleanup.
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
    // Weighted path (GCPs): heap-allocate and track for cleanup.
    auto* cf = IntrinsicsToCostFunction(intrinsic, observation, weight);
    fallback_storage.emplace_back(cf);
    return cf;
  }

  switch (intrinsic->getType())
  {
    case PINHOLE_CAMERA:
      return arena.Create<AnalyticCostFunction_Pinhole>(
        observation[0], observation[1]);

    case PINHOLE_CAMERA_RADIAL1:
      return arena.Create<AnalyticCostFunction_Pinhole_Radial_K1>(
        observation[0], observation[1]);

    case PINHOLE_CAMERA_RADIAL3:
      return arena.Create<AnalyticCostFunction_Pinhole_Radial_K3>(
        observation[0], observation[1]);

    case PINHOLE_CAMERA_BROWN:
      return arena.Create<AnalyticCostFunction_Pinhole_Brown_T2>(
        observation[0], observation[1]);

    default:
    {
      // Fisheye, spherical, and any future models: heap-allocate and track.
      auto* cf = IntrinsicsToCostFunction(intrinsic, observation, 0.0);
      fallback_storage.emplace_back(cf);
      return cf;
    }
  }
}

Bundle_Adjustment_Ceres::BA_Ceres_options::BA_Ceres_options
(
  const bool bVerbose,
  bool bmultithreaded
)
: bVerbose_(bVerbose),
  nb_threads_(1),
  parameter_tolerance_(1.5e-3),
  function_tolerance_(1.5e-3 * 32),
  gradient_tolerance_(1.5e-3 * 32 * 1e-4),
  bUse_loss_function_(true),
  max_num_iterations_(5),
  max_linear_solver_iterations_(500),
  use_nonmonotonic_steps_(true),
  max_consecutive_nonmonotonic_steps_(2),
  initial_trust_region_radius_(1e4),
  max_trust_region_radius_(1e15),
  min_trust_region_radius_(1e-31),
  max_num_consecutive_invalid_steps_(5)
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
  SfM_Data& sfm_data,     // the SfM scene to refine
  const Optimize_Options& options
)
{
  //----------
  // Add camera parameters
  // - intrinsics
  // - poses [R|t]

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  //----------

  // Arena for bulk-allocating cost functions.
  // All cost functions created via the arena overload live here;
  // we tell Ceres DO_NOT_TAKE_OWNERSHIP so ~Problem won't delete them individually.
  // The arena destructor runs after the Problem is destroyed, cleaning up in bulk.
  CostFunctionArena cost_function_arena;
  // Fallback cost functions for non-pinhole camera types (fisheye, spherical).
  // These are heap-allocated by IntrinsicsToCostFunction and must be manually
  // freed after the Problem is solved (Problem uses DO_NOT_TAKE_OWNERSHIP).
  std::vector<std::unique_ptr<ceres::CostFunction>> fallback_cost_functions;

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
      for (const auto& view_it : sfm_data.GetViews())
      {
        const sfm::ViewPriors* prior = dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
        if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
        {
          X_SfM.push_back(sfm_data.GetPoses().at(prior->id_pose).center());
          X_GPS.push_back(prior->pose_center_);
        }
      }
      openMVG::geometry::Similarity3 sim;

      // Compute the registration:
      if (X_GPS.size() > 3)
      {
        const Mat X_SfM_Mat = Eigen::Map<Mat>(X_SfM[0].data(), 3, X_SfM.size());
        const Mat X_GPS_Mat = Eigen::Map<Mat>(X_GPS[0].data(), 3, X_GPS.size());
        geometry::kernel::Similarity3_Kernel kernel(X_SfM_Mat, X_GPS_Mat);
        const double lmeds_median = openMVG::robust::LeastMedianOfSquares(kernel, &sim);
        if (lmeds_median != std::numeric_limits<double>::max())
        {
          b_usable_prior = true; // PRIOR can be used safely

          // Compute the median residual error once the registration is applied
          for (Vec3& pos : X_SfM) // Transform SfM poses for residual computation
          {
            pos = sim(pos);
          }
          Vec residual = (Eigen::Map<Mat3X>(X_SfM[0].data(), 3, X_SfM.size()) - Eigen::Map<Mat3X>(X_GPS[0].data(), 3, X_GPS.size())).colwise().norm();
          std::sort(residual.data(), residual.data() + residual.size());
          pose_center_robust_fitting_error = residual(residual.size() / 2);

          // Apply the found transformation to the SfM Data Scene
          openMVG::sfm::ApplySimilarity(sim, sfm_data);

          // Move entire scene to center for better numerical stability
          Vec3 pose_centroid = Vec3::Zero();
          for (const auto& pose_it : sfm_data.poses)
          {
            pose_centroid += (pose_it.second.center() / (double)sfm_data.poses.size());
          }
          sim_to_center = openMVG::geometry::Similarity3(openMVG::sfm::Pose3(Mat3::Identity(), pose_centroid), 1.0);
          openMVG::sfm::ApplySimilarity(sim_to_center, sfm_data, true);
        }
      }
      else
      {
        OPENMVG_LOG_WARNING << "Cannot used the motion prior, insufficient number of motion priors/poses";
      }
    }
  }

  ceres::Problem::Options problem_options;
  problem_options.enable_fast_removal = false;
  problem_options.disable_all_safety_checks = true;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

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
  for (const auto& pose_it : sfm_data.poses)
  {
    const IndexT indexPose = pose_it.first;

    const Pose3& pose = pose_it.second;
    const Mat3 R = pose.rotation();
    const Vec3 t = pose.translation();

    // Guard against NaN/Inf poses (e.g. from degenerate resection or triangulation)
    if (!R.allFinite() || !t.allFinite())
    {
      OPENMVG_LOG_WARNING << "Pose " << indexPose << " contains non-finite values, skipping in BA.";
      continue;
    }

    double angleAxis[3];
    ceres::RotationMatrixToAngleAxis((const double*)R.data(), angleAxis);

    // Verify angle-axis conversion produced finite values
    if (!std::isfinite(angleAxis[0]) || !std::isfinite(angleAxis[1]) || !std::isfinite(angleAxis[2]))
    {
      OPENMVG_LOG_WARNING << "Pose " << indexPose << " produced non-finite angle-axis, skipping in BA.";
      continue;
    }

    // angleAxis + translation
    map_poses[indexPose] = { angleAxis[0], angleAxis[1], angleAxis[2], t(0), t(1), t(2) };

    double* parameter_block = &map_poses.at(indexPose)[0];
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
        vec_constant_extrinsic.insert(vec_constant_extrinsic.end(), { 0,1,2 });
      }
      // If we adjust only the rotation, we must set TRANSLATION as constant
      if (options.extrinsics_opt == Extrinsic_Parameter_Type::ADJUST_ROTATION)
      {
        // Subset translation parametrization
        vec_constant_extrinsic.insert(vec_constant_extrinsic.end(), { 3,4,5 });
      }
      if (!vec_constant_extrinsic.empty())
      {
#if OPENMVG_CERES_HAS_MANIFOLD
        auto* subset_manifold = new ceres::SubsetManifold(6, vec_constant_extrinsic);
        problem.SetManifold(parameter_block, subset_manifold);
#else
        auto* subset_parameterization =
          new ceres::SubsetParameterization(6, vec_constant_extrinsic);
        problem.SetParameterization(parameter_block, subset_parameterization);
#endif
      }
    }
  }

  // Setup Intrinsics data & subparametrization
  for (const auto& intrinsic_it : sfm_data.intrinsics)
  {
    const IndexT indexCam = intrinsic_it.first;

    if (isValid(intrinsic_it.second->getType()))
    {
      map_intrinsics[indexCam] = intrinsic_it.second->getParams();
      if (!map_intrinsics.at(indexCam).empty())
      {
        double* parameter_block = &map_intrinsics.at(indexCam)[0];
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
            auto* subset_parameterization =
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

  // Pre-reserve arena entries and Ceres-internal vectors to avoid
  // repeated heap reallocations during bulk problem construction.
  {
    size_t total_observations = 0;
    for (const auto & landmark_it : sfm_data.structure)
      total_observations += landmark_it.second.obs.size();
    // Each observation needs 2 arena entries (functor + AutoDiffCostFunction).
    cost_function_arena.Reserve(total_observations * 2);
    // Pre-reserve Ceres-internal vectors.
    // Parameter blocks: one per landmark + one per pose + one per intrinsic.
    const size_t estimated_parameter_blocks =
      sfm_data.structure.size() + map_poses.size() + map_intrinsics.size();
    problem.ReserveResidualBlocks(total_observations);
    problem.ReserveParameterBlocks(estimated_parameter_blocks);
  }

  // For all visibility add reprojections errors:
  for (auto & structure_landmark_it : sfm_data.structure)
  {
    const Observations & obs = structure_landmark_it.second.obs;

    // Skip landmarks with non-finite 3D positions
    if (!structure_landmark_it.second.X.allFinite())
      continue;

    bool landmark_has_residuals = false;
    double* landmark_data = structure_landmark_it.second.X.data();

    for (const auto & obs_it : obs)
    {
      // Build the residual block corresponding to the track observation:
      const View * view = sfm_data.views.at(obs_it.first).get();

      // Skip observations that reference a pose not in the BA problem
      // (e.g. because the pose had NaN values and was excluded)
      const auto pose_it = map_poses.find(view->id_pose);
      if (pose_it == map_poses.end())
        continue;

      const auto intrinsic_it = map_intrinsics.find(view->id_intrinsic);

      // Each Residual block takes a point and a camera as input and outputs a 2
      // dimensional residual. Internally, the cost function stores the observed
      // image location and compares the reprojection against the observation.
      ceres::CostFunction* cost_function =
        IntrinsicsToCostFunction(cost_function_arena,
                                 fallback_cost_functions,
                                 sfm_data.intrinsics.at(view->id_intrinsic).get(),
                                 obs_it.second.x);

      if (cost_function)
      {
        if (intrinsic_it != map_intrinsics.end() && !intrinsic_it->second.empty())
        {
          problem.AddResidualBlock(cost_function,
            p_LossFunction.get(),
            &intrinsic_it->second[0],
            &pose_it->second[0],
            landmark_data);
        }
        else
        {
          problem.AddResidualBlock(cost_function,
            p_LossFunction.get(),
            &pose_it->second[0],
            landmark_data);
        }
        landmark_has_residuals = true;
      }
      else
      {
        OPENMVG_LOG_ERROR << "Cannot create a CostFunction for this camera model.";
        return false;
      }
    }
    if (landmark_has_residuals && options.structure_opt == Structure_Parameter_Type::NONE)
      problem.SetParameterBlockConstant(landmark_data);
  }

  if (options.control_point_opt.bUse_control_points)
  {
    // Use Ground Control Point:
    // - fixed 3D points with weighted observations
    for (auto & gcp_landmark_it : sfm_data.control_points)
    {
      const Observations & obs = gcp_landmark_it.second.obs;

      bool gcp_has_residuals = false;

      for (const auto & obs_it : obs)
      {
        // Build the residual block corresponding to the track observation:
        const View * view = sfm_data.views.at(obs_it.first).get();

        // Skip observations that reference a pose not in the BA problem
        if (map_poses.count(view->id_pose) == 0)
          continue;

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
          gcp_has_residuals = true;
        }
      }
      if (obs.empty())
      {
        OPENMVG_LOG_ERROR
          << "Cannot use this GCP id: " << gcp_landmark_it.first
          << ". There is not linked image observation.";
      }
      else if (gcp_has_residuals)
      {
        // Set the 3D point as FIXED (it's a valid GCP)
        problem.SetParameterBlockConstant(gcp_landmark_it.second.X.data());
      }
    }
  }

  // Add Pose prior constraints if any
  // Note: prior cost functions and loss functions are heap-allocated because their
  // number is small. We track them for manual cleanup since ownership is DO_NOT_TAKE.
  std::vector<std::unique_ptr<ceres::CostFunction>> prior_cost_functions;
  std::vector<std::unique_ptr<ceres::LossFunction>> prior_loss_functions;
  if (b_usable_prior)
  {
    for (const auto& view_it : sfm_data.GetViews())
    {
      const sfm::ViewPriors* prior = dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
      if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
      {
        // Skip if this pose was excluded from the BA (e.g. due to NaN values)
        if (map_poses.count(prior->id_pose) == 0)
          continue;

        // Add the cost functor (distance from Pose prior to the SfM_Data Pose center)
        auto* cost_function =
          new ceres::AutoDiffCostFunction<PoseCenterConstraintCostFunction, 3, 6>(
            new PoseCenterConstraintCostFunction(prior->pose_center_, prior->center_weight_));
        prior_cost_functions.emplace_back(cost_function);

        auto* loss = new ceres::HuberLoss(Square(pose_center_robust_fitting_error));
        prior_loss_functions.emplace_back(loss);

        problem.AddResidualBlock(
          cost_function,
          loss,
          &map_poses.at(prior->id_pose)[0]);
      }
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
  ceres_config_options.minimizer_progress_to_stdout = ceres_options_.bVerbose_;
  ceres_config_options.logging_type =
#if 0 // JPB WIP BUG Debugging
    ceres::PER_MINIMIZER_ITERATION;
#else
    ceres::SILENT;
#endif
  ceres_config_options.num_threads = ceres_options_.nb_threads_;
#if CERES_VERSION_MAJOR < 2
  ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
#endif
  ceres_config_options.parameter_tolerance = ceres_options_.parameter_tolerance_;
  ceres_config_options.function_tolerance = ceres_options_.function_tolerance_;
  ceres_config_options.gradient_tolerance = ceres_options_.gradient_tolerance_;
  ceres_config_options.use_nonmonotonic_steps = ceres_options_.use_nonmonotonic_steps_;
  ceres_config_options.max_consecutive_nonmonotonic_steps = ceres_options_.max_consecutive_nonmonotonic_steps_;
  ceres_config_options.initial_trust_region_radius = ceres_options_.initial_trust_region_radius_;
  ceres_config_options.max_trust_region_radius = ceres_options_.max_trust_region_radius_;
  ceres_config_options.min_trust_region_radius = ceres_options_.min_trust_region_radius_;
  ceres_config_options.max_num_consecutive_invalid_steps = ceres_options_.max_num_consecutive_invalid_steps_;

#if 0 // JPB WIP BUG Debugging
  // Log per-iteration cost to see convergence rate
  ceres_config_options.minimizer_progress_to_stdout = true;
#endif

  // Solve BA
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_config_options, &problem, &summary);
  if (ceres_options_.bCeres_summary_)
    OPENMVG_LOG_INFO << summary.FullReport();

  // If no error, get back refined parameters
  if (!summary.IsSolutionUsable())
  {
    OPENMVG_LOG_ERROR << "IsSolutionUsable is false. Bundle Adjustment failed."
      << " termination: " << ceres::TerminationTypeToString(summary.termination_type)
      << " message: " << summary.message
      << " #residuals: " << summary.num_residuals
      << " #params: " << summary.num_parameters
      << " initial_cost: " << summary.initial_cost
      << " final_cost: " << summary.final_cost
      << " #iterations: " << summary.iterations.size()
      << "\n" << summary.FullReport();

    return false;
  }
  else // Solution is usable
  {
    if (ceres_options_.bVerbose_)
    {
      // Display statistics about the minimization
      OPENMVG_LOG_INFO
        << "\nBundle Adjustment statistics (approximated RMSE):\n"
        << " #views: " << sfm_data.views.size() << "\n"
        << " #poses: " << sfm_data.poses.size() << "\n"
        << " #intrinsics: " << sfm_data.intrinsics.size() << "\n"
        << " #tracks: " << sfm_data.structure.size() << "\n"
        << " #residuals: " << summary.num_residuals << "\n"
        << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
        << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
        << " Time (s): " << summary.total_time_in_seconds
        << " \n--\n"
        << " Used motion prior: " << static_cast<int>(b_usable_prior);
    }

    // Update camera poses with refined data
    if (options.extrinsics_opt != Extrinsic_Parameter_Type::NONE)
    {
      for (auto& pose_it : sfm_data.poses)
      {
        const IndexT indexPose = pose_it.first;

        // Skip poses that were not included in the BA (e.g. due to NaN values)
        if (map_poses.count(indexPose) == 0)
          continue;

        Mat3 R_refined;
        ceres::AngleAxisToRotationMatrix(&map_poses.at(indexPose)[0], R_refined.data());
        Vec3 t_refined(map_poses.at(indexPose)[3], map_poses.at(indexPose)[4], map_poses.at(indexPose)[5]);
        // Update the pose
        Pose3& pose = pose_it.second;
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
      for (auto& intrinsic_it : sfm_data.intrinsics)
      {
        const IndexT indexCam = intrinsic_it.first;

        const std::vector<double>& vec_params = map_intrinsics.at(indexCam);
        intrinsic_it.second->updateFromParams(vec_params);
      }
    }

    // Structure is already updated directly if needed (no data wrapping)

    if (b_usable_prior)
    {
      // set back to the original scene centroid
      openMVG::sfm::ApplySimilarity(sim_to_center.inverse(), sfm_data, true);

      //--
      // - Compute some fitting statistics
      //--

      // Collect corresponding camera centers
      std::vector<Vec3> X_SfM, X_GPS;
      for (const auto& view_it : sfm_data.GetViews())
      {
        const sfm::ViewPriors* prior = dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
        if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
        {
          X_SfM.push_back(sfm_data.GetPoses().at(prior->id_pose).center());
          X_GPS.push_back(prior->pose_center_);
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
    }
    return true;
  }
}

} // namespace sfm
} // namespace openMVG
