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

#include <iostream>
#include <limits>

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
  initial_trust_region_radius_(1e6),
  max_trust_region_radius_(1e15),
  min_trust_region_radius_(1e-31),
  max_num_consecutive_invalid_steps_(1)
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

    double angleAxis[3];
    ceres::RotationMatrixToAngleAxis((const double*)R.data(), angleAxis);
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

  // Build a per-view cache to avoid repeated hash lookups in the residual loop
  // Maps view_id -> { intrinsic pointer, intrinsic parameter block, pose parameter block }
  struct ViewResidualCache {
    IntrinsicBase* intrinsic;
    double* intrinsic_block;  // nullptr if intrinsic params are empty
    double* pose_block;
  };
  Hash_Map<IndexT, ViewResidualCache> view_residual_cache;
  view_residual_cache.reserve(sfm_data.views.size());
  for (const auto& view_it : sfm_data.views)
  {
    const View* v = view_it.second.get();
    const auto pose_map_it = map_poses.find(v->id_pose);
    if (pose_map_it == map_poses.end())
      continue;
    const auto intrinsic_sfm_it = sfm_data.intrinsics.find(v->id_intrinsic);
    if (intrinsic_sfm_it == sfm_data.intrinsics.end())
      continue;
    const auto intrinsic_map_it = map_intrinsics.find(v->id_intrinsic);
    if (intrinsic_map_it == map_intrinsics.end())
      continue;
    view_residual_cache[view_it.first] = {
      intrinsic_sfm_it->second.get(),
      intrinsic_map_it->second.empty() ? nullptr : &intrinsic_map_it->second[0],
      &pose_map_it->second[0]
    };
  }

  // For all visibility add reprojections errors:
  for (auto& structure_landmark_it : sfm_data.structure)
  {
    const Observations& obs = structure_landmark_it.second.obs;

    for (const auto& obs_it : obs)
    {
      const auto cache_it = view_residual_cache.find(obs_it.first);
      if (cache_it == view_residual_cache.end())
        continue;
      const auto& vc = cache_it->second;

      ceres::CostFunction* cost_function =
        IntrinsicsToCostFunction(vc.intrinsic, obs_it.second.x);

      if (cost_function)
      {
        if (vc.intrinsic_block)
        {
          problem.AddResidualBlock(cost_function,
            p_LossFunction.get(),
            vc.intrinsic_block,
            vc.pose_block,
            structure_landmark_it.second.X.data());
        }
        else
        {
          problem.AddResidualBlock(cost_function,
            p_LossFunction.get(),
            vc.pose_block,
            structure_landmark_it.second.X.data());
        }
      }
      else
      {
        OPENMVG_LOG_ERROR << "Cannot create a CostFunction for this camera model.";
        return false;
      }
    }
    if (options.structure_opt == Structure_Parameter_Type::NONE)
      problem.SetParameterBlockConstant(structure_landmark_it.second.X.data());
  }

  if (options.control_point_opt.bUse_control_points)
  {
    for (auto& gcp_landmark_it : sfm_data.control_points)
    {
      const Observations& obs = gcp_landmark_it.second.obs;

      for (const auto& obs_it : obs)
      {
        const auto cache_it = view_residual_cache.find(obs_it.first);
        if (cache_it == view_residual_cache.end())
          continue;
        const auto& vc = cache_it->second;

        ceres::CostFunction* cost_function =
          IntrinsicsToCostFunction(
            vc.intrinsic,
            obs_it.second.x,
            options.control_point_opt.weight);

        if (cost_function)
        {
          if (vc.intrinsic_block)
          {
            problem.AddResidualBlock(cost_function,
              nullptr,
              vc.intrinsic_block,
              vc.pose_block,
              gcp_landmark_it.second.X.data());
          }
          else
          {
            problem.AddResidualBlock(cost_function,
              nullptr,
              vc.pose_block,
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
        problem.SetParameterBlockConstant(gcp_landmark_it.second.X.data());
      }
    }
  }

  // Add Pose prior constraints if any
  if (b_usable_prior)
  {
    for (const auto& view_it : sfm_data.GetViews())
    {
      const sfm::ViewPriors* prior = dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
      if (prior != nullptr && prior->b_use_pose_center_ && sfm_data.IsPoseAndIntrinsicDefined(prior))
      {
        // Add the cost functor (distance from Pose prior to the SfM_Data Pose center)
        ceres::CostFunction* cost_function =
          new ceres::AutoDiffCostFunction<PoseCenterConstraintCostFunction, 3, 6>(
            new PoseCenterConstraintCostFunction(prior->pose_center_, prior->center_weight_));

        problem.AddResidualBlock(
          cost_function,
          new ceres::HuberLoss(
            Square(pose_center_robust_fitting_error)),
          &map_poses.at(prior->id_view)[0]);
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
  ceres_config_options.logging_type = ceres::SILENT;
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


  // Solve BA
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_config_options, &problem, &summary);
  if (ceres_options_.bCeres_summary_)
    OPENMVG_LOG_INFO << summary.FullReport();

  // If no error, get back refined parameters
  if (!summary.IsSolutionUsable())
  {
    OPENMVG_LOG_ERROR << "IsSolutionUsable is false. Bundle Adjustment failed.";
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
