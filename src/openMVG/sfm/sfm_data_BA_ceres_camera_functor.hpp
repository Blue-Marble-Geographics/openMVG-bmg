// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_DATA_BA_CERES_CAMERA_FUNCTOR_HPP
#define OPENMVG_SFM_SFM_DATA_BA_CERES_CAMERA_FUNCTOR_HPP

#include <memory>
#include <cmath>
#include <limits>
#include <cstring>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/cameras/Camera_Pinhole.hpp"
#include "openMVG/cameras/Camera_Pinhole_Radial.hpp"
#include "openMVG/cameras/Camera_Pinhole_Brown.hpp"
#include "openMVG/cameras/Camera_Pinhole_Fisheye.hpp"
#include "openMVG/cameras/Camera_Spherical.hpp"

//--
//- Define ceres Cost_functor for each OpenMVG camera model
//--

namespace openMVG {
namespace sfm {

// ============================================================================
// Helper: Analytic Jacobian of AngleAxisRotatePoint
// ============================================================================
// Given angle-axis vector w[3] and point X[3], computes:
//   P = R(w) * X
//   dP/dw  (3x3 matrix, row-major in dP_dw[9])
//   dP/dX  = R  (3x3 matrix, row-major in dP_dX[9])
//
// Uses the Rodrigues formula and its derivative.
// R = I + sin(theta)/theta * [w]x + (1-cos(theta))/theta^2 * [w]x^2
// dR/dwi is computed via the derivative of the Rodrigues formula.
namespace analytic_detail {

// Helper to index the wx_X components by j
inline double wx_X(int j, double wx0, double wx1, double wx2)
{
  return (j == 0) ? wx0 : ((j == 1) ? wx1 : wx2);
}

inline void AngleAxisRotatePointAndJacobian(
    const double* w, const double* X,
    double* P,         // output: R*X (3)
    double* dP_dw,     // output: d(R*X)/dw row-major 3x3 (may be nullptr)
    double* dP_dX)     // output: d(R*X)/dX = R row-major 3x3 (may be nullptr)
{
  const double theta2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];

  if (theta2 > ::std::numeric_limits<double>::epsilon())
  {
    const double theta = ::std::sqrt(theta2);
    const double inv_theta = 1.0 / theta;
    const double costh = ::std::cos(theta);
    const double sinth = ::std::sin(theta);

    // Rodrigues: R*X = X*cos(theta) + (w x X)*sin(theta)/theta + w*(w.X)*(1-cos(theta))/theta^2
    const double k = sinth * inv_theta;       // sin(theta)/theta
    const double k2 = (1.0 - costh) / theta2; // (1-cos(theta))/theta^2

    // Unit axis
    const double ax = w[0] * inv_theta;
    const double ay = w[1] * inv_theta;
    const double az = w[2] * inv_theta;

    // w x X
    const double wx_X0 = w[1]*X[2] - w[2]*X[1];
    const double wx_X1 = w[2]*X[0] - w[0]*X[2];
    const double wx_X2 = w[0]*X[1] - w[1]*X[0];

    // w . X
    const double wdotX = w[0]*X[0] + w[1]*X[1] + w[2]*X[2];

    // P = R*X using Rodrigues
    P[0] = X[0]*costh + wx_X0*k + w[0]*wdotX*k2;
    P[1] = X[1]*costh + wx_X1*k + w[1]*wdotX*k2;
    P[2] = X[2]*costh + wx_X2*k + w[2]*wdotX*k2;

    if (dP_dX)
    {
      // R = I*cos(theta) + (1-cos(theta))*a*a^T + sin(theta)*[a]x
      // where a = w/theta is the unit axis
      const double omc = 1.0 - costh;  // 1 - cos(theta)
      // Row-major R
      dP_dX[0] = costh + omc*ax*ax;      dP_dX[1] = omc*ax*ay - sinth*az;   dP_dX[2] = omc*ax*az + sinth*ay;
      dP_dX[3] = omc*ay*ax + sinth*az;   dP_dX[4] = costh + omc*ay*ay;      dP_dX[5] = omc*ay*az - sinth*ax;
      dP_dX[6] = omc*az*ax - sinth*ay;   dP_dX[7] = omc*az*ay + sinth*ax;   dP_dX[8] = costh + omc*az*az;
    }

    if (dP_dw)
    {
      // Derivative of Rodrigues rotation w.r.t. angle-axis w.
      // d(R*X)/dw_i for each component i.
      //
      // Using: P = cos(theta)*X + sin(theta)/theta * (w x X) + (1-cos(theta))/theta^2 * w*(w.X)
      //
      // Let s = sin(theta), c = cos(theta), th = theta
      // ds/dw_i = cos(theta) * w_i / theta
      // dc/dw_i = -sin(theta) * w_i / theta
      // d(theta)/dw_i = w_i / theta
      //
      // d(s/th)/dw_i = (c*th - s)/(th^3) * w_i = (c/th^2 - s/th^3) * w_i
      // d((1-c)/th^2)/dw_i = (s*th - 2*(1-c))/(th^4) * w_i = (s/th^3 - 2*(1-c)/th^4) * w_i

      const double inv_theta2 = inv_theta * inv_theta;
      const double inv_theta3 = inv_theta2 * inv_theta;

      // Coefficients for chain rule
      // dc/dw_i = -sinth * w_i * inv_theta
      // dk/dw_i = d(sinth/theta)/dw_i = (costh - sinth*inv_theta) * inv_theta * w_i * inv_theta
      //         = (costh*inv_theta2 - sinth*inv_theta3) * w_i
      const double dk_coeff = costh * inv_theta2 - sinth * inv_theta3;
      // dk2/dw_i = d((1-costh)/theta^2)/dw_i = (sinth*inv_theta3 - 2*(1-costh)/(theta^4)) * w_i
      //          = (sinth*inv_theta3 - 2*k2*inv_theta2) * w_i
      const double dk2_coeff = sinth * inv_theta3 - 2.0 * k2 * inv_theta2;
      const double dc_coeff = -sinth * inv_theta; // dc/dw_i = dc_coeff * w_i

      for (int i = 0; i < 3; ++i)
      {
        const double wi = w[i];

        // d(cos(theta)*X)/dw_i = dc_coeff * wi * X
        // d(k * (w x X))/dw_i = dk_coeff * wi * (w x X) + k * d(w x X)/dw_i
        // d(k2 * w * (w.X))/dw_i = dk2_coeff * wi * w*(w.X) + k2 * (e_i*(w.X) + w*(X_i))
        //   where e_i is the i-th unit vector

        // d(w x X)/dw_i: cross product derivative
        // w x X = [w1*X2 - w2*X1, w2*X0 - w0*X2, w0*X1 - w1*X0]
        double dwxX[3] = {0, 0, 0};
        if (i == 0) { dwxX[1] = -X[2]; dwxX[2] =  X[1]; }
        if (i == 1) { dwxX[0] =  X[2]; dwxX[2] = -X[0]; }
        if (i == 2) { dwxX[0] = -X[1]; dwxX[1] =  X[0]; }

        // For the w*(w.X) term:
        // d(w*(w.X))/dw_i = e_i*(w.X) + w*X_i
        double dw_wdotX[3];
        dw_wdotX[0] = w[0] * X[i];
        dw_wdotX[1] = w[1] * X[i];
        dw_wdotX[2] = w[2] * X[i];
        dw_wdotX[i] += wdotX;

        for (int j = 0; j < 3; ++j)
        {
          dP_dw[j * 3 + i] =
            dc_coeff * wi * X[j] +                    // d(cos*X)/dw_i
            dk_coeff * wi * wx_X(j, wx_X0, wx_X1, wx_X2) + // d(k)/dw_i * (wxX)
            k * dwxX[j] +                             // k * d(wxX)/dw_i
            dk2_coeff * wi * w[j] * wdotX +            // d(k2)/dw_i * w*(w.X)
            k2 * dw_wdotX[j];                         // k2 * d(w*(w.X))/dw_i
        }
      }
    }
  }
  else
  {
    // Near zero: R ≈ I + [w]x, so R*X ≈ X + w x X
    P[0] = X[0] + w[1]*X[2] - w[2]*X[1];
    P[1] = X[1] + w[2]*X[0] - w[0]*X[2];
    P[2] = X[2] + w[0]*X[1] - w[1]*X[0];

    if (dP_dX)
    {
      // R ≈ I + [w]x
      dP_dX[0] = 1.0;    dP_dX[1] = -w[2];  dP_dX[2] = w[1];
      dP_dX[3] = w[2];   dP_dX[4] = 1.0;    dP_dX[5] = -w[0];
      dP_dX[6] = -w[1];  dP_dX[7] = w[0];   dP_dX[8] = 1.0;
    }

    if (dP_dw)
    {
      // d(w x X)/dw is the cross product derivative
      // P = X + w x X, so dP/dw = d(w x X)/dw = -[X]x
      // dP/dw0 = [0, -X2, X1]
      // dP/dw1 = [X2, 0, -X0]
      // dP/dw2 = [-X1, X0, 0]
      // Stored row-major: dP_dw[j*3+i] = dP_j/dw_i
      dP_dw[0*3+0] =  0.0;   dP_dw[0*3+1] =  X[2];  dP_dw[0*3+2] = -X[1];
      dP_dw[1*3+0] = -X[2];  dP_dw[1*3+1] =  0.0;   dP_dw[1*3+2] =  X[0];
      dP_dw[2*3+0] =  X[1];  dP_dw[2*3+1] = -X[0];  dP_dw[2*3+2] =  0.0;
    }
  }
}

} // namespace analytic_detail

// ============================================================================
// Analytic cost function for Pinhole_Intrinsic_Radial_K3
// Parameter blocks: <2, 6, 6, 3> (residuals, intrinsics, extrinsics, point)
//
// This is the most commonly used camera model and the biggest performance win.
// For this model, AutoDiff uses Jet<double, 15> — every intermediate variable
// carries a 15-element derivative vector through sin/cos/sqrt/div operations.
// The analytic version computes the same 2×15 Jacobian directly with ~4x fewer
// FLOPs by exploiting the chain-rule structure.
// ============================================================================
class AnalyticCostFunction_Pinhole_Radial_K3
  : public ceres::SizedCostFunction<2, 6, 6, 3>
{
public:
  explicit AnalyticCostFunction_Pinhole_Radial_K3(
    const double obs_x, const double obs_y)
    : obs_x_(obs_x), obs_y_(obs_y)
  {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override
  {
    const double* intrinsics = parameters[0]; // [f, ppx, ppy, k1, k2, k3]
    const double* extrinsics = parameters[1]; // [r0, r1, r2, tx, ty, tz]
    const double* point      = parameters[2]; // [X, Y, Z]

    const double f   = intrinsics[0];
    const double ppx = intrinsics[1];
    const double ppy = intrinsics[2];
    const double k1  = intrinsics[3];
    const double k2  = intrinsics[4];
    const double k3  = intrinsics[5];

    const double* w = extrinsics;     // angle-axis [r0, r1, r2]
    const double* t = extrinsics + 3; // translation [tx, ty, tz]

    double P[3];
    double dP_dw[9];
    double dP_dX[9];
    const bool need_jac = (jacobians != nullptr) &&
                          (jacobians[0] || jacobians[1] || jacobians[2]);

    analytic_detail::AngleAxisRotatePointAndJacobian(
      w, point, P,
      need_jac ? dP_dw : nullptr,
      need_jac ? dP_dX : nullptr);

    const double Qx = P[0] + t[0];
    const double Qy = P[1] + t[1];
    const double Qz = P[2] + t[2];

    const double inv_z = 1.0 / Qz;
    const double x = Qx * inv_z;
    const double y = Qy * inv_z;

    const double r2 = x*x + y*y;
    const double r4 = r2*r2;
    const double r6 = r4*r2;
    const double rc = 1.0 + k1*r2 + k2*r4 + k3*r6;

    const double xd = x * rc;
    const double yd = y * rc;

    residuals[0] = ppx + xd * f - obs_x_;
    residuals[1] = ppy + yd * f - obs_y_;

    if (!jacobians) return true;

    const double drc_dr2 = k1 + 2.0*k2*r2 + 3.0*k3*r4;
    const double dxd_dx = rc + 2.0*x*x*drc_dr2;
    const double dxd_dy = 2.0*x*y*drc_dr2;
    const double dyd_dx = dxd_dy;
    const double dyd_dy = rc + 2.0*y*y*drc_dr2;

    const double A00 = f * dxd_dx, A01 = f * dxd_dy;
    const double A10 = f * dyd_dx, A11 = f * dyd_dy;

    const double B00 = A00 * inv_z;
    const double B01 = A01 * inv_z;
    const double B02 = -(A00*x + A01*y) * inv_z;
    const double B10 = A10 * inv_z;
    const double B11 = A11 * inv_z;
    const double B12 = -(A10*x + A11*y) * inv_z;

    if (jacobians[0])
    {
      double* J = jacobians[0]; // 2x6 row-major
      J[0] = xd;        J[1] = 1.0;  J[2] = 0.0;
      J[3] = f*x*r2;    J[4] = f*x*r4;  J[5] = f*x*r6;
      J[6] = yd;        J[7] = 0.0;  J[8] = 1.0;
      J[9] = f*y*r2;    J[10]= f*y*r4;  J[11]= f*y*r6;
    }

    if (jacobians[1])
    {
      double* J = jacobians[1]; // 2x6 row-major
      for (int i = 0; i < 3; ++i)
      {
        const double dQx_dwi = dP_dw[0*3 + i];
        const double dQy_dwi = dP_dw[1*3 + i];
        const double dQz_dwi = dP_dw[2*3 + i];
        J[i]     = B00*dQx_dwi + B01*dQy_dwi + B02*dQz_dwi;
        J[6 + i] = B10*dQx_dwi + B11*dQy_dwi + B12*dQz_dwi;
      }
      J[3] = B00;  J[4] = B01;  J[5] = B02;
      J[9] = B10;  J[10]= B11;  J[11]= B12;
    }

    if (jacobians[2])
    {
      double* J = jacobians[2]; // 2x3 row-major
      for (int i = 0; i < 3; ++i)
      {
        const double dQx_dXi = dP_dX[0*3 + i];
        const double dQy_dXi = dP_dX[1*3 + i];
        const double dQz_dXi = dP_dX[2*3 + i];
        J[i]     = B00*dQx_dXi + B01*dQy_dXi + B02*dQz_dXi;
        J[3 + i] = B10*dQx_dXi + B11*dQy_dXi + B12*dQz_dXi;
      }
    }

    return true;
  }

private:
  double obs_x_, obs_y_;
};

// ============================================================================
// Analytic cost function for Pinhole_Intrinsic (no distortion)
// Parameter blocks: <2, 3, 6, 3>
// ============================================================================
class AnalyticCostFunction_Pinhole
  : public ceres::SizedCostFunction<2, 3, 6, 3>
{
public:
  explicit AnalyticCostFunction_Pinhole(
    const double obs_x, const double obs_y)
    : obs_x_(obs_x), obs_y_(obs_y)
  {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override
  {
    const double* intrinsics = parameters[0]; // [f, ppx, ppy]
    const double* extrinsics = parameters[1]; // [r0, r1, r2, tx, ty, tz]
    const double* point      = parameters[2]; // [X, Y, Z]

    const double f   = intrinsics[0];
    const double ppx = intrinsics[1];
    const double ppy = intrinsics[2];

    const double* w = extrinsics;
    const double* t = extrinsics + 3;

    double P[3], dP_dw[9], dP_dX[9];
    const bool need_jac = (jacobians != nullptr) &&
                          (jacobians[0] || jacobians[1] || jacobians[2]);

    analytic_detail::AngleAxisRotatePointAndJacobian(
      w, point, P,
      need_jac ? dP_dw : nullptr,
      need_jac ? dP_dX : nullptr);

    const double Qx = P[0] + t[0];
    const double Qy = P[1] + t[1];
    const double Qz = P[2] + t[2];
    const double inv_z = 1.0 / Qz;
    const double x = Qx * inv_z;
    const double y = Qy * inv_z;

    residuals[0] = ppx + x * f - obs_x_;
    residuals[1] = ppy + y * f - obs_y_;

    if (!jacobians) return true;

    // d(res)/d(x,y) is simply [f, 0; 0, f] (no distortion)
    // d(x,y)/d(Q) as before
    const double B00 = f * inv_z;
    const double B01 = 0.0;
    const double B02 = -f * x * inv_z;
    const double B10 = 0.0;
    const double B11 = f * inv_z;
    const double B12 = -f * y * inv_z;

    if (jacobians[0])
    {
      double* J = jacobians[0]; // 2x3
      J[0] = x;    J[1] = 1.0;  J[2] = 0.0;
      J[3] = y;    J[4] = 0.0;  J[5] = 1.0;
    }

    if (jacobians[1])
    {
      double* J = jacobians[1]; // 2x6
      for (int i = 0; i < 3; ++i)
      {
        const double dQx = dP_dw[0*3+i], dQy = dP_dw[1*3+i], dQz = dP_dw[2*3+i];
        J[i]     = B00*dQx + B01*dQy + B02*dQz;
        J[6 + i] = B10*dQx + B11*dQy + B12*dQz;
      }
      J[3] = B00;  J[4] = B01;  J[5] = B02;
      J[9] = B10;  J[10]= B11;  J[11]= B12;
    }

    if (jacobians[2])
    {
      double* J = jacobians[2]; // 2x3
      for (int i = 0; i < 3; ++i)
      {
        const double dQx = dP_dX[0*3+i], dQy = dP_dX[1*3+i], dQz = dP_dX[2*3+i];
        J[i]     = B00*dQx + B01*dQy + B02*dQz;
        J[3 + i] = B10*dQx + B11*dQy + B12*dQz;
      }
    }

    return true;
  }

private:
  double obs_x_, obs_y_;
};

// ============================================================================
// Analytic cost function for Pinhole_Intrinsic_Radial_K1
// Parameter blocks: <2, 4, 6, 3>
// ============================================================================
class AnalyticCostFunction_Pinhole_Radial_K1
  : public ceres::SizedCostFunction<2, 4, 6, 3>
{
public:
  explicit AnalyticCostFunction_Pinhole_Radial_K1(
    const double obs_x, const double obs_y)
    : obs_x_(obs_x), obs_y_(obs_y)
  {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override
  {
    const double* intrinsics = parameters[0]; // [f, ppx, ppy, k1]
    const double* extrinsics = parameters[1];
    const double* point      = parameters[2];

    const double f   = intrinsics[0];
    const double ppx = intrinsics[1];
    const double ppy = intrinsics[2];
    const double k1  = intrinsics[3];

    const double* w = extrinsics;
    const double* t = extrinsics + 3;

    double P[3], dP_dw[9], dP_dX[9];
    const bool need_jac = (jacobians != nullptr) &&
                          (jacobians[0] || jacobians[1] || jacobians[2]);

    analytic_detail::AngleAxisRotatePointAndJacobian(
      w, point, P,
      need_jac ? dP_dw : nullptr,
      need_jac ? dP_dX : nullptr);

    const double Qx = P[0] + t[0];
    const double Qy = P[1] + t[1];
    const double Qz = P[2] + t[2];
    const double inv_z = 1.0 / Qz;
    const double x = Qx * inv_z;
    const double y = Qy * inv_z;

    const double r2 = x*x + y*y;
    const double rc = 1.0 + k1*r2;

    const double xd = x * rc;
    const double yd = y * rc;

    residuals[0] = ppx + xd * f - obs_x_;
    residuals[1] = ppy + yd * f - obs_y_;

    if (!jacobians) return true;

    const double drc_dr2 = k1;
    const double dxd_dx = rc + 2.0*x*x*drc_dr2;
    const double dxd_dy = 2.0*x*y*drc_dr2;
    const double dyd_dx = dxd_dy;
    const double dyd_dy = rc + 2.0*y*y*drc_dr2;

    const double A00 = f * dxd_dx, A01 = f * dxd_dy;
    const double A10 = f * dyd_dx, A11 = f * dyd_dy;

    const double B00 = A00 * inv_z;
    const double B01 = A01 * inv_z;
    const double B02 = -(A00*x + A01*y) * inv_z;
    const double B10 = A10 * inv_z;
    const double B11 = A11 * inv_z;
    const double B12 = -(A10*x + A11*y) * inv_z;

    if (jacobians[0])
    {
      double* J = jacobians[0]; // 2x4
      J[0] = xd;        J[1] = 1.0;  J[2] = 0.0;  J[3] = f*x*r2;
      J[4] = yd;        J[5] = 0.0;  J[6] = 1.0;  J[7] = f*y*r2;
    }

    if (jacobians[1])
    {
      double* J = jacobians[1]; // 2x6
      for (int i = 0; i < 3; ++i)
      {
        const double dQx = dP_dw[0*3+i], dQy = dP_dw[1*3+i], dQz = dP_dw[2*3+i];
        J[i]     = B00*dQx + B01*dQy + B02*dQz;
        J[6 + i] = B10*dQx + B11*dQy + B12*dQz;
      }
      J[3] = B00;  J[4] = B01;  J[5] = B02;
      J[9] = B10;  J[10]= B11;  J[11]= B12;
    }

    if (jacobians[2])
    {
      double* J = jacobians[2]; // 2x3
      for (int i = 0; i < 3; ++i)
      {
        const double dQx = dP_dX[0*3+i], dQy = dP_dX[1*3+i], dQz = dP_dX[2*3+i];
        J[i]     = B00*dQx + B01*dQy + B02*dQz;
        J[3 + i] = B10*dQx + B11*dQy + B12*dQz;
      }
    }

    return true;
  }

private:
  double obs_x_, obs_y_;
};

// ============================================================================
// Analytic cost function for Pinhole_Intrinsic_Brown_T2
// Parameter blocks: <2, 8, 6, 3>
// ============================================================================
class AnalyticCostFunction_Pinhole_Brown_T2
  : public ceres::SizedCostFunction<2, 8, 6, 3>
{
public:
  explicit AnalyticCostFunction_Pinhole_Brown_T2(
    const double obs_x, const double obs_y)
    : obs_x_(obs_x), obs_y_(obs_y)
  {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override
  {
    const double* intrinsics = parameters[0]; // [f, ppx, ppy, k1, k2, k3, t1, t2]
    const double* extrinsics = parameters[1];
    const double* point      = parameters[2];

    const double f   = intrinsics[0];
    const double ppx = intrinsics[1];
    const double ppy = intrinsics[2];
    const double k1  = intrinsics[3];
    const double k2  = intrinsics[4];
    const double k3  = intrinsics[5];
    const double t1  = intrinsics[6];
    const double t2  = intrinsics[7];

    const double* w = extrinsics;
    const double* t_ext = extrinsics + 3;

    double P[3], dP_dw[9], dP_dX[9];
    const bool need_jac = (jacobians != nullptr) &&
                          (jacobians[0] || jacobians[1] || jacobians[2]);

    analytic_detail::AngleAxisRotatePointAndJacobian(
      w, point, P,
      need_jac ? dP_dw : nullptr,
      need_jac ? dP_dX : nullptr);

    const double Qx = P[0] + t_ext[0];
    const double Qy = P[1] + t_ext[1];
    const double Qz = P[2] + t_ext[2];
    const double inv_z = 1.0 / Qz;
    const double xu = Qx * inv_z;
    const double yu = Qy * inv_z;

    const double r2 = xu*xu + yu*yu;
    const double r4 = r2*r2;
    const double r6 = r4*r2;
    const double rc = 1.0 + k1*r2 + k2*r4 + k3*r6;
    const double tx_val = t2*(r2 + 2.0*xu*xu) + 2.0*t1*xu*yu;
    const double ty_val = t1*(r2 + 2.0*yu*yu) + 2.0*t2*xu*yu;

    const double xd = xu*rc + tx_val;
    const double yd = yu*rc + ty_val;

    residuals[0] = ppx + xd * f - obs_x_;
    residuals[1] = ppy + yd * f - obs_y_;

    if (!jacobians) return true;

    const double drc_dr2 = k1 + 2.0*k2*r2 + 3.0*k3*r4;

    const double dxd_dxu = rc + 2.0*xu*xu*drc_dr2 + 6.0*t2*xu + 2.0*t1*yu;
    const double dxd_dyu = 2.0*xu*yu*drc_dr2 + 2.0*t2*yu + 2.0*t1*xu;
    const double dyd_dxu = 2.0*xu*yu*drc_dr2 + 2.0*t1*xu + 2.0*t2*yu;
    const double dyd_dyu = rc + 2.0*yu*yu*drc_dr2 + 6.0*t1*yu + 2.0*t2*xu;

    const double A00 = f*dxd_dxu, A01 = f*dxd_dyu;
    const double A10 = f*dyd_dxu, A11 = f*dyd_dyu;

    const double B00 = A00*inv_z, B01 = A01*inv_z, B02 = -(A00*xu+A01*yu)*inv_z;
    const double B10 = A10*inv_z, B11 = A11*inv_z, B12 = -(A10*xu+A11*yu)*inv_z;

    if (jacobians[0])
    {
      double* J = jacobians[0]; // 2x8
      J[0] = xd;  J[1] = 1.0;  J[2] = 0.0;
      J[3] = f*xu*r2;  J[4] = f*xu*r4;  J[5] = f*xu*r6;
      J[6] = f*(2.0*xu*yu);  J[7] = f*(r2 + 2.0*xu*xu);

      J[8] = yd;  J[9] = 0.0;  J[10] = 1.0;
      J[11]= f*yu*r2;  J[12]= f*yu*r4;  J[13]= f*yu*r6;
      J[14]= f*(r2 + 2.0*yu*yu);  J[15]= f*(2.0*xu*yu);
    }

    if (jacobians[1])
    {
      double* J = jacobians[1]; // 2x6
      for (int i = 0; i < 3; ++i)
      {
        const double dQx = dP_dw[0*3+i], dQy = dP_dw[1*3+i], dQz = dP_dw[2*3+i];
        J[i]     = B00*dQx + B01*dQy + B02*dQz;
        J[6 + i] = B10*dQx + B11*dQy + B12*dQz;
      }
      J[3] = B00;  J[4] = B01;  J[5] = B02;
      J[9] = B10;  J[10]= B11;  J[11]= B12;
    }

    if (jacobians[2])
    {
      double* J = jacobians[2]; // 2x3
      for (int i = 0; i < 3; ++i)
      {
        const double dQx = dP_dX[0*3+i], dQy = dP_dX[1*3+i], dQz = dP_dX[2*3+i];
        J[i]     = B00*dQx + B01*dQy + B02*dQz;
        J[3 + i] = B10*dQx + B11*dQy + B12*dQz;
      }
    }

    return true;
  }

private:
  double obs_x_, obs_y_;
};


// Decorator used to Weight a given cost camera functor
/// i.e useful to weight GCP (Ground Control Points)
template <typename CostFunctor>
struct WeightedCostFunction
{
  WeightedCostFunction(): weight_(1.0) {}

  explicit WeightedCostFunction
  (
    CostFunctor * func,
    const double weight
  ):
    functor_(func), weight_(weight)
  {}

  template <typename T>
  bool operator()
  (
    const T* const cam_intrinsic,
    const T* const cam_extrinsics,
    const T* const pos_3dpoint,
    T* out_residuals
  ) const
  {
    if (functor_->operator()(cam_intrinsic, cam_extrinsics, pos_3dpoint, out_residuals))
    {
      // Reweight the residual values
      for (int i = 0; i < CostFunctor::num_residuals(); ++i)
      {
        out_residuals[i] *= T(weight_);
      }
      return true;
    }
    return false;
  }

  template <typename T>
  bool operator()
  (
    const T* const cam_extrinsics,
    const T* const pos_3dpoint,
    T* out_residuals
  ) const
  {
    if (functor_->operator()(cam_extrinsics, pos_3dpoint, out_residuals))
    {
      // Reweight the residual values
      for (int i = 0; i < CostFunctor::num_residuals(); ++i)
      {
        out_residuals[i] *= T(weight_);
      }
      return true;
    }
    return false;
  }

  std::unique_ptr<CostFunctor> functor_;
  const double weight_;
};

/**
 * @brief Ceres functor to use a Pinhole_Intrinsic (pinhole camera model K[R[t]) and a 3D point.
 *
 *  Data parameter blocks are the following <2,3,6,3>
 *  - 2 => dimension of the residuals,
 *  - 3 => the intrinsic data block [focal, principal point x, principal point y],
 *  - 6 => the camera extrinsic data block (camera orientation and position) [R;t],
 *         - rotation(angle axis), and translation [rX,rY,rZ,tx,ty,tz].
 *  - 3 => a 3D point data block.
 *
 */
struct ResidualErrorFunctor_Pinhole_Intrinsic
{
  explicit ResidualErrorFunctor_Pinhole_Intrinsic(const double* const pos_2dpoint)
  :m_pos_2dpoint(pos_2dpoint)
  {
  }

  // Enum to map intrinsics parameters between openMVG & ceres camera data parameter block.
  enum : uint8_t {
    OFFSET_FOCAL_LENGTH = 0,
    OFFSET_PRINCIPAL_POINT_X = 1,
    OFFSET_PRINCIPAL_POINT_Y = 2
  };

  template <typename T>
  bool operator()(
    const T* const cam_intrinsics,
    const T* const cam_extrinsics,
    const T* const pos_3dpoint,
    T* out_residuals) const
  {
    T p[3];
    ceres::AngleAxisRotatePoint(cam_extrinsics, pos_3dpoint, p);

    p[0] += cam_extrinsics[3];
    p[1] += cam_extrinsics[4];
    p[2] += cam_extrinsics[5];

    const T inv_z = T(1.0) / p[2];
    const T x = p[0] * inv_z;
    const T y = p[1] * inv_z;

    const T& focal = cam_intrinsics[OFFSET_FOCAL_LENGTH];
    const T& ppx = cam_intrinsics[OFFSET_PRINCIPAL_POINT_X];
    const T& ppy = cam_intrinsics[OFFSET_PRINCIPAL_POINT_Y];

    out_residuals[0] = ppx + x * focal - m_pos_2dpoint[0];
    out_residuals[1] = ppy + y * focal - m_pos_2dpoint[1];
    return true;
  }

  static int num_residuals() { return 2; }

  static ceres::CostFunction* Create
  (
    const Vec2 & observation,
    const double weight = 0.0
  )
  {
    if (weight == 0.0)
    {
      return new AnalyticCostFunction_Pinhole(observation[0], observation[1]);
    }
    else
    {
      return
        (new ceres::AutoDiffCostFunction
          <WeightedCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic>, 2, 3, 6, 3>
          (new WeightedCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic>
            (new ResidualErrorFunctor_Pinhole_Intrinsic(observation.data()), weight)));
    }
  }

  const double * m_pos_2dpoint; // The 2D observation
};

/**
 * @brief Ceres functor to use a Pinhole_Intrinsic_Radial_K1
 *
 *  Data parameter blocks are the following <2,4,6,3>
 *  - 2 => dimension of the residuals,
 *  - 4 => the intrinsic data block [focal, principal point x, principal point y, K1],
 *  - 6 => the camera extrinsic data block (camera orientation and position) [R;t],
 *         - rotation(angle axis), and translation [rX,rY,rZ,tx,ty,tz].
 *  - 3 => a 3D point data block.
 *
 */
struct ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1
{
  explicit ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1(const double* const pos_2dpoint)
  :m_pos_2dpoint(pos_2dpoint)
  {
  }

  // Enum to map intrinsics parameters between openMVG & ceres camera data parameter block.
  enum : uint8_t {
    OFFSET_FOCAL_LENGTH = 0,
    OFFSET_PRINCIPAL_POINT_X = 1,
    OFFSET_PRINCIPAL_POINT_Y = 2,
    OFFSET_DISTO_K1 = 3
  };

  template <typename T>
  bool operator()(
    const T* const cam_intrinsics,
    const T* const cam_extrinsics,
    const T* const pos_3dpoint,
    T* out_residuals) const
  {
    T p[3];
    ceres::AngleAxisRotatePoint(cam_extrinsics, pos_3dpoint, p);

    p[0] += cam_extrinsics[3];
    p[1] += cam_extrinsics[4];
    p[2] += cam_extrinsics[5];

    const T inv_z = T(1.0) / p[2];
    const T x = p[0] * inv_z;
    const T y = p[1] * inv_z;

    const T& focal = cam_intrinsics[OFFSET_FOCAL_LENGTH];
    const T& ppx = cam_intrinsics[OFFSET_PRINCIPAL_POINT_X];
    const T& ppy = cam_intrinsics[OFFSET_PRINCIPAL_POINT_Y];
    const T& k1 = cam_intrinsics[OFFSET_DISTO_K1];

    const T r2 = x * x + y * y;
    const T r_coeff = T(1.0) + k1 * r2;

    out_residuals[0] = ppx + (x * r_coeff) * focal - m_pos_2dpoint[0];
    out_residuals[1] = ppy + (y * r_coeff) * focal - m_pos_2dpoint[1];

    return true;
  }

  static int num_residuals() { return 2; }

  static ceres::CostFunction* Create
  (
    const Vec2 & observation,
    const double weight = 0.0
  )
  {
    if (weight == 0.0)
    {
      return new AnalyticCostFunction_Pinhole_Radial_K1(observation[0], observation[1]);
    }
    else
    {
      return
        (new ceres::AutoDiffCostFunction
          <WeightedCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1>, 2, 4, 6, 3>
          (new WeightedCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1>
            (new ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1(observation.data()), weight)));
    }
  }

  const double * m_pos_2dpoint; // The 2D observation
};

/**
 * @brief Ceres functor to use a Pinhole_Intrinsic_Radial_K3
 *
 *  Data parameter blocks are the following <2,6,6,3>
 *  - 2 => dimension of the residuals,
 *  - 6 => the intrinsic data block [focal, principal point x, principal point y, K1, K2, K3],
 *  - 6 => the camera extrinsic data block (camera orientation and position) [R;t],
 *         - rotation(angle axis), and translation [rX,rY,rZ,tx,ty,tz].
 *  - 3 => a 3D point data block.
 *
 */
struct ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3
{
  explicit ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3(const double* const pos_2dpoint)
  :m_pos_2dpoint(pos_2dpoint)
  {
  }

  // Enum to map intrinsics parameters between openMVG & ceres camera data parameter block.
  enum : uint8_t {
    OFFSET_FOCAL_LENGTH = 0,
    OFFSET_PRINCIPAL_POINT_X = 1,
    OFFSET_PRINCIPAL_POINT_Y = 2,
    OFFSET_DISTO_K1 = 3,
    OFFSET_DISTO_K2 = 4,
    OFFSET_DISTO_K3 = 5,
  };

  template <typename T>
  bool operator()(
    const T* const cam_intrinsics,
    const T* const cam_extrinsics,
    const T* const pos_3dpoint,
    T* out_residuals) const
  {
    T p[3];
    ceres::AngleAxisRotatePoint(cam_extrinsics, pos_3dpoint, p);

    p[0] += cam_extrinsics[3];
    p[1] += cam_extrinsics[4];
    p[2] += cam_extrinsics[5];

    const T inv_z = T(1.0) / p[2];
    const T x = p[0] * inv_z;
    const T y = p[1] * inv_z;

    const T& focal = cam_intrinsics[OFFSET_FOCAL_LENGTH];
    const T& ppx = cam_intrinsics[OFFSET_PRINCIPAL_POINT_X];
    const T& ppy = cam_intrinsics[OFFSET_PRINCIPAL_POINT_Y];
    const T& k1 = cam_intrinsics[OFFSET_DISTO_K1];
    const T& k2 = cam_intrinsics[OFFSET_DISTO_K2];
    const T& k3 = cam_intrinsics[OFFSET_DISTO_K3];

    const T r2 = x * x + y * y;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;
    const T r_coeff = (T(1.0) + k1 * r2 + k2 * r4 + k3 * r6);

    out_residuals[0] = ppx + (x * r_coeff) * focal - m_pos_2dpoint[0];
    out_residuals[1] = ppy + (y * r_coeff) * focal - m_pos_2dpoint[1];

    return true;
  }

  static int num_residuals() { return 2; }

  static ceres::CostFunction* Create
  (
    const Vec2 & observation,
    const double weight = 0.0
  )
  {
    if (weight == 0.0)
    {
      return new AnalyticCostFunction_Pinhole_Radial_K3(observation[0], observation[1]);
    }
    else
    {
      return
        (new ceres::AutoDiffCostFunction
          <WeightedCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3>, 2, 6, 6, 3>
          (new WeightedCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3>
            (new ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3(observation.data()), weight)));
    }
  }

  const double * m_pos_2dpoint; // The 2D observation
};

/**
 * @brief Ceres functor with constrained 3D points to use a Pinhole_Intrinsic_Brown_T2
 *
 *  Data parameter blocks are the following <2,8,6,3>
 */
struct ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2
{
  explicit ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2(const double* const pos_2dpoint)
  :m_pos_2dpoint(pos_2dpoint)
  {
  }

  // Enum to map intrinsics parameters between openMVG & ceres camera data parameter block.
  enum : uint8_t {
    OFFSET_FOCAL_LENGTH = 0,
    OFFSET_PRINCIPAL_POINT_X = 1,
    OFFSET_PRINCIPAL_POINT_Y = 2,
    OFFSET_DISTO_K1 = 3,
    OFFSET_DISTO_K2 = 4,
    OFFSET_DISTO_K3 = 5,
    OFFSET_DISTO_T1 = 6,
    OFFSET_DISTO_T2 = 7,
  };

  template <typename T>
  bool operator()(
    const T* const cam_intrinsics,
    const T* const cam_extrinsics,
    const T* const pos_3dpoint,
    T* out_residuals) const
  {
    T p[3];
    ceres::AngleAxisRotatePoint(cam_extrinsics, pos_3dpoint, p);

    p[0] += cam_extrinsics[3];
    p[1] += cam_extrinsics[4];
    p[2] += cam_extrinsics[5];

    const T inv_z = T(1.0) / p[2];
    const T x_u = p[0] * inv_z;
    const T y_u = p[1] * inv_z;

    const T& focal = cam_intrinsics[OFFSET_FOCAL_LENGTH];
    const T& ppx = cam_intrinsics[OFFSET_PRINCIPAL_POINT_X];
    const T& ppy = cam_intrinsics[OFFSET_PRINCIPAL_POINT_Y];
    const T& k1 = cam_intrinsics[OFFSET_DISTO_K1];
    const T& k2 = cam_intrinsics[OFFSET_DISTO_K2];
    const T& k3 = cam_intrinsics[OFFSET_DISTO_K3];
    const T& t1 = cam_intrinsics[OFFSET_DISTO_T1];
    const T& t2 = cam_intrinsics[OFFSET_DISTO_T2];

    const T r2 = x_u * x_u + y_u * y_u;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;
    const T r_coeff = (T(1.0) + k1 * r2 + k2 * r4 + k3 * r6);
    const T t_x = t2 * (r2 + T(2.0) * x_u * x_u) + T(2.0) * t1 * x_u * y_u;
    const T t_y = t1 * (r2 + T(2.0) * y_u * y_u) + T(2.0) * t2 * x_u * y_u;

    out_residuals[0] = ppx + (x_u * r_coeff + t_x) * focal - m_pos_2dpoint[0];
    out_residuals[1] = ppy + (y_u * r_coeff + t_y) * focal - m_pos_2dpoint[1];

    return true;
  }

  static int num_residuals() { return 2; }

  static ceres::CostFunction* Create
  (
    const Vec2 & observation,
    const double weight = 0.0
  )
  {
    if (weight == 0.0)
    {
      return new AnalyticCostFunction_Pinhole_Brown_T2(observation[0], observation[1]);
    }
    else
    {
      return
        (new ceres::AutoDiffCostFunction
          <WeightedCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2>, 2, 8, 6, 3>
          (new WeightedCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2>
            (new ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2(observation.data()), weight)));
    }
  }

  const double * m_pos_2dpoint; // The 2D observation
};


/**
 * @brief Ceres functor with constrained 3D points to use a Pinhole_Intrinsic_Fisheye
 *
 *  Data parameter blocks are the following <2,7,6,3>
 *  - 2 => dimension of the residuals,
 *  - 7 => the intrinsic data block [focal, principal point x, principal point y, K1, K2, K3, K4],
 *  - 6 => the camera extrinsic data block (camera orientation and position) [R;t],
 *         - rotation(angle axis), and translation [rX,rY,rZ,tx,ty,tz].
 *  - 3 => a 3D point data block.
 *
 */

struct ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye
{
  explicit ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye(const double* const pos_2dpoint)
  :m_pos_2dpoint(pos_2dpoint)
  {
  }

  // Enum to map intrinsics parameters between openMVG & ceres camera data parameter block.
  enum : uint8_t {
    OFFSET_FOCAL_LENGTH = 0,
    OFFSET_PRINCIPAL_POINT_X = 1,
    OFFSET_PRINCIPAL_POINT_Y = 2,
    OFFSET_DISTO_K1 = 3,
    OFFSET_DISTO_K2 = 4,
    OFFSET_DISTO_K3 = 5,
    OFFSET_DISTO_K4 = 6,
  };

  template <typename T>
  bool operator()(
    const T* const cam_intrinsics,
    const T* const cam_extrinsics,
    const T* const pos_3dpoint,
    T* out_residuals) const
  {
    T p[3];
    ceres::AngleAxisRotatePoint(cam_extrinsics, pos_3dpoint, p);

    p[0] += cam_extrinsics[3];
    p[1] += cam_extrinsics[4];
    p[2] += cam_extrinsics[5];

    const T inv_z = T(1.0) / p[2];
    const T x = p[0] * inv_z;
    const T y = p[1] * inv_z;

    const T& focal = cam_intrinsics[OFFSET_FOCAL_LENGTH];
    const T& ppx = cam_intrinsics[OFFSET_PRINCIPAL_POINT_X];
    const T& ppy = cam_intrinsics[OFFSET_PRINCIPAL_POINT_Y];
    const T& k1 = cam_intrinsics[OFFSET_DISTO_K1];
    const T& k2 = cam_intrinsics[OFFSET_DISTO_K2];
    const T& k3 = cam_intrinsics[OFFSET_DISTO_K3];
    const T& k4 = cam_intrinsics[OFFSET_DISTO_K4];

    const T r2 = x * x + y * y;
    const T r = sqrt(r2);
    const T
      theta = atan(r),
      theta2 = theta*theta,
      theta3 = theta2*theta,
      theta4 = theta2*theta2,
      theta5 = theta4*theta,
      theta7 = theta3*theta3*theta, //theta6*theta
      theta8 = theta4*theta4,
      theta9 = theta8*theta;
    const T theta_dist = theta + k1*theta3 + k2*theta5 + k3*theta7 + k4*theta9;
    const T inv_r = r > T(1e-8) ? T(1.0)/r : T(1.0);
    const T cdist = r > T(1e-8) ? theta_dist * inv_r : T(1.0);

    out_residuals[0] = ppx + (x * cdist) * focal - m_pos_2dpoint[0];
    out_residuals[1] = ppy + (y * cdist) * focal - m_pos_2dpoint[1];

    return true;
  }

  static int num_residuals() { return 2; }

  static ceres::CostFunction* Create
  (
    const Vec2 & observation,
    const double weight = 0.0
  )
  {
    if (weight == 0.0)
    {
      return
        (new ceres::AutoDiffCostFunction
          <ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye, 2, 7, 6, 3>(
            new ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye(observation.data())));
    }
    else
    {
      return
        (new ceres::AutoDiffCostFunction
          <WeightedCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye>, 2, 7, 6, 3>
          (new WeightedCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye>
            (new ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye(observation.data()), weight)));
    }
  }

  const double * m_pos_2dpoint; // The 2D observation
};

struct ResidualErrorFunctor_Intrinsic_Spherical
{
  explicit ResidualErrorFunctor_Intrinsic_Spherical
  (
    const double* const pos_2dpoint,
    const uint32_t imageSize_w,
    const uint32_t imageSize_h
  )
  : m_pos_2dpoint(pos_2dpoint),
    m_imageSize{imageSize_w, imageSize_h}
  {
  }

  template <typename T>
  bool operator()
  (
    const T* const cam_extrinsics,
    const T* const pos_3dpoint,
    T* out_residuals
  )
  const
  {
    // Rotate point by angle-axis
    T p[3];
    ceres::AngleAxisRotatePoint(cam_extrinsics, pos_3dpoint, p);

    // Add translation
    p[0] += cam_extrinsics[3];
    p[1] += cam_extrinsics[4];
    p[2] += cam_extrinsics[5];

    // Transform the coord into Image space
    const T lon = ceres::atan2(p[0], p[2]);
    const T lat = ceres::atan2(-p[1], sqrt(p[0] * p[0] + p[2] * p[2]));
    const T coord[] = {lon / T(2 * M_PI), - lat / T(2 * M_PI)};

    const T size ( std::max(m_imageSize[0], m_imageSize[1]) );
    const T projected_x = coord[0] * size + m_imageSize[0] / 2.0;
    const T projected_y = coord[1] * size + m_imageSize[1] / 2.0;

    out_residuals[0] = projected_x - m_pos_2dpoint[0];
    out_residuals[1] = projected_y - m_pos_2dpoint[1];

    return true;
  }

  static int num_residuals() { return 2; }

  static ceres::CostFunction* Create
  (
    const cameras::IntrinsicBase * cameraInterface,
    const Vec2 & observation,
    const double weight = 0.0
  )
  {
    if (weight == 0.0)
    {
      return
          new ceres::AutoDiffCostFunction
            <ResidualErrorFunctor_Intrinsic_Spherical, 2, 6, 3>(
              new ResidualErrorFunctor_Intrinsic_Spherical(
                observation.data(),
                cameraInterface->w(),
                cameraInterface->h()
              )
            );
    }
    else
    {
      return
        new ceres::AutoDiffCostFunction
          <WeightedCostFunction<ResidualErrorFunctor_Intrinsic_Spherical>, 2, 6, 3>
            (new WeightedCostFunction<ResidualErrorFunctor_Intrinsic_Spherical>
              (new ResidualErrorFunctor_Intrinsic_Spherical(
                observation.data(),
                cameraInterface->w(),
                cameraInterface->h()),
              weight)
            );
    }
  }

  const double * m_pos_2dpoint;  // The 2D observation
  size_t         m_imageSize[2]; // The image width and height
};

} // namespace sfm
} // namespace openMVG

#endif // OPENMVG_SFM_SFM_DATA_BA_CERES_CAMERA_FUNCTOR_HPP