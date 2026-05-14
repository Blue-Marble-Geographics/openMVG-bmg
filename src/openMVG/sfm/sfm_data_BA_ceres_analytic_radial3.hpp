// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.
//
// Analytic-Jacobian replacement for the Pinhole + Radial3 reprojection cost
// previously evaluated via ceres::AutoDiffCostFunction<...>. The forward pass
// is bit-equivalent to the AutoDiff functor in
// `sfm_data_BA_ceres_camera_functor.hpp`; the Jacobian is derived in closed
// form so we don't pay for Jet propagation at runtime.
//
// Why this exists:
//   AutoDiff with K=15 partials propagates 15-component Jets through every
//   multiply / sin / cos in the projection chain. On the slow datasets each
//   Jacobian eval was ~3.1s of 16.7s total final-BA wall time, repeated
//   across ~20 intermediate BAs. This class computes the same numerical
//   Jacobian without Jet machinery; expected ~3-5x faster Jacobian eval.
//
// Output equivalence:
//   Residuals are computed with the same scalar arithmetic as the AutoDiff
//   path -> bit-identical residuals.
//   Jacobians are the closed-form analytical derivative of the same forward
//   pass; AutoDiff is mathematically exact, so AutoDiff and analytic agree
//   to ~1e-10 (FP rounding only). See RunSelfTest() at the bottom.
//
// Toggle:
//   See `kUseAnalyticJacobian_Radial3` in sfm_data_BA_ceres.cpp. Default
//   false until the self-test has been run on a representative dataset.
//
// Coverage:
//   PINHOLE_CAMERA_RADIAL3 only. All other intrinsic models continue to use
//   the AutoDiff path. Weighted observations (weight != 0.0) also fall back
//   to AutoDiff for simplicity.

#pragma once

#include <ceres/sized_cost_function.h>
#include <cmath>

namespace openMVG {
namespace sfm {

class AnalyticReprojectionCost_Radial3 final
    : public ceres::SizedCostFunction<2, 6, 6, 3>
{
 public:
  // observation = (u, v) in pixels.
  explicit AnalyticReprojectionCost_Radial3(const double* obs)
      : observation_x_(obs[0]), observation_y_(obs[1]) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override
  {
    // Parameter blocks (matches functor layout exactly):
    //   parameters[0] = intrinsics: [f, cx, cy, k1, k2, k3]
    //   parameters[1] = extrinsics: [r0, r1, r2, t0, t1, t2]  (angle-axis + t)
    //   parameters[2] = world point: [X, Y, Z]
    const double* intr = parameters[0];
    const double* extr = parameters[1];
    const double* X    = parameters[2];

    const double w0 = extr[0], w1 = extr[1], w2 = extr[2];
    const double t0 = extr[3], t1 = extr[4], t2 = extr[5];

    // ----- Forward pass: Rodrigues rotation + translation -----
    // R(omega) X = cos(theta) X + (sin(theta)/theta) (omega x X)
    //            + ((1 - cos(theta))/theta^2) (omega . X) omega
    //
    // For numerical stability near theta = 0, we use limiting forms of the
    // (sin/theta), (1-cos)/theta^2 expressions. cos itself is well-behaved.
    const double theta2 = w0*w0 + w1*w1 + w2*w2;
    const double theta  = std::sqrt(theta2);

    const double alpha = std::cos(theta);                  // cos(theta)
    double beta;                                           // sin(theta)/theta
    double gamma;                                          // (1-cos(theta))/theta^2
    if (theta2 < 1e-12) {
      // Taylor: sin/theta = 1 - theta^2/6 + ..., (1-cos)/theta^2 = 1/2 - theta^2/24 + ...
      beta  = 1.0 - theta2 * (1.0/6.0);
      gamma = 0.5 - theta2 * (1.0/24.0);
    } else {
      const double sin_theta = std::sin(theta);
      beta  = sin_theta / theta;
      gamma = (1.0 - alpha) / theta2;
    }

    const double cx0 = w1*X[2] - w2*X[1];   // (omega x X).x
    const double cx1 = w2*X[0] - w0*X[2];   // (omega x X).y
    const double cx2 = w0*X[1] - w1*X[0];   // (omega x X).z
    const double wdotX = w0*X[0] + w1*X[1] + w2*X[2];

    const double RX0 = alpha*X[0] + beta*cx0 + gamma*wdotX*w0;
    const double RX1 = alpha*X[1] + beta*cx1 + gamma*wdotX*w1;
    const double RX2 = alpha*X[2] + beta*cx2 + gamma*wdotX*w2;

    const double P0 = RX0 + t0;
    const double P1 = RX1 + t1;
    const double P2 = RX2 + t2;

    // Perspective divide. The AutoDiff functor returns true unconditionally
    // (uses hnormalized() with no behind-camera guard); we mirror that for
    // strict bit-equivalence.
    const double inv_z = 1.0 / P2;
    const double xn = P0 * inv_z;   // x in normalized image plane
    const double yn = P1 * inv_z;

    const double r2 = xn*xn + yn*yn;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double f   = intr[0];
    const double cxp = intr[1];
    const double cyp = intr[2];
    const double k1  = intr[3];
    const double k2  = intr[4];
    const double k3  = intr[5];

    const double d  = 1.0 + k1*r2 + k2*r4 + k3*r6;
    const double xd = xn * d;
    const double yd = yn * d;

    residuals[0] = cxp + f*xd - observation_x_;
    residuals[1] = cyp + f*yd - observation_y_;

    if (!jacobians) return true;

    // ----- Jacobians -----
    //
    // Block layout per Ceres convention: jacobians[i] is row-major,
    // size = num_residuals (2) x parameter_block_size_[i].
    //   jacobians[0]: 2 x 6 (intrinsics)
    //   jacobians[1]: 2 x 6 (extrinsics: [omega(3), t(3)])
    //   jacobians[2]: 2 x 3 (world point)

    // d's derivative w.r.t. r^2:
    //   d_d_dr2 := dd/d(r^2) = k1 + 2 k2 r^2 + 3 k3 r^4
    // Then ∂xd/∂xn = d + 2 xn^2 * d_d_dr2, etc.
    const double d_d_dr2 = k1 + 2.0*k2*r2 + 3.0*k3*r4;

    // 2x2 Jacobian of distorted-normalized w.r.t. undistorted-normalized:
    //   [[d + 2*xn^2*D, 2*xn*yn*D],
    //    [2*xn*yn*D,    d + 2*yn^2*D]]
    const double D_xx = d + 2.0*xn*xn*d_d_dr2;
    const double D_xy =     2.0*xn*yn*d_d_dr2;
    const double D_yy = d + 2.0*yn*yn*d_d_dr2;

    // ----- Block 0: ∂residual / ∂intrinsics (2 x 6) -----
    if (jacobians[0]) {
      double* J = jacobians[0];
      // Row 0 (rx)
      J[0]  = xd;            // ∂rx/∂f
      J[1]  = 1.0;           // ∂rx/∂cx
      J[2]  = 0.0;           // ∂rx/∂cy
      J[3]  = f * xn * r2;   // ∂rx/∂k1
      J[4]  = f * xn * r4;   // ∂rx/∂k2
      J[5]  = f * xn * r6;   // ∂rx/∂k3
      // Row 1 (ry)
      J[6]  = yd;
      J[7]  = 0.0;
      J[8]  = 1.0;
      J[9]  = f * yn * r2;
      J[10] = f * yn * r4;
      J[11] = f * yn * r6;
    }

    // ----- Common: ∂residual / ∂P_camera (2 x 3) -----
    // ∂(xd, yd) / ∂P = D_2x2 * Jproj_2x3 where
    //   Jproj = (1/Pz) * [[1, 0, -xn], [0, 1, -yn]]
    // Then ∂residual/∂P = f * ∂(xd,yd)/∂P.
    //
    // We need this for both the extrinsics and the point block; cache it.
    const double f_inv_z = f * inv_z;
    const double dr_dP[2][3] = {
      { f_inv_z * D_xx,
        f_inv_z * D_xy,
        f_inv_z * (-D_xx*xn - D_xy*yn) },
      { f_inv_z * D_xy,                            // (D_yx == D_xy)
        f_inv_z * D_yy,
        f_inv_z * (-D_xy*xn - D_yy*yn) }
    };

    // ----- Block 1: ∂residual / ∂extrinsics (2 x 6) -----
    // First 3 columns: ∂residual/∂omega (rotation axis-angle).
    // Last 3 columns:  ∂residual/∂t = ∂residual/∂P  (since ∂P/∂t = I).
    if (jacobians[1]) {
      double* J = jacobians[1];

      // Compute ∂(R X) / ∂omega_i for i = 0,1,2.
      //
      // Differentiating Rodrigues:
      //   alpha     = cos(theta)
      //   beta      = sin(theta)/theta
      //   gamma     = (1 - cos(theta))/theta^2
      //   ∂alpha/∂omega_i = -beta * omega_i
      //   ∂beta /∂omega_i = omega_i * (alpha - beta) / theta^2
      //   ∂gamma/∂omega_i = omega_i * (beta  - 2*gamma) / theta^2
      //
      // Numerically-stable Taylor expansions for small theta:
      //   (alpha - beta)/theta^2  -> -1/3 + theta^2/30   (cos - sinc form)
      //   (beta  - 2*gamma)/theta^2 -> -1/12 + theta^2/180
      double dbeta_factor;
      double dgamma_factor;
      if (theta2 < 1e-6) {
        dbeta_factor  = -1.0/3.0  + theta2 * (1.0/30.0);
        dgamma_factor = -1.0/12.0 + theta2 * (1.0/180.0);
      } else {
        dbeta_factor  = (alpha - beta) / theta2;
        dgamma_factor = (beta  - 2.0*gamma) / theta2;
      }
      const double dalpha_factor = -beta;

      // ∂(R X) / ∂omega_i = dalpha*omega_i*X
      //                   + dbeta *omega_i*(omega x X)
      //                   + beta  *(e_i x X)
      //                   + dgamma*omega_i*(omega.X)*omega
      //                   + gamma *X_i*omega
      //                   + gamma *(omega.X)*e_i
      //
      // We compute ∂P/∂omega as a 3x3 matrix dPdw[3][3] where
      //   dPdw[row][i] = d P_row / d omega_i.
      double dPdw[3][3];
      for (int i = 0; i < 3; ++i) {
        const double wi = (i == 0) ? w0 : (i == 1) ? w1 : w2;
        const double Xi = X[i];
        // (e_i x X) where (a,b,c) x (x,y,z) = (b*z - c*y, c*x - a*z, a*y - b*x):
        //   e_0 x X = ( 0,    -X[2],  X[1])
        //   e_1 x X = ( X[2],  0,    -X[0])
        //   e_2 x X = (-X[1],  X[0],  0   )
        const double ei_cross_X[3] = {
          (i == 1) ?  X[2] : (i == 2) ? -X[1] : 0.0,
          (i == 0) ? -X[2] : (i == 2) ?  X[0] : 0.0,
          (i == 0) ?  X[1] : (i == 1) ? -X[0] : 0.0
        };
        for (int row = 0; row < 3; ++row) {
          const double wrow  = (row == 0) ? w0 : (row == 1) ? w1 : w2;
          const double Xrow  = X[row];
          const double crow  = (row == 0) ? cx0 : (row == 1) ? cx1 : cx2;
          const double e_ir  = (row == i) ? 1.0 : 0.0;
          dPdw[row][i] = dalpha_factor * wi * Xrow
                       + dbeta_factor  * wi * crow
                       + beta          * ei_cross_X[row]
                       + dgamma_factor * wi * wdotX * wrow
                       + gamma         * Xi * wrow
                       + gamma         * wdotX * e_ir;
        }
      }

      // ∂residual/∂omega = ∂residual/∂P * ∂P/∂omega   (2x3 = (2x3)*(3x3))
      for (int i = 0; i < 3; ++i) {
        J[0*6 + i] = dr_dP[0][0]*dPdw[0][i] + dr_dP[0][1]*dPdw[1][i] + dr_dP[0][2]*dPdw[2][i];
        J[1*6 + i] = dr_dP[1][0]*dPdw[0][i] + dr_dP[1][1]*dPdw[1][i] + dr_dP[1][2]*dPdw[2][i];
      }
      // ∂residual/∂t = ∂residual/∂P
      J[0*6 + 3] = dr_dP[0][0];  J[0*6 + 4] = dr_dP[0][1];  J[0*6 + 5] = dr_dP[0][2];
      J[1*6 + 3] = dr_dP[1][0];  J[1*6 + 4] = dr_dP[1][1];  J[1*6 + 5] = dr_dP[1][2];
    }

    // ----- Block 2: ∂residual / ∂X_world (2 x 3) -----
    // ∂P/∂X = R, so ∂residual/∂X = ∂residual/∂P * R.
    if (jacobians[2]) {
      double* J = jacobians[2];

      // R[i][j] = alpha * delta_ij + gamma * w_i*w_j + beta * [omega]_x_{ij}
      // [omega]_x = [[0,-w2,w1],[w2,0,-w0],[-w1,w0,0]]
      const double R[3][3] = {
        { alpha + gamma*w0*w0,       gamma*w0*w1 - beta*w2,  gamma*w0*w2 + beta*w1 },
        { gamma*w1*w0 + beta*w2,     alpha + gamma*w1*w1,    gamma*w1*w2 - beta*w0 },
        { gamma*w2*w0 - beta*w1,     gamma*w2*w1 + beta*w0,  alpha + gamma*w2*w2  }
      };
      for (int j = 0; j < 3; ++j) {
        J[0*3 + j] = dr_dP[0][0]*R[0][j] + dr_dP[0][1]*R[1][j] + dr_dP[0][2]*R[2][j];
        J[1*3 + j] = dr_dP[1][0]*R[0][j] + dr_dP[1][1]*R[1][j] + dr_dP[1][2]*R[2][j];
      }
    }

    return true;
  }

 private:
  const double observation_x_;
  const double observation_y_;
};

// ---------------------------------------------------------------------------
// Self-test: validate analytic Jacobians against AutoDiff over random inputs.
// Returns max |delta| seen across residuals & jacobians; pass threshold ~1e-8.
// Implemented in sfm_data_BA_ceres.cpp (needs the AutoDiff functor visible).
// Default argument lives on the declaration in sfm_data_BA_ceres.hpp; this
// secondary declaration is unqualified to avoid a redefinition diagnostic.
// ---------------------------------------------------------------------------
double RunSelfTest_AnalyticReprojectionCost_Radial3(int trials);

}  // namespace sfm
}  // namespace openMVG
