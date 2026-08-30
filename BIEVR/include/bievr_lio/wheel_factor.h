#ifndef BIEVR_LIO_WHEEL_FACTOR_H_
#define BIEVR_LIO_WHEEL_FACTOR_H_

#include <ceres/ceres.h>

#include "bievr_lio/common.h"
#include "bievr_lio/wheel_integrator.h"

namespace bievr {

// State and residual shared by the wheel velocity factors (plain and
// scale-estimating variants). The residual is the difference between the
// average IMU-frame velocity implied by the state velocities (trapezoidal rule
// over the fixed poses) and the wheel-measured average velocity
//
//   r = weight * (0.5 * (R_i^T * v_i + R_j^T * v_j) - scale * delta_p_I / dt).
//
// In planar mode the body-frame z row is zeroed, constraining only the x/y
// motion. Jacobian pointers may be null; non-null arrays are row-major
// (jac_v_i/jac_v_j: 3x3, jac_scale: 1x3), matching ceres' layout.
namespace wheel_factor_internal {
struct ResidualData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Rotation R_i = Rotation::Identity();
  Rotation R_j = Rotation::Identity();
  V3 wheel_velocity_I = V3::Zero();
  double weight = 1.0;
  bool planar_only = true;
};

ResidualData makeResidualData(const Quaternion& q_i, const Quaternion& q_j,
                              const WheelIntegrator& integrator, double weight, bool planar_only);

// Evaluates the residual and optionally computes the Jacobians.
void evaluateResidual(const ResidualData& data, const V3& v_i, const V3& v_j, double scale,
                      double* residuals, double* jac_v_i, double* jac_v_j, double* jac_scale);
}  // namespace wheel_factor_internal

// Velocity constraint from integrated wheel odometry between two consecutive
// states. The rotations are captured at construction time: the pipeline keeps
// every pose block constant inside the sliding-window optimization, so R_i and
// R_j cannot change between construction and solve — only v_i and v_j are
// optimized.
class WheelVelocityFactor : public ceres::SizedCostFunction<3, 3, 3> {
 public:
  WheelVelocityFactor() = delete;
  WheelVelocityFactor(const Quaternion& q_i, const Quaternion& q_j,
                      const WheelIntegrator& integrator, double weight, bool planar_only)
      : data_(wheel_factor_internal::makeResidualData(q_i, q_j, integrator, weight,
                                                      planar_only)) {}
  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;

 protected:
  wheel_factor_internal::ResidualData data_;
};

// Same residual with an additional shared multiplicative wheel scale parameter
// (wheel radius error), optimized jointly across all wheel factors in the
// window. Pair with WheelScalePriorFactor to keep the scale near 1.
class WheelVelocityScaleFactor : public ceres::SizedCostFunction<3, 3, 3, 1> {
 public:
  WheelVelocityScaleFactor() = delete;
  WheelVelocityScaleFactor(const Quaternion& q_i, const Quaternion& q_j,
                           const WheelIntegrator& integrator, double weight, bool planar_only)
      : data_(wheel_factor_internal::makeResidualData(q_i, q_j, integrator, weight,
                                                      planar_only)) {}
  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;

 protected:
  wheel_factor_internal::ResidualData data_;
};

// 1-dim prior pulling the wheel scale toward 1, weighted by 1/sigma.
class WheelScalePriorFactor : public ceres::SizedCostFunction<1, 1> {
 public:
  WheelScalePriorFactor() = delete;
  explicit WheelScalePriorFactor(double sigma) : weight_(1.0 / sigma) {}
  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override;

 private:
  double weight_;
};

}  // namespace bievr

#endif  // BIEVR_LIO_WHEEL_FACTOR_H_
