#include "bievr_lio/wheel_factor.h"

namespace bievr {

namespace wheel_factor_internal {

ResidualData makeResidualData(const Quaternion& q_i, const Quaternion& q_j,
                              const WheelIntegrator& integrator, double weight,
                              bool planar_only) {
  ResidualData data;
  data.R_i = q_i.toRotationMatrix();
  data.R_j = q_j.toRotationMatrix();
  data.wheel_velocity_I = integrator.displacement() / integrator.durationS();
  data.weight = weight;
  data.planar_only = planar_only;
  return data;
}

void evaluateResidual(const ResidualData& data, const V3& v_i, const V3& v_j, double scale,
                      double* residuals, double* jac_v_i, double* jac_v_j, double* jac_scale) {
  Eigen::Map<V3> residual(residuals);
  residual = data.weight *
             (0.5 * (data.R_i.transpose() * v_i + data.R_j.transpose() * v_j) -
              scale * data.wheel_velocity_I);

  if (jac_v_i) {
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jac_v_i);
    jac.noalias() = data.weight * 0.5 * data.R_i.transpose();
  }
  if (jac_v_j) {
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jac_v_j);
    jac.noalias() = data.weight * 0.5 * data.R_j.transpose();
  }
  if (jac_scale) {
    Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jac(jac_scale);
    jac = -data.weight * data.wheel_velocity_I.transpose();
  }

  if (data.planar_only) {
    // Zero the body-frame z row: the vertical direction stays with IMU/gravity.
    residual.z() = 0.0;
    if (jac_v_i) Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(jac_v_i).row(2).setZero();
    if (jac_v_j) Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(jac_v_j).row(2).setZero();
    if (jac_scale) Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>>(jac_scale)(2) = 0.0;
  }
}

}  // namespace wheel_factor_internal

bool WheelVelocityFactor::Evaluate(double const* const* parameters, double* residuals,
                                   double** jacobians) const {
  const Eigen::Map<const V3> v_i(parameters[0]);
  const Eigen::Map<const V3> v_j(parameters[1]);
  wheel_factor_internal::evaluateResidual(data_, v_i, v_j, 1.0, residuals,
                                          jacobians ? jacobians[0] : nullptr,
                                          jacobians ? jacobians[1] : nullptr, nullptr);
  return true;
}

bool WheelVelocityScaleFactor::Evaluate(double const* const* parameters, double* residuals,
                                        double** jacobians) const {
  const Eigen::Map<const V3> v_i(parameters[0]);
  const Eigen::Map<const V3> v_j(parameters[1]);
  const double scale = parameters[2][0];
  wheel_factor_internal::evaluateResidual(data_, v_i, v_j, scale, residuals,
                                          jacobians ? jacobians[0] : nullptr,
                                          jacobians ? jacobians[1] : nullptr,
                                          jacobians ? jacobians[2] : nullptr);
  return true;
}

bool WheelScalePriorFactor::Evaluate(double const* const* parameters, double* residuals,
                                     double** jacobians) const {
  residuals[0] = weight_ * (parameters[0][0] - 1.0);
  if (jacobians && jacobians[0]) {
    jacobians[0][0] = weight_;
  }
  return true;
}

}  // namespace bievr
