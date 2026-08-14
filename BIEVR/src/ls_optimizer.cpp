// This Levenberg-Marquardt Optimizer builds upon the code used in the nanogicp module in DLIO
// https://github.com/vectr-ucla/direct_lidar_inertial_odometry

#include "bievr_lio/ls_optimizer.h"

#include <Eigen/Eigenvalues>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>

#include "bievr_lio/log++.h"
#include "bievr_lio/utils.h"

namespace bievr {
namespace {

// Eigenvector signs are mathematically arbitrary. Canonicalizing the largest
// component to be positive prevents harmless frame-to-frame sign flips from
// making the logged weak-direction axis difficult to plot.
template <typename MatrixType>
void canonicalizeEigenvectorSigns(MatrixType& eigenvectors) {
  for (Eigen::Index col = 0; col < eigenvectors.cols(); ++col) {
    Eigen::Index dominant_row = 0;
    eigenvectors.col(col).cwiseAbs().maxCoeff(&dominant_row);
    if (eigenvectors(dominant_row, col) < 0.0) {
      eigenvectors.col(col) *= -1.0;
    }
  }
}

template <typename VectorType>
void spectrumRatios(const VectorType& eigenvalues, double& condition_number,
                    double& degeneracy_ratio) {
  const double lambda_max = std::max(0.0, eigenvalues(eigenvalues.size() - 1));
  const double lambda_min = std::max(0.0, eigenvalues(0));
  if (lambda_max <= 0.0) {
    condition_number = 0.0;
    degeneracy_ratio = 0.0;
    return;
  }

  degeneracy_ratio = lambda_min / lambda_max;
  // Keep the CSV finite even for an exactly singular Hessian. The cap is only
  // for representation; degeneracy_ratio still becomes exactly zero.
  const double floor = std::max(1e-12, lambda_max * 1e-12);
  condition_number = lambda_max / std::max(lambda_min, floor);
}

}  // namespace

LsqRegistration::LsqRegistration(const BIEVRMap& map, const Pointcloud& source,
                                 const RegistrationConfig& config)
    : config_(config), map_(map), points_j_(source) {}

Transform LsqRegistration::computeTransformation(const Transform& T_W_L_init) {
  Transform x0 = T_W_L_init;

  skew_points_j_.resize(points_j_.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, points_j_.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        skew_points_j_[i] = skew(points_j_[i]);
                      }
                    });

  if (config_.lm_debug_print) {
    LOG(I, "***************** optimize *****************");
  }

  for (int i = 0; i < config_.max_iterations && !converged_; i++) {
    ++num_iterations_;
    Transform delta;
    if (!stepLm(x0, delta)) {
      LOG(W, "lm not converged!!");
      break;
    }
    converged_ = isConverged(delta);
  }

  if (config_.collect_diagnostics) {
    Matrix66 H;
    Vector6 b;
    Accumulator statistics;
    linearize(x0, &H, &b, &statistics);
    updateDiagnostics(H, b, statistics);
  }

  return x0;
}

bool LsqRegistration::isConverged(const Transform& delta) const {
  Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = delta.translation();

  Eigen::Matrix3d r_delta = 1.0 / config_.rotation_epsilon * R.array().abs();
  Eigen::Vector3d t_delta = 1.0 / config_.transformation_epsilon * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

double LsqRegistration::linearize(const Transform& T_W_L, Matrix66* H, Vector6* b,
                                  Accumulator* statistics) {
  const bool compute_jacobians = (H != nullptr && b != nullptr);

  Accumulator identity_accumulator;
  identity_accumulator.huber_delta = config_.huber_delta;

  Accumulator total = tbb::parallel_deterministic_reduce(
      tbb::blocked_range<size_t>(0, points_j_.size()),
      identity_accumulator,  // identity
      [&](const tbb::blocked_range<size_t>& r, Accumulator local_acc) -> Accumulator {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          Point p_W = T_W_L.linear() * points_j_[i] + T_W_L.translation();
          size_t hash = map_.hashIndex(p_W);
          const Voxel* voxel = map_.getVoxel(hash);
          if (!voxel) {
            if (!map_.nearestVoxel(p_W, hash)) continue;
            voxel = map_.getVoxel(hash);
            if (!voxel) continue;
          }

          const double inv_size = map_.inv_px_size;
          const auto& T_C_W = voxel->T_C_W_;

          Rotation R_o_j = T_C_W.linear() * T_W_L.linear();
          const Point p_o = T_C_W * p_W;

          const double x = p_o.x() * inv_size;
          const double y = p_o.y() * inv_size;

          double I = 0.0;

          if (!compute_jacobians) {
            if (config_.img_residual) {
              if (!getSubPixelValue(voxel, x, y, I)) continue;
            }
            const double r = p_o.z() - I;
            local_acc.add(r, nullptr);
            continue;
          }

          double dIdx = 0.0;
          double dIdy = 0.0;
          if (config_.img_residual) {
            if (!sampleValueAndGradient(voxel, x, y, I, dIdx, dIdy)) continue;
          }

          Eigen::Matrix<double, 3, 6> SE3_Jac;
          SE3_Jac.block<3, 3>(0, 3) = R_o_j;  // d p_o / d t
          SE3_Jac.block<3, 3>(0, 0).noalias() = -R_o_j * skew_points_j_[i];

          Eigen::Matrix<double, 1, 2> I_Jac;
          if (config_.img_jacobian) {
            I_Jac(0, 0) = dIdx;
            I_Jac(0, 1) = dIdy;
            I_Jac *= inv_size;
          }

          Row6 J = SE3_Jac.row(2) - (I_Jac * SE3_Jac.topRows<2>());

          const double r = p_o.z() - I;
          local_acc.add(r, &J);
        }

        return local_acc;
      },
      [](const Accumulator& a, const Accumulator& b) -> Accumulator {
        Accumulator out = a;
        out.merge(b);
        return out;
      });

  if (compute_jacobians) {
    *H = total.H;
    *b = total.b;
    // Remember how many points contributed correspondences in this (Jacobian)
    // linearization so the pipeline can report the effective point count.
    num_effective_points_ = total.count;
  }
  if (statistics) {
    *statistics = total;
  }

  return total.error_sum;
}

void LsqRegistration::updateDiagnostics(const Matrix66& H, const Vector6& b,
                                        const Accumulator& statistics) {
  diagnostics_ = RegistrationDiagnostics();
  diagnostics_.converged = converged_;
  diagnostics_.iterations = num_iterations_;
  diagnostics_.source_points = points_j_.size();
  diagnostics_.effective_points = statistics.count;
  diagnostics_.huber_inlier_points = statistics.huber_inlier_count;
  diagnostics_.robust_cost = statistics.error_sum;
  diagnostics_.weight_sum = statistics.weight_sum;
  diagnostics_.gradient_norm = b.norm();
  diagnostics_.hard_projection_enabled = config_.hard_project_weakest_translation;
  diagnostics_.weak_translation_retention_alpha =
      config_.weak_translation_retention_alpha;
  diagnostics_.hard_projection_ratio_threshold =
      config_.translation_degeneracy_ratio_threshold;
  diagnostics_.lm_trial_steps = lm_trial_steps_;
  diagnostics_.lm_rejected_steps = lm_rejected_steps_;
  diagnostics_.hard_projection_trial_steps = hard_projection_trial_steps_;
  diagnostics_.hard_projection_accepted_steps = hard_projection_accepted_steps_;
  diagnostics_.hard_projection_accepted_weak_before_abs_max_m =
      hard_projection_accepted_weak_before_abs_max_m_;
  diagnostics_.hard_projection_accepted_removed_abs_sum_m =
      hard_projection_accepted_removed_abs_sum_m_;
  diagnostics_.hard_projection_accepted_removed_abs_max_m =
      hard_projection_accepted_removed_abs_max_m_;
  diagnostics_.hard_projection_accepted_weak_after_abs_max_m =
      hard_projection_accepted_weak_after_abs_max_m_;

  if (!points_j_.empty()) {
    diagnostics_.effective_ratio =
        static_cast<double>(statistics.count) / static_cast<double>(points_j_.size());
  }
  if (statistics.count > 0) {
    diagnostics_.huber_inlier_ratio = static_cast<double>(statistics.huber_inlier_count) /
                                      static_cast<double>(statistics.count);
    diagnostics_.residual_rmse =
        std::sqrt(statistics.squared_error_sum / static_cast<double>(statistics.count));
  }

  const Matrix66 H_symmetric = 0.5 * (H + H.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix66> hessian_solver(H_symmetric);

  const M3 H_translation = H_symmetric.bottomRightCorner<3, 3>();
  Eigen::SelfAdjointEigenSolver<M3> translation_solver(H_translation);

  if (hessian_solver.info() != Eigen::Success || translation_solver.info() != Eigen::Success ||
      statistics.count == 0) {
    return;
  }

  diagnostics_.hessian_eigenvalues = hessian_solver.eigenvalues();
  diagnostics_.hessian_eigenvectors = hessian_solver.eigenvectors();
  canonicalizeEigenvectorSigns(diagnostics_.hessian_eigenvectors);
  spectrumRatios(diagnostics_.hessian_eigenvalues, diagnostics_.hessian_condition_number,
                 diagnostics_.hessian_degeneracy_ratio);

  diagnostics_.translation_eigenvalues = translation_solver.eigenvalues();
  diagnostics_.translation_eigenvectors = translation_solver.eigenvectors();
  canonicalizeEigenvectorSigns(diagnostics_.translation_eigenvectors);
  spectrumRatios(diagnostics_.translation_eigenvalues,
                 diagnostics_.translation_condition_number,
                 diagnostics_.translation_degeneracy_ratio);
  diagnostics_.weakest_translation_information_per_point =
      std::max(0.0, diagnostics_.translation_eigenvalues(0)) /
      static_cast<double>(statistics.count);
  diagnostics_.valid = true;
}

namespace {

Eigen::Quaterniond so3_exp(const Eigen::Vector3d& omega) {
  double theta_sq = omega.dot(omega);

  double theta;
  double imag_factor;
  double real_factor;
  if (theta_sq < 1e-10) {
    theta = 0;
    double theta_quad = theta_sq * theta_sq;
    imag_factor = 0.5 - 1.0 / 48.0 * theta_sq + 1.0 / 3840.0 * theta_quad;
    real_factor = 1.0 - 1.0 / 8.0 * theta_sq + 1.0 / 384.0 * theta_quad;
  } else {
    theta = std::sqrt(theta_sq);
    double half_theta = 0.5 * theta;
    imag_factor = std::sin(half_theta) / theta;
    real_factor = std::cos(half_theta);
  }

  return Eigen::Quaterniond(real_factor, imag_factor * omega.x(), imag_factor * omega.y(),
                            imag_factor * omega.z());
}

}  // namespace

bool LsqRegistration::stepLm(Transform& x0, Transform& delta) {
  Matrix66 H;
  Vector6 b;
  double y0 = linearize(x0, &H, &b);

  bool weak_translation_attenuation_active = false;
  V3 weakest_translation_direction = V3::Zero();
  if (config_.hard_project_weakest_translation) {
    // 从完整 6*6 Hessian 取出 H_{tt}
    const M3 H_translation = H.bottomRightCorner<3, 3>();
    // 强制构造对称矩阵，弥补浮点精度误差
    const M3 H_translation_symmetric =
        0.5 * (H_translation + H_translation.transpose());
    Eigen::SelfAdjointEigenSolver<M3> translation_solver(H_translation_symmetric);
    if (translation_solver.info() == Eigen::Success) {
      const V3 eigenvalues = translation_solver.eigenvalues(); // 特征值
      const double lambda_min = std::max(0.0, eigenvalues(0));
      const double lambda_max = std::max(0.0, eigenvalues(2));
      if (lambda_max > 0.0) {
        const double ratio = lambda_min / lambda_max;
        // 退化程度比达到阈值时，取最小特征值对应的特征向量
        if (ratio < config_.translation_degeneracy_ratio_threshold) {
          weakest_translation_direction = translation_solver.eigenvectors().col(0);
          weak_translation_attenuation_active = true;
        }
      }
    }
  }

  if (lm_lambda_ < 0.0) {
    lm_lambda_ = config_.lm_init_lambda_factor * H.diagonal().array().abs().maxCoeff();
  }

  double nu = 2.0;
  // 开始LM求解
  for (int i = 0; i < config_.lm_max_iterations; i++) {
    ++lm_trial_steps_;
    // 防奇异法求解 6DoF 位姿增量 d = [d_rotation; d_translation]
    Eigen::LDLT<Matrix66> solver(H + lm_lambda_ * Matrix66::Identity());
    Vector6 d = solver.solve(-b);

    // 将最弱约束方向的平移增量保留为原来的 alpha 倍
    bool weak_translation_attenuation_applied = false;
    double weak_translation_before = 0.0;
    double weak_translation_after = 0.0;
    if (weak_translation_attenuation_active) {
      ++hard_projection_trial_steps_;
      V3 translation = d.tail<3>();
      weak_translation_before = weakest_translation_direction.dot(translation);
      const double removed_weak_translation =
          (1.0 - config_.weak_translation_retention_alpha) * weak_translation_before;
      translation -= weakest_translation_direction * removed_weak_translation;
      weak_translation_after = weakest_translation_direction.dot(translation);
      d.tail<3>() = translation;
      weak_translation_attenuation_applied = true;
    }

    delta.setIdentity();
    delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
    delta.translation() = d.tail<3>();

    Transform xi = x0 * delta;
    double yi = linearize(xi);
    double rho = (y0 - yi) / (d.dot(lm_lambda_ * d - b)); // 模型线性化程度

    if (rho < 0) {
      ++lm_rejected_steps_;
      if (isConverged(delta)) {
        return true;
      }

      lm_lambda_ *= nu;
      nu *= 2;
      continue;
    }

    // 记录软约束前后弱方向分量，便于核对实际保留比例
    if (weak_translation_attenuation_applied) {
      ++hard_projection_accepted_steps_;
      hard_projection_accepted_weak_before_abs_max_m_ =
          std::max(hard_projection_accepted_weak_before_abs_max_m_,
                   std::abs(weak_translation_before));
      const double removed_abs = std::abs(weak_translation_before - weak_translation_after);
      hard_projection_accepted_removed_abs_sum_m_ += removed_abs;
      hard_projection_accepted_removed_abs_max_m_ =
          std::max(hard_projection_accepted_removed_abs_max_m_, removed_abs);
      hard_projection_accepted_weak_after_abs_max_m_ =
          std::max(hard_projection_accepted_weak_after_abs_max_m_,
                   std::abs(weak_translation_after));
    }

    x0 = xi;
    lm_lambda_ *= std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
    return true;
  }

  return false;
}

}  // namespace bievr
