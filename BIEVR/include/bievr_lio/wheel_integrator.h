#ifndef BIEVR_LIO_WHEEL_INTEGRATOR_H_
#define BIEVR_LIO_WHEEL_INTEGRATOR_H_

#include <memory>

#include "bievr_lio/common.h"
#include "bievr_lio/utils.h"

namespace bievr {

// Wheel-odometry fusion parameters. Topic, enable flag and extrinsic are
// sensor properties and live in the sensor config; the noise/behaviour knobs
// live in params.yaml (see config_loader.h for the split).
struct WheelConfig {
  bool enabled = false;
  // Transform from the wheel-odometry child frame into the IMU frame: a point
  // expressed in the wheel frame maps into the IMU frame via T_I_W.
  Transform T_I_W = Transform::Identity();
  // 1-sigma of the wheel velocity measurement [m/s]; the factor weight is
  // 1/sigma_v.
  double sigma_v = 0.15;
  // Maximum sensor-time lag tolerated while waiting for a wheel sample at or
  // after the LiDAR frame end. The synchronizer then degrades gracefully and
  // processes the frame without an end bracket instead of stalling the LIO.
  double max_sync_delay_s = 0.05;
  // Ground-robot mode: constrain only the body-frame x/y displacement, leaving
  // the vertical direction to IMU/gravity.
  bool planar_only = true;
  // Estimate a shared multiplicative scale (wheel radius error) in the window
  // optimization, regularized by a prior of scale_prior_sigma around 1.
  bool estimate_scale = false;
  double scale_prior_sigma = 0.1;
};

// Integrates the wheel-odometry twist over one state interval [t_start, t_end]
// into the IMU-frame displacement of the IMU origin.
//
// The wheel twist is the velocity of the wheel frame origin expressed in the
// wheel frame axes, so the IMU-frame velocity is
//   v_I = R_I_W * v_W - omega_I x t_I_W,
// where t_I_W is the wheel frame origin position in IMU coordinates (the
// translation of T_I_W) and omega_I the IMU angular rate (gyro): a rotation
// about the IMU origin moves the wheel frame origin, and the lever-arm term
// removes that contribution so pure rotation integrates to zero displacement.
// The lever arm is applied once in finalize() with the interval-mean gyro,
// which keeps the per-sample path a plain frame rotation.
//
// Displacements integrate trapezoidally. The interval boundaries are bracked
// with the nearest samples outside the interval (trapezoid across the gap);
// when no start bracket is available the first in-interval velocity is held
// back to t_start. Brackets are only used while fresh (kMaxBracketAgeS), so a
// wheel publisher that dies mid-run degrades to fewer constraints instead of
// freezing an old velocity.
class WheelIntegrator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<WheelIntegrator>;

  WheelIntegrator(const WheelConfig& config, uint64_t t_start, uint64_t t_end)
      : config_(config), t_start_(t_start), t_end_(t_end) {}

  // Adds one ordered sample: at or before t_start updates the start bracket,
  // after t_end the end bracket (consumed by finalize), inside the interval it
  // accumulates. The Synchronizer hands over an ordered slice, so out-of-order
  // input is only dropped defensively.
  void addMeasurement(const WheelMeasurement& wheel) {
    const Rotation R_I_W = config_.T_I_W.rotation();
    const V3 v_I = R_I_W * wheel.v_body;
    yaw_rate_sum_ += wheel.yaw_rate;
    ++yaw_rate_count_;
    if (wheel.stamp <= t_start_) {
      if (t_start_ - wheel.stamp <= sToNs(kMaxBracketAgeS)) {
        has_start_bracket_ = true;
        v_start_I_ = v_I;
      }
      return;
    }
    if (wheel.stamp > t_end_) {
      if (wheel.stamp - t_end_ <= sToNs(kMaxBracketAgeS)) {
        has_end_bracket_ = true;
        v_end_I_ = v_I;
      }
      return;
    }
    if (num_samples_ > 0 && wheel.stamp <= last_stamp_) return;
    if (num_samples_ == 0) {
      const double dt_lead = nsToS(wheel.stamp - t_start_);
      // Bridge t_start to the first sample: trapezoid when a fresh bracket
      // gives the velocity at t_start, hold the first velocity otherwise.
      delta_p_I_ += (has_start_bracket_ ? 0.5 * (v_start_I_ + v_I) : v_I) * dt_lead;
    } else {
      delta_p_I_ += 0.5 * (last_v_I_ + v_I) * nsToS(wheel.stamp - last_stamp_);
    }
    last_stamp_ = wheel.stamp;
    last_v_I_ = v_I;
    ++num_samples_;
  }

  // Applies the lever-arm correction with the interval-mean IMU angular rate
  // and bridges the tail up to t_end with the end bracket (hold when absent).
  // Must be called once after all samples were added, before displacement()
  // and valid() are read.
  void finalize(const V3& mean_gyro_I) {
    delta_p_I_ -= skew(mean_gyro_I) * config_.T_I_W.translation() * durationS();
    if (num_samples_ > 0) {
      const double dt_tail = nsToS(t_end_ - last_stamp_);
      delta_p_I_ += (has_end_bracket_ ? 0.5 * (last_v_I_ + v_end_I_) : last_v_I_) * dt_tail;
    }
    finalized_ = true;
  }

  // True when at least one in-interval sample covers the interval, so the
  // displacement spans [t_start, t_end] and can constrain the factor.
  bool valid() const { return finalized_ && num_samples_ > 0 && t_end_ > t_start_; }

  const V3& displacement() const { return delta_p_I_; }
  double durationS() const { return nsToS(t_end_ - t_start_); }
  int numSamples() const { return num_samples_; }
  // Mean raw yaw rate over all samples handed to this interval (brackets
  // included). 0 when no sample carried a yaw rate.
  double meanYawRate() const {
    return yaw_rate_count_ > 0 ? yaw_rate_sum_ / yaw_rate_count_ : 0.0;
  }
  uint64_t startTime() const { return t_start_; }
  uint64_t endTime() const { return t_end_; }

 private:
  static constexpr double kMaxBracketAgeS = 0.5;

  WheelConfig config_;
  uint64_t t_start_;
  uint64_t t_end_;
  V3 delta_p_I_ = V3::Zero();
  V3 last_v_I_ = V3::Zero();
  V3 v_start_I_ = V3::Zero();
  V3 v_end_I_ = V3::Zero();
  uint64_t last_stamp_ = 0;
  int num_samples_ = 0;
  double yaw_rate_sum_ = 0.0;
  int yaw_rate_count_ = 0;
  bool has_start_bracket_ = false;
  bool has_end_bracket_ = false;
  bool finalized_ = false;
};

}  // namespace bievr

#endif  // BIEVR_LIO_WHEEL_INTEGRATOR_H_
