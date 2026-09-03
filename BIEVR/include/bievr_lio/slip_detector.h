#ifndef BIEVR_LIO_SLIP_DETECTOR_H_
#define BIEVR_LIO_SLIP_DETECTOR_H_

#include <deque>

#include "bievr_lio/common.h"

namespace bievr {

// Slip-detection state machine. When wheel fusion is enabled the pipeline
// adds wheel factors only while state == Normal; suggested_weight is still
// recorded for offline analysis.
enum class SlipState {
  Normal = 0,
  Suspect = 1,
  Slip = 2,
  Recovering = 3,
};

// Thresholds and hysteresis. Defaults are simulation first guesses for a
// small differential-drive robot.
struct SlipConfig {
  // Translation high thresholds. The translation channel is high when BOTH
  // |r_x| > abs_threshold_mps AND |r_x|/v_ref > rel_threshold (when the
  // relative check is active); the dual condition prevents the relative ratio
  // from firing at near-zero speed.
  double abs_threshold_mps = 0.20;
  double rel_threshold = 0.35;
  // High threshold of the yaw-rate channel |wheel_wz - gyro_wz|. A frame is
  // high if the translation test fires OR this yaw test fires, so one-sided
  // differential-drive slip can enter SLIP without a large r_x. Simulation
  // first guess: ~0.5 rad/s injection is clearly above this, clean-bag |r_wz|
  // is ~0.
  double wz_threshold_rps = 0.25;
  // Low thresholds: entering SUSPECT needs any channel exceeded; recovering
  // requires every channel below for recover_time_s.
  double suspect_abs_mps = 0.12;
  double suspect_rel = 0.20;
  double suspect_wz_rps = 0.12;
  // LiDAR velocity is obtained by differentiating consecutive registered
  // poses, so a short time-weighted history suppresses single-frame spikes.
  int lidar_velocity_window_frames = 3;
  // SLIP entry uses a K-of-N vote instead of requiring all high residuals to be
  // consecutive. The old slip_n_high_frames key is accepted as a legacy alias
  // for high_votes_required by the YAML loader.
  int high_window_frames = 5;
  int high_votes_required = 3;
  // Relative residuals are disabled below this actual speed. This prevents a
  // small pose jitter divided by a near-zero speed floor from becoming large.
  double relative_min_speed_mps = 0.15;
  double recover_time_s = 0.75;
  // Floor of the reference speed in the relative check: at near-zero speed
  // the ratio |r_x|/|v| is meaningless and the absolute check carries alone.
  double speed_floor_mps = 0.1;
};

// Per-frame inputs, all expressed in the body (IMU) frame at the interval
// start, averaged over the state interval [t_i, t_j].
struct SlipDetectionInput {
  double dt = 0.0;                        // interval duration [s]
  V3 wheel_v = V3::Zero();                // wheel mean velocity (displacement/dt)
  double wheel_wz = 0.0;                  // wheel mean yaw rate [rad/s]
  double gyro_wz = 0.0;                   // IMU mean yaw rate [rad/s]
  V3 lidar_v = V3::Zero();                // fused-trajectory mean velocity
  double imu_pred_vx = 0.0;               // forward IMU-predicted velocity [m/s]
  double imu_ax = 0.0;                    // mean gravity-compensated forward specific force
  double htt_forward_information = 0.0;   // per-effective-point forward translation information
  double htt_degeneracy_ratio = 0.0;
  double n_effective_points = 0.0;
};

struct SlipDetectionOutput {
  double r_x = 0.0;             // wheel_v.x() - lidar_v.x()
  double r_y = 0.0;
  double raw_r_x = 0.0;         // wheel_v.x() - unfiltered lidar_v.x()
  double raw_r_y = 0.0;
  double r_wz = 0.0;            // wheel_wz - gyro_wz
  double v_ref = 0.0;           // max(|wheel_v|, |lidar_v|, speed floor)
  V3 lidar_v_filtered = V3::Zero();
  bool relative_speed_active = false;
  int high_votes = 0;
  int high_window_size = 0;
  double score = 0.0;           // max-normalized residual, >=1 at SLIP-entry level
  double suggested_weight = 1.0;
  SlipState state = SlipState::Normal;
};

// Consistency-based wheel-slip monitor. Each frame it compares the wheel
// odometry against the fused trajectory (translation) and the gyro (yaw
// rate), normalizes the residuals by the configured thresholds, and runs a
// hysteresis state machine:
//   NORMAL --(any low exceeded)--> SUSPECT --(K-of-N high vote)--> SLIP
//   SLIP --(all channels below low)--> RECOVERING --(low for recover_time_s)--> NORMAL
//   RECOVERING --(high vote again)--> SLIP
// A high frame is translation dual-high OR |r_wz| above wz_threshold_rps.
class SlipDetector {
 public:
  SlipDetector() = delete;
  explicit SlipDetector(const SlipConfig& config);

  SlipDetectionOutput update(const SlipDetectionInput& input);
  SlipState state() const { return state_; }

 private:
  struct LidarVelocitySample {
    V3 velocity = V3::Zero();
    double dt = 0.0;
  };

  V3 filteredLidarVelocity(const SlipDetectionInput& input);
  bool relativeSpeedActive(const SlipDetectionInput& input) const;
  double speedReference(const SlipDetectionInput& input) const;
  double relativeResidual(const SlipDetectionInput& input, double residual) const;
  double yawResidual(const SlipDetectionInput& input) const;
  // SUSPECT-entry: any low threshold exceeded (translation or yaw).
  bool aboveLow(const SlipDetectionInput& input) const;
  // Fully quiet: translation and yaw all below their low thresholds.
  bool belowLow(const SlipDetectionInput& input) const;
  // Translation SLIP-entry: absolute AND relative high thresholds both exceeded.
  bool aboveHighX(const SlipDetectionInput& input) const;
  // Yaw SLIP-entry: |r_wz| above the high yaw threshold.
  bool aboveHighWz(const SlipDetectionInput& input) const;
  bool aboveHigh(const SlipDetectionInput& input) const;

  SlipConfig config_;
  SlipState state_ = SlipState::Normal;
  std::deque<LidarVelocitySample> lidar_velocity_history_;
  std::deque<bool> high_history_;
  double below_low_time_s_ = 0.0;
};

}  // namespace bievr

#endif  // BIEVR_LIO_SLIP_DETECTOR_H_
