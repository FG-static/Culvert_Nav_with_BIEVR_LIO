#include "bievr_lio/slip_detector.h"

#include <algorithm>
#include <cmath>

namespace bievr {

SlipDetector::SlipDetector(const SlipConfig& config) : config_(config) {}

V3 SlipDetector::filteredLidarVelocity(const SlipDetectionInput& input) {
  const int max_history = std::max(1, config_.lidar_velocity_window_frames);
  if (input.dt > 0.0 && std::isfinite(input.dt)) {
    lidar_velocity_history_.push_back({input.lidar_v, input.dt});
    while (static_cast<int>(lidar_velocity_history_.size()) > max_history) {
      lidar_velocity_history_.pop_front();
    }
  }

  V3 weighted_sum = V3::Zero();
  double total_dt = 0.0;
  for (const auto& sample : lidar_velocity_history_) {
    weighted_sum += sample.velocity * sample.dt;
    total_dt += sample.dt;
  }
  return total_dt > 0.0 ? weighted_sum / total_dt : input.lidar_v;
}

bool SlipDetector::relativeSpeedActive(const SlipDetectionInput& input) const {
  return std::max(input.wheel_v.norm(), input.lidar_v.norm()) >
         config_.relative_min_speed_mps;
}

double SlipDetector::speedReference(const SlipDetectionInput& input) const {
  return std::max({input.wheel_v.norm(), input.lidar_v.norm(), config_.speed_floor_mps});
}

double SlipDetector::relativeResidual(const SlipDetectionInput& input, double residual) const {
  if (!relativeSpeedActive(input)) return 0.0;
  return std::abs(residual) / speedReference(input);
}

bool SlipDetector::aboveLowX(const SlipDetectionInput& input) const {
  const double r_x = input.wheel_v.x() - input.lidar_v.x();
  return std::abs(r_x) > config_.suspect_abs_mps ||
         relativeResidual(input, r_x) > config_.suspect_rel;
}

bool SlipDetector::belowLowX(const SlipDetectionInput& input) const {
  const double r_x = input.wheel_v.x() - input.lidar_v.x();
  return std::abs(r_x) <= config_.suspect_abs_mps &&
         relativeResidual(input, r_x) <= config_.suspect_rel;
}

bool SlipDetector::aboveHighX(const SlipDetectionInput& input) const {
  const double r_x = input.wheel_v.x() - input.lidar_v.x();
  // At low speed the relative test is intentionally disabled; the absolute
  // threshold is then sufficient to detect a meaningful wheel motion.
  return std::abs(r_x) > config_.abs_threshold_mps &&
         (!relativeSpeedActive(input) ||
          relativeResidual(input, r_x) > config_.rel_threshold);
}

SlipDetectionOutput SlipDetector::update(const SlipDetectionInput& input) {
  const V3 lidar_v_filtered = filteredLidarVelocity(input);
  SlipDetectionInput effective_input = input;
  effective_input.lidar_v = lidar_v_filtered;

  SlipDetectionOutput out;
  out.raw_r_x = input.wheel_v.x() - input.lidar_v.x();
  out.raw_r_y = input.wheel_v.y() - input.lidar_v.y();
  out.lidar_v_filtered = lidar_v_filtered;
  out.r_x = input.wheel_v.x() - lidar_v_filtered.x();
  out.r_y = input.wheel_v.y() - lidar_v_filtered.y();
  out.r_wz = input.wheel_wz - input.gyro_wz;
  out.v_ref = speedReference(effective_input);
  out.relative_speed_active = relativeSpeedActive(effective_input);

  // Max-normalized score: each channel divided by its high threshold, so a
  // score >= 1 means some channel is at SLIP-entry level. Raw residuals are
  // logged alongside, so any other scoring (e.g. calibrated chi-square) can
  // be recomputed offline.
  const double relative_score =
      out.relative_speed_active ? relativeResidual(effective_input, out.r_x) /
                                     config_.rel_threshold
                                : 0.0;
  out.score = std::max({std::abs(out.r_x) / config_.abs_threshold_mps, relative_score,
                        std::abs(out.r_wz) / config_.wz_threshold_rps});
  // Log-only hint for a future gate: 1 below SLIP-entry level, ramping to 0
  // by twice that level. Pure function of the score, no hysteresis.
  out.suggested_weight = std::clamp(2.0 - out.score, 0.0, 1.0);

  const bool high = aboveHighX(effective_input);
  const int max_high_history = std::max(1, config_.high_window_frames);
  high_history_.push_back(high);
  while (static_cast<int>(high_history_.size()) > max_high_history) {
    high_history_.pop_front();
  }
  out.high_window_size = static_cast<int>(high_history_.size());
  out.high_votes = static_cast<int>(std::count(high_history_.begin(), high_history_.end(), true));
  const bool high_vote = out.high_votes >= std::max(1, config_.high_votes_required);

  switch (state_) {
    case SlipState::Normal:
    case SlipState::Suspect:
      if (high_vote) {
        state_ = SlipState::Slip;
      } else if (aboveLowX(effective_input)) {
        state_ = SlipState::Suspect;
      } else {
        state_ = SlipState::Normal;
      }
      break;

    case SlipState::Slip:
    case SlipState::Recovering:
      if (high_vote) {
        // Re-enter (or stay in) SLIP; the recovery clock restarts.
        below_low_time_s_ = 0.0;
        state_ = SlipState::Slip;
      } else if (belowLowX(effective_input)) {
        below_low_time_s_ += input.dt;
        state_ = SlipState::Recovering;
        if (below_low_time_s_ >= config_.recover_time_s) {
          below_low_time_s_ = 0.0;
          state_ = SlipState::Normal;
        }
      } else {
        // Between low and high: hold the current state, restart the clock.
        below_low_time_s_ = 0.0;
      }
      break;
  }

  out.state = state_;
  return out;
}

}  // namespace bievr
