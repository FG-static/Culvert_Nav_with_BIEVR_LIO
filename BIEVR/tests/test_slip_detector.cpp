#include "bievr_lio/slip_detector.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using bievr::SlipConfig;
using bievr::SlipDetectionInput;
using bievr::SlipDetector;
using bievr::SlipState;

SlipDetectionInput makeInput(double wheel_x, double lidar_x, double dt = 0.1) {
  SlipDetectionInput input;
  input.dt = dt;
  input.wheel_v = bievr::V3(wheel_x, 0.0, 0.0);
  input.lidar_v = bievr::V3(lidar_x, 0.0, 0.0);
  return input;
}

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void expectNear(double actual, double expected, double tolerance, const std::string& message) {
  expect(std::abs(actual - expected) <= tolerance,
         message + " (actual=" + std::to_string(actual) + ", expected=" +
             std::to_string(expected) + ")");
}

void testLowSpeedJitterDoesNotBecomeSuspect() {
  SlipConfig config;
  config.lidar_velocity_window_frames = 1;
  SlipDetector detector(config);

  for (int i = 0; i < 5; ++i) {
    const auto output = detector.update(makeInput(0.05, 0.0));
    expect(output.state == SlipState::Normal, "low-speed jitter must remain NORMAL");
    expect(!output.relative_speed_active, "relative test must be disabled at low speed");
  }
}

void testSingleLidarSpikeIsSmoothed() {
  SlipConfig config;
  config.lidar_velocity_window_frames = 3;
  SlipDetector detector(config);

  detector.update(makeInput(1.0, 1.0));
  detector.update(makeInput(1.0, 1.0));
  const auto output = detector.update(makeInput(1.0, 0.0));

  expectNear(output.lidar_v_filtered.x(), 2.0 / 3.0, 1e-12,
             "LiDAR velocity must use the time-weighted short-window average");
  expectNear(output.raw_r_x, 1.0, 1e-12, "raw residual must preserve the unsmoothed spike");
  expectNear(output.r_x, 1.0 / 3.0, 1e-12,
             "state residual must use the smoothed LiDAR velocity");
  expect(output.state != SlipState::Slip, "one LiDAR spike must not enter SLIP");
}

void testThreeOfFiveHighFramesEnterSlip() {
  SlipConfig config;
  config.lidar_velocity_window_frames = 1;
  config.high_window_frames = 5;
  config.high_votes_required = 3;
  SlipDetector detector(config);

  detector.update(makeInput(1.0, 0.0));  // high
  detector.update(makeInput(1.0, 0.0));  // high
  detector.update(makeInput(1.0, 0.8));  // not high
  detector.update(makeInput(1.0, 0.0));  // third high
  const auto output = detector.update(makeInput(1.0, 0.8));  // still 3/5

  expect(output.high_votes == 3, "K-of-N vote must count three high frames");
  expect(output.high_window_size == 5, "vote log must expose the current window size");
  expect(output.state == SlipState::Slip, "three of five high frames must enter SLIP");
}

void testTwoOfFiveHighFramesDoNotEnterSlip() {
  SlipConfig config;
  config.lidar_velocity_window_frames = 1;
  config.high_window_frames = 5;
  config.high_votes_required = 3;
  SlipDetector detector(config);

  detector.update(makeInput(1.0, 0.0));  // high
  detector.update(makeInput(1.0, 0.8));  // not high
  detector.update(makeInput(1.0, 0.8));  // not high
  detector.update(makeInput(1.0, 0.0));  // second high
  const auto output = detector.update(makeInput(1.0, 0.8));  // still 2/5

  expect(output.high_votes == 2, "vote count must remain below K with only two highs");
  expect(output.state != SlipState::Slip, "two of five high frames must not enter SLIP");
}

void testSlipRecoveryHysteresis() {
  SlipConfig config;
  config.lidar_velocity_window_frames = 1;
  config.high_window_frames = 1;
  config.high_votes_required = 1;
  config.recover_time_s = 0.25;
  SlipDetector detector(config);

  const auto slip = detector.update(makeInput(1.0, 0.0));
  expect(slip.state == SlipState::Slip, "a single high frame must enter SLIP for K=N=1");

  const auto recovering = detector.update(makeInput(1.0, 1.0, 0.1));
  expect(recovering.state == SlipState::Recovering, "quiet data must start RECOVERING");
  detector.update(makeInput(1.0, 1.0, 0.1));
  const auto normal = detector.update(makeInput(1.0, 1.0, 0.1));
  expect(normal.state == SlipState::Normal, "quiet data for recover_time_s must return NORMAL");
}

void testLowSpeedAbsoluteCheckStillWorks() {
  SlipConfig config;
  config.lidar_velocity_window_frames = 1;
  config.high_window_frames = 1;
  config.high_votes_required = 1;
  config.relative_min_speed_mps = 0.15;
  SlipDetector detector(config);

  const auto output = detector.update(makeInput(0.15, -0.15));
  expect(!output.relative_speed_active,
         "relative test must be disabled when both speeds are at the low-speed boundary");
  expect(output.state == SlipState::Slip,
         "absolute residual must still detect a meaningful low-speed disagreement");
}

}  // namespace

int main() {
  testLowSpeedJitterDoesNotBecomeSuspect();
  testSingleLidarSpikeIsSmoothed();
  testThreeOfFiveHighFramesEnterSlip();
  testTwoOfFiveHighFramesDoNotEnterSlip();
  testSlipRecoveryHysteresis();
  testLowSpeedAbsoluteCheckStillWorks();
  std::cout << "SlipDetector tests passed\n";
  return EXIT_SUCCESS;
}
