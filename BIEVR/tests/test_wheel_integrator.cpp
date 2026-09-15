#include "bievr_lio/wheel_integrator.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void expectNear(double actual, double expected, double tolerance, const char* message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " (actual=" << actual << ", expected=" << expected << ")\n";
    std::exit(EXIT_FAILURE);
  }
}

void testAngularVelocityUsesWheelExtrinsic() {
  bievr::WheelConfig config;
  // Wheel-frame +Y maps to IMU-frame +Z.
  config.T_I_W.linear() << 1.0, 0.0, 0.0,
                           0.0, 0.0, -1.0,
                           0.0, 1.0, 0.0;

  constexpr uint64_t kStart = 1000000000ULL;
  constexpr uint64_t kEnd = 2000000000ULL;
  bievr::WheelIntegrator integrator(config, kStart, kEnd);
  integrator.addMeasurement(
      bievr::WheelMeasurement{1500000000ULL, bievr::V3::Zero(), bievr::V3(0.0, 1.0, 0.0)});
  integrator.finalize(bievr::V3::Zero());

  expectNear(integrator.meanYawRate(), 1.0, 1e-12,
             "wheel angular velocity must be rotated into the IMU frame");
}

}  // namespace

int main() {
  testAngularVelocityUsesWheelExtrinsic();
  std::cout << "WheelIntegrator tests passed\n";
  return EXIT_SUCCESS;
}
