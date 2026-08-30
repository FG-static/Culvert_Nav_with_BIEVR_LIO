#ifndef BIEVR_LIO_SYNCHRONIZER_H_
#define BIEVR_LIO_SYNCHRONIZER_H_

#include <deque>

#include "bievr_lio/pipeline.h"

namespace bievr {

class Synchronizer {
 public:
  explicit Synchronizer(std::shared_ptr<Pipeline> pipeline, bool wheel_input_enabled = false,
                        double wheel_max_sync_delay_s = 0.05)
      : wheel_input_enabled_(wheel_input_enabled),
        wheel_max_sync_delay_ns_(sToNs(wheel_max_sync_delay_s)),
        pipeline_(pipeline) {}
  virtual ~Synchronizer() = default;
  bool addImu(const ImuMeasurement& imu);
  bool addPointcloud(const StampedIntensityPointcloud& cloud);
  bool addWheelOdometry(const WheelMeasurement& wheel);

 private:
  void synchronizeData();
  std::deque<ImuMeasurement> imu_queue_;
  std::deque<StampedIntensityPointcloud> point_queue_;
  std::deque<WheelMeasurement> wheel_queue_;
  // Last wheel sample handed to a frame, prepended to the next slice so the
  // WheelIntegrator can bridge the interval start by trapezoid instead of
  // holding the first in-interval velocity.
  WheelMeasurement last_wheel_ = WheelMeasurement{0, V3::Zero(), 0.0};
  bool has_last_wheel_ = false;
  bool wheel_input_enabled_ = false;
  uint64_t wheel_max_sync_delay_ns_ = 0;
  std::shared_ptr<Pipeline> pipeline_;
};
}  // namespace bievr

#endif  // BIEVR_LIO_SYNCHRONIZER_H_
