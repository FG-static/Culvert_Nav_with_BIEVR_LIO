#include "bievr_lio/pipeline.h"

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "bievr_lio/inertial_factor.h"
#include "bievr_lio/prior_factor.h"
#include "bievr_lio/timing.h"
#include "bievr_lio/undistort.h"
#include "bievr_lio/utils.h"
#include "bievr_lio/wheel_factor.h"

namespace bievr {

Pipeline::Pipeline(const Config& config) : config_(config), slip_detector_(config.slip) {
  map_ = std::make_shared<BIEVRMap>(config_.map);

  if (!config_.log_path.empty()) {
    LOG(I, "Logging to " << config_.log_path);
    tum_log_ = std::make_shared<std::ofstream>(config_.log_path, std::ios::trunc);
    if (!tum_log_->is_open()) {
      LOG(E, "Error opening output file.");
    }
  }

  if (!config_.registration_metrics_path.empty()) {
    LOG(I, "Registration metrics logging to " << config_.registration_metrics_path);
    registration_metrics_log_ =
        std::make_shared<std::ofstream>(config_.registration_metrics_path, std::ios::trunc);
    if (!registration_metrics_log_->is_open()) {
      LOG(E, "Error opening registration metrics file '"
                 << config_.registration_metrics_path << "'. Diagnostics disabled.");
      registration_metrics_log_.reset();
    } else {
      config_.registration.collect_diagnostics = true;
      writeRegistrationMetricsHeader();
    }
  }

  if (!config_.wheel_slip_path.empty()) {
    if (!config_.wheel.enabled) {
      LOG(W, "Wheel slip logging requested but wheel odometry is disabled; "
             "the slip log will stay empty.");
    }
    LOG(I, "Wheel slip logging to " << config_.wheel_slip_path
                                    << (config_.slip_gate_factors
                                            ? " (gating wheel factors to NORMAL)"
                                            : " (ungated ablation, all wheel factors kept)"));
    slip_log_ = std::make_shared<std::ofstream>(config_.wheel_slip_path, std::ios::trunc);
    if (!slip_log_->is_open()) {
      LOG(E, "Error opening wheel slip log file '" << config_.wheel_slip_path
                                                   << "'. Slip logging disabled.");
      slip_log_.reset();
    } else {
      // The Htt information metrics come from the registration diagnostics.
      config_.registration.collect_diagnostics = true;
      writeSlipLogHeader();
    }
  }

  if (!config_.odometry_twist_path.empty()) {
    LOG(I, "Odometry twist logging to " << config_.odometry_twist_path);
    odometry_twist_log_ =
        std::make_shared<std::ofstream>(config_.odometry_twist_path, std::ios::trunc);
    if (!odometry_twist_log_->is_open()) {
      LOG(E, "Error opening odometry twist log file '" << config_.odometry_twist_path << "'.");
      odometry_twist_log_.reset();
    } else {
      writeOdometryTwistHeader();
    }
  }

  if (config_.print_dashboard && !config_.dashboard_ascii_path.empty()) {
    std::ifstream ascii_file(config_.dashboard_ascii_path);
    if (ascii_file.is_open()) {
      // Indent every line so the art sits a few characters off the left margin.
      const std::string indent(6, ' ');
      std::string line;
      std::ostringstream art;
      while (std::getline(ascii_file, line)) {
        art << indent << line << "\n";
      }
      dashboard_.ascii = art.str();
    } else {
      LOG(W, "Could not open dashboard ASCII art at " << config_.dashboard_ascii_path);
    }
  }

  if (config_.print_dashboard) {
    printDashboardBanner(dashboard_.ascii, "BIEVR-LIO  WAITING FOR DATA");
  }
}

void Pipeline::processFrame(const std::vector<ImuMeasurement>& imu_data,
                            const StampedIntensityPointcloud& points_L,
                            const std::vector<WheelMeasurement>& wheel_data) {
  const auto frame_wall_start = std::chrono::steady_clock::now();
  timing::Timer step_timer("step");

  if (imu_data.empty()) {
    LOG(W, "No IMU data to process.");
    return;
  }

  // Cache the latest gyro reading so the odometry twist can report the current
  // angular velocity (already expressed in the body/IMU frame).
  latest_gyro_ = imu_data.back().gyro;

  // Remove points outside of intended range
  timing::Timer filter_timer("01_filter");
  StampedIntensityPointcloud points_filtered_L;
  filterMinMaxRange(points_L, points_filtered_L, config_.preprocess.min_range,
                    config_.preprocess.max_range);
  // The per-point time and intensity stay in points_filtered_L; we only take
  // zero-copy views onto those rows. Undistortion consumes the time view, intensity
  // rides through to publishing. Both views remain valid for the whole frame and
  // stay aligned by index with the spatial clouds derived below, since every
  // transform downstream preserves point order.
  const StampedIntensityPointcloud& filtered_L = points_filtered_L;
  const TimeView times = filtered_L.times();
  const IntensityView intensities = filtered_L.intensities();
  // Transform only the spatial coordinates into the IMU frame (single 3xN write);
  // time and intensity are never rewritten.
  const Pointcloud points_filtered_I = transformPoints(config_.T_I_L, points_filtered_L);
  filter_timer.Stop();

  if (phase_ == Phase::NeedBias) {
    // Estimate initial biases and orientation based on zero velocity assumption.
    // This also resolves imu_acc_scale_ (normalization detection).
    if (!initializeBias(imu_data, points_filtered_I)) return;
  }

  // Apply the accelerometer scale resolved during bias estimation if necessary.
  std::vector<ImuMeasurement> imu_scaled;
  const std::vector<ImuMeasurement>* imu_ptr = &imu_data;
  if (imu_acc_scale_ != 1.0) {
    imu_scaled = imu_data;
    for (auto& imu : imu_scaled) imu.acc *= imu_acc_scale_;
    imu_ptr = &imu_scaled;
  }
  const std::vector<ImuMeasurement>& imu = *imu_ptr;

  const State& x_i = states_.rbegin()->second;
  const Header header{points_filtered_L.end_stamp, static_cast<uint32_t>(x_i.id + 1),
                      config_.map_frame};

  // Propagate state based on IMU data
  timing::Timer preint_timer("02_preint");
  ImuIntegratorPtr imu_integrator =
      std::make_shared<ImuIntegrator>(config_.imu, acc_bias_, gyro_bias_);
  imu_integrator->integrate(imu);
  V3 G = gravity_dir_ * kGMagnitude;
  State x_j_pred;
  imu_integrator->predict(x_i.quat, x_i.p, x_i.v, G, x_j_pred.quat, x_j_pred.p, x_j_pred.v);
  preint_timer.Stop();

  if (points_filtered_I.empty()) {
    LOG(W, "No pointcloud data to process.");
    addState(imu.back().stamp, x_j_pred.quat, x_j_pred.p, x_j_pred.v);
    publishLatestState(header);
    return;
  }

  // Undistort pointcloud based on IMU integration
  timing::Timer undistortion_timer("03_undistortion");
  Pointcloud points_undistorted_I;
  undistortCloud(imu_integrator, x_i, points_filtered_I, times, points_filtered_L.stamp, G,
                 points_undistorted_I);
  undistortion_timer.Stop();

  std::vector<double> ranges;
  calculateRanges(points_undistorted_I, ranges);
  const Transform T_W_I_init(x_j_pred.quat, x_j_pred.p);
  if (phase_ == Phase::NeedMap) {
    tryInitMap(imu_data.back().stamp, x_j_pred, T_W_I_init, points_undistorted_I, intensities,
               ranges, header);
    return;
  }

  // Select points from the source cloud that will be used for registration
  timing::Timer voxel_timer("04_sampling");
  Pointcloud source_filtered, source_coarse, source_fine;
  sampleSource(points_undistorted_I, T_W_I_init, source_filtered, source_coarse, source_fine);
  voxel_timer.Stop();

  // Perform the actual registration
  timing::Timer align_timer("05_registration");
  const auto registration_wall_start = std::chrono::steady_clock::now();
  LsqRegistration optimizer(*map_, source_filtered, config_.registration);
  const Transform T_W_I = optimizer.computeTransformation(T_W_I_init);
  const double registration_time_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                               registration_wall_start)
          .count();
  const int n_effective_points = optimizer.numEffectivePoints();
  align_timer.Stop();

  // Transform the full cloud using the estimated pose and add it to the map
  timing::Timer map_timer("06_map");
  const Pointcloud points_registered = T_W_I * points_undistorted_I;
  map_->integratePoints(points_registered, &ranges);
  map_timer.Stop();

  // Integrate the wheel-odometry slice over [t_i, t_j] for the window factor.
  // The interval endpoints match the state times by construction (a state is
  // added at every cloud end stamp), so one slice maps to one interval.
  WheelIntegrator::Ptr wheel_integrator;
  if (config_.wheel.enabled) {
    V3 mean_gyro = V3::Zero();
    for (const auto& imu_meas : imu) mean_gyro += imu_meas.gyro;
    mean_gyro /= static_cast<double>(imu.size());
    wheel_integrator = std::make_shared<WheelIntegrator>(config_.wheel, states_.rbegin()->first,
                                                         imu.back().stamp);
    // Add wheel measurements to the integrator and finalize it.
    for (const auto& wheel : wheel_data) wheel_integrator->addMeasurement(wheel);
    wheel_integrator->finalize(mean_gyro);
  }

  // Bookkeeping and optimization of the intertial part of the state
  addState(imu.back().stamp, T_W_I.quaternion(), T_W_I.translation(), x_j_pred.v);

  // Wheel-slip monitor: run before optimization so wheel factors are only added
  // while the state machine reads NORMAL. Compares the wheel interval against the
  // fused trajectory, gyro, and registration forward observability.
  SlipDetectionOutput slip_output;
  slip_output.state = slip_detector_.state();
  double slip_dt_s = 0.0;
  SlipDetectionInput slip_input;
  if (config_.wheel.enabled && wheel_integrator && wheel_integrator->valid() &&
      states_.size() >= 2) {
    const State& state_i = states_.at(wheel_integrator->startTime());
    const State& state_j = states_.at(wheel_integrator->endTime());
    slip_dt_s = wheel_integrator->durationS();
    slip_input = buildSlipInput(*wheel_integrator, state_i, state_j, x_j_pred.v, imu,
                                optimizer.diagnostics());
    slip_output = slip_detector_.update(slip_input);
    if (slip_log_) {
      logSlip(header.stamp, header.seq, slip_dt_s, slip_input, slip_output);
    }
  }
  const bool wheel_factors_allowed =
      !config_.slip_gate_factors || slip_output.state == SlipState::Normal;

  timing::Timer imu_opt("07_imu_opt");
  if (map_->size() > config_.min_map_size_for_imu_opt) {
    addImuIntegrator(imu_integrator);
    if (wheel_factors_allowed && wheel_integrator && wheel_integrator->valid()) {
      addWheelIntegrator(wheel_integrator);
    }
  }
  optimizeInertialWindow();
  imu_opt.Stop();

  if (odometry_twist_log_) {
    logOdometryTwist(header.stamp);
  }

  // Publish results
  timing::Timer pub_time("08_publish");
  publishFrame(header, T_W_I, points_registered, source_filtered, source_coarse, source_fine,
               points_undistorted_I, intensities);
  pub_time.Stop();

  step_timer.Stop();

  if (registration_metrics_log_) {
    const double frame_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                 frame_wall_start)
            .count();
    logRegistrationMetrics(header.stamp, header.seq, T_W_I_init, T_W_I, optimizer.diagnostics(),
                           points_undistorted_I.size(), source_filtered.size(), source_fine.size(),
                           source_coarse.size(), frame_time_ms, registration_time_ms);
  }

  if (!config_.log_path.empty()) {
    logTUM(nsToS(points_L.end_stamp), T_W_I);
  }

  if (config_.print_timing) {
    LOG(I, "Timings:\n" << timing::Timing::Print());
  }

  if (config_.print_dashboard) {
    dashboard_.wheel_enabled = config_.wheel.enabled;
    dashboard_.wheel_residual_rms = wheel_residual_rms_;
    dashboard_.wheel_scale = wheel_scale_[0];
    printDashboard(dashboard_, header.stamp, T_W_I, x_j_pred.v, acc_bias_, gyro_bias_,
                   timing::Timing::GetMeanSeconds("step"), timing::Timing::GetMaxSeconds("step"),
                   n_effective_points);
  }
}

bool Pipeline::initializeBias(const std::vector<ImuMeasurement>& imu_data,
                              const Pointcloud& pointcloud) {
  if (!bias_initializer_) {
    bias_initializer_ =
        std::make_unique<BiasInitializer>(config_.imu.t_init, config_.imu.normalized);
  }
  for (const auto& imu : imu_data) {
    if (bias_initializer_->addMeasurement(imu)) break;
  }
  if (!bias_initializer_->isReady()) return false;

  acc_bias_ = bias_initializer_->accBias();
  gyro_bias_ = bias_initializer_->gyroBias();
  imu_acc_scale_ = bias_initializer_->accScale();
  const Rotation R_est = bias_initializer_->initialOrientation();
  bias_initializer_.reset();

  LOG(I, "Bias initialized: " << acc_bias_.transpose() << " " << gyro_bias_.transpose());
  const Eigen::Vector3d euler = R_est.eulerAngles(2, 1, 0);  // yaw, pitch, roll
  LOG(I, "Initial Pitch: " << (180. / M_PI) * euler[1]);
  LOG(I, "Initial Roll: " << (180. / M_PI) * euler[2]);

  State x_init;
  x_init.quat = R_est;
  x_init.p = V3::Zero();
  x_init.v = V3::Zero();

  addState(imu_data.back().stamp, x_init.quat, x_init.p, x_init.v);

  if (pointcloud.empty()) {
    phase_ = Phase::NeedMap;
  } else {
    const Transform T_W_I_init(x_init.quat, x_init.p);
    std::vector<double> ranges(pointcloud.size(), 0.f);
    for (size_t i = 0; i < pointcloud.size(); ++i) {
      ranges[i] = pointcloud[i].head<3>().norm();
    }
    map_->integratePoints(T_W_I_init * pointcloud, &ranges);
    phase_ = Phase::Running;
  }
  return true;
}

void Pipeline::tryInitMap(uint64_t stamp, const State& x_j_pred, const Transform& T_W_I_init,
                          const Pointcloud& undistorted, const IntensityView& intensities,
                          std::vector<double>& ranges, const Header& header) {
  if (undistorted.size() < config_.min_points_for_map_init) return;
  const Pointcloud registered = T_W_I_init * undistorted;
  map_->integratePoints(registered, &ranges);
  addState(stamp, x_j_pred.quat, x_j_pred.p, x_j_pred.v);
  publishLatestState(header);
  publish(IntensityPointcloud(registered, intensities), header, "points/registered");
  if (map_->size() > config_.map_size_running_threshold) {
    phase_ = Phase::Running;
  }
}

void Pipeline::sampleSource(const Pointcloud& undistorted, const Transform& T_W_I_init,
                            Pointcloud& filtered, Pointcloud& coarse, Pointcloud& fine) const {
  Pointcloud source_down;
  voxelDownsample(undistorted, source_down, config_.preprocess.downsample_resolution);
  if (config_.preprocess.informed_sampling) {
    sampleInformed(*map_, T_W_I_init, source_down, coarse, fine,
                   config_.preprocess.downsample_resolution, config_.informed_sample_count);
    filtered = fine + coarse;
  } else {
    filtered = source_down;
  }
}

bool Pipeline::addState(const uint64_t time, const Quaternion& quat, const V3& p, const V3& v) {
  if (states_.find(time) != states_.end()) {
    LOG(D, "Time " << time << " already exists in the Pipeline. Skipping.");
    return false;
  }

  states_[time] = {seq_counter_++, quat, p, v};

  const double allowed_oldest_time_s = nsToS(time) - config_.imu.window_length_s;
  if (allowed_oldest_time_s < 0) return true;

  const uint64_t allowed_oldest_time = sToNs(allowed_oldest_time_s);
  for (auto it = states_.begin(); it != states_.end();) {
    // Remove states that are outside of the window length, along with their corresponding IMU
    // integrators and wheel integrators
    if (it->first >= allowed_oldest_time) break;
    imu_integrators_.erase(it->first);
    wheel_integrators_.erase(it->first);
    it = states_.erase(it);
  }
  return true;
}

bool Pipeline::addImuIntegrator(ImuIntegratorPtr imu_integrator) {
  uint64_t time = imu_integrator->getFirstTime();
  if (states_.find(time) == states_.end()) {
    LOG(D, "Time " << time << " does not exist in the Pipeline. Skipping.");
    return false;
  }

  imu_integrators_[time] = imu_integrator;
  return true;
}

bool Pipeline::addWheelIntegrator(WheelIntegrator::Ptr wheel_integrator) {
  uint64_t time = wheel_integrator->startTime();
  if (states_.find(time) == states_.end()) {
    LOG(D, "Time " << time << " does not exist in the Pipeline. Skipping wheel integrator.");
    return false;
  }

  wheel_integrators_[time] = wheel_integrator;
  return true;
}

bool Pipeline::optimizeInertialWindow() {
  const size_t n_states = states_.size();
  const size_t n_imu = imu_integrators_.size();
  if (n_states == 0 || n_imu == 0) {
    LOG(D, "No states or IMU integrators to optimize.");
    return false;
  }

  // Skip if there are not enough IMU integrators to prevent instable optimization
  if (n_imu < config_.min_imu_integrators_for_opt) return false;

  ceres::Problem problem;
  ceres::Solver::Options options;
  options.logging_type = ceres::SILENT;
  ceres::LossFunction* loss_function = new ceres::TrivialLoss();

  bool first = true;
  for (auto integrator_it : imu_integrators_) {
    ImuIntegratorPtr& imu_integrator = integrator_it.second;
    const uint64_t t_i = imu_integrator->getFirstTime();
    const uint64_t t_j = imu_integrator->getLastTime();
    State& state_i = states_.at(t_i);
    State& state_j = states_.at(t_j);

    ceres::CostFunction* cost_function = new InertialFactor(imu_integrator);

    // Add preintegrated IMU cost factors between consecutive states, with shared bias and gravity
    // parameters
    problem.AddResidualBlock(cost_function, loss_function, state_i.quat.coeffs().data(),
                             state_i.p.data(), state_i.v.data(), state_j.quat.coeffs().data(),
                             state_j.p.data(), state_j.v.data(), acc_bias_.data(),
                             gyro_bias_.data(), gravity_dir_.data());
    // Keep poses fixed as we only want to optimize over velocities, biases and gravity direction
    problem.SetParameterBlockConstant(state_i.quat.coeffs().data());
    problem.SetParameterBlockConstant(state_j.quat.coeffs().data());
    problem.SetParameterBlockConstant(state_i.p.data());
    problem.SetParameterBlockConstant(state_j.p.data());

    if (first) {
      problem.SetParameterBlockConstant(state_i.v.data());
      first = false;
    }
  }

  // Wheel-odometry velocity factors between the same consecutive states the
  // IMU factors span. Poses are constant blocks (set above), which the wheel
  // factor relies on by capturing the rotations at construction time. The
  // map contains only intervals accepted by the slip gate; an observation
  // becoming SUSPECT/SLIP does not remove previously accepted history.
  wheel_residual_rms_ = 0.0;
  if (config_.wheel.enabled &&
      !wheel_integrators_.empty()) {
    const double weight = 1.0 / config_.wheel.sigma_v;
    size_t n_factors = 0;
    for (const auto& integrator_it : wheel_integrators_) {
      const WheelIntegrator::Ptr& wheel_integrator = integrator_it.second;
      if (!wheel_integrator->valid()) continue;
      State& state_i = states_.at(wheel_integrator->startTime());
      State& state_j = states_.at(wheel_integrator->endTime());
      if (config_.wheel.estimate_scale) {
        ceres::CostFunction* wheel_cost =
            new WheelVelocityScaleFactor(state_i.quat, state_j.quat, *wheel_integrator, weight,
                                         config_.wheel.planar_only);
        problem.AddResidualBlock(wheel_cost, loss_function, state_i.v.data(), state_j.v.data(),
                                 wheel_scale_.data());
      } else {
        ceres::CostFunction* wheel_cost =
            new WheelVelocityFactor(state_i.quat, state_j.quat, *wheel_integrator, weight,
                                    config_.wheel.planar_only);
        problem.AddResidualBlock(wheel_cost, loss_function, state_i.v.data(), state_j.v.data());
      }
      ++n_factors;
    }
    if (config_.wheel.estimate_scale && n_factors > 0) {
      // Keep the shared scale close to 1 so it cannot absorb real velocity
      // information when the wheel constraints get sparse.
      ceres::CostFunction* scale_prior_cost =
          new WheelScalePriorFactor(config_.wheel.scale_prior_sigma);
      problem.AddResidualBlock(scale_prior_cost, loss_function, wheel_scale_.data());
    }

    // Post-solve residual RMS for the dashboard, evaluated with the optimized
    // velocities (and scale). Weight 1 keeps the value in physical m/s.
    size_t n_residual_rows = 0;
    double squared_sum = 0.0;
    for (const auto& integrator_it : wheel_integrators_) {
      const WheelIntegrator::Ptr& wheel_integrator = integrator_it.second;
      if (!wheel_integrator->valid()) continue;
      const State& state_i = states_.at(wheel_integrator->startTime());
      const State& state_j = states_.at(wheel_integrator->endTime());
      std::array<double, 3> residuals{};
      wheel_factor_internal::evaluateResidual(
          wheel_factor_internal::makeResidualData(state_i.quat, state_j.quat, *wheel_integrator,
                                                  1.0, config_.wheel.planar_only),
          state_i.v, state_j.v, wheel_scale_[0], residuals.data(), nullptr, nullptr, nullptr);
      const int n_rows = config_.wheel.planar_only ? 2 : 3;
      for (int k = 0; k < n_rows; ++k) squared_sum += residuals[k] * residuals[k];
      n_residual_rows += n_rows;
    }
    if (n_residual_rows > 0) {
      wheel_residual_rms_ = std::sqrt(squared_sum / static_cast<double>(n_residual_rows));
    }
  }

  // Add constant prior to add some rigidity to the optimization problem and prevent gravity from
  // drifting
  ceres::CostFunction* prior_cost_function =
      new PriorFactor(gravity_dir_, config_.gravity_prior_weight);
  problem.AddResidualBlock(prior_cost_function, loss_function, gravity_dir_.data());
  ceres::Manifold* gravity_manifold = new ceres::SphereManifold<3>();
  problem.SetManifold(gravity_dir_.data(), gravity_manifold);

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return true;
}

void Pipeline::publishFrame(const Header& header, const Transform& T_W_I,
                            const Pointcloud& full_registered, const Pointcloud& source_filtered,
                            const Pointcloud& source_coarse, const Pointcloud& source_fine,
                            const Pointcloud& undistorted, const IntensityView& intensities) {
  if (config_.publish_all_clouds) {
    publishDebugClouds(source_filtered, source_coarse, source_fine, undistorted, intensities, T_W_I,
                       header);
  }
  publish(IntensityPointcloud(full_registered, intensities), header, "points/registered");
  publishLatestState(header);
}

void Pipeline::publishLatestState(const Header& header) {
  if (states_.empty()) {
    LOG(W, "No states to publish.");
    return;
  }

  const State& state = states_.rbegin()->second;

  // Internal state is the IMU pose/velocity. Publish T_W_B and the twist of B
  // expressed in B, so Nav2 can consume odom -> base_footprint without rotating
  // the LiDAR-IMU calibration.
  const Transform T_W_I(state.quat, state.p);
  const Transform T_W_B = T_W_I * config_.T_I_B;
  const Rotation R_B_I = config_.T_I_B.rotation().transpose();

  const V3 velocity_I = state.quat.conjugate() * state.v;
  const V3 velocity_at_B_I =
      velocity_I + latest_gyro_.cross(config_.T_I_B.translation());

  Odometry odom;
  odom.pose = T_W_B;
  odom.linear_velocity = R_B_I * velocity_at_B_I;
  odom.angular_velocity = R_B_I * latest_gyro_;
  publish(odom, header, "odom", config_.body_frame);
  publish(acc_bias_, header, "bias/acc");
  publish(gyro_bias_, header, "bias/gyro");
}

void Pipeline::publishDebugClouds(const Pointcloud& source_filtered,
                                  const Pointcloud& source_coarse, const Pointcloud& source_fine,
                                  const Pointcloud& undistorted_cloud,
                                  const IntensityView& intensities, const Transform& T_W_I,
                                  const Header& header) {
  Pointcloud source_registered = T_W_I * source_filtered;
  Pointcloud fine_registered = T_W_I * source_fine;
  Pointcloud coarse_registered = T_W_I * source_coarse;
  publish(fine_registered, header, "points/fine");
  publish(coarse_registered, header, "points/coarse");
  publish(source_registered, header, "points/effective");
  Header body_header = header;
  body_header.frame = config_.body_frame;
  // The undistorted cloud keeps its original point order, so the snapshotted
  // intensity row still lines up with it.
  publish(IntensityPointcloud(undistorted_cloud, intensities), body_header, "points/undistorted");
}

void Pipeline::logTUM(double timestamp, const Transform& pose) {
  const Quaternion q(pose.linear());
  const V3& t = pose.translation();
  if (!tum_log_->is_open()) {
    LOG(E, "Error: Output file is not open for logging.");
    return;
  }

  (*tum_log_) << std::fixed;
  (*tum_log_) << timestamp << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
              << q.y() << " " << q.z() << " " << q.w() << "\n";
}

void Pipeline::writeRegistrationMetricsHeader() {
  if (!registration_metrics_log_ || !registration_metrics_log_->is_open()) return;

  constexpr std::array<const char*, 6> kAxes = {"rx", "ry", "rz", "tx", "ty", "tz"};
  std::ofstream& out = *registration_metrics_log_;
  out << "schema_version,stamp_ns,stamp_s,frame_id,diagnostics_valid,lm_converged,lm_iterations,"
         "lm_trial_steps,lm_rejected_steps,hard_projection_enabled,"
         "hard_projection_retention_alpha,hard_projection_ratio_threshold,"
         "hard_projection_trial_steps,"
         "hard_projection_accepted_steps,hard_projection_accepted_removed_abs_sum_m,"
         "hard_projection_accepted_weak_before_abs_max_m,"
         "hard_projection_accepted_removed_abs_max_m,"
         "hard_projection_accepted_weak_after_abs_max_m,"
         "frame_time_ms,registration_time_ms,undistorted_points,selected_points,fine_points,"
         "coarse_points,effective_points,huber_inlier_points,effective_ratio,huber_inlier_ratio,"
         "robust_cost,residual_rmse,robust_weight_sum,gradient_norm";
  for (int i = 0; i < 6; ++i) out << ",h6_eigenvalue_" << i;
  out << ",h6_condition_number,h6_degeneracy_ratio";
  for (int eigenvector = 0; eigenvector < 6; ++eigenvector) {
    for (int axis = 0; axis < 6; ++axis) {
      out << ",h6_eigenvector_" << eigenvector << "_" << kAxes[axis];
    }
  }
  for (int i = 0; i < 3; ++i) out << ",htt_eigenvalue_" << i;
  out << ",htt_condition_number,htt_degeneracy_ratio,htt_weak_information_per_effective_point,"
         "htt_weak_body_x,htt_weak_body_y,htt_weak_body_z,"
         "htt_weak_world_x,htt_weak_world_y,htt_weak_world_z,"
         "imu_lidar_rotation_correction_body_rx,imu_lidar_rotation_correction_body_ry,"
         "imu_lidar_rotation_correction_body_rz,imu_lidar_rotation_correction_norm_rad,"
         "imu_lidar_translation_correction_body_x,imu_lidar_translation_correction_body_y,"
         "imu_lidar_translation_correction_body_z,imu_lidar_translation_correction_body_norm_m,"
         "imu_lidar_translation_correction_world_x,imu_lidar_translation_correction_world_y,"
         "imu_lidar_translation_correction_world_z,imu_lidar_translation_correction_world_norm_m\n";
  out.flush();
}

void Pipeline::logRegistrationMetrics(uint64_t stamp_ns, uint32_t frame_id,
                                      const Transform& T_W_I_init, const Transform& T_W_I,
                                      const RegistrationDiagnostics& diagnostics,
                                      size_t undistorted_points, size_t selected_points,
                                      size_t fine_points, size_t coarse_points,
                                      double frame_time_ms, double registration_time_ms) {
  if (!registration_metrics_log_ || !registration_metrics_log_->is_open()) return;

  const V3 weak_translation_body = diagnostics.translation_eigenvectors.col(0);
  const V3 weak_translation_world = T_W_I.linear() * weak_translation_body;
  const Transform T_init_final = T_W_I_init.inverse() * T_W_I;
  const V3 rotation_correction_body = logMap(T_init_final.linear());
  const V3 translation_correction_body = T_init_final.translation();
  const V3 translation_correction_world = T_W_I.translation() - T_W_I_init.translation();

  std::ofstream& out = *registration_metrics_log_;
  out << std::setprecision(17);
  out << 3 << ',' << stamp_ns << ',' << nsToS(stamp_ns) << ',' << frame_id << ','
      << static_cast<int>(diagnostics.valid) << ',' << static_cast<int>(diagnostics.converged) << ','
      << diagnostics.iterations << ',' << diagnostics.lm_trial_steps << ','
      << diagnostics.lm_rejected_steps << ','
      << static_cast<int>(diagnostics.hard_projection_enabled) << ','
      << diagnostics.weak_translation_retention_alpha << ','
      << diagnostics.hard_projection_ratio_threshold << ','
      << diagnostics.hard_projection_trial_steps << ','
      << diagnostics.hard_projection_accepted_steps << ','
      << diagnostics.hard_projection_accepted_removed_abs_sum_m << ','
      << diagnostics.hard_projection_accepted_weak_before_abs_max_m << ','
      << diagnostics.hard_projection_accepted_removed_abs_max_m << ','
      << diagnostics.hard_projection_accepted_weak_after_abs_max_m << ',' << frame_time_ms << ','
      << registration_time_ms << ','
      << undistorted_points << ',' << selected_points << ',' << fine_points << ',' << coarse_points
      << ',' << diagnostics.effective_points << ',' << diagnostics.huber_inlier_points << ','
      << diagnostics.effective_ratio << ',' << diagnostics.huber_inlier_ratio << ','
      << diagnostics.robust_cost << ',' << diagnostics.residual_rmse << ','
      << diagnostics.weight_sum << ',' << diagnostics.gradient_norm;

  for (int i = 0; i < 6; ++i) out << ',' << diagnostics.hessian_eigenvalues(i);
  out << ',' << diagnostics.hessian_condition_number << ','
      << diagnostics.hessian_degeneracy_ratio;
  for (int eigenvector = 0; eigenvector < 6; ++eigenvector) {
    for (int axis = 0; axis < 6; ++axis) {
      out << ',' << diagnostics.hessian_eigenvectors(axis, eigenvector);
    }
  }
  for (int i = 0; i < 3; ++i) out << ',' << diagnostics.translation_eigenvalues(i);
  out << ',' << diagnostics.translation_condition_number << ','
      << diagnostics.translation_degeneracy_ratio << ','
      << diagnostics.weakest_translation_information_per_point << ','
      << weak_translation_body.x() << ',' << weak_translation_body.y() << ','
      << weak_translation_body.z() << ',' << weak_translation_world.x() << ','
      << weak_translation_world.y() << ',' << weak_translation_world.z() << ','
      << rotation_correction_body.x() << ',' << rotation_correction_body.y() << ','
      << rotation_correction_body.z() << ',' << rotation_correction_body.norm() << ','
      << translation_correction_body.x() << ',' << translation_correction_body.y() << ','
      << translation_correction_body.z() << ',' << translation_correction_body.norm() << ','
      << translation_correction_world.x() << ',' << translation_correction_world.y() << ','
      << translation_correction_world.z() << ',' << translation_correction_world.norm() << '\n';
  // Preserve the last diagnostic frame even if an experimental run terminates early.
  out.flush();
}

SlipDetectionInput Pipeline::buildSlipInput(const WheelIntegrator& wheel_integrator,
                                            const State& state_i, const State& state_j,
                                            const V3& imu_pred_v,
                                            const std::vector<ImuMeasurement>& imu,
                                            const RegistrationDiagnostics& diag) const {

  const double dt_s = wheel_integrator.durationS();
  const Rotation R_i = state_i.quat.toRotationMatrix();

  SlipDetectionInput slip_input;
  slip_input.dt = dt_s;
  slip_input.wheel_v = wheel_integrator.displacement() / dt_s;
  slip_input.wheel_wz = wheel_integrator.meanYawRate();
  slip_input.lidar_v = R_i.transpose() * (state_j.p - state_i.p) / dt_s;
  slip_input.imu_pred_vx = (state_i.quat.inverse() * imu_pred_v).x();

  V3 acc_mean = V3::Zero();
  V3 gyro_mean = V3::Zero();
  for (const auto& imu_meas : imu) {
    acc_mean += imu_meas.acc;
    gyro_mean += imu_meas.gyro;
  }
  acc_mean /= static_cast<double>(imu.size());
  gyro_mean /= static_cast<double>(imu.size());
  const V3 gravity_body = R_i.transpose() * gravity_dir_ * kGMagnitude;
  slip_input.imu_ax = (acc_mean - acc_bias_ - gravity_body).x();
  slip_input.gyro_wz = (gyro_mean - gyro_bias_).z();

  if (diag.valid) {
    double forward_info = 0.0;
    for (int i = 0; i < 3; ++i) {
      const double d = diag.translation_eigenvectors.col(i).x();
      forward_info += diag.translation_eigenvalues(i) * d * d;
    }
    slip_input.htt_forward_information =
        forward_info / static_cast<double>(std::max<size_t>(diag.effective_points, 1));
    slip_input.htt_degeneracy_ratio = diag.translation_degeneracy_ratio;
    slip_input.n_effective_points = static_cast<double>(diag.effective_points);
  }
  return slip_input;
}

void Pipeline::writeSlipLogHeader() {
  if (!slip_log_ || !slip_log_->is_open()) return;

  std::ofstream& out = *slip_log_;
  out << "schema_version,stamp_ns,stamp_s,frame_id,dt_s,"
         "wheel_vx,wheel_vy,wheel_wz,lidar_vx,lidar_vy,lidar_v_filtered_x,lidar_v_filtered_y,"
         "imu_pred_vx,imu_ax,gyro_wz,r_x,r_y,raw_r_x,raw_r_y,r_wz,v_ref,"
         "relative_speed_active,high_votes,high_window_size,"
         "htt_forward_information,htt_degeneracy_ratio,n_effective_points,"
         "slip_score,slip_state,suggested_wheel_weight\n";
  out.flush();
}

void Pipeline::logSlip(uint64_t stamp_ns, uint32_t frame_id, double dt_s,
                       const SlipDetectionInput& input, const SlipDetectionOutput& output) {
  if (!slip_log_ || !slip_log_->is_open()) return;

  std::ofstream& out = *slip_log_;
  out << std::setprecision(9);
  out << 2 << ',' << stamp_ns << ',' << nsToS(stamp_ns) << ',' << frame_id << ',' << dt_s << ','
      << input.wheel_v.x() << ',' << input.wheel_v.y() << ',' << input.wheel_wz << ','
      << input.lidar_v.x() << ',' << input.lidar_v.y() << ',' << output.lidar_v_filtered.x() << ','
      << output.lidar_v_filtered.y() << ',' << input.imu_pred_vx << ',' << input.imu_ax << ','
      << input.gyro_wz << ',' << output.r_x << ',' << output.r_y << ',' << output.raw_r_x << ','
      << output.raw_r_y << ',' << output.r_wz << ',' << output.v_ref << ','
      << static_cast<int>(output.relative_speed_active) << ',' << output.high_votes << ','
      << output.high_window_size << ',' << input.htt_forward_information << ','
      << input.htt_degeneracy_ratio << ',' << input.n_effective_points << ',' << output.score << ','
      << static_cast<int>(output.state) << ',' << output.suggested_weight << '\n';
  // Preserve the last row even if a run terminates early.
  out.flush();
}

void Pipeline::writeOdometryTwistHeader() {
  if (!odometry_twist_log_ || !odometry_twist_log_->is_open()) return;
  std::ofstream& out = *odometry_twist_log_;
  out << "stamp_ns,stamp_s,vx,vy,vz,wx,wy,wz\n";
  out.flush();
}

void Pipeline::logOdometryTwist(uint64_t stamp_ns) {
  if (!odometry_twist_log_ || !odometry_twist_log_->is_open() || states_.empty()) return;
  const State& latest_state = states_.rbegin()->second;
  const V3 v_body = latest_state.quat.conjugate() * latest_state.v;
  std::ofstream& out = *odometry_twist_log_;
  out << std::setprecision(9);
  out << stamp_ns << ',' << nsToS(stamp_ns) << ',' << v_body.x() << ',' << v_body.y() << ','
      << v_body.z() << ',' << latest_gyro_.x() << ',' << latest_gyro_.y() << ',' << latest_gyro_.z()
      << '\n';
  out.flush();
}

}  // namespace bievr
