#include "t_parking_sim/direction_locked_rpp.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "nav2_core/controller_exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2/utils.hpp"

namespace t_parking_sim
{

namespace
{

double euclideanDistance(
  const geometry_msgs::msg::PoseStamped & first,
  const geometry_msgs::msg::PoseStamped & second)
{
  return std::hypot(
    first.pose.position.x - second.pose.position.x,
    first.pose.position.y - second.pose.position.y);
}

double calculateCurvature(const geometry_msgs::msg::Point & point)
{
  const double squared_distance = point.x * point.x + point.y * point.y;
  return squared_distance > 0.001 ? 2.0 * point.y / squared_distance : 0.0;
}

constexpr double kPi = 3.14159265358979323846;

double radiansToDegrees(double radians)
{
  return radians * 180.0 / kPi;
}

double degreesToRadians(double degrees)
{
  return degrees * kPi / 180.0;
}

double normalizeAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

}  // namespace

void DirectionLockedRPP::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  RegulatedPurePursuitController::configure(parent, name, std::move(tf), costmap_ros);
  auto node = parent.lock();
  if (!node) {
    throw nav2_core::ControllerException("Unable to lock controller node");
  }

  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".locked_direction", rclcpp::ParameterValue(1));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".segment_state_topic",
    rclcpp::ParameterValue(std::string("/t_parking/active_segment")));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".reverse_wheel_base", rclcpp::ParameterValue(0.77));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".reverse_hard_steering_limit_deg",
    rclcpp::ParameterValue(27.0));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".reverse_profile_window",
    rclcpp::ParameterValue(0.25));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".reverse_lateral_gain",
    rclcpp::ParameterValue(0.35));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".reverse_heading_gain",
    rclcpp::ParameterValue(0.8));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".reverse_lateral_correction_limit_deg",
    rclcpp::ParameterValue(3.0));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".reverse_soft_limit_margin_deg",
    rclcpp::ParameterValue(1.5));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".reverse_soft_limit_override_deg",
    rclcpp::ParameterValue(0.0));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".terminal_capture_distance",
    rclcpp::ParameterValue(0.0));
  node->get_parameter(plugin_name_ + ".locked_direction", locked_direction_);
  node->get_parameter(plugin_name_ + ".segment_state_topic", segment_state_topic_);
  node->get_parameter(plugin_name_ + ".reverse_wheel_base", reverse_wheel_base_);
  node->get_parameter(
    plugin_name_ + ".reverse_hard_steering_limit_deg",
    reverse_hard_steering_limit_deg_);
  node->get_parameter(plugin_name_ + ".reverse_profile_window", reverse_profile_window_);
  node->get_parameter(plugin_name_ + ".reverse_lateral_gain", reverse_lateral_gain_);
  node->get_parameter(plugin_name_ + ".reverse_heading_gain", reverse_heading_gain_);
  node->get_parameter(
    plugin_name_ + ".reverse_lateral_correction_limit_deg",
    reverse_lateral_correction_limit_deg_);
  node->get_parameter(
    plugin_name_ + ".reverse_soft_limit_margin_deg",
    reverse_soft_limit_margin_deg_);
  node->get_parameter(
    plugin_name_ + ".reverse_soft_limit_override_deg",
    reverse_soft_limit_override_deg_);
  node->get_parameter(
    plugin_name_ + ".terminal_capture_distance",
    terminal_capture_distance_);
  if (locked_direction_ != 1 && locked_direction_ != -1) {
    throw nav2_core::ControllerException(
            plugin_name_ + ".locked_direction must be +1 or -1");
  }
  if (reverse_wheel_base_ <= 0.0 || reverse_profile_window_ <= 0.0 ||
    reverse_hard_steering_limit_deg_ <= 0.0 ||
    reverse_hard_steering_limit_deg_ > 27.0 || reverse_lateral_gain_ < 0.0 ||
    reverse_heading_gain_ < 0.0 ||
    reverse_lateral_correction_limit_deg_ < 0.0 ||
    reverse_soft_limit_margin_deg_ < 0.0 || terminal_capture_distance_ < 0.0)
  {
    throw nav2_core::ControllerException(
            plugin_name_ + " has invalid reverse steering geometry parameters");
  }

  rclcpp::QoS qos(1);
  qos.reliable().transient_local();
  segment_state_sub_ = node->create_subscription<std_msgs::msg::Int32MultiArray>(
    segment_state_topic_, qos,
    std::bind(&DirectionLockedRPP::segmentStateCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    logger_,
    "[RPP-LOCK] controller=%s locked_direction=%s; allow_reversing remains configured "
    "but carrot x no longer selects gear",
    plugin_name_.c_str(), locked_direction_ > 0 ? "FORWARD" : "REVERSE");
}

void DirectionLockedRPP::cleanup()
{
  segment_state_sub_.reset();
  RegulatedPurePursuitController::cleanup();
}

void DirectionLockedRPP::segmentStateCallback(
  const std_msgs::msg::Int32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 4) {
    RCLCPP_WARN(logger_, "[RPP-LOCK] ignoring malformed active segment state");
    return;
  }
  segment_number_.store(msg->data[0]);
  metadata_direction_.store(msg->data[1]);
  full_start_index_.store(msg->data[2]);
  full_end_index_.store(msg->data[3]);
}

void DirectionLockedRPP::setPlan(const nav_msgs::msg::Path & path)
{
  if (path.poses.size() < 2) {
    throw nav2_core::InvalidPath("DirectionLockedRPP requires at least two poses");
  }
  RegulatedPurePursuitController::setPlan(path);
  std::lock_guard<std::mutex> lock(plan_mutex_);
  if (!segment_plan_.poses.empty()) {
    logTrackingSummary("new plan");
  }
  segment_plan_ = path;
  resetTrackingState();
  median_spacing_ = medianPathSpacing();
  cumulative_arc_length_.assign(segment_plan_.poses.size(), 0.0);
  for (std::size_t index = 1; index < segment_plan_.poses.size(); ++index) {
    cumulative_arc_length_[index] = cumulative_arc_length_[index - 1] +
      euclideanDistance(segment_plan_.poses[index - 1], segment_plan_.poses[index]);
  }
  progress_search_distance_ = std::max(
    params_->max_lookahead_dist, 3.0 * median_spacing_);

  const int metadata_direction = metadata_direction_.load();
  if (metadata_direction != 0 && metadata_direction != locked_direction_) {
    RCLCPP_ERROR(
      logger_,
      "[RPP-LOCK] controller_id direction mismatch: controller=%s metadata=%s",
      locked_direction_ > 0 ? "FORWARD" : "REVERSE",
      metadata_direction > 0 ? "FORWARD" : "REVERSE");
  }
  RCLCPP_INFO(
    logger_,
    "[RPP-LOCK] setPlan controller=%s segment=%d direction=%s poses=%zu "
    "full_range=%d..%d median_spacing=%.4f search_distance=%.4f "
    "terminal_capture_distance=%.4f",
    plugin_name_.c_str(), segment_number_.load(),
    locked_direction_ > 0 ? "FORWARD" : "REVERSE", segment_plan_.poses.size(),
    full_start_index_.load(), full_end_index_.load(), median_spacing_,
    progress_search_distance_, terminal_capture_distance_);
  diagnoseSelfProximity();
  if (locked_direction_ < 0) {
    buildReverseSteeringProfile();
    logReverseSteeringProfile();
  }
}

void DirectionLockedRPP::resetTrackingState()
{
  progress_initialized_ = false;
  last_progress_index_ = 0;
  rollback_candidate_count_ = 0;
  accepted_rollback_count_ = 0;
  direction_conflict_samples_ = 0;
  direction_conflict_events_ = 0;
  direction_conflict_active_ = false;
  lateral_error_samples_ = 0;
  lateral_error_sum_ = 0.0;
  max_lateral_error_ = 0.0;
  reverse_projection_initialized_ = false;
  last_projection_arc_length_ = 0.0;
  reverse_steering_samples_ = 0;
  reverse_saturation_samples_ = 0;
  reverse_saturation_events_ = 0;
  reverse_saturation_active_ = false;
  terminal_goal_reached_logged_ = false;
  last_track_log_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  last_terminal_log_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
}

void DirectionLockedRPP::logTrackingSummary(const char * context)
{
  if (segment_plan_.poses.empty()) {
    return;
  }
  const double average_error = lateral_error_samples_ == 0 ? 0.0 :
    lateral_error_sum_ / static_cast<double>(lateral_error_samples_);
  RCLCPP_INFO(
    logger_,
    "[RPP-LOCK][SUMMARY] context=%s controller=%s segment=%d direction=%s "
    "local_index=%zu full_index=%d rollback_candidates=%" PRIu64
    " accepted_rollbacks=%" PRIu64 " direction_conflict_samples=%" PRIu64
    " direction_conflict_events=%" PRIu64 " "
    "lateral_error_avg=%.4f lateral_error_max=%.4f "
    "steering_samples=%" PRIu64 " saturation_samples=%" PRIu64
    " saturation_events=%" PRIu64 " saturation_ratio=%.4f",
    context, plugin_name_.c_str(), segment_number_.load(),
    locked_direction_ > 0 ? "FORWARD" : "REVERSE", last_progress_index_,
    fullIndex(last_progress_index_), rollback_candidate_count_, accepted_rollback_count_,
    direction_conflict_samples_, direction_conflict_events_, average_error,
    max_lateral_error_, reverse_steering_samples_, reverse_saturation_samples_,
    reverse_saturation_events_, reverse_steering_samples_ == 0 ? 0.0 :
    static_cast<double>(reverse_saturation_samples_) /
    static_cast<double>(reverse_steering_samples_));
}

void DirectionLockedRPP::reset()
{
  std::lock_guard<std::mutex> lock(plan_mutex_);
  logTrackingSummary("reset");
  segment_plan_ = nav_msgs::msg::Path();
  resetTrackingState();
  RegulatedPurePursuitController::reset();
}

double DirectionLockedRPP::medianPathSpacing() const
{
  std::vector<double> spacings;
  spacings.reserve(segment_plan_.poses.size() - 1);
  for (std::size_t index = 1; index < segment_plan_.poses.size(); ++index) {
    const double spacing = euclideanDistance(
      segment_plan_.poses[index - 1], segment_plan_.poses[index]);
    if (spacing > std::numeric_limits<double>::epsilon()) {
      spacings.push_back(spacing);
    }
  }
  if (spacings.empty()) {
    return 0.0;
  }
  const auto middle = spacings.begin() + spacings.size() / 2;
  std::nth_element(spacings.begin(), middle, spacings.end());
  return *middle;
}

double DirectionLockedRPP::percentile(
  std::vector<double> values, double fraction) const
{
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double position = std::clamp(fraction, 0.0, 1.0) *
    static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double ratio = position - static_cast<double>(lower);
  return values[lower] + ratio * (values[upper] - values[lower]);
}

double DirectionLockedRPP::threePointCurvature(std::size_t index) const
{
  if (segment_plan_.poses.size() < 3 || cumulative_arc_length_.size() < 3) {
    return 0.0;
  }
  const double half_window = 0.5 * reverse_profile_window_;
  std::size_t first = index;
  std::size_t third = index;
  while (first > 0 &&
    cumulative_arc_length_[index] - cumulative_arc_length_[first] < half_window)
  {
    --first;
  }
  while (third + 1 < segment_plan_.poses.size() &&
    cumulative_arc_length_[third] - cumulative_arc_length_[index] < half_window)
  {
    ++third;
  }
  if (first == index) {
    third = std::min(segment_plan_.poses.size() - 1, index + 2);
  }
  if (third == index) {
    first = index >= 2 ? index - 2 : 0;
  }
  if (first == index || third == index || first == third) {
    return 0.0;
  }

  const auto & a = segment_plan_.poses[first].pose.position;
  const auto & b = segment_plan_.poses[index].pose.position;
  const auto & c = segment_plan_.poses[third].pose.position;
  const double ab = std::hypot(b.x - a.x, b.y - a.y);
  const double bc = std::hypot(c.x - b.x, c.y - b.y);
  const double ac = std::hypot(c.x - a.x, c.y - a.y);
  const double denominator = ab * bc * ac;
  if (denominator < 1.0e-6) {
    return 0.0;
  }
  const double cross =
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  return 2.0 * cross / denominator;
}

void DirectionLockedRPP::buildReverseSteeringProfile()
{
  std::vector<double> raw_curvature(segment_plan_.poses.size(), 0.0);
  for (std::size_t index = 0; index < segment_plan_.poses.size(); ++index) {
    // Path indices increase in the direction of travel.  During reverse,
    // ds_vehicle = -ds_path, so bicycle curvature tan(delta) / L is the
    // negative of the geometric path curvature.
    raw_curvature[index] = -threePointCurvature(index);
  }

  reverse_controller_curvature_.assign(segment_plan_.poses.size(), 0.0);
  const double smoothing_radius = 0.5 * reverse_profile_window_;
  for (std::size_t index = 0; index < segment_plan_.poses.size(); ++index) {
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t sample = 0; sample < segment_plan_.poses.size(); ++sample) {
      const double separation = std::abs(
        cumulative_arc_length_[sample] - cumulative_arc_length_[index]);
      if (separation > smoothing_radius) {
        continue;
      }
      const double weight = 1.0 - separation / std::max(smoothing_radius, 1.0e-6);
      weighted_sum += weight * raw_curvature[sample];
      weight_sum += weight;
    }
    reverse_controller_curvature_[index] =
      weight_sum > 0.0 ? weighted_sum / weight_sum : raw_curvature[index];
  }

  std::vector<double> absolute_steering;
  absolute_steering.reserve(reverse_controller_curvature_.size());
  reverse_profile_min_curvature_ = *std::min_element(
    reverse_controller_curvature_.begin(), reverse_controller_curvature_.end());
  reverse_profile_max_curvature_ = *std::max_element(
    reverse_controller_curvature_.begin(), reverse_controller_curvature_.end());
  for (const double curvature : reverse_controller_curvature_) {
    absolute_steering.push_back(std::abs(radiansToDegrees(
      std::atan(reverse_wheel_base_ * curvature))));
  }
  reverse_profile_min_steering_deg_ = *std::min_element(
    absolute_steering.begin(), absolute_steering.end());
  reverse_profile_max_steering_deg_ = *std::max_element(
    absolute_steering.begin(), absolute_steering.end());
  reverse_profile_mean_steering_deg_ = std::accumulate(
    absolute_steering.begin(), absolute_steering.end(), 0.0) /
    static_cast<double>(absolute_steering.size());
  reverse_profile_median_steering_deg_ = percentile(absolute_steering, 0.50);
  reverse_profile_p90_steering_deg_ = percentile(absolute_steering, 0.90);
  reverse_profile_p95_steering_deg_ = percentile(absolute_steering, 0.95);
  const double profile_soft_limit = std::max(
    reverse_profile_p95_steering_deg_, reverse_profile_max_steering_deg_) +
    reverse_soft_limit_margin_deg_;
  reverse_soft_limit_deg_ = reverse_soft_limit_override_deg_ > 0.0 ?
    reverse_soft_limit_override_deg_ : profile_soft_limit;
  reverse_soft_limit_deg_ = std::clamp(
    reverse_soft_limit_deg_, reverse_profile_p95_steering_deg_,
    reverse_hard_steering_limit_deg_);
}

void DirectionLockedRPP::logReverseSteeringProfile() const
{
  const double maximum_absolute_curvature = std::max(
    std::abs(reverse_profile_min_curvature_),
    std::abs(reverse_profile_max_curvature_));
  const double minimum_radius = maximum_absolute_curvature > 1.0e-6 ?
    1.0 / maximum_absolute_curvature : std::numeric_limits<double>::infinity();
  double first_two_min = std::numeric_limits<double>::max();
  double first_two_max = 0.0;
  for (std::size_t index = 0; index < reverse_controller_curvature_.size(); ++index) {
    if (cumulative_arc_length_[index] > 2.0) {
      break;
    }
    const double steering = std::abs(radiansToDegrees(std::atan(
      reverse_wheel_base_ * reverse_controller_curvature_[index])));
    first_two_min = std::min(first_two_min, steering);
    first_two_max = std::max(first_two_max, steering);
  }
  if (!std::isfinite(first_two_min)) {
    first_two_min = 0.0;
  }
  RCLCPP_INFO(
    logger_,
    "[REVERSE-PROFILE] curvature_min=%+.6f curvature_max=%+.6f "
    "minimum_radius=%.4f steering_min=%.3f steering_max=%.3f "
    "steering_mean=%.3f steering_median=%.3f steering_p90=%.3f "
    "steering_p95=%.3f first_2m_steering=%.3f..%.3f soft_limit=%.3f "
    "wheel_base=%.3f window=%.3f",
    reverse_profile_min_curvature_, reverse_profile_max_curvature_, minimum_radius,
    reverse_profile_min_steering_deg_, reverse_profile_max_steering_deg_,
    reverse_profile_mean_steering_deg_, reverse_profile_median_steering_deg_,
    reverse_profile_p90_steering_deg_, reverse_profile_p95_steering_deg_,
    first_two_min, first_two_max, reverse_soft_limit_deg_, reverse_wheel_base_,
    reverse_profile_window_);
  for (std::size_t index = 0; index < reverse_controller_curvature_.size(); index += 5) {
    const double curvature = reverse_controller_curvature_[index];
    const double radius = std::abs(curvature) > 1.0e-6 ?
      1.0 / std::abs(curvature) : std::numeric_limits<double>::infinity();
    RCLCPP_INFO(
      logger_,
      "[REVERSE-PROFILE-POINT] index=%zu arc_length=%.3f curvature=%+.6f "
      "turning_radius=%.3f required_steering=%+.3f",
      index, cumulative_arc_length_[index], curvature, radius,
      radiansToDegrees(std::atan(reverse_wheel_base_ * curvature)));
  }
}

DirectionLockedRPP::PathProjection DirectionLockedRPP::projectReversePath(
  const geometry_msgs::msg::PoseStamped & robot_plan_pose)
{
  PathProjection best;
  best.distance = std::numeric_limits<double>::max();
  const std::size_t search_end = std::max(
    last_progress_index_ + 1,
    forwardWindowEnd(last_progress_index_, progress_search_distance_));
  const double px = robot_plan_pose.pose.position.x;
  const double py = robot_plan_pose.pose.position.y;
  for (std::size_t index = last_progress_index_;
    index < search_end && index + 1 < segment_plan_.poses.size(); ++index)
  {
    const auto & a = segment_plan_.poses[index].pose.position;
    const auto & b = segment_plan_.poses[index + 1].pose.position;
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double squared_length = dx * dx + dy * dy;
    if (squared_length <= 1.0e-12) {
      continue;
    }
    const double ratio = std::clamp(
      ((px - a.x) * dx + (py - a.y) * dy) / squared_length, 0.0, 1.0);
    const double x = a.x + ratio * dx;
    const double y = a.y + ratio * dy;
    const double distance = std::hypot(px - x, py - y);
    if (distance < best.distance) {
      const double length = std::sqrt(squared_length);
      best.segment_index = index;
      best.ratio = ratio;
      best.x = x;
      best.y = y;
      best.arc_length = cumulative_arc_length_[index] + ratio * length;
      best.signed_lateral_error = (dx * (py - y) - dy * (px - x)) / length;
      best.distance = distance;
    }
  }
  if (!std::isfinite(best.distance)) {
    throw nav2_core::NoValidControl("Unable to project robot onto reverse path");
  }

  if (reverse_projection_initialized_ && best.arc_length < last_projection_arc_length_) {
    ++rollback_candidate_count_;
    best.arc_length = last_projection_arc_length_;
    const auto upper = std::upper_bound(
      cumulative_arc_length_.begin(), cumulative_arc_length_.end(), best.arc_length);
    best.segment_index = upper == cumulative_arc_length_.begin() ? 0 :
      std::min(
      static_cast<std::size_t>(std::distance(cumulative_arc_length_.begin(), upper) - 1),
      segment_plan_.poses.size() - 2);
    const auto & a = segment_plan_.poses[best.segment_index].pose.position;
    const auto & b = segment_plan_.poses[best.segment_index + 1].pose.position;
    const double length = std::max(
      cumulative_arc_length_[best.segment_index + 1] -
      cumulative_arc_length_[best.segment_index], 1.0e-9);
    best.ratio = std::clamp(
      (best.arc_length - cumulative_arc_length_[best.segment_index]) / length,
      0.0, 1.0);
    best.x = a.x + best.ratio * (b.x - a.x);
    best.y = a.y + best.ratio * (b.y - a.y);
    best.distance = std::hypot(px - best.x, py - best.y);
    best.signed_lateral_error =
      ((b.x - a.x) * (py - best.y) - (b.y - a.y) * (px - best.x)) / length;
  }
  reverse_projection_initialized_ = true;
  last_projection_arc_length_ = best.arc_length;
  last_progress_index_ = std::max(last_progress_index_, best.segment_index);
  return best;
}

geometry_msgs::msg::PoseStamped DirectionLockedRPP::reverseArcTarget(
  double target_arc_length, const builtin_interfaces::msg::Time & stamp) const
{
  target_arc_length = std::clamp(
    target_arc_length, 0.0, cumulative_arc_length_.back());
  const auto upper = std::upper_bound(
    cumulative_arc_length_.begin(), cumulative_arc_length_.end(), target_arc_length);
  const std::size_t index = upper == cumulative_arc_length_.begin() ? 0 :
    std::min(
    static_cast<std::size_t>(std::distance(cumulative_arc_length_.begin(), upper) - 1),
    segment_plan_.poses.size() - 2);
  const double length = std::max(
    cumulative_arc_length_[index + 1] - cumulative_arc_length_[index], 1.0e-9);
  const double ratio = std::clamp(
    (target_arc_length - cumulative_arc_length_[index]) / length, 0.0, 1.0);
  geometry_msgs::msg::PoseStamped target = segment_plan_.poses[index];
  target.header.frame_id = segment_plan_.header.frame_id;
  target.header.stamp = stamp;
  target.pose.position.x += ratio *
    (segment_plan_.poses[index + 1].pose.position.x - target.pose.position.x);
  target.pose.position.y += ratio *
    (segment_plan_.poses[index + 1].pose.position.y - target.pose.position.y);
  return target;
}

double DirectionLockedRPP::reverseControllerCurvature(double arc_length) const
{
  arc_length = std::clamp(arc_length, 0.0, cumulative_arc_length_.back());
  const auto upper = std::upper_bound(
    cumulative_arc_length_.begin(), cumulative_arc_length_.end(), arc_length);
  const std::size_t index = upper == cumulative_arc_length_.begin() ? 0 :
    std::min(
    static_cast<std::size_t>(std::distance(cumulative_arc_length_.begin(), upper) - 1),
    reverse_controller_curvature_.size() - 2);
  const double length = std::max(
    cumulative_arc_length_[index + 1] - cumulative_arc_length_[index], 1.0e-9);
  const double ratio = std::clamp(
    (arc_length - cumulative_arc_length_[index]) / length, 0.0, 1.0);
  return reverse_controller_curvature_[index] + ratio *
         (reverse_controller_curvature_[index + 1] - reverse_controller_curvature_[index]);
}

std::size_t DirectionLockedRPP::forwardWindowEnd(
  std::size_t first, double distance) const
{
  if (segment_plan_.poses.empty()) {
    return 0;
  }
  double integrated = 0.0;
  std::size_t last = first;
  while (last + 1 < segment_plan_.poses.size() && integrated < distance) {
    integrated += euclideanDistance(segment_plan_.poses[last], segment_plan_.poses[last + 1]);
    ++last;
  }
  return last;
}

std::size_t DirectionLockedRPP::closestPose(
  const geometry_msgs::msg::PoseStamped & robot_plan_pose,
  std::size_t first, std::size_t last) const
{
  std::size_t closest = first;
  double closest_distance = std::numeric_limits<double>::max();
  for (std::size_t index = first; index <= last; ++index) {
    const double distance = euclideanDistance(robot_plan_pose, segment_plan_.poses[index]);
    if (distance < closest_distance) {
      closest = index;
      closest_distance = distance;
    }
  }
  return closest;
}

double DirectionLockedRPP::pointToSegmentDistance(
  double px, double py, std::size_t first, std::size_t second) const
{
  const auto & a = segment_plan_.poses[first].pose.position;
  const auto & b = segment_plan_.poses[second].pose.position;
  const double dx = b.x - a.x;
  const double dy = b.y - a.y;
  const double squared_length = dx * dx + dy * dy;
  if (squared_length <= std::numeric_limits<double>::epsilon()) {
    return std::hypot(px - a.x, py - a.y);
  }
  const double ratio = std::clamp(
    ((px - a.x) * dx + (py - a.y) * dy) / squared_length, 0.0, 1.0);
  return std::hypot(px - (a.x + ratio * dx), py - (a.y + ratio * dy));
}

double DirectionLockedRPP::calculateLateralError(
  const geometry_msgs::msg::PoseStamped & robot_plan_pose,
  std::size_t first, std::size_t last) const
{
  if (segment_plan_.poses.size() < 2) {
    return 0.0;
  }
  first = first > 0 ? first - 1 : first;
  last = std::min(last, segment_plan_.poses.size() - 1);
  double error = std::numeric_limits<double>::max();
  for (std::size_t index = first; index < last; ++index) {
    error = std::min(
      error, pointToSegmentDistance(
        robot_plan_pose.pose.position.x, robot_plan_pose.pose.position.y,
        index, index + 1));
  }
  return std::isfinite(error) ? error : 0.0;
}

DirectionLockedRPP::ProgressUpdate DirectionLockedRPP::updateProgress(
  const geometry_msgs::msg::PoseStamped & robot_plan_pose)
{
  ProgressUpdate update;
  update.previous_index = last_progress_index_;
  update.unrestricted_candidate = closestPose(
    robot_plan_pose, 0, segment_plan_.poses.size() - 1);

  if (!progress_initialized_) {
    const std::size_t start_end = forwardWindowEnd(0, progress_search_distance_);
    std::size_t selected = closestPose(robot_plan_pose, 0, start_end);
    const double meaningful_distance = 0.5 * median_spacing_;
    for (std::size_t index = 0; index <= start_end; ++index) {
      geometry_msgs::msg::PoseStamped stamped = segment_plan_.poses[index];
      stamped.header.frame_id = segment_plan_.header.frame_id;
      stamped.header.stamp = robot_plan_pose.header.stamp;
      geometry_msgs::msg::PoseStamped base_pose;
      if (path_handler_->transformPose(costmap_ros_->getBaseFrameID(), stamped, base_pose) &&
        locked_direction_ * base_pose.pose.position.x >= meaningful_distance)
      {
        selected = index;
        break;
      }
    }
    last_progress_index_ = std::min(selected, segment_plan_.poses.size() - 2);
    progress_initialized_ = true;
    RCLCPP_INFO(
      logger_,
      "[RPP-LOCK][START] segment=%d direction=%s local_index=%zu full_index=%d "
      "start_window=0..%zu meaningful_distance=%.4f",
      segment_number_.load(), locked_direction_ > 0 ? "FORWARD" : "REVERSE",
      last_progress_index_, fullIndex(last_progress_index_), start_end,
      meaningful_distance);
  } else {
    if (update.unrestricted_candidate < last_progress_index_) {
      ++rollback_candidate_count_;
      auto node = node_.lock();
      if (node) {
        RCLCPP_WARN_THROTTLE(
          logger_, *node->get_clock(), 1000,
          "[PATH INDEX ERROR] segment=%d previous=%zu candidate=%zu candidate rejected",
          segment_number_.load(), last_progress_index_, update.unrestricted_candidate);
      }
    }
    const std::size_t search_end = forwardWindowEnd(
      last_progress_index_, progress_search_distance_);
    const std::size_t candidate = closestPose(
      robot_plan_pose, last_progress_index_, search_end);
    if (candidate < last_progress_index_) {
      ++accepted_rollback_count_;
    } else {
      last_progress_index_ = monotonicPathIndex(
        last_progress_index_, candidate, segment_plan_.poses.size() - 2);
    }
  }

  const std::size_t error_window_end = forwardWindowEnd(
    last_progress_index_, progress_search_distance_);
  update.lateral_error = calculateLateralError(
    robot_plan_pose, last_progress_index_, error_window_end);
  ++lateral_error_samples_;
  lateral_error_sum_ += update.lateral_error;
  max_lateral_error_ = std::max(max_lateral_error_, update.lateral_error);
  update.local_index = last_progress_index_;
  return update;
}

nav_msgs::msg::Path DirectionLockedRPP::transformActivePlan(
  const geometry_msgs::msg::PoseStamped & robot_plan_pose,
  const builtin_interfaces::msg::Time & command_stamp)
{
  nav_msgs::msg::Path transformed;
  transformed.header.frame_id = costmap_ros_->getBaseFrameID();
  transformed.header.stamp = command_stamp;
  const double max_extent = std::max(
    costmap_->getSizeInMetersX(), costmap_->getSizeInMetersY()) / 2.0;

  for (std::size_t index = last_progress_index_; index < segment_plan_.poses.size(); ++index) {
    const double distance = euclideanDistance(robot_plan_pose, segment_plan_.poses[index]);
    if (distance > max_extent && transformed.poses.size() >= 2) {
      break;
    }
    geometry_msgs::msg::PoseStamped stamped = segment_plan_.poses[index];
    stamped.header.frame_id = segment_plan_.header.frame_id;
    stamped.header.stamp = command_stamp;
    geometry_msgs::msg::PoseStamped base_pose;
    if (!path_handler_->transformPose(costmap_ros_->getBaseFrameID(), stamped, base_pose)) {
      throw nav2_core::ControllerTFError("Unable to transform active segment pose");
    }
    base_pose.pose.position.z = 0.0;
    transformed.poses.push_back(base_pose);
  }
  if (transformed.poses.size() < 2) {
    throw nav2_core::InvalidPath("DirectionLockedRPP transformed plan has fewer than two poses");
  }
  return transformed;
}

geometry_msgs::msg::PoseStamped DirectionLockedRPP::getDirectionLockedLookAheadPoint(
  double lookahead_distance, const nav_msgs::msg::Path & transformed_plan) const
{
  geometry_msgs::msg::PoseStamped carrot;
  if (findDirectionLockedLookAheadPoint(
      lookahead_distance, transformed_plan, carrot))
  {
    return carrot;
  }
  throw nav2_core::NoValidControl(
          "No active-segment carrot exists in the locked motion half-plane");
}

bool DirectionLockedRPP::findDirectionLockedLookAheadPoint(
  double lookahead_distance, const nav_msgs::msg::Path & transformed_plan,
  geometry_msgs::msg::PoseStamped & carrot) const
{
  const geometry_msgs::msg::PoseStamped * best_directional_pose = nullptr;
  double best_directional_x = 0.0;
  const geometry_msgs::msg::PoseStamped * previous_pose = nullptr;
  double previous_directional_x = 0.0;

  for (const auto & candidate : transformed_plan.poses) {
    const double directional_x = locked_direction_ * candidate.pose.position.x;
    if (directional_x > best_directional_x) {
      best_directional_x = directional_x;
      best_directional_pose = &candidate;
    }
    if (directional_x >= lookahead_distance) {
      carrot = candidate;
      if (previous_pose != nullptr && previous_directional_x < lookahead_distance &&
        directional_x > previous_directional_x)
      {
        const double ratio = std::clamp(
          (lookahead_distance - previous_directional_x) /
          (directional_x - previous_directional_x), 0.0, 1.0);
        carrot.pose.position.x = previous_pose->pose.position.x + ratio *
          (candidate.pose.position.x - previous_pose->pose.position.x);
        carrot.pose.position.y = previous_pose->pose.position.y + ratio *
          (candidate.pose.position.y - previous_pose->pose.position.y);
      }
      return true;
    }
    previous_pose = &candidate;
    previous_directional_x = directional_x;
  }

  // Near the goal the remaining longitudinal distance can be smaller than the
  // configured lookahead. Use the pose furthest into the locked motion
  // half-plane, never a lateral pose from the opposite half-plane.
  if (best_directional_pose != nullptr) {
    carrot = *best_directional_pose;
    return true;
  }
  return false;
}

void DirectionLockedRPP::diagnoseSelfProximity()
{
  const double proximity_distance = params_->max_lookahead_dist;
  const double minimum_arc_separation = 2.0 * proximity_distance;
  std::vector<double> cumulative(segment_plan_.poses.size(), 0.0);
  for (std::size_t index = 1; index < segment_plan_.poses.size(); ++index) {
    cumulative[index] = cumulative[index - 1] + euclideanDistance(
      segment_plan_.poses[index - 1], segment_plan_.poses[index]);
  }

  std::size_t best_first = 0;
  std::size_t best_second = 0;
  double best_distance = std::numeric_limits<double>::max();
  std::uint64_t cases = 0;
  for (std::size_t first = 0; first < segment_plan_.poses.size(); ++first) {
    for (std::size_t second = first + 1; second < segment_plan_.poses.size(); ++second) {
      if (cumulative[second] - cumulative[first] <= minimum_arc_separation) {
        continue;
      }
      const double distance = euclideanDistance(
        segment_plan_.poses[first], segment_plan_.poses[second]);
      if (distance < proximity_distance) {
        ++cases;
        if (distance < best_distance) {
          best_distance = distance;
          best_first = first;
          best_second = second;
        }
      }
    }
  }
  if (cases > 0) {
    RCLCPP_INFO(
      logger_,
      "[PATH SELF-PROXIMITY] segment=%d cases=%" PRIu64
      " local=%zu/%zu full=%d/%d "
      "distance=%.4f index_difference=%zu arc_separation=%.4f",
      segment_number_.load(), cases, best_first, best_second,
      fullIndex(best_first), fullIndex(best_second), best_distance,
      best_second - best_first, cumulative[best_second] - cumulative[best_first]);
  } else {
    RCLCPP_INFO(
      logger_, "[PATH SELF-PROXIMITY] segment=%d cases=0 threshold=%.4f",
      segment_number_.load(), proximity_distance);
  }
}

int DirectionLockedRPP::fullIndex(std::size_t local_index) const
{
  return full_start_index_.load() + static_cast<int>(local_index);
}

geometry_msgs::msg::TwistStamped DirectionLockedRPP::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & speed,
  nav2_core::GoalChecker * goal_checker)
{
  std::lock_guard<std::mutex> parameter_lock(param_handler_->getMutex());
  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> costmap_lock(
    *(costmap_->getMutex()));
  std::lock_guard<std::mutex> plan_lock(plan_mutex_);
  if (segment_plan_.poses.size() < 2) {
    throw nav2_core::InvalidPath("DirectionLockedRPP has no active segment");
  }

  geometry_msgs::msg::Pose pose_tolerance;
  geometry_msgs::msg::Twist velocity_tolerance;
  double xy_goal_tolerance = -1.0;
  double yaw_goal_tolerance = -1.0;
  if (goal_checker->getTolerances(pose_tolerance, velocity_tolerance)) {
    goal_dist_tol_ = pose_tolerance.position.x;
    xy_goal_tolerance = std::abs(pose_tolerance.position.x);
    // GoalChecker encodes angular tolerance as a yaw-only quaternion.
    // Decode its angle; orientation.z alone is sin(yaw_tolerance / 2).
    yaw_goal_tolerance = yawToleranceFromQuaternion(
      pose_tolerance.orientation.z, pose_tolerance.orientation.w);
  } else {
    RCLCPP_WARN(logger_, "Unable to retrieve goal checker tolerances");
  }

  geometry_msgs::msg::PoseStamped robot_plan_pose;
  if (!path_handler_->transformPose(segment_plan_.header.frame_id, pose, robot_plan_pose)) {
    throw nav2_core::ControllerTFError("Unable to transform robot pose into segment frame");
  }
  ProgressUpdate progress;
  PathProjection reverse_projection;
  if (locked_direction_ < 0) {
    progress.previous_index = last_progress_index_;
    progress.unrestricted_candidate = closestPose(
      robot_plan_pose, 0, segment_plan_.poses.size() - 1);
    if (progress.unrestricted_candidate < last_progress_index_) {
      auto node = node_.lock();
      if (node) {
        RCLCPP_WARN_THROTTLE(
          logger_, *node->get_clock(), 1000,
          "[PATH INDEX ERROR] segment=%d previous=%zu candidate=%zu candidate rejected",
          segment_number_.load(), last_progress_index_, progress.unrestricted_candidate);
      }
    }
    reverse_projection = projectReversePath(robot_plan_pose);
    progress.local_index = last_progress_index_;
    progress.lateral_error = std::abs(reverse_projection.signed_lateral_error);
    ++lateral_error_samples_;
    lateral_error_sum_ += progress.lateral_error;
    max_lateral_error_ = std::max(max_lateral_error_, progress.lateral_error);
  } else {
    progress = updateProgress(robot_plan_pose);
  }
  const auto transformed_plan = transformActivePlan(robot_plan_pose, pose.header.stamp);
  global_path_pub_->publish(transformed_plan);

  const double lookahead_distance = getLookAheadDistance(speed);
  const auto raw_carrot_pose = getLookAheadPoint(lookahead_distance, transformed_plan);
  const double raw_directional_x = locked_direction_ * raw_carrot_pose.pose.position.x;
  const double unambiguous_x = 0.5 * median_spacing_;
  auto carrot_pose = raw_carrot_pose;
  bool normal_carrot_available = false;
  double target_plan_x = std::numeric_limits<double>::quiet_NaN();
  double target_plan_y = std::numeric_limits<double>::quiet_NaN();
  if (locked_direction_ < 0) {
    const auto target_in_plan = reverseArcTarget(
      reverse_projection.arc_length + lookahead_distance, pose.header.stamp);
    target_plan_x = target_in_plan.pose.position.x;
    target_plan_y = target_in_plan.pose.position.y;
    if (!path_handler_->transformPose(
        costmap_ros_->getBaseFrameID(), target_in_plan, carrot_pose))
    {
      throw nav2_core::ControllerTFError("Unable to transform reverse arc target");
    }
    normal_carrot_available = isInLockedMotionHalfPlane(
      locked_direction_, carrot_pose.pose.position.x);
  } else if (raw_directional_x < unambiguous_x) {
    normal_carrot_available = findDirectionLockedLookAheadPoint(
      lookahead_distance, transformed_plan, carrot_pose);
  } else {
    normal_carrot_available = true;
  }

  geometry_msgs::msg::PoseStamped endpoint_plan = segment_plan_.poses.back();
  endpoint_plan.header.frame_id = segment_plan_.header.frame_id;
  endpoint_plan.header.stamp = pose.header.stamp;
  geometry_msgs::msg::PoseStamped endpoint_base;
  if (!path_handler_->transformPose(
      costmap_ros_->getBaseFrameID(), endpoint_plan, endpoint_base))
  {
    throw nav2_core::ControllerTFError("Unable to transform active segment endpoint");
  }
  endpoint_base.pose.position.z = 0.0;

  const std::size_t last_index = segment_plan_.poses.size() - 1;
  const std::size_t remaining_poses = last_index - std::min(
    progress.local_index, last_index);
  const double progress_arc_length = locked_direction_ < 0 ?
    reverse_projection.arc_length : cumulative_arc_length_[progress.local_index];
  const double remaining_arc_length = std::max(
    0.0, cumulative_arc_length_.back() - progress_arc_length);
  const double endpoint_distance = std::hypot(
    endpoint_base.pose.position.x, endpoint_base.pose.position.y);
  const double position_error = euclideanDistance(robot_plan_pose, endpoint_plan);
  const double goal_yaw = tf2::getYaw(endpoint_plan.pose.orientation);
  const double actual_yaw = tf2::getYaw(robot_plan_pose.pose.orientation);
  const double yaw_error = normalizeAngle(actual_yaw - goal_yaw);
  const TerminalCaptureMode terminal_mode = selectTerminalCaptureMode(
    remaining_arc_length, terminal_capture_distance_, locked_direction_,
    endpoint_base.pose.position.x, position_error, yaw_error,
    xy_goal_tolerance, yaw_goal_tolerance);
  const bool terminal_carrot_used =
    terminal_mode == TerminalCaptureMode::kCaptureEndpoint;
  if (terminal_carrot_used) {
    carrot_pose = endpoint_base;
  }

  auto terminal_node = node_.lock();
  if (terminal_node && terminal_capture_distance_ > 0.0 && remaining_poses <= 5) {
    const rclcpp::Time now = terminal_node->now();
    if (last_terminal_log_.nanoseconds() == 0 ||
      (now - last_terminal_log_).seconds() >= 0.5)
    {
      last_terminal_log_ = now;
      RCLCPP_INFO(
        logger_,
        "[T-PARK][TERMINAL TRACK] segment=%d direction=%s local_index=%zu "
        "last_index=%zu remaining_poses=%zu remaining_arc_length=%.4f "
        "goal=(%.4f,%.4f,%.4f) actual=(%.4f,%.4f,%.4f) "
        "endpoint_base_x=%+.4f endpoint_base_y=%+.4f endpoint_distance=%.4f "
        "position_error=%.4f yaw_error=%+.4f xy_tolerance=%.4f yaw_tolerance=%.4f "
        "normal_carrot_available=%s terminal_carrot_used=%s terminal_mode=%s",
        segment_number_.load(), locked_direction_ > 0 ? "FORWARD" : "REVERSE",
        progress.local_index, last_index, remaining_poses, remaining_arc_length,
        endpoint_plan.pose.position.x, endpoint_plan.pose.position.y, goal_yaw,
        robot_plan_pose.pose.position.x, robot_plan_pose.pose.position.y, actual_yaw,
        endpoint_base.pose.position.x, endpoint_base.pose.position.y, endpoint_distance,
        position_error, yaw_error, xy_goal_tolerance, yaw_goal_tolerance,
        normal_carrot_available ? "true" : "false",
        terminal_carrot_used ? "true" : "false",
        terminalCaptureModeName(terminal_mode));
    }
  }

  if (terminal_mode == TerminalCaptureMode::kGoalToleranceZero) {
    if (terminal_node && !terminal_goal_reached_logged_) {
      terminal_goal_reached_logged_ = true;
      RCLCPP_INFO(
        logger_,
        "[T-PARK][TERMINAL GOAL REACHED] segment=%d direction=%s "
        "position_error=%.4f yaw_error=%+.4f xy_tolerance=%.4f "
        "yaw_tolerance=%.4f",
        segment_number_.load(), locked_direction_ > 0 ? "FORWARD" : "REVERSE",
        position_error, yaw_error, xy_goal_tolerance, yaw_goal_tolerance);
    }
    is_rotating_to_heading_ = false;
    carrot_pub_->publish(createCarrotMsg(endpoint_base));
    std_msgs::msg::Bool rotating_message;
    rotating_message.data = false;
    is_rotating_to_heading_pub_->publish(rotating_message);
    geometry_msgs::msg::TwistStamped zero_command;
    zero_command.header = pose.header;
    return zero_command;
  }
  if (terminal_mode == TerminalCaptureMode::kOvershootAbort) {
    if (terminal_node) {
      RCLCPP_ERROR_THROTTLE(
        logger_, *terminal_node->get_clock(), 1000,
        "[T-PARK][TERMINAL] CUSP_ENDPOINT_OVERSHOOT segment=%d direction=%s "
        "remaining_arc_length=%.4f endpoint_base=(%+.4f,%+.4f) "
        "position_error=%.4f yaw_error=%+.4f",
        segment_number_.load(), locked_direction_ > 0 ? "FORWARD" : "REVERSE",
        remaining_arc_length, endpoint_base.pose.position.x,
        endpoint_base.pose.position.y, position_error, yaw_error);
    }
    throw nav2_core::NoValidControl("CUSP_ENDPOINT_OVERSHOOT");
  }
  if (!normal_carrot_available) {
    throw nav2_core::NoValidControl(
            "No active-segment carrot exists in the locked motion half-plane");
  }
  auto rotation_carrot = carrot_pose;
  carrot_pub_->publish(createCarrotMsg(carrot_pose));

  const double lookahead_curvature = calculateCurvature(carrot_pose.pose.position);
  double regulation_curvature = lookahead_curvature;
  double path_curvature = regulation_curvature;
  double delta_path_deg = radiansToDegrees(std::atan(
    reverse_wheel_base_ * path_curvature));
  double delta_controller_deg = delta_path_deg;
  double delta_output_deg = delta_path_deg;
  double reverse_heading_error = 0.0;
  if (locked_direction_ < 0) {
    const double curvature_sample_arc = std::min(
      reverse_projection.arc_length + 0.5 * lookahead_distance,
      cumulative_arc_length_.back());
    path_curvature = reverseControllerCurvature(curvature_sample_arc);
    const double delta_feedforward = std::atan(reverse_wheel_base_ * path_curvature);
    const auto & projection_start =
      segment_plan_.poses[reverse_projection.segment_index].pose.position;
    const auto & projection_end =
      segment_plan_.poses[reverse_projection.segment_index + 1].pose.position;
    const double path_travel_heading = std::atan2(
      projection_end.y - projection_start.y,
      projection_end.x - projection_start.x);
    const double desired_vehicle_heading = normalizeAngle(path_travel_heading + kPi);
    reverse_heading_error = normalizeAngle(
      tf2::getYaw(robot_plan_pose.pose.orientation) - desired_vehicle_heading);
    const double delta_feedback = std::clamp(
      reverse_lateral_gain_ * reverse_projection.signed_lateral_error +
      reverse_heading_gain_ * reverse_heading_error,
      -degreesToRadians(reverse_lateral_correction_limit_deg_),
      degreesToRadians(reverse_lateral_correction_limit_deg_));
    const double requested_delta = delta_feedforward + delta_feedback;
    const double soft_limited_delta = std::clamp(
      requested_delta, -degreesToRadians(reverse_soft_limit_deg_),
      degreesToRadians(reverse_soft_limit_deg_));
    const double output_delta = std::clamp(
      soft_limited_delta, -degreesToRadians(reverse_hard_steering_limit_deg_),
      degreesToRadians(reverse_hard_steering_limit_deg_));
    delta_path_deg = radiansToDegrees(delta_feedforward);
    delta_controller_deg = radiansToDegrees(requested_delta);
    delta_output_deg = radiansToDegrees(output_delta);
    regulation_curvature = std::tan(output_delta) / reverse_wheel_base_;

    ++reverse_steering_samples_;
    const bool saturated =
      std::abs(delta_output_deg) >= reverse_hard_steering_limit_deg_ - 0.5;
    if (saturated) {
      ++reverse_saturation_samples_;
      if (!reverse_saturation_active_) {
        ++reverse_saturation_events_;
        RCLCPP_WARN(
          logger_, "[STEERING SATURATION] requested=%+.3f deg output=%+.3f deg",
          delta_controller_deg, delta_output_deg);
      }
    }
    reverse_saturation_active_ = saturated;
  } else if (params_->use_fixed_curvature_lookahead && !terminal_carrot_used) {
    const auto raw_curvature_carrot = getLookAheadPoint(
      params_->curvature_lookahead_dist, transformed_plan,
      params_->interpolate_curvature_after_goal);
    const double curvature_directional_x =
      locked_direction_ * raw_curvature_carrot.pose.position.x;
    auto curvature_carrot = curvature_directional_x >= unambiguous_x ?
      raw_curvature_carrot : getDirectionLockedLookAheadPoint(
      params_->curvature_lookahead_dist, transformed_plan);
    rotation_carrot = curvature_carrot;
    regulation_curvature = calculateCurvature(curvature_carrot.pose.position);
    path_curvature = regulation_curvature;
    curvature_carrot_pub_->publish(createCarrotMsg(curvature_carrot));
  }

  const int raw_direction = raw_carrot_pose.pose.position.x >= 0.0 ? 1 : -1;
  const bool conflict = raw_direction != locked_direction_;
  if (conflict) {
    ++direction_conflict_samples_;
    if (!direction_conflict_active_) {
      ++direction_conflict_events_;
      RCLCPP_WARN(
        logger_,
        "[DIRECTION CONFLICT] segment=%d expected=%s raw_direction=%s "
        "local_index=%zu full_index=%d raw_carrot_base=(%+.4f,%+.4f) "
        "locked_carrot_base=(%+.4f,%+.4f)",
        segment_number_.load(), locked_direction_ > 0 ? "FORWARD" : "REVERSE",
        raw_direction > 0 ? "FORWARD" : "REVERSE", progress.local_index,
        fullIndex(progress.local_index), raw_carrot_pose.pose.position.x,
        raw_carrot_pose.pose.position.y, carrot_pose.pose.position.x,
        carrot_pose.pose.position.y);
    }
  }
  direction_conflict_active_ = conflict;

  double linear_velocity = params_->desired_linear_vel;
  double angular_velocity = 0.0;
  double angle_to_heading = 0.0;
  double locked_sign = static_cast<double>(locked_direction_);
  if (shouldRotateToGoalHeading(carrot_pose)) {
    is_rotating_to_heading_ = true;
    const double angle_to_goal = tf2::getYaw(transformed_plan.poses.back().pose.orientation);
    rotateToHeading(linear_velocity, angular_velocity, angle_to_goal, speed);
  } else if (shouldRotateToPath(rotation_carrot, angle_to_heading, locked_sign)) {
    is_rotating_to_heading_ = true;
    rotateToHeading(linear_velocity, angular_velocity, angle_to_heading, speed);
  } else {
    is_rotating_to_heading_ = false;
    applyConstraints(
      regulation_curvature, speed,
      collision_checker_->costAtPose(pose.pose.position.x, pose.pose.position.y),
      transformed_plan, linear_velocity, locked_sign);

    if (cancelling_) {
      linear_velocity = speed.linear.x - locked_sign * control_duration_ *
        params_->cancel_deceleration;
      if ((locked_sign > 0.0 && linear_velocity <= 0.0) ||
        (locked_sign < 0.0 && linear_velocity >= 0.0))
      {
        linear_velocity = 0.0;
        finished_cancelling_ = true;
      }
    }

    // Direction is applied before angular velocity so omega / v remains the
    // curvature selected from the same active-segment carrot.
    angular_velocity = linear_velocity * regulation_curvature;
  }

  const double carrot_distance = std::hypot(
    carrot_pose.pose.position.x, carrot_pose.pose.position.y);
  if (params_->use_collision_detection && collision_checker_->isCollisionImminent(
      pose, linear_velocity, angular_velocity, carrot_distance))
  {
    throw nav2_core::NoValidControl("DirectionLockedRPP detected collision ahead");
  }

  std_msgs::msg::Bool rotating_message;
  rotating_message.data = is_rotating_to_heading_;
  is_rotating_to_heading_pub_->publish(rotating_message);

  geometry_msgs::msg::TwistStamped command;
  command.header = pose.header;
  command.twist.linear.x = linear_velocity;
  command.twist.angular.z = angular_velocity;

  auto node = node_.lock();
  if (node) {
    const rclcpp::Time now = node->now();
    if (last_track_log_.nanoseconds() == 0 || (now - last_track_log_).seconds() >= 0.5) {
      last_track_log_ = now;
      RCLCPP_INFO(
        logger_,
        "[TRACK] segment=%s local_index=%zu full_index=%d previous_index=%zu "
        "unrestricted_candidate=%zu raw_carrot_base=(%+.4f,%+.4f) "
        "locked_carrot_base=(%+.4f,%+.4f) direction=%s raw_direction=%s "
        "cmd_v=%+.4f cmd_w=%+.4f lateral_error=%.4f",
        locked_direction_ > 0 ? "FORWARD" : "REVERSE", progress.local_index,
        fullIndex(progress.local_index), progress.previous_index,
        progress.unrestricted_candidate, raw_carrot_pose.pose.position.x,
        raw_carrot_pose.pose.position.y, carrot_pose.pose.position.x,
        carrot_pose.pose.position.y, locked_direction_ > 0 ? "FORWARD" : "REVERSE",
        raw_direction > 0 ? "FORWARD" : "REVERSE", linear_velocity,
        angular_velocity, progress.lateral_error);
      if (locked_direction_ < 0) {
        RCLCPP_INFO(
          logger_,
          "[REVERSE-STEER] progress_index=%zu robot_pose=(%.4f,%.4f,%.4f) "
          "projection=(%.4f,%.4f,s=%.4f,segment=%zu) "
          "target=(%.4f,%.4f) target_base_x=%+.4f target_base_y=%+.4f "
          "path_curvature=%+.6f controller_curvature=%+.6f "
          "delta_path_deg=%+.3f delta_controller_deg=%+.3f "
          "delta_output_deg=%+.3f lateral_error=%+.4f heading_error=%+.4f "
          "soft_limit_deg=%.3f",
          progress.local_index, robot_plan_pose.pose.position.x,
          robot_plan_pose.pose.position.y, tf2::getYaw(robot_plan_pose.pose.orientation),
          reverse_projection.x, reverse_projection.y, reverse_projection.arc_length,
          reverse_projection.segment_index, target_plan_x,
          target_plan_y, carrot_pose.pose.position.x,
          carrot_pose.pose.position.y, path_curvature, regulation_curvature,
          delta_path_deg, delta_controller_deg, delta_output_deg,
          reverse_projection.signed_lateral_error, reverse_heading_error,
          reverse_soft_limit_deg_);
      }
    }
  }
  return command;
}

}  // namespace t_parking_sim

PLUGINLIB_EXPORT_CLASS(t_parking_sim::DirectionLockedRPP, nav2_core::Controller)
