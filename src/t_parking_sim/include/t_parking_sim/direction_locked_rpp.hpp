#ifndef T_PARKING_SIM__DIRECTION_LOCKED_RPP_HPP_
#define T_PARKING_SIM__DIRECTION_LOCKED_RPP_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav2_core/goal_checker.hpp"
#include "nav2_regulated_pure_pursuit_controller/regulated_pure_pursuit_controller.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "t_parking_sim/terminal_capture.hpp"

namespace t_parking_sim
{

class DirectionLockedRPP
  : public nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController
{
public:
  DirectionLockedRPP() = default;
  ~DirectionLockedRPP() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & speed,
    nav2_core::GoalChecker * goal_checker) override;

  void reset() override;

private:
  struct ProgressUpdate
  {
    std::size_t previous_index{0};
    std::size_t local_index{0};
    std::size_t unrestricted_candidate{0};
    double lateral_error{0.0};
  };

  struct PathProjection
  {
    std::size_t segment_index{0};
    double ratio{0.0};
    double x{0.0};
    double y{0.0};
    double arc_length{0.0};
    double signed_lateral_error{0.0};
    double distance{0.0};
  };

  void segmentStateCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg);
  void resetTrackingState();
  void logTrackingSummary(const char * context);
  void diagnoseSelfProximity();
  void buildReverseSteeringProfile();
  void logReverseSteeringProfile() const;

  ProgressUpdate updateProgress(
    const geometry_msgs::msg::PoseStamped & robot_plan_pose);
  nav_msgs::msg::Path transformActivePlan(
    const geometry_msgs::msg::PoseStamped & robot_plan_pose,
    const builtin_interfaces::msg::Time & command_stamp);
  geometry_msgs::msg::PoseStamped getDirectionLockedLookAheadPoint(
    double lookahead_distance, const nav_msgs::msg::Path & transformed_plan) const;
  bool findDirectionLockedLookAheadPoint(
    double lookahead_distance, const nav_msgs::msg::Path & transformed_plan,
    geometry_msgs::msg::PoseStamped & carrot) const;
  PathProjection projectReversePath(
    const geometry_msgs::msg::PoseStamped & robot_plan_pose);
  geometry_msgs::msg::PoseStamped reverseArcTarget(
    double target_arc_length, const builtin_interfaces::msg::Time & stamp) const;
  double reverseControllerCurvature(double arc_length) const;
  double threePointCurvature(std::size_t index) const;
  double percentile(std::vector<double> values, double fraction) const;

  std::size_t forwardWindowEnd(std::size_t first, double distance) const;
  std::size_t closestPose(
    const geometry_msgs::msg::PoseStamped & robot_plan_pose,
    std::size_t first, std::size_t last) const;
  double pointToSegmentDistance(
    double px, double py, std::size_t first, std::size_t second) const;
  double calculateLateralError(
    const geometry_msgs::msg::PoseStamped & robot_plan_pose,
    std::size_t first, std::size_t last) const;
  double medianPathSpacing() const;
  int fullIndex(std::size_t local_index) const;

  nav_msgs::msg::Path segment_plan_;
  mutable std::mutex plan_mutex_;
  bool progress_initialized_{false};
  std::size_t last_progress_index_{0};
  double median_spacing_{0.0};
  double progress_search_distance_{0.0};
  std::vector<double> cumulative_arc_length_;
  std::vector<double> reverse_controller_curvature_;
  bool reverse_projection_initialized_{false};
  double last_projection_arc_length_{0.0};

  double reverse_wheel_base_{0.73};
  double reverse_hard_steering_limit_deg_{22.0};
  bool reverse_fault_on_steering_limit_{false};
  double reverse_profile_window_{0.25};
  double reverse_lateral_gain_{0.35};
  double reverse_heading_gain_{0.8};
  double reverse_lateral_correction_limit_deg_{3.0};
  double reverse_soft_limit_margin_deg_{1.5};
  double reverse_soft_limit_override_deg_{0.0};
  double reverse_soft_limit_deg_{22.0};
  double reverse_profile_min_curvature_{0.0};
  double reverse_profile_max_curvature_{0.0};
  double reverse_profile_min_steering_deg_{0.0};
  double reverse_profile_max_steering_deg_{0.0};
  double reverse_profile_mean_steering_deg_{0.0};
  double reverse_profile_median_steering_deg_{0.0};
  double reverse_profile_p90_steering_deg_{0.0};
  double reverse_profile_p95_steering_deg_{0.0};
  double terminal_capture_distance_{0.0};

  int locked_direction_{1};
  std::atomic<int> segment_number_{-1};
  std::atomic<int> metadata_direction_{0};
  std::atomic<int> full_start_index_{0};
  std::atomic<int> full_end_index_{0};
  std::string segment_state_topic_;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr segment_state_sub_;

  std::uint64_t rollback_candidate_count_{0};
  std::uint64_t accepted_rollback_count_{0};
  std::uint64_t direction_conflict_samples_{0};
  std::uint64_t direction_conflict_events_{0};
  bool direction_conflict_active_{false};
  std::uint64_t lateral_error_samples_{0};
  double lateral_error_sum_{0.0};
  double max_lateral_error_{0.0};
  std::uint64_t reverse_steering_samples_{0};
  std::uint64_t reverse_saturation_samples_{0};
  std::uint64_t reverse_saturation_events_{0};
  bool reverse_saturation_active_{false};
  bool terminal_goal_reached_logged_{false};
  rclcpp::Time last_track_log_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_terminal_log_{0, 0, RCL_ROS_TIME};
};

}  // namespace t_parking_sim

#endif  // T_PARKING_SIM__DIRECTION_LOCKED_RPP_HPP_
