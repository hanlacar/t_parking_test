#ifndef T_PARKING_SIM__TERMINAL_CAPTURE_HPP_
#define T_PARKING_SIM__TERMINAL_CAPTURE_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace t_parking_sim
{

enum class TerminalCaptureMode
{
  kInactive,
  kCaptureEndpoint,
  kGoalToleranceZero,
  kOvershootAbort,
};

inline bool isInLockedMotionHalfPlane(int locked_direction, double base_x)
{
  return static_cast<double>(locked_direction) * base_x > 0.0;
}

inline double yawToleranceFromQuaternion(double z, double w)
{
  return std::abs(2.0 * std::atan2(z, w));
}

inline TerminalCaptureMode selectTerminalCaptureMode(
  double remaining_arc_length, double terminal_capture_distance,
  int locked_direction, double endpoint_base_x,
  double position_error, double yaw_error,
  double xy_goal_tolerance, double yaw_goal_tolerance)
{
  if (terminal_capture_distance <= 0.0 ||
    remaining_arc_length > terminal_capture_distance)
  {
    return TerminalCaptureMode::kInactive;
  }

  const bool tolerances_available =
    xy_goal_tolerance >= 0.0 && yaw_goal_tolerance >= 0.0;
  if (tolerances_available && position_error <= xy_goal_tolerance &&
    std::abs(yaw_error) <= yaw_goal_tolerance)
  {
    return TerminalCaptureMode::kGoalToleranceZero;
  }
  if (isInLockedMotionHalfPlane(locked_direction, endpoint_base_x)) {
    return TerminalCaptureMode::kCaptureEndpoint;
  }
  return TerminalCaptureMode::kOvershootAbort;
}

inline const char * terminalCaptureModeName(TerminalCaptureMode mode)
{
  switch (mode) {
    case TerminalCaptureMode::kInactive:
      return "INACTIVE";
    case TerminalCaptureMode::kCaptureEndpoint:
      return "CAPTURE_ENDPOINT";
    case TerminalCaptureMode::kGoalToleranceZero:
      return "GOAL_TOLERANCE_ZERO";
    case TerminalCaptureMode::kOvershootAbort:
      return "CUSP_ENDPOINT_OVERSHOOT";
  }
  return "UNKNOWN";
}

inline std::size_t monotonicPathIndex(
  std::size_t previous, std::size_t candidate, std::size_t last)
{
  return std::min(std::max(previous, candidate), last);
}

}  // namespace t_parking_sim

#endif  // T_PARKING_SIM__TERMINAL_CAPTURE_HPP_
