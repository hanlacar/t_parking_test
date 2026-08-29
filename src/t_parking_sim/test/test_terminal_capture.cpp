#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "t_parking_sim/terminal_capture.hpp"

namespace t_parking_sim
{

TEST(TerminalCapture, IsInactiveOutsideTerminalRange)
{
  EXPECT_EQ(
    selectTerminalCaptureMode(0.351, 0.35, 1, 0.30, 0.30, 0.0, 0.03, 0.10),
    TerminalCaptureMode::kInactive);
}

TEST(TerminalCapture, DecodesGoalCheckerYawToleranceQuaternion)
{
  constexpr double tolerance = 0.16;
  EXPECT_NEAR(
    yawToleranceFromQuaternion(
      std::sin(0.5 * tolerance), std::cos(0.5 * tolerance)),
    tolerance, 1.0e-12);
}

TEST(TerminalCapture, CapturesForwardAndReverseEndpointsOnlyInLockedHalfPlane)
{
  EXPECT_EQ(
    selectTerminalCaptureMode(0.30, 0.35, 1, 0.25, 0.25, 0.02, 0.03, 0.10),
    TerminalCaptureMode::kCaptureEndpoint);
  EXPECT_EQ(
    selectTerminalCaptureMode(0.30, 0.35, -1, -0.25, 0.25, 0.02, 0.03, 0.10),
    TerminalCaptureMode::kCaptureEndpoint);
  EXPECT_FALSE(isInLockedMotionHalfPlane(1, -0.01));
  EXPECT_FALSE(isInLockedMotionHalfPlane(-1, 0.01));
}

TEST(TerminalCapture, ReturnsSafeZeroInsideStrictCuspTolerance)
{
  EXPECT_EQ(
    selectTerminalCaptureMode(0.02, 0.35, 1, -0.01, 0.029, 0.099, 0.03, 0.10),
    TerminalCaptureMode::kGoalToleranceZero);
}

TEST(TerminalCapture, AcceptsObservedBenchCuspResidualBeforeOvershoot)
{
  EXPECT_EQ(
    selectTerminalCaptureMode(
      0.009, 0.35, 1, -0.0026, 0.009, 0.145, 0.03, 0.16),
    TerminalCaptureMode::kGoalToleranceZero);
}

TEST(TerminalCapture, RejectsYawOutsideBenchCuspTolerance)
{
  EXPECT_EQ(
    selectTerminalCaptureMode(
      0.009, 0.35, 1, 0.0026, 0.009, 0.17, 0.03, 0.16),
    TerminalCaptureMode::kCaptureEndpoint);
}

TEST(TerminalCapture, RejectsPositionOutsideCuspTolerance)
{
  EXPECT_EQ(
    selectTerminalCaptureMode(
      0.031, 0.35, 1, 0.031, 0.031, 0.10, 0.03, 0.16),
    TerminalCaptureMode::kCaptureEndpoint);
}

TEST(TerminalCapture, RejectsBadEndpointOvershoot)
{
  EXPECT_EQ(
    selectTerminalCaptureMode(0.02, 0.35, 1, -0.04, 0.04, 0.02, 0.03, 0.10),
    TerminalCaptureMode::kOvershootAbort);
  EXPECT_EQ(
    selectTerminalCaptureMode(0.02, 0.35, -1, 0.04, 0.04, 0.02, 0.03, 0.10),
    TerminalCaptureMode::kOvershootAbort);
}

TEST(TerminalCapture, ProgressIndexNeverRollsBack)
{
  std::size_t index = 0;
  for (const std::size_t candidate : {1u, 3u, 5u, 4u, 8u, 7u, 10u}) {
    const std::size_t previous = index;
    index = monotonicPathIndex(index, candidate, 10u);
    EXPECT_GE(index, previous);
  }
  EXPECT_EQ(index, 10u);
}

TEST(TerminalCapture, VirtualForwardCuspReverseSequenceCompletesSafely)
{
  std::vector<std::string> events;

  for (const double endpoint_base_x : {0.30, 0.18, 0.08}) {
    EXPECT_EQ(
      selectTerminalCaptureMode(
        endpoint_base_x, 0.35, 1, endpoint_base_x,
        endpoint_base_x, 0.02, 0.03, 0.10),
      TerminalCaptureMode::kCaptureEndpoint);
  }
  EXPECT_EQ(
    selectTerminalCaptureMode(0.009, 0.35, 1, -0.0026, 0.009, 0.145, 0.03, 0.16),
    TerminalCaptureMode::kGoalToleranceZero);
  events.emplace_back("FORWARD_SUCCEEDED");

  for (int sample = 1; sample <= 3; ++sample) {
    events.emplace_back("CUSP_ZERO_" + std::to_string(sample));
  }
  events.emplace_back("ParkingReverse");

  for (const double endpoint_base_x : {-0.30, -0.18, -0.08}) {
    EXPECT_EQ(
      selectTerminalCaptureMode(
        std::abs(endpoint_base_x), 0.35, -1, endpoint_base_x,
        std::abs(endpoint_base_x), 0.02, 0.03, 0.10),
      TerminalCaptureMode::kCaptureEndpoint);
  }
  EXPECT_EQ(
    selectTerminalCaptureMode(0.02, 0.35, -1, 0.01, 0.02, 0.02, 0.03, 0.10),
    TerminalCaptureMode::kGoalToleranceZero);
  events.emplace_back("REVERSE_SUCCEEDED");
  events.emplace_back("FINAL_ZERO");

  const std::vector<std::string> expected{
    "FORWARD_SUCCEEDED", "CUSP_ZERO_1", "CUSP_ZERO_2", "CUSP_ZERO_3",
    "ParkingReverse", "REVERSE_SUCCEEDED", "FINAL_ZERO"};
  EXPECT_EQ(events, expected);
}

}  // namespace t_parking_sim
