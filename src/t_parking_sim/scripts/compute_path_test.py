#!/usr/bin/env python3
"""Request a Nav2 path without commanding vehicle motion."""

import math
import sys

from builtin_interfaces.msg import Time
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


ERROR_NAMES = {
    ComputePathToPose.Result.NONE: 'NONE',
    ComputePathToPose.Result.UNKNOWN: 'UNKNOWN',
    ComputePathToPose.Result.INVALID_PLANNER: 'INVALID_PLANNER',
    ComputePathToPose.Result.TF_ERROR: 'TF_ERROR',
    ComputePathToPose.Result.START_OUTSIDE_MAP: 'START_OUTSIDE_MAP',
    ComputePathToPose.Result.GOAL_OUTSIDE_MAP: 'GOAL_OUTSIDE_MAP',
    ComputePathToPose.Result.START_OCCUPIED: 'START_OCCUPIED',
    ComputePathToPose.Result.GOAL_OCCUPIED: 'GOAL_OCCUPIED',
    ComputePathToPose.Result.TIMEOUT: 'TIMEOUT',
    ComputePathToPose.Result.NO_VALID_PATH: 'NO_VALID_PATH',
}


def yaw_from_pose(pose) -> float:
    """Extract planar yaw from a pose quaternion."""
    quaternion = pose.orientation
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(sin_yaw, cos_yaw)


class ComputePathTest(Node):
    """Send one ComputePathToPose goal and retain the resulting Path."""

    def __init__(self) -> None:
        super().__init__('compute_path_test')
        self.declare_parameter('goal_x', 3.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_yaw', 0.0)
        self.declare_parameter('planner_id', 'GridBased')
        self.declare_parameter('action_timeout', 30.0)

        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.goal_yaw = float(self.get_parameter('goal_yaw').value)
        self.planner_id = str(self.get_parameter('planner_id').value)
        self.action_timeout = float(self.get_parameter('action_timeout').value)

        path_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.path_publisher = self.create_publisher(Path, '/planned_path', path_qos)
        self.action_client = ActionClient(
            self, ComputePathToPose, '/compute_path_to_pose'
        )
        self.path = None
        self.republish_timer = None

    def request_path(self) -> bool:
        """Request and publish a path; return whether planning succeeded."""
        self.get_logger().info(
            'Waiting for /compute_path_to_pose '
            f'(timeout={self.action_timeout:.1f}s)'
        )
        if not self.action_client.wait_for_server(timeout_sec=self.action_timeout):
            self.get_logger().error('/compute_path_to_pose action server unavailable')
            return False

        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = 'map'
        # Zero time asks TF for the latest transform and avoids wall/sim-time mismatch.
        goal.goal.header.stamp = Time()
        goal.goal.pose.position.x = self.goal_x
        goal.goal.pose.position.y = self.goal_y
        goal.goal.pose.orientation.z = math.sin(self.goal_yaw * 0.5)
        goal.goal.pose.orientation.w = math.cos(self.goal_yaw * 0.5)
        goal.planner_id = self.planner_id
        goal.use_start = False

        self.get_logger().info(
            f'Goal: frame=map x={self.goal_x:.3f} y={self.goal_y:.3f} '
            f'yaw={self.goal_yaw:.3f}; planner_id={self.planner_id}; '
            'use_start=false'
        )

        goal_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('ComputePathToPose goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            self.get_logger().error('ComputePathToPose returned no action result')
            return False

        result = wrapped_result.result
        planning_time = result.planning_time.sec + result.planning_time.nanosec * 1e-9
        if result.error_code != ComputePathToPose.Result.NONE:
            error_name = ERROR_NAMES.get(result.error_code, 'UNRECOGNIZED')
            self.get_logger().error(
                f'Path planning failed: error_code={result.error_code} '
                f'({error_name}), error_msg={result.error_msg!r}, '
                f'planning_time={planning_time:.6f}s'
            )
            return False

        if not result.path.poses:
            self.get_logger().error(
                'Planner reported success but returned an empty Path'
            )
            return False

        self.path = result.path
        self.path_publisher.publish(self.path)
        first = self.path.poses[0].pose.position
        last = self.path.poses[-1].pose.position
        self.get_logger().info(
            f'Path planning succeeded: poses={len(self.path.poses)}, '
            f'planning_time={planning_time:.6f}s'
        )
        self.get_logger().info(
            f'Path start: x={first.x:.3f} y={first.y:.3f}; '
            f'Path end: x={last.x:.3f} y={last.y:.3f}'
        )
        self._log_path_geometry()
        self.get_logger().info(
            'Published /planned_path; keeping the node alive for late RViz '
            'subscribers'
        )
        self.republish_timer = self.create_timer(1.0, self._republish_path)
        return True

    def _log_path_geometry(self) -> None:
        """Report bounds and discrete curvature for repeatable safety checks."""
        poses = self.path.poses
        x_values = [item.pose.position.x for item in poses]
        y_values = [item.pose.position.y for item in poses]
        path_length = 0.0
        max_curvature = 0.0
        reverse_segments = 0

        for previous, current in zip(poses, poses[1:]):
            dx = current.pose.position.x - previous.pose.position.x
            dy = current.pose.position.y - previous.pose.position.y
            segment_length = math.hypot(dx, dy)
            if segment_length <= 1e-6:
                continue

            path_length += segment_length
            previous_yaw = yaw_from_pose(previous.pose)
            if dx * math.cos(previous_yaw) + dy * math.sin(previous_yaw) < 0.0:
                reverse_segments += 1

        # Smac headings are angle-bin quantized, so estimate the path geometry
        # from each three-position circumcircle instead of differentiating yaw.
        for first, middle, last in zip(poses, poses[1:], poses[2:]):
            first_point = first.pose.position
            middle_point = middle.pose.position
            last_point = last.pose.position
            side_a = math.hypot(
                middle_point.x - first_point.x,
                middle_point.y - first_point.y,
            )
            side_b = math.hypot(
                last_point.x - middle_point.x,
                last_point.y - middle_point.y,
            )
            side_c = math.hypot(
                last_point.x - first_point.x,
                last_point.y - first_point.y,
            )
            denominator = side_a * side_b * side_c
            if denominator <= 1e-9:
                continue
            twice_area = abs(
                (middle_point.x - first_point.x)
                * (last_point.y - first_point.y)
                - (middle_point.y - first_point.y)
                * (last_point.x - first_point.x)
            )
            max_curvature = max(
                max_curvature, 2.0 * twice_area / denominator
            )

        self.get_logger().info(
            f'Path bounds: x=[{min(x_values):.3f}, {max(x_values):.3f}] '
            f'y=[{min(y_values):.3f}, {max(y_values):.3f}]'
        )
        self.get_logger().info(
            f'Path geometry: length={path_length:.3f}m, '
            f'max_position_curvature={max_curvature:.3f} 1/m, '
            f'reverse_segments={reverse_segments}'
        )

    def _republish_path(self) -> None:
        if self.path is not None:
            self.path_publisher.publish(self.path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ComputePathTest()
    succeeded = node.request_path()
    try:
        if succeeded:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if not succeeded:
        sys.exit(1)


if __name__ == '__main__':
    main()
