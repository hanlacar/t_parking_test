#!/usr/bin/env python3
"""Animate the existing T-parking Path without producing drive commands."""

import math
from typing import List, Sequence

from geometry_msgs.msg import (
    Pose,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_pose(pose: Pose) -> float:
    quaternion = pose.orientation
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z
               + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y
                     + quaternion.z * quaternion.z),
    )


def set_pose_yaw(pose: Pose, yaw: float) -> None:
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = math.sin(0.5 * yaw)
    pose.orientation.w = math.cos(0.5 * yaw)


class RvizFakeVehicle(Node):
    """Publish odometry and TF while interpolating an already-planned Path."""

    def __init__(self) -> None:
        super().__init__('rviz_fake_vehicle')
        self.declare_parameter('initial_pose_x', 0.0)
        self.declare_parameter('initial_pose_y', 0.0)
        self.declare_parameter('initial_pose_yaw', 0.0)
        self.declare_parameter('forward_speed_mps', 0.30)
        self.declare_parameter('reverse_speed_mps', 0.20)
        self.declare_parameter('update_rate_hz', 20.0)
        self.declare_parameter('cusp_pause_sec', 0.5)
        self.declare_parameter('exit_goal.x', 0.0)
        self.declare_parameter('exit_goal.y', 0.0)
        self.declare_parameter('exit_goal.yaw', 0.0)
        self.declare_parameter('exit_goal.position_tolerance', 0.05)
        self.declare_parameter('exit_goal.yaw_tolerance', 0.05)
        self.declare_parameter('wheel_base', 0.73)
        self.declare_parameter('wheel_radius', 0.135)

        self.x = float(self.get_parameter('initial_pose_x').value)
        self.y = float(self.get_parameter('initial_pose_y').value)
        self.yaw = float(self.get_parameter('initial_pose_yaw').value)
        self.forward_speed = max(
            0.01, float(self.get_parameter('forward_speed_mps').value))
        self.reverse_speed = max(
            0.01, float(self.get_parameter('reverse_speed_mps').value))
        self.update_rate = max(
            1.0, float(self.get_parameter('update_rate_hz').value))
        self.cusp_pause = max(
            0.0, float(self.get_parameter('cusp_pause_sec').value))
        self.exit_goal_x = float(self.get_parameter('exit_goal.x').value)
        self.exit_goal_y = float(self.get_parameter('exit_goal.y').value)
        self.exit_goal_yaw = float(self.get_parameter('exit_goal.yaw').value)
        self.exit_goal_position_tolerance = max(
            0.0, float(
                self.get_parameter('exit_goal.position_tolerance').value))
        self.exit_goal_yaw_tolerance = max(
            0.0, float(
                self.get_parameter('exit_goal.yaw_tolerance').value))
        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.wheel_radius = float(self.get_parameter('wheel_radius').value)

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.odom_publisher = self.create_publisher(Odometry, '/odom', 10)
        self.joint_publisher = self.create_publisher(
            JointState, '/joint_states', 10)
        self.status_publisher = self.create_publisher(
            String, '/t_parking/fake_vehicle_status', transient_qos)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            Path, '/t_parking/planned_path', self._path_callback,
            transient_qos)
        self.create_subscription(
            Path, '/t_parking/forward_exit_path', self._exit_path_callback,
            transient_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose',
            self._initial_pose_callback, 10)

        self.poses: List[Pose] = []
        self.directions: List[int] = []
        self.edge_index = 0
        self.edge_progress = 0.0
        self.pause_remaining = 0.0
        self.playback_kind = 'idle'
        self.exit_poses: List[Pose] = []
        self.exit_directions: List[int] = []
        self.exit_pending = False
        self.finish_pending = False
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.wheel_rotation = 0.0

        self._publish_status('WAITING_FOR_PATH')
        self._publish_state()
        self.timer = self.create_timer(1.0 / self.update_rate, self._timer)
        self.get_logger().info(
            'RViz-only fake vehicle ready: publishes /odom, TF, and '
            '/joint_states only; no drive-command publisher is created')

    def _publish_status(self, value: str) -> None:
        message = String()
        message.data = value
        self.status_publisher.publish(message)

    @staticmethod
    def _edge_directions(poses: Sequence[Pose]) -> List[int]:
        raw: List[int] = []
        lengths: List[float] = []
        for first, second in zip(poses, poses[1:]):
            dx = second.position.x - first.position.x
            dy = second.position.y - first.position.y
            length = math.hypot(dx, dy)
            lengths.append(length)
            if length < 1.0e-6:
                raw.append(0)
                continue
            score = dx * math.cos(yaw_from_pose(first)) + dy * math.sin(
                yaw_from_pose(first))
            raw.append(0 if abs(score) < 1.0e-4 else (1 if score > 0.0 else -1))
        first_nonzero = next((value for value in raw if value), 1)
        previous = first_nonzero
        for index, value in enumerate(raw):
            if value == 0:
                raw[index] = previous
            else:
                previous = value

        nonzero_lengths = sorted(value for value in lengths if value > 1.0e-6)
        median = nonzero_lengths[len(nonzero_lengths) // 2] if nonzero_lengths else 0.0
        confirmation_length = max(0.10, 3.0 * median)
        while raw:
            runs = []
            start = 0
            for index in range(1, len(raw) + 1):
                if index < len(raw) and raw[index] == raw[start]:
                    continue
                runs.append((start, index, raw[start]))
                start = index
            merged = False
            for run_index, (first, last, _direction) in enumerate(runs):
                if last - first >= 3 and sum(lengths[first:last]) >= confirmation_length:
                    continue
                if (0 < run_index < len(runs) - 1
                        and runs[run_index - 1][2] == runs[run_index + 1][2]):
                    raw[first:last] = [runs[run_index - 1][2]] * (last - first)
                    merged = True
                    break
            if not merged:
                break
        return raw

    def _path_callback(self, message: Path) -> None:
        if len(message.poses) < 2:
            self.get_logger().warning('ignoring a parking Path with fewer than two poses')
            return
        poses = [item.pose for item in message.poses]
        directions = self._edge_directions(poses)
        if len(directions) != len(poses) - 1:
            self.get_logger().error('could not classify parking Path directions')
            return

        self.exit_poses = []
        self.exit_directions = []

        self._start_playback(poses, directions, 'parking')
        forward_edges = sum(value > 0 for value in directions)
        reverse_edges = sum(value < 0 for value in directions)
        cusps = sum(a != b for a, b in zip(directions, directions[1:]))
        self.get_logger().info(
            f'parking Path accepted: poses={len(poses)} '
            f'forward_edges={forward_edges} reverse_edges={reverse_edges} '
            f'cusps={cusps}; exit goal=({self.exit_goal_x:.3f}, '
            f'{self.exit_goal_y:.3f}, {self.exit_goal_yaw:.3f})')

    def _exit_path_callback(self, message: Path) -> None:
        if len(message.poses) < 2:
            self.get_logger().error(
                'ignoring ForwardExit Path with fewer than two poses')
            self._publish_status('EXIT_PLAN_FAILED')
            return
        poses = [item.pose for item in message.poses]
        directions = self._edge_directions(poses)
        if not directions or any(value < 0 for value in directions):
            self.get_logger().error(
                'ForwardExit Path is not a continuous forward-only path')
            self._publish_status('EXIT_PLAN_FAILED')
            return

        endpoint = poses[-1]
        position_error = math.hypot(
            endpoint.position.x - self.exit_goal_x,
            endpoint.position.y - self.exit_goal_y)
        yaw_error = abs(normalize_angle(
            yaw_from_pose(endpoint) - self.exit_goal_yaw))
        if (position_error > self.exit_goal_position_tolerance
                or yaw_error > self.exit_goal_yaw_tolerance):
            self.get_logger().error(
                'ForwardExit endpoint does not satisfy configured exit_goal: '
                f'position_error={position_error:.4f} '
                f'yaw_error={yaw_error:.4f}')
            self._publish_status('EXIT_PLAN_FAILED')
            return

        self.exit_poses = poses
        self.exit_directions = directions
        total_length = sum(
            math.hypot(second.position.x - first.position.x,
                       second.position.y - first.position.y)
            for first, second in zip(poses, poses[1:]))
        cusps = sum(a != b for a, b in zip(directions, directions[1:]))
        self.get_logger().info(
            f'planner ForwardExit Path accepted: poses={len(poses)} '
            f'length={total_length:.3f}m cusps={cusps} '
            f'endpoint_position_error={position_error:.4f} '
            f'endpoint_yaw_error={yaw_error:.4f}')

    def _start_playback(
            self, poses: Sequence[Pose], directions: Sequence[int],
            kind: str) -> None:
        self.poses = list(poses)
        self.directions = list(directions)
        self.edge_index = 0
        self.edge_progress = 0.0
        self.pause_remaining = 0.0
        self.playback_kind = kind
        self.exit_pending = False
        self.finish_pending = False
        self._publish_status(f'PLAYING_{kind.upper()}')

    def _initial_pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        frame = message.header.frame_id.lstrip('/')
        if frame and frame not in ('map', 'odom'):
            self.get_logger().warning(
                f'ignoring /initialpose in unsupported frame {frame!r}')
            return
        self.x = message.pose.pose.position.x
        self.y = message.pose.pose.position.y
        self.yaw = yaw_from_pose(message.pose.pose)
        self.poses = []
        self.directions = []
        self.exit_poses = []
        self.exit_directions = []
        self.exit_pending = False
        self.finish_pending = False
        self.playback_kind = 'idle'
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self._publish_status('POSE_RESET')
        self._publish_state()
        self.get_logger().info(
            f'2D Pose Estimate applied: x={self.x:.3f} y={self.y:.3f} '
            f'yaw={self.yaw:.3f}')

    def _timer(self) -> None:
        dt = 1.0 / self.update_rate
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        if self.pause_remaining > 0.0:
            self.pause_remaining = max(0.0, self.pause_remaining - dt)
            if self.pause_remaining == 0.0 and self.exit_pending:
                self._begin_exit()
            elif self.pause_remaining == 0.0 and self.finish_pending:
                self.finish_pending = False
                self.playback_kind = 'idle'
                self._publish_status('FINISHED')
            self._publish_state()
            return

        if self.edge_index < len(self.directions):
            self._advance_edge(dt)
        self._publish_state()

    def _advance_edge(self, dt: float) -> None:
        first = self.poses[self.edge_index]
        second = self.poses[self.edge_index + 1]
        dx = second.position.x - first.position.x
        dy = second.position.y - first.position.y
        length = math.hypot(dx, dy)
        direction = self.directions[self.edge_index]
        speed = self.forward_speed if direction > 0 else self.reverse_speed
        if length < 1.0e-6:
            self.edge_progress = length
        else:
            self.edge_progress = min(length, self.edge_progress + speed * dt)
        ratio = 1.0 if length < 1.0e-6 else self.edge_progress / length
        first_yaw = yaw_from_pose(first)
        yaw_delta = normalize_angle(yaw_from_pose(second) - first_yaw)
        new_yaw = normalize_angle(first_yaw + ratio * yaw_delta)
        self.x = first.position.x + ratio * dx
        self.y = first.position.y + ratio * dy
        self.angular_velocity = normalize_angle(new_yaw - self.yaw) / dt
        self.yaw = new_yaw
        self.linear_velocity = speed * direction

        if ratio < 1.0:
            return
        previous_direction = direction
        self.edge_index += 1
        self.edge_progress = 0.0
        if self.edge_index < len(self.directions):
            if self.directions[self.edge_index] != previous_direction:
                self.pause_remaining = self.cusp_pause
                self.linear_velocity = 0.0
                self.angular_velocity = 0.0
                self._publish_status('STOPPED_AT_CUSP')
            return

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        if self.playback_kind == 'parking':
            self.exit_pending = True
            self.pause_remaining = self.cusp_pause
            self._publish_status('PARKED_PAUSE')
            return
        if self.playback_kind == 'exit':
            position_error = math.hypot(
                self.x - self.exit_goal_x, self.y - self.exit_goal_y)
            yaw_error = abs(normalize_angle(
                self.yaw - self.exit_goal_yaw))
            if (position_error > self.exit_goal_position_tolerance
                    or yaw_error > self.exit_goal_yaw_tolerance):
                self.get_logger().error(
                    'actual return failed: '
                    f'final=({self.x:.4f}, {self.y:.4f}, {self.yaw:.4f}) '
                    f'position_error={position_error:.4f} '
                    f'yaw_error={yaw_error:.4f}')
                self.playback_kind = 'idle'
                self._publish_status('RETURN_VALIDATION_FAILED')
                return
            self.get_logger().info(
                'EXIT_GOAL_REACHED: '
                f'final=({self.x:.4f}, {self.y:.4f}, {self.yaw:.4f}) '
                f'position_error={position_error:.4f} '
                f'yaw_error={yaw_error:.4f}; vehicle faces east')
            self.finish_pending = True
            self.pause_remaining = self.cusp_pause
            self._publish_status('RETURNED_TO_START')

    def _begin_exit(self) -> None:
        if (len(self.exit_poses) < 2
                or len(self.exit_directions) != len(self.exit_poses) - 1):
            self.get_logger().error(
                'parking completed but no valid planner ForwardExit Path exists')
            self.playback_kind = 'idle'
            self.exit_pending = False
            self._publish_status('EXIT_PLAN_FAILED')
            return
        start = self.exit_poses[0]
        join_position_error = math.hypot(
            start.position.x - self.x, start.position.y - self.y)
        join_yaw_error = abs(normalize_angle(yaw_from_pose(start) - self.yaw))
        if join_position_error > 0.05 or join_yaw_error > 0.05:
            self.get_logger().error(
                'ForwardExit Path does not join the actual parked pose: '
                f'position_error={join_position_error:.4f} '
                f'yaw_error={join_yaw_error:.4f}')
            self.playback_kind = 'idle'
            self.exit_pending = False
            self._publish_status('EXIT_PLAN_FAILED')
            return
        self._start_playback(
            self.exit_poses, self.exit_directions, 'exit')

    def _publish_state(self) -> None:
        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        set_pose_yaw(odom.pose.pose, self.yaw)
        odom.twist.twist.linear.x = self.linear_velocity
        odom.twist.twist.angular.z = self.angular_velocity
        self.odom_publisher.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_footprint'
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

        steering = 0.0
        if abs(self.linear_velocity) > 1.0e-3:
            steering = math.atan(
                self.wheel_base * self.angular_velocity / self.linear_velocity)
            steering = max(-math.radians(22.0), min(math.radians(22.0), steering))
        self.wheel_rotation += (
            self.linear_velocity / max(1.0e-3, self.wheel_radius)
            / self.update_rate)
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = [
            'front_left_steering_joint', 'front_right_steering_joint',
            'front_left_wheel_joint', 'front_right_wheel_joint',
            'rear_left_wheel_joint', 'rear_right_wheel_joint',
        ]
        joints.position = [
            steering, steering, self.wheel_rotation, self.wheel_rotation,
            self.wheel_rotation, self.wheel_rotation,
        ]
        self.joint_publisher.publish(joints)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RvizFakeVehicle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # ros2 launch may deliver a second SIGINT while rclpy is tearing down.
        # Treat it as the same requested shutdown instead of surfacing a
        # traceback and an exit-code -2 for this hardware-free helper.
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
