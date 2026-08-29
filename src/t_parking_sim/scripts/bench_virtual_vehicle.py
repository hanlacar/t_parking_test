#!/usr/bin/env python3
"""Publish isolated BENCH odometry by integrating the real Nav2 Twist."""

import math
import time
from typing import Optional

from bench_support import (
    BENCH_BASE_FRAME,
    bench_motion_allowed,
    BENCH_ODOM_FRAME,
    BENCH_ODOM_TOPIC,
    integrate_bicycle,
    Pose2D,
)
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


class BenchVirtualVehicle(Node):
    """Close the BENCH Nav2 pose loop without touching production odometry."""

    def __init__(self) -> None:
        super().__init__('bench_virtual_vehicle')
        self.declare_parameter('bench_mode', False)
        self.declare_parameter('wheels_off_ground', False)
        self.declare_parameter('execute', False)
        self.declare_parameter('input_topic', '/t_parking/cmd_vel_control')
        self.declare_parameter('odom_topic', BENCH_ODOM_TOPIC)
        self.declare_parameter('odom_frame', BENCH_ODOM_FRAME)
        self.declare_parameter('base_frame', BENCH_BASE_FRAME)
        self.declare_parameter('wheel_base', 0.77)
        self.declare_parameter('steering_limit_deg', 27.0)
        self.declare_parameter('stopped_speed_epsilon', 0.01)
        self.declare_parameter('input_timeout_sec', 0.50)
        self.declare_parameter('publish_frequency', 50.0)

        parameter = self.get_parameter
        self.bench_mode = bool(parameter('bench_mode').value)
        self.wheels_off_ground = bool(parameter('wheels_off_ground').value)
        self.execute = bool(parameter('execute').value)
        self.motion_allowed = bench_motion_allowed(
            self.bench_mode, self.wheels_off_ground, self.execute)
        self.input_topic = str(parameter('input_topic').value)
        self.odom_topic = str(parameter('odom_topic').value)
        self.odom_frame = str(parameter('odom_frame').value)
        self.base_frame = str(parameter('base_frame').value)
        self.wheel_base = float(parameter('wheel_base').value)
        self.steering_limit_deg = float(
            parameter('steering_limit_deg').value)
        self.stopped_speed_epsilon = float(
            parameter('stopped_speed_epsilon').value)
        self.input_timeout_sec = float(parameter('input_timeout_sec').value)
        self.publish_frequency = float(parameter('publish_frequency').value)
        if self.wheel_base <= 0.0 or self.steering_limit_deg <= 0.0:
            raise ValueError('BENCH virtual vehicle geometry must be positive')
        if self.input_timeout_sec <= 0.0 or self.publish_frequency < 10.0:
            raise ValueError('invalid BENCH virtual vehicle timing')
        if self.odom_frame in ('odom', 'map') or self.base_frame == 'base_link':
            raise ValueError('BENCH frames must not reuse production TF names')

        self.pose = Pose2D(0.0, 0.0, 0.0)
        self.command = Twist()
        self.last_command_time: Optional[float] = None
        self.last_tick_time = time.monotonic()
        self.last_log_time = 0.0
        self.path_progress = 0.0
        self.estop = False
        self.real_odom_start: Optional[Pose2D] = None
        self.real_odom_latest: Optional[Pose2D] = None

        self.odom_publisher = self.create_publisher(
            Odometry, self.odom_topic, 20)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Twist, self.input_topic, self._cmd_callback, 20)
        self.create_subscription(Bool, '/estop_lock', self._estop_callback, 10)
        self.create_subscription(
            Bool, '/t_parking/emergency_stop_request',
            self._estop_callback, 10)
        self.create_subscription(Odometry, '/odom', self._real_odom_callback, 10)
        self.timer = self.create_timer(
            1.0 / self.publish_frequency,
            self._tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

        self.get_logger().warning(
            '[BENCH VIRTUAL VEHICLE] physical chassis remains stationary; '
            'only isolated bench_odom -> bench_base_link moves')
        if not self.motion_allowed:
            self.get_logger().warning('BENCH VIRTUAL MOTION INHIBITED')

    @staticmethod
    def _odom_pose(msg: Odometry) -> Pose2D:
        quaternion = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (quaternion.w * quaternion.z
                   + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y
                         + quaternion.z * quaternion.z),
        )
        return Pose2D(
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            yaw,
        )

    def _cmd_callback(self, msg: Twist) -> None:
        self.command = msg
        self.last_command_time = time.monotonic()

    def _estop_callback(self, msg: Bool) -> None:
        self.estop = bool(msg.data)

    def _real_odom_callback(self, msg: Odometry) -> None:
        pose = self._odom_pose(msg)
        if self.real_odom_start is None:
            self.real_odom_start = pose
        self.real_odom_latest = pose

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - self.last_tick_time))
        self.last_tick_time = now
        command_fresh = (
            self.last_command_time is not None
            and now - self.last_command_time <= self.input_timeout_sec)
        command = self.command if command_fresh and not self.estop else Twist()
        step = integrate_bicycle(
            self.pose,
            float(command.linear.x),
            float(command.angular.z),
            dt,
            wheel_base=self.wheel_base,
            steering_limit_deg=self.steering_limit_deg,
            stopped_speed_epsilon=self.stopped_speed_epsilon,
            motion_allowed=self.motion_allowed,
        )
        self.pose = step.pose
        self.path_progress += abs(step.linear_velocity) * dt
        self._publish(step.linear_velocity, step.yaw_rate)

        if now - self.last_log_time >= 0.5:
            self.last_log_time = now
            real_delta = 0.0
            if self.real_odom_start is not None and self.real_odom_latest is not None:
                real_delta = math.hypot(
                    self.real_odom_latest.x - self.real_odom_start.x,
                    self.real_odom_latest.y - self.real_odom_start.y,
                )
            self.get_logger().info(
                '[BENCH VIRTUAL VEHICLE]\n'
                f'cmd_v={float(command.linear.x):+.4f}\n'
                f'cmd_w={float(command.angular.z):+.4f}\n'
                f'raw_delta_deg={step.raw_delta_deg:+.3f}\n'
                f'clamped_delta_deg={step.clamped_delta_deg:+.3f}\n'
                f'virtual_x={self.pose.x:+.4f}\n'
                f'virtual_y={self.pose.y:+.4f}\n'
                f'virtual_yaw={self.pose.yaw:+.4f}\n'
                f'path_progress={self.path_progress:.4f}\n'
                f'real_odom_delta={real_delta:.4f}\n'
                f'virtual_odom_delta={math.hypot(self.pose.x, self.pose.y):.4f}')

    def _publish(self, linear_velocity: float, yaw_rate: float) -> None:
        stamp = self.get_clock().now().to_msg()
        sine = math.sin(0.5 * self.pose.yaw)
        cosine = math.cos(0.5 * self.pose.yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.pose.x
        odom.pose.pose.position.y = self.pose.y
        odom.pose.pose.orientation.z = sine
        odom.pose.pose.orientation.w = cosine
        odom.twist.twist.linear.x = linear_velocity
        odom.twist.twist.angular.z = yaw_rate
        self.odom_publisher.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = self.pose.x
        transform.transform.translation.y = self.pose.y
        transform.transform.rotation.z = sine
        transform.transform.rotation.w = cosine
        self.tf_broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BenchVirtualVehicle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
