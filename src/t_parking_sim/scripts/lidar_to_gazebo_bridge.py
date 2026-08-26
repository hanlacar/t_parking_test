#!/usr/bin/env python3
"""Translate the shared lidar/MCU command contract to Gazebo Ackermann Twist."""

from dataclasses import dataclass
import math
import time
from typing import Optional

from geometry_msgs.msg import Twist
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32


@dataclass(frozen=True)
class GazeboCommand:
    """A converted Gazebo command and the clamped ROS steering angle."""

    linear_x: float
    angular_z: float
    steering_deg_ros: float


def convert_lidar_command(
        drive_stage: float,
        wheel_deg_mcu: int,
        drive_scale_mps: float,
        wheel_base: float,
        wheel_limit_deg: int) -> GazeboCommand:
    """
    Convert MCU convention (-left/+right) with the bicycle model.

    The Gazebo Ackermann plugin consumes a Twist whose angular.z is yaw rate,
    not a steering angle.  ROS positive yaw is left, opposite to the MCU wheel
    sign, hence ``delta_ros = -wheel_deg_mcu``.
    """
    if not math.isfinite(drive_stage):
        return GazeboCommand(0.0, 0.0, 0.0)

    clamped_wheel = max(
        -wheel_limit_deg, min(wheel_limit_deg, int(wheel_deg_mcu)))
    linear_x = drive_stage * drive_scale_mps
    if abs(linear_x) <= 1.0e-12:
        return GazeboCommand(0.0, 0.0, 0.0)

    steering_deg_ros = -float(clamped_wheel)
    steering_rad_ros = math.radians(steering_deg_ros)
    angular_z = linear_x * math.tan(steering_rad_ros) / wheel_base
    return GazeboCommand(linear_x, angular_z, steering_deg_ros)


class LidarToGazeboBridge(Node):
    """Sole ROS publisher of the Gazebo vehicle's /cmd_vel command."""

    def __init__(self) -> None:
        super().__init__('lidar_to_gazebo_bridge')
        self.declare_parameter('drive_scale_mps', 0.30)
        self.declare_parameter('wheel_base', -1.0)
        self.declare_parameter('wheel_limit_deg', 27)
        self.declare_parameter('command_timeout_sec', 0.50)
        self.declare_parameter('publish_frequency', 20.0)

        self.drive_scale_mps = float(
            self.get_parameter('drive_scale_mps').value)
        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.wheel_limit_deg = int(
            self.get_parameter('wheel_limit_deg').value)
        self.command_timeout_sec = float(
            self.get_parameter('command_timeout_sec').value)
        publish_frequency = float(
            self.get_parameter('publish_frequency').value)

        if self.drive_scale_mps <= 0.0:
            raise ValueError('drive_scale_mps must be > 0')
        if self.wheel_base <= 0.0:
            raise ValueError('wheel_base must be supplied from the vehicle xacro')
        if self.wheel_limit_deg <= 0:
            raise ValueError('wheel_limit_deg must be > 0')
        if self.command_timeout_sec <= 0.0:
            raise ValueError('command_timeout_sec must be > 0')
        if publish_frequency <= 0.0:
            raise ValueError('publish_frequency must be > 0')

        self.drive_stage: Optional[float] = None
        self.wheel_deg: Optional[int] = None
        self.stop_active: Optional[bool] = None
        self.last_drive_time: Optional[float] = None
        self.last_wheel_time: Optional[float] = None
        self.last_stop_time: Optional[float] = None
        self.watchdog_stopped = True

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            Float32, '/lidar_drive', self._drive_callback, 10)
        self.create_subscription(
            Int32, '/lidar_wheel', self._wheel_callback, 10)
        self.create_subscription(
            Bool, '/lidar_stop', self._stop_callback, 10)
        self.timer = self.create_timer(
            1.0 / publish_frequency,
            self._publish_tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME))

        self.get_logger().info('Gazebo lidar command bridge started')
        self.get_logger().info(
            'drive scale: 1.0 -> %.2f m/s' % self.drive_scale_mps)
        self.get_logger().info(
            'wheel range: -%d ~ +%d deg'
            % (self.wheel_limit_deg, self.wheel_limit_deg))
        self.get_logger().info('wheelbase: %.3f m' % self.wheel_base)
        self.get_logger().info(
            'wheel sign: negative=left, positive=right')
        self.get_logger().info(
            'watchdog: drive, wheel, and stop must update within %.2f s'
            % self.command_timeout_sec)

    def _drive_callback(self, msg: Float32) -> None:
        value = float(msg.data)
        if not math.isfinite(value):
            self.get_logger().warning(
                'non-finite /lidar_drive rejected; commanding stop',
                throttle_duration_sec=2.0)
            value = 0.0
        self.drive_stage = value
        self.last_drive_time = time.monotonic()

    def _wheel_callback(self, msg: Int32) -> None:
        raw = int(msg.data)
        clamped = max(
            -self.wheel_limit_deg, min(self.wheel_limit_deg, raw))
        if clamped != raw:
            self.get_logger().warning(
                '/lidar_wheel clamped: %d -> %d' % (raw, clamped),
                throttle_duration_sec=1.0)
        self.wheel_deg = clamped
        self.last_wheel_time = time.monotonic()

    def _stop_callback(self, msg: Bool) -> None:
        self.stop_active = bool(msg.data)
        self.last_stop_time = time.monotonic()

    def _commands_fresh(self, now: float) -> bool:
        return (
            self.drive_stage is not None
            and self.wheel_deg is not None
            and self.stop_active is not None
            and self.last_drive_time is not None
            and self.last_wheel_time is not None
            and self.last_stop_time is not None
            and now - self.last_drive_time <= self.command_timeout_sec
            and now - self.last_wheel_time <= self.command_timeout_sec
            and now - self.last_stop_time <= self.command_timeout_sec)

    def _publish_tick(self) -> None:
        now = time.monotonic()
        msg = Twist()
        if self._commands_fresh(now) and not self.stop_active:
            command = convert_lidar_command(
                drive_stage=self.drive_stage,
                wheel_deg_mcu=self.wheel_deg,
                drive_scale_mps=self.drive_scale_mps,
                wheel_base=self.wheel_base,
                wheel_limit_deg=self.wheel_limit_deg,
            )
            msg.linear.x = command.linear_x
            msg.angular.z = command.angular_z
            self.watchdog_stopped = False
        else:
            if not self.watchdog_stopped:
                self.get_logger().warning(
                    'lidar stop active or command watchdog timeout; '
                    'publishing /cmd_vel = 0')
            self.watchdog_stopped = True
        self.publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarToGazeboBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Best effort immediate stop before the publisher disappears.
        if rclpy.ok():
            node.publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
