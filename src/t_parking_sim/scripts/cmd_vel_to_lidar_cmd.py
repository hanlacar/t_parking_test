#!/usr/bin/env python3
"""Convert the private final Nav2 Twist into the shared lidar MCU protocol."""

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
class ConvertedCommand:
    """One paired command plus the ROS-convention virtual steering angle."""

    drive_stage: float
    wheel_deg: int
    steering_deg_ros: float


def convert_command(
        linear_x: float,
        angular_z: float,
        wheel_base: float,
        steering_limit_deg: float,
        mcu_wheel_limit_deg: int,
        stopped_speed_epsilon: float,
        forward_drive_stage: float,
        reverse_drive_stage: float) -> ConvertedCommand:
    """
    Convert Twist values to the established mcu_manager lidar contract.

    ROS and Gazebo use positive yaw/steering for a left turn.  The existing
    MCU protocol is the opposite: negative wheel degrees mean left.  Keeping
    ``linear_x`` in the denominator also gives the correct virtual steering
    sign while reversing.
    """
    if not math.isfinite(linear_x) or not math.isfinite(angular_z):
        return ConvertedCommand(0.0, 0, 0.0)

    if abs(linear_x) < stopped_speed_epsilon:
        return ConvertedCommand(0.0, 0, 0.0)

    steering_rad = math.atan(wheel_base * angular_z / linear_x)
    steering_deg_ros = math.degrees(steering_rad)
    steering_deg_ros = max(
        -steering_limit_deg,
        min(steering_limit_deg, steering_deg_ros),
    )

    # Existing /lidar_wheel -> /mcu_wheel contract: -left, +right, degrees.
    wheel_deg = int(round(-steering_deg_ros))
    wheel_deg = max(-mcu_wheel_limit_deg, min(mcu_wheel_limit_deg, wheel_deg))
    drive_stage = forward_drive_stage if linear_x > 0.0 else reverse_drive_stage
    return ConvertedCommand(drive_stage, wheel_deg, steering_deg_ros)


class CmdVelToLidarCmd(Node):
    """Publish the final three-topic lidar vehicle command at a fixed rate."""

    def __init__(self) -> None:
        super().__init__('cmd_vel_to_lidar_cmd')

        # Vehicle geometry is supplied by auto_t_parking.launch.py after it
        # reads the canonical xacro properties.  Invalid defaults make an
        # accidental standalone launch fail instead of silently guessing.
        self.declare_parameter('wheel_base', -1.0)
        self.declare_parameter('steering_limit_deg', -1.0)
        # Keep /cmd_vel as the standalone/parallel compatibility default.
        # auto_t_parking.launch.py overrides this with a private topic.
        self.declare_parameter('input_topic', '/cmd_vel')

        # These values are the existing mcu_manager/mcu_bridge protocol.
        self.declare_parameter('mcu_wheel_limit_deg', 27)
        self.declare_parameter('stopped_speed_epsilon', 0.01)
        self.declare_parameter('forward_drive_stage', 1.0)
        self.declare_parameter('reverse_drive_stage', -1.0)
        self.declare_parameter('output_frequency', 10.0)
        self.declare_parameter('input_timeout_sec', 0.50)

        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.steering_limit_deg = float(
            self.get_parameter('steering_limit_deg').value)
        self.mcu_wheel_limit_deg = int(
            self.get_parameter('mcu_wheel_limit_deg').value)
        self.stopped_speed_epsilon = float(
            self.get_parameter('stopped_speed_epsilon').value)
        self.forward_drive_stage = float(
            self.get_parameter('forward_drive_stage').value)
        self.reverse_drive_stage = float(
            self.get_parameter('reverse_drive_stage').value)
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_frequency = float(
            self.get_parameter('output_frequency').value)
        self.input_timeout_sec = float(
            self.get_parameter('input_timeout_sec').value)

        if self.wheel_base <= 0.0:
            raise ValueError('wheel_base must be supplied from the vehicle xacro')
        if self.steering_limit_deg <= 0.0:
            raise ValueError(
                'steering_limit_deg must be supplied from the vehicle xacro')
        if self.mcu_wheel_limit_deg <= 0:
            raise ValueError('mcu_wheel_limit_deg must be > 0')
        if self.stopped_speed_epsilon <= 0.0:
            raise ValueError('stopped_speed_epsilon must be > 0')
        if not self.input_topic:
            raise ValueError('input_topic must not be empty')
        if self.forward_drive_stage not in (1.0, 2.0, 3.0):
            raise ValueError('forward_drive_stage must be 1.0, 2.0, or 3.0')
        if self.reverse_drive_stage != -1.0:
            raise ValueError('reverse_drive_stage must be -1.0')
        if self.output_frequency < 4.0:
            raise ValueError('output_frequency must be at least 4 Hz')
        if self.input_timeout_sec <= 0.0:
            raise ValueError('input_timeout_sec must be > 0')

        self.current_command = ConvertedCommand(0.0, 0, 0.0)
        self.last_input_time: Optional[float] = None
        self.emergency_stop = False

        self.drive_publisher = self.create_publisher(
            Float32, '/lidar_drive', 10)
        self.wheel_publisher = self.create_publisher(
            Int32, '/lidar_wheel', 10)
        self.stop_publisher = self.create_publisher(
            Bool, '/lidar_stop', 10)
        self.cmd_subscription = self.create_subscription(
            Twist, self.input_topic, self._cmd_vel_callback, 10)
        self.stop_subscription = self.create_subscription(
            Bool, '/t_parking/emergency_stop_request',
            self._emergency_stop_callback, 10)
        self.output_timer = self.create_timer(
            1.0 / self.output_frequency,
            self._publish_tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME))

        self.get_logger().info(
            '[LIDAR_COMMAND] %s -> '
            '/lidar_drive + /lidar_wheel + /lidar_stop'
            % self.input_topic)
        self.get_logger().info(
            '[CMD_SPLITTER] geometry: wheel_base=%.3fm, '
            'vehicle_limit=%.2fdeg, MCU_limit=%ddeg'
            % (
                self.wheel_base,
                self.steering_limit_deg,
                self.mcu_wheel_limit_deg,
            ))
        self.get_logger().info(
            '[LIDAR_COMMAND] fixed output rate=%.1fHz, input timeout=%.2fs'
            % (self.output_frequency, self.input_timeout_sec))

    def _cmd_vel_callback(self, msg: Twist) -> None:
        linear_x = float(msg.linear.x)
        angular_z = float(msg.angular.z)
        self.current_command = convert_command(
            linear_x=linear_x,
            angular_z=angular_z,
            wheel_base=self.wheel_base,
            steering_limit_deg=self.steering_limit_deg,
            mcu_wheel_limit_deg=self.mcu_wheel_limit_deg,
            stopped_speed_epsilon=self.stopped_speed_epsilon,
            forward_drive_stage=self.forward_drive_stage,
            reverse_drive_stage=self.reverse_drive_stage,
        )

        if not math.isfinite(linear_x) or not math.isfinite(angular_z):
            self.get_logger().warning(
                '[CMD_SPLITTER] non-finite Twist rejected; publishing stop/center',
                throttle_duration_sec=2.0,
            )

        self.last_input_time = time.monotonic()

    def _emergency_stop_callback(self, msg: Bool) -> None:
        self.emergency_stop = bool(msg.data)

    def _publish_tick(self) -> None:
        command = self.current_command
        input_fresh = (
            self.last_input_time is not None
            and time.monotonic() - self.last_input_time
            <= self.input_timeout_sec)
        if self.emergency_stop:
            command = ConvertedCommand(0.0, 0, 0.0)
        elif not input_fresh:
            command = ConvertedCommand(0.0, 0, 0.0)

        drive_msg = Float32()
        drive_msg.data = command.drive_stage
        wheel_msg = Int32()
        wheel_msg.data = command.wheel_deg
        stop_msg = Bool()
        stop_msg.data = self.emergency_stop
        # Publish steering first.  Both the Gazebo bridge and mcu_manager run
        # periodic output loops, so they observe the newly paired values on
        # their next tick rather than a new drive value with old steering.
        self.wheel_publisher.publish(wheel_msg)
        self.drive_publisher.publish(drive_msg)
        self.stop_publisher.publish(stop_msg)

        self.get_logger().debug(
            '[CMD_SPLITTER] steering=%+.2fdeg drive_out=%+.1f '
            'wheel_out=%+d stop=%s fresh=%s'
            % (
                command.steering_deg_ros,
                command.drive_stage,
                command.wheel_deg,
                self.emergency_stop,
                input_fresh,
            ))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelToLidarCmd()
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
