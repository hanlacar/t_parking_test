#!/usr/bin/env python3
"""Convert the private final Nav2 Twist into the shared lidar MCU protocol."""

from dataclasses import dataclass
import math
import time
from typing import Optional

from bench_support import bench_motion_allowed
from geometry_msgs.msg import Twist
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String


@dataclass(frozen=True)
class ConvertedCommand:
    """One paired command plus the ROS-convention virtual steering angle."""

    drive_stage: float
    wheel_deg: int
    steering_deg_ros: float


ZERO_COMMAND = ConvertedCommand(0.0, 0, 0.0)
DEFAULT_PARKING_MODES = frozenset({'T_PARK', 'PARALLEL_PARK'})


def any_stop_active(*stop_requests: bool) -> bool:
    """Combine independent stop owners without letting one clear another."""
    return any(bool(request) for request in stop_requests)


def mode_allows_lidar_commands(
        vehicle_mode: str,
        parking_modes=DEFAULT_PARKING_MODES) -> bool:
    """Return whether the MCU has granted drive and wheel to LiDAR."""
    return str(vehicle_mode).strip().upper() in parking_modes


def select_output_command(
        command: ConvertedCommand,
        last_input_time: Optional[float],
        now: float,
        input_timeout_sec: float,
        emergency_stop: bool) -> ConvertedCommand:
    """Select a safe heartbeat command without changing production policy."""
    input_fresh = (
        last_input_time is not None
        and now - last_input_time <= input_timeout_sec)
    if emergency_stop or not input_fresh:
        return ZERO_COMMAND
    return command


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
        # Production keeps the request topic as its compatibility default.
        # BENCH overrides this with the MCU manager's applied-mode topic.
        self.declare_parameter('mode_topic', '/vehicle_mode')
        # The front motion detector owns this immediate 0.5 m ROI safety
        # result.  Keep it separate from the parking controller's stop so a
        # false message from either producer cannot clear the other one.
        self.declare_parameter(
            'lidar_safety_stop_topic', '/lidar/stop_required')
        self.declare_parameter(
            'parking_modes', ['T_PARK', 'PARALLEL_PARK'])
        # Send a brief paired zero on parking exit, then become silent so the
        # MCU's 0.5 s freshness timeout cannot be held by an idle heartbeat.
        self.declare_parameter('mode_exit_zero_duration_sec', 0.20)
        # Disabled by default so the production command converter retains its
        # exact existing behavior.  bench_t_parking.launch.py explicitly
        # enables this independent, fail-closed second motion gate.
        self.declare_parameter('bench_interlock_enabled', False)
        self.declare_parameter('bench_mode', False)
        self.declare_parameter('wheels_off_ground', False)
        self.declare_parameter('execute', False)

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
        self.mode_topic = str(self.get_parameter('mode_topic').value)
        self.lidar_safety_stop_topic = str(
            self.get_parameter('lidar_safety_stop_topic').value)
        self.parking_modes = {
            str(value).strip().upper()
            for value in self.get_parameter('parking_modes').value
            if str(value).strip()
        }
        self.mode_exit_zero_duration_sec = float(
            self.get_parameter('mode_exit_zero_duration_sec').value)
        self.bench_interlock_enabled = bool(
            self.get_parameter('bench_interlock_enabled').value)
        self.bench_mode = bool(self.get_parameter('bench_mode').value)
        self.wheels_off_ground = bool(
            self.get_parameter('wheels_off_ground').value)
        self.execute = bool(self.get_parameter('execute').value)
        self.bench_motion_allowed = bench_motion_allowed(
            self.bench_mode, self.wheels_off_ground, self.execute)

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
        if not self.mode_topic:
            raise ValueError('mode_topic must not be empty')
        if not self.lidar_safety_stop_topic:
            raise ValueError('lidar_safety_stop_topic must not be empty')
        if not self.parking_modes:
            raise ValueError('parking_modes must not be empty')
        if self.mode_exit_zero_duration_sec < 0.0:
            raise ValueError('mode_exit_zero_duration_sec must be >= 0')

        self.current_command = ZERO_COMMAND
        self.last_input_time: Optional[float] = None
        self.emergency_stop = False
        self.lidar_safety_stop = False
        self.current_mode = ''
        self.mode_exit_zero_until: Optional[float] = None
        self.stop_release_pending = False

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
        self.lidar_safety_stop_subscription = self.create_subscription(
            Bool, self.lidar_safety_stop_topic,
            self._lidar_safety_stop_callback, 10)
        self.mode_subscription = self.create_subscription(
            String, self.mode_topic, self._mode_callback, 10)
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
        self.get_logger().info(
            '[LIDAR_COMMAND] mode source=%s' % self.mode_topic)
        self.get_logger().info(
            '[LIDAR_COMMAND] silent outside vehicle modes: %s'
            % sorted(self.parking_modes))
        self.get_logger().info(
            '[LIDAR_SAFETY] %s -> /lidar_stop (all vehicle modes)'
            % self.lidar_safety_stop_topic)
        if self.bench_interlock_enabled and not self.bench_motion_allowed:
            self.get_logger().warning(
                'BENCH MOTION INHIBITED: bench_mode and wheels_off_ground '
                'must both be true; converter output is locked to zero')
        elif self.bench_interlock_enabled:
            self.get_logger().info(
                '[BENCH STARTUP] no fresh Twist means zero heartbeat; '
                'motion still requires a later FollowPath command')

    def _cmd_vel_callback(self, msg: Twist) -> None:
        linear_x = float(msg.linear.x)
        angular_z = float(msg.angular.z)
        converted = convert_command(
            linear_x=linear_x,
            angular_z=angular_z,
            wheel_base=self.wheel_base,
            steering_limit_deg=self.steering_limit_deg,
            mcu_wheel_limit_deg=self.mcu_wheel_limit_deg,
            stopped_speed_epsilon=self.stopped_speed_epsilon,
            forward_drive_stage=self.forward_drive_stage,
            reverse_drive_stage=self.reverse_drive_stage,
        )
        if self.bench_interlock_enabled and not self.bench_motion_allowed:
            self.current_command = ZERO_COMMAND
            if abs(linear_x) >= self.stopped_speed_epsilon:
                self.get_logger().error(
                    'BENCH MOTION INHIBITED: bench_mode and '
                    'wheels_off_ground must both be true',
                    throttle_duration_sec=2.0,
                )
        else:
            self.current_command = converted

        if not math.isfinite(linear_x) or not math.isfinite(angular_z):
            self.get_logger().warning(
                '[CMD_SPLITTER] non-finite Twist rejected; publishing stop/center',
                throttle_duration_sec=2.0,
            )

        self.last_input_time = time.monotonic()

    def _emergency_stop_callback(self, msg: Bool) -> None:
        previous = self._stop_active()
        self.emergency_stop = bool(msg.data)
        self._record_stop_transition(previous)

    def _lidar_safety_stop_callback(self, msg: Bool) -> None:
        previous = self._stop_active()
        self.lidar_safety_stop = bool(msg.data)
        self._record_stop_transition(previous)

    def _stop_active(self) -> bool:
        return any_stop_active(self.emergency_stop, self.lidar_safety_stop)

    def _record_stop_transition(self, previous: bool) -> None:
        if previous and not self._stop_active():
            # /lidar_stop is global in mcu_manager. Publish one explicit false
            # to release this node's request even while drive/wheel are silent.
            self.stop_release_pending = True

    def _mode_callback(self, msg: String) -> None:
        previous_mode = self.current_mode
        previous_active = mode_allows_lidar_commands(
            previous_mode, self.parking_modes)
        new_mode = str(msg.data).strip().upper()
        new_active = mode_allows_lidar_commands(new_mode, self.parking_modes)
        self.current_mode = new_mode

        if new_mode != previous_mode:
            self.get_logger().info(
                '[LIDAR_COMMAND] current mode received: %s'
                % (new_mode or 'UNKNOWN'))

        if previous_active and not new_active:
            self.current_command = ZERO_COMMAND
            self.last_input_time = None
            self.mode_exit_zero_until = (
                time.monotonic() + self.mode_exit_zero_duration_sec)
            self.get_logger().info(
                '[LIDAR_COMMAND] vehicle mode %s: brief zero then silent'
                % (new_mode or 'UNKNOWN'))
        elif not previous_active and new_active:
            self.mode_exit_zero_until = None
            if self.bench_interlock_enabled:
                # BENCH startup must establish an explicit zero heartbeat
                # before any later Nav2 command can be accepted.
                self.current_command = ZERO_COMMAND
                self.last_input_time = None
                self.get_logger().info(
                    '[BENCH STARTUP] zero heartbeat enabled for %s'
                    % new_mode)
            else:
                self.get_logger().info(
                    '[LIDAR_COMMAND] vehicle mode %s: command output enabled'
                    % new_mode)

    def _publish_stop(self, active: bool) -> None:
        self.stop_publisher.publish(Bool(data=bool(active)))

    def _publish_tick(self) -> None:
        now = time.monotonic()
        stop_active = self._stop_active()
        parking_active = mode_allows_lidar_commands(
            self.current_mode, self.parking_modes)
        exit_zero_active = (
            self.mode_exit_zero_until is not None
            and now <= self.mode_exit_zero_until)

        if not parking_active and not exit_zero_active:
            self.mode_exit_zero_until = None
            # Preserve the MCU's global /lidar_stop safety semantics without
            # keeping drive/wheel fresh in NORMAL.
            if stop_active:
                self._publish_stop(True)
            elif self.stop_release_pending:
                self._publish_stop(False)
                self.stop_release_pending = False
            return

        input_fresh = (
            self.last_input_time is not None
            and now - self.last_input_time <= self.input_timeout_sec)
        if parking_active:
            command = select_output_command(
                self.current_command,
                self.last_input_time,
                now,
                self.input_timeout_sec,
                stop_active,
            )
        else:
            command = ZERO_COMMAND

        drive_msg = Float32()
        drive_msg.data = command.drive_stage
        wheel_msg = Int32()
        wheel_msg.data = command.wheel_deg
        # Publish steering first.  Both the Gazebo bridge and mcu_manager run
        # periodic output loops, so they observe the newly paired values on
        # their next tick rather than a new drive value with old steering.
        self.wheel_publisher.publish(wheel_msg)
        self.drive_publisher.publish(drive_msg)
        self._publish_stop(stop_active)
        if not stop_active:
            self.stop_release_pending = False

        self.get_logger().debug(
            '[CMD_SPLITTER] steering=%+.2fdeg drive_out=%+.1f '
            'wheel_out=%+d stop=%s fresh=%s'
            % (
                command.steering_deg_ros,
                command.drive_stage,
                command.wheel_deg,
                stop_active,
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
