#!/usr/bin/env python3
"""Bridge the T-parking lidar command topics to T870 FIELD serial commands."""

import math
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
import serial
from std_msgs.msg import Bool, Float32, Int32, String
from std_srvs.srv import Trigger


FIELD_DRIVE_COMMANDS = {
    -1.0: b'b\n',
    0.0: b'0',
    1.0: b'w\n',
    2.0: b'e\n',
    3.0: b'r\n',
}
MECHANICAL_STEERING_MAX_DEG = 22
COMMAND_STEERING_MAX_DEG = 22


def encode_drive_command(value: float) -> bytes:
    """Encode one exact FIELD drive stage, rejecting every other value."""
    value = float(value)
    if not math.isfinite(value) or value not in FIELD_DRIVE_COMMANDS:
        raise ValueError(f'invalid FIELD drive stage: {value!r}')
    return FIELD_DRIVE_COMMANDS[value]


def encode_wheel_command(
        value: int, limit_deg: int = COMMAND_STEERING_MAX_DEG) -> bytes:
    """Encode signed wheel degrees without hiding an over-limit request."""
    value = int(value)
    if abs(value) > limit_deg:
        raise ValueError(
            f'FIELD steering request {value:+d}deg exceeds +/-{limit_deg}deg')
    if value < 0:
        return f'L{-value}\n'.encode('ascii')
    if value > 0:
        return f'R{value}\n'.encode('ascii')
    return b'C\n'


def command_inputs_fresh(
        now: float,
        drive_received_at: Optional[float],
        wheel_received_at: Optional[float],
        stop_received_at: Optional[float],
        timeout_sec: float) -> bool:
    """Require a fresh complete three-topic command set."""
    received = (drive_received_at, wheel_received_at, stop_received_at)
    return all(
        stamp is not None and 0.0 <= now - stamp <= timeout_sec
        for stamp in received)


class T870FieldSerialBridge(Node):
    """Fail-closed 115200-baud adapter for T870_FIELD_v9_0904 firmware."""

    def __init__(self) -> None:
        super().__init__('t870_field_serial_bridge')
        self.declare_parameter('serial_port', '/dev/t870_mcu')
        self.declare_parameter('serial_baud', 115200)
        self.declare_parameter('watchdog_timeout_sec', 0.50)
        self.declare_parameter('output_frequency', 10.0)
        self.declare_parameter(
            'steering_limit_deg', COMMAND_STEERING_MAX_DEG)
        self.declare_parameter('shutdown_stop_repeats', 5)

        self.serial_port = str(self.get_parameter('serial_port').value)
        self.serial_baud = int(self.get_parameter('serial_baud').value)
        self.watchdog_timeout_sec = float(
            self.get_parameter('watchdog_timeout_sec').value)
        self.output_frequency = float(
            self.get_parameter('output_frequency').value)
        self.steering_limit_deg = int(
            self.get_parameter('steering_limit_deg').value)
        self.shutdown_stop_repeats = int(
            self.get_parameter('shutdown_stop_repeats').value)

        if not self.serial_port:
            raise ValueError('serial_port must not be empty')
        if self.serial_baud != 115200:
            raise ValueError('T870 FIELD firmware requires serial_baud=115200')
        if self.watchdog_timeout_sec <= 0.0:
            raise ValueError('watchdog_timeout_sec must be > 0')
        if self.output_frequency < 4.0:
            raise ValueError('output_frequency must be at least 4 Hz')
        if self.steering_limit_deg != COMMAND_STEERING_MAX_DEG:
            raise ValueError(
                'T870 FIELD software command limit is 22 degrees')
        if self.shutdown_stop_repeats < 1:
            raise ValueError('shutdown_stop_repeats must be >= 1')

        self._lock = threading.Lock()
        self._serial: Optional[serial.Serial] = None
        self._closing = False
        self._fault_latched = False
        self._drive = 0.0
        self._wheel = 0
        self._stop = True
        self._drive_received_at: Optional[float] = None
        self._wheel_received_at: Optional[float] = None
        self._stop_received_at: Optional[float] = None
        self._last_output_state = ''

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._ready_publisher = self.create_publisher(
            Bool, '/t870_field_bridge/ready', status_qos)
        self._fault_publisher = self.create_publisher(
            String, '/t870_field_bridge/fault', status_qos)
        self._emergency_publisher = self.create_publisher(
            Bool, '/t_parking/emergency_stop_request', 10)
        self._cancel_client = self.create_client(
            Trigger, '/t_parking/cancel')

        self.create_subscription(
            Float32, '/lidar_drive', self._drive_callback, 10)
        self.create_subscription(
            Int32, '/lidar_wheel', self._wheel_callback, 10)
        self.create_subscription(
            Bool, '/lidar_stop', self._stop_callback, 10)

        try:
            self._serial = serial.Serial(
                port=self.serial_port,
                baudrate=self.serial_baud,
                timeout=0.0,
                write_timeout=0.20,
            )
        except (OSError, serial.SerialException) as exc:
            self._publish_ready(False)
            raise RuntimeError(
                f'cannot open FIELD serial port {self.serial_port}: {exc}') from exc

        self._best_effort_stop(repeats=3)
        if self._fault_latched:
            try:
                self._serial.close()
            except (OSError, serial.SerialException):
                pass
            raise RuntimeError(
                f'FIELD serial port {self.serial_port} did not accept the '
                'startup stop sequence')
        self._publish_ready(True)
        self._timer = self.create_timer(
            1.0 / self.output_frequency, self._output_tick)
        self.get_logger().info(
            '[FIELD BRIDGE] ready: port=%s baud=%d watchdog=%.2fs '
            'command_limit=+/-%ddeg mechanical_max~+/-%ddeg'
            % (
                self.serial_port, self.serial_baud,
                self.watchdog_timeout_sec, self.steering_limit_deg,
                MECHANICAL_STEERING_MAX_DEG))

    def _publish_ready(self, ready: bool) -> None:
        message = Bool()
        message.data = ready
        self._ready_publisher.publish(message)

    def _publish_fault(self, reason: str) -> None:
        message = String()
        message.data = reason
        self._fault_publisher.publish(message)

    def _write(self, payload: bytes) -> bool:
        try:
            if self._serial is None or not self._serial.is_open:
                raise serial.SerialException('serial port is not open')
            written = self._serial.write(payload)
            self._serial.flush()
            if written != len(payload):
                raise serial.SerialTimeoutException(
                    f'short write: {written}/{len(payload)} bytes')
            return True
        except (OSError, serial.SerialException) as exc:
            first_fault = not self._fault_latched
            if not self._closing and first_fault:
                self._latch_fault(f'serial write failed: {exc}', send_stop=False)
                # A transient write fault may still leave the link usable.
                # Try every stop write even if an earlier attempt failed.
                self._best_effort_stop(repeats=3)
            return False

    def _best_effort_stop(self, repeats: Optional[int] = None) -> None:
        count = self.shutdown_stop_repeats if repeats is None else repeats
        for _ in range(max(1, count)):
            self._write(b'0')

    def _request_parking_cancel(self) -> None:
        emergency = Bool()
        emergency.data = True
        self._emergency_publisher.publish(emergency)
        if self._cancel_client.service_is_ready():
            self._cancel_client.call_async(Trigger.Request())

    def _latch_fault(self, reason: str, send_stop: bool = True) -> None:
        if self._fault_latched:
            return
        self._fault_latched = True
        if send_stop:
            self._best_effort_stop(repeats=3)
        self._publish_ready(False)
        self._publish_fault(reason)
        self._request_parking_cancel()
        self.get_logger().error(f'[FIELD BRIDGE FAULT] {reason}; serial drive stopped')

    def _drive_callback(self, message: Float32) -> None:
        value = float(message.data)
        try:
            encode_drive_command(value)
        except ValueError as exc:
            self._latch_fault(str(exc))
            return
        with self._lock:
            self._drive = value
            self._drive_received_at = time.monotonic()

    def _wheel_callback(self, message: Int32) -> None:
        value = int(message.data)
        try:
            encode_wheel_command(value, self.steering_limit_deg)
        except ValueError as exc:
            self._latch_fault(str(exc))
            return
        with self._lock:
            self._wheel = value
            self._wheel_received_at = time.monotonic()

    def _stop_callback(self, message: Bool) -> None:
        stop = bool(message.data)
        with self._lock:
            self._stop = stop
            self._stop_received_at = time.monotonic()
        if stop:
            self._best_effort_stop(repeats=1)

    def _output_tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            drive = self._drive
            wheel = self._wheel
            stop = self._stop
            fresh = command_inputs_fresh(
                now,
                self._drive_received_at,
                self._wheel_received_at,
                self._stop_received_at,
                self.watchdog_timeout_sec,
            )

        if self._fault_latched:
            self._best_effort_stop(repeats=1)
            state = 'FAULT_STOP'
        elif stop:
            self._best_effort_stop(repeats=1)
            state = 'STOP_REQUEST'
        elif not fresh:
            self._best_effort_stop(repeats=1)
            state = 'WATCHDOG_STOP'
        else:
            try:
                wheel_command = encode_wheel_command(
                    wheel, self.steering_limit_deg)
                drive_command = encode_drive_command(drive)
            except ValueError as exc:
                self._latch_fault(str(exc))
                return
            if not self._write(wheel_command):
                return
            if not self._write(drive_command):
                return
            state = f'DRIVE={drive:+.0f} WHEEL={wheel:+d}'

        if state != self._last_output_state:
            if state == 'WATCHDOG_STOP':
                self.get_logger().warning(
                    '[FIELD BRIDGE WATCHDOG] stale command; serial drive stopped')
            else:
                self.get_logger().info(f'[FIELD BRIDGE] {state}')
            self._last_output_state = state

    def shutdown_serial(self) -> None:
        """Send repeated immediate stops before closing the serial device."""
        if self._closing:
            return
        self._closing = True
        self._best_effort_stop()
        self._publish_ready(False)
        if self._serial is not None:
            try:
                self._serial.close()
            except (OSError, serial.SerialException):
                pass
        self.get_logger().warning(
            '[FIELD BRIDGE] shutdown final: repeated serial 0 sent')


def main(args=None) -> None:
    rclpy.init(
        args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = T870FieldSerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_serial()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
