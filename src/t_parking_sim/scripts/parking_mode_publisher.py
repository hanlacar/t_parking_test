#!/usr/bin/env python3
"""
Own the simulated vehicle mode for one parking state machine.

The real vehicle's mission/mode manager owns the applied mode. Gazebo has no
such manager, so the parking launch owns this lifecycle adapter instead: it
grants the parking command bridge access while a run is active and restores
the non-parking mode after every terminal result.
"""

from typing import FrozenSet

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_msgs.msg import String


TERMINAL_STATUSES: FrozenSet[str] = frozenset({
    'FINISHED', 'PARKING_SUCCESS', 'SUCCESS', 'FAILED', 'CANCELLED',
})


def mode_for_status(
        status_message: str, active_mode: str, restore_mode: str) -> str:
    """Return the requested mode for a parking status message."""
    status = str(status_message).partition(':')[0].strip().upper()
    if not status or status == 'WAITING' or status in TERMINAL_STATUSES:
        return restore_mode
    return active_mode


class ParkingModePublisher(Node):
    """Publish a parking mode only for the lifetime of an active run."""

    def __init__(self) -> None:
        """Initialize parameters, lifecycle subscription, and mode output."""
        super().__init__('parking_mode_publisher')
        self.declare_parameter('status_topic', '/t_parking/status')
        self.declare_parameter('mode_topic', '/vehicle_mode')
        self.declare_parameter('active_mode', 'T_PARK')
        self.declare_parameter('restore_mode', 'NORMAL')
        self.declare_parameter('publish_hz', 2.0)

        status_topic = str(self.get_parameter('status_topic').value).strip()
        mode_topic = str(self.get_parameter('mode_topic').value).strip()
        self.active_mode = str(
            self.get_parameter('active_mode').value).strip().upper()
        self.restore_mode = str(
            self.get_parameter('restore_mode').value).strip().upper()
        frequency = float(self.get_parameter('publish_hz').value)
        if not status_topic or not mode_topic:
            raise ValueError('status_topic and mode_topic must not be empty')
        if not self.active_mode or not self.restore_mode:
            raise ValueError('active_mode and restore_mode must not be empty')
        if frequency <= 0.0:
            raise ValueError('publish_hz must be > 0')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(String, mode_topic, qos)
        self.subscription = self.create_subscription(
            String, status_topic, self._status_callback, qos)
        self.current_mode = self.restore_mode
        self.timer = self.create_timer(1.0 / frequency, self._publish_mode)
        self._publish_mode()
        self.get_logger().info(
            'parking mode lifecycle: %s=%s while active; restore=%s; status=%s'
            % (mode_topic, self.active_mode, self.restore_mode, status_topic))

    def _status_callback(self, message: String) -> None:
        requested = mode_for_status(
            message.data, self.active_mode, self.restore_mode)
        if requested != self.current_mode:
            self.get_logger().info(
                'vehicle mode transition: %s -> %s (parking status=%s)'
                % (self.current_mode, requested, message.data))
            self.current_mode = requested
            self._publish_mode()

    def _publish_mode(self) -> None:
        self.publisher.publish(String(data=self.current_mode))


def main(args=None) -> None:
    """Run the Gazebo parking-mode lifecycle adapter."""
    rclpy.init(args=args)
    node = ParkingModePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.current_mode = node.restore_mode
            node._publish_mode()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
