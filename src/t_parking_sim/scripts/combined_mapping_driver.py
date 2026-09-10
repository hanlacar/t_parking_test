#!/usr/bin/env python3
"""Drive one continuous, obstacle-free survey route for combined mapping."""

import math
from typing import List, Optional, Tuple

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, String


def yaw_from_quaternion(quaternion) -> float:
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(sin_yaw, cos_yaw)


class CombinedMappingDriver(Node):
    """Follow a smooth odom-frame route without resetting odom or SLAM."""

    def __init__(self) -> None:
        super().__init__('combined_mapping_driver')
        self.declare_parameter('linear_speed', 0.35)
        self.declare_parameter('lookahead_distance', 0.60)
        self.declare_parameter('wheel_base', 0.73)
        self.declare_parameter('steering_limit_deg', 22.0)
        self.declare_parameter('start_delay', 5.0)
        self.declare_parameter('goal_tolerance', 0.20)

        self.odom: Optional[Odometry] = None
        self.map_received = False
        self.vehicle_ready = False
        self.ready_time = None
        self.progress_index = 0
        self.finished = False
        self.path = self._build_route()

        self.command_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_publisher = self.create_publisher(
            String, '/combined_mapping/status', status_qos)
        self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos_profile_sensor_data)
        self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, status_qos)
        self.create_subscription(
            Bool, '/parking_practice/ready', self._ready_callback, status_qos)
        self.timer = self.create_timer(0.05, self._control)
        self._publish_status('WAITING_FOR_ODOM_AND_MAP')

    @staticmethod
    def _line(
            start: Tuple[float, float], end: Tuple[float, float],
            spacing: float = 0.20) -> List[Tuple[float, float]]:
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        count = max(1, int(math.ceil(distance / spacing)))
        return [
            (
                start[0] + (end[0] - start[0]) * index / count,
                start[1] + (end[1] - start[1]) * index / count,
            )
            for index in range(count)
        ]

    @staticmethod
    def _arc(
            center: Tuple[float, float], radius: float,
            start_angle: float, end_angle: float,
            spacing: float = 0.12) -> List[Tuple[float, float]]:
        count = max(
            1, int(math.ceil(abs(end_angle - start_angle) * radius / spacing)))
        return [
            (
                center[0] + radius * math.cos(
                    start_angle + (end_angle - start_angle) * index / count),
                center[1] + radius * math.sin(
                    start_angle + (end_angle - start_angle) * index / count),
            )
            for index in range(count)
        ]

    def _build_route(self) -> List[Tuple[float, float]]:
        # full_course spawn is world (-5, 0, yaw=0), so this odom route maps
        # to world by world_x=odom_x-5 and world_y=odom_y.  It observes the T
        # bay, crosses the x=[7,8] connector, surveys the vertical leg, then
        # traverses the complete upper lane and both parallel slots.
        # 1.60 m arcs stay above the measured 1.511 m physical minimum while
        # preserving the connector and upper-lane centre lines.
        route = self._line((0.0, 0.0), (13.15, 0.0))
        route.extend(self._arc((13.15, 1.60), 1.60, -math.pi / 2.0, 0.0))
        route.extend(self._line((14.75, 1.60), (14.75, 8.65)))
        route.extend(self._arc((13.15, 8.65), 1.60, 0.0, math.pi / 2.0))
        route.extend(self._line((13.15, 10.25), (0.50, 10.25)))
        route.append((0.50, 10.25))
        return route

    def _odom_callback(self, message: Odometry) -> None:
        self.odom = message

    def _map_callback(self, _message: OccupancyGrid) -> None:
        self.map_received = True

    def _ready_callback(self, message: Bool) -> None:
        if message.data and not self.vehicle_ready:
            self.vehicle_ready = True
            self.ready_time = self.get_clock().now()
            self._publish_status('READY_DELAY')

    def _publish_status(self, value: str) -> None:
        self.status_publisher.publish(String(data=value))
        self.get_logger().info(value)

    def _stop(self) -> None:
        self.command_publisher.publish(Twist())

    def _control(self) -> None:
        if self.finished:
            self._stop()
            return
        if (self.odom is None or not self.map_received
                or not self.vehicle_ready or self.ready_time is None):
            return
        elapsed = (
            self.get_clock().now() - self.ready_time).nanoseconds * 1.0e-9
        if elapsed < float(self.get_parameter('start_delay').value):
            return

        pose = self.odom.pose.pose
        x = pose.position.x
        y = pose.position.y
        yaw = yaw_from_quaternion(pose.orientation)
        search_end = min(len(self.path), self.progress_index + 50)
        nearest = min(
            range(self.progress_index, search_end),
            key=lambda index: math.hypot(
                self.path[index][0] - x, self.path[index][1] - y))
        self.progress_index = max(self.progress_index, nearest)

        goal_x, goal_y = self.path[-1]
        tolerance = float(self.get_parameter('goal_tolerance').value)
        if (self.progress_index >= len(self.path) - 3
                and math.hypot(goal_x - x, goal_y - y) <= tolerance):
            self.finished = True
            self._stop()
            self._publish_status('SURVEY_COMPLETE')
            return

        lookahead = float(self.get_parameter('lookahead_distance').value)
        target_index = self.progress_index
        accumulated = 0.0
        while target_index + 1 < len(self.path) and accumulated < lookahead:
            accumulated += math.hypot(
                self.path[target_index + 1][0] - self.path[target_index][0],
                self.path[target_index + 1][1] - self.path[target_index][1])
            target_index += 1
        target_x, target_y = self.path[target_index]
        dx = target_x - x
        dy = target_y - y
        base_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        base_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        distance_squared = max(base_x * base_x + base_y * base_y, 1.0e-6)
        curvature = 2.0 * base_y / distance_squared
        wheel_base = float(self.get_parameter('wheel_base').value)
        steering_limit = math.radians(float(
            self.get_parameter('steering_limit_deg').value))
        maximum_curvature = math.tan(steering_limit) / wheel_base
        curvature = max(-maximum_curvature, min(maximum_curvature, curvature))

        remaining = math.hypot(goal_x - x, goal_y - y)
        speed = float(self.get_parameter('linear_speed').value)
        if remaining < 1.0:
            speed = max(0.12, speed * remaining)
        command = Twist()
        command.linear.x = speed
        command.angular.z = speed * curvature
        self.command_publisher.publish(command)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CombinedMappingDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
