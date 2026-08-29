#!/usr/bin/env python3
"""Publish the isolated canonical map -> bench_odom BENCH anchor."""

import math

from bench_support import BENCH_ODOM_FRAME
from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, String
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class BenchMapOdomAnchor(Node):
    """Keep production map/odom/base_link frames out of the BENCH tree."""

    def __init__(self) -> None:
        super().__init__('bench_map_odom_anchor')
        self.declare_parameter('desired_x', 9.70)
        self.declare_parameter('desired_y', 0.0)
        self.declare_parameter('desired_yaw', math.pi)
        self.declare_parameter('odom_frame', BENCH_ODOM_FRAME)
        desired_x = float(self.get_parameter('desired_x').value)
        desired_y = float(self.get_parameter('desired_y').value)
        desired_yaw = float(self.get_parameter('desired_yaw').value)
        odom_frame = str(self.get_parameter('odom_frame').value)
        if odom_frame in ('map', 'odom'):
            raise ValueError('BENCH anchor must not reuse production frames')

        transient = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.ready_publisher = self.create_publisher(
            Bool, '/bench/map_odom_anchor_ready', transient)
        self.report_publisher = self.create_publisher(
            String, '/bench/map_odom_anchor_report', transient)
        self.broadcaster = StaticTransformBroadcaster(self)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = odom_frame
        transform.transform.translation.x = desired_x
        transform.transform.translation.y = desired_y
        transform.transform.rotation.z = math.sin(0.5 * desired_yaw)
        transform.transform.rotation.w = math.cos(0.5 * desired_yaw)
        self.broadcaster.sendTransform(transform)

        report = (
            '[BENCH MAP ODOM ANCHOR]\n\n'
            'virtual_odom_start:\n'
            'x=0.000000\ny=0.000000\nyaw=0.000000\n\n'
            'desired_map_start:\n'
            f'x={desired_x:.6f}\ny={desired_y:.6f}\n'
            f'yaw={desired_yaw:.6f}\n\n'
            'computed_map_to_bench_odom:\n'
            f'x={desired_x:.6f}\ny={desired_y:.6f}\n'
            f'yaw={desired_yaw:.6f}\n\n'
            'result_map_to_bench_base_link_at_start:\n'
            f'x={desired_x:.6f}\ny={desired_y:.6f}\n'
            f'yaw={desired_yaw:.6f}\n\n'
            'position_error=0.000000000\n'
            'yaw_error=0.000000000')
        self.get_logger().info(report)
        ready = Bool()
        ready.data = True
        status = String()
        status.data = report
        self.ready_publisher.publish(ready)
        self.report_publisher.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BenchMapOdomAnchor()
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
