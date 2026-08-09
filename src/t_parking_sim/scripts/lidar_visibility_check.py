#!/usr/bin/env python3
"""Report directional LaserScan visibility and near-range blockage statistics."""

import math
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


SECTORS = (
    ('front', -15.0, 15.0),
    ('left_front', 30.0, 75.0),
    ('left_side', 75.0, 105.0),
    ('right_front', -75.0, -30.0),
    ('right_side', -105.0, -75.0),
    ('left_rear', 105.0, 135.0),
    ('right_rear', -135.0, -105.0),
)


class LidarVisibilityCheck(Node):
    """Accumulate a few scans and print per-sector statistics once."""

    def __init__(self):
        super().__init__('lidar_visibility_check')
        self.declare_parameter('frame_count', 5)
        self.declare_parameter('timeout', 10.0)
        self.declare_parameter('scan_topic', '/scan')
        self.frame_count = int(self.get_parameter('frame_count').value)
        self.timeout = float(self.get_parameter('timeout').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.started = time.monotonic()
        self.frames = 0
        self.range_min = None
        self.samples = {name: [] for name, _, _ in SECTORS}
        self.create_subscription(
            LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.create_timer(0.2, self.check_timeout)
        self.get_logger().info(f'checking LaserScan topic {self.scan_topic}')

    def on_scan(self, msg):
        self.frames += 1
        self.range_min = msg.range_min
        for index, value in enumerate(msg.ranges):
            if not math.isfinite(value):
                continue
            angle_deg = math.degrees(msg.angle_min + index * msg.angle_increment)
            for name, lower, upper in SECTORS:
                if lower <= angle_deg <= upper:
                    self.samples[name].append(float(value))
                    break
        if self.frames >= self.frame_count:
            self.report_and_stop()

    def check_timeout(self):
        if time.monotonic() - self.started >= self.timeout:
            self.get_logger().error(
                f'timed out after {self.timeout:.1f}s; received {self.frames} scans')
            rclpy.shutdown()

    def report_and_stop(self):
        near_limit = self.range_min + 0.03
        print(
            f'topic={self.scan_topic} frames={self.frames} '
            f'range_min={self.range_min:.3f} '
            f'near_limit={near_limit:.3f}')
        blocked = False
        side_medians = {}
        for name, lower, upper in SECTORS:
            values = self.samples[name]
            if not values:
                print(f'{name} [{lower:+.0f},{upper:+.0f}] valid=0')
                if name in ('left_side', 'right_side'):
                    blocked = True
                continue
            near_count = sum(value <= near_limit for value in values)
            near_ratio = near_count / len(values)
            median = statistics.median(values)
            mean = statistics.fmean(values)
            print(
                f'{name} [{lower:+.0f},{upper:+.0f}] valid={len(values)} '
                f'min={min(values):.3f} median={median:.3f} mean={mean:.3f} '
                f'near={near_count} near_ratio={near_ratio:.4f}')
            if name in ('left_side', 'right_side'):
                side_medians[name] = median
                blocked |= near_ratio > 0.50

        if len(side_medians) == 2:
            asymmetry = abs(
                side_medians['left_side'] - side_medians['right_side'])
            print(f'side_median_asymmetry={asymmetry:.3f}')
        print(f'verdict={"BLOCKED" if blocked else "CLEAR"}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LidarVisibilityCheck()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if node.frames < node.frame_count:
        sys.exit(1)


if __name__ == '__main__':
    main()
