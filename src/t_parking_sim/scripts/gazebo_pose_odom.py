#!/usr/bin/env python3
"""Publish spawn-relative odometry from Gazebo's physical model pose."""

import math

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster


def yaw_from_quaternion(quaternion):
    """Return planar yaw from a geometry_msgs quaternion."""
    return math.atan2(
        2.0 * (
            quaternion.w * quaternion.z
            + quaternion.x * quaternion.y),
        1.0 - 2.0 * (
            quaternion.y * quaternion.y
            + quaternion.z * quaternion.z),
    )


def normalize_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class GazeboPoseOdom(Node):
    """Transform absolute Gazebo pose into the project's local odom frame."""

    def __init__(self):
        super().__init__('gazebo_pose_odom')
        self.declare_parameter(
            'source_topic', '/gazebo_ground_truth/odom_absolute')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')

        self._spawn_pose = None
        source_topic = self.get_parameter('source_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value

        spawn_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            PoseStamped,
            '/parking_practice/spawn_pose',
            self._spawn_pose_callback,
            spawn_qos,
        )
        self.create_subscription(
            Odometry, source_topic, self._source_callback, 10)
        self._odom_publisher = self.create_publisher(
            Odometry, odom_topic, 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info(
            f'waiting for {source_topic} and /parking_practice/spawn_pose; '
            f'will exclusively publish {odom_topic} and '
            f'{self._odom_frame} -> {self._base_frame}')

    def _spawn_pose_callback(self, message):
        pose = message.pose
        self._spawn_pose = (
            pose.position.x,
            pose.position.y,
            yaw_from_quaternion(pose.orientation),
        )
        self.get_logger().info(
            'odom origin set from spawn pose: '
            f'x={self._spawn_pose[0]:.3f}, '
            f'y={self._spawn_pose[1]:.3f}, '
            f'yaw={self._spawn_pose[2]:.3f}')

    def _source_callback(self, source):
        if self._spawn_pose is None:
            return

        source_pose = source.pose.pose
        world_yaw = yaw_from_quaternion(source_pose.orientation)
        origin_x, origin_y, origin_yaw = self._spawn_pose
        delta_x = source_pose.position.x - origin_x
        delta_y = source_pose.position.y - origin_y
        cosine = math.cos(origin_yaw)
        sine = math.sin(origin_yaw)

        odom = Odometry()
        odom.header.stamp = source.header.stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = cosine * delta_x + sine * delta_y
        odom.pose.pose.position.y = -sine * delta_x + cosine * delta_y
        odom.pose.pose.position.z = 0.0
        relative_yaw = normalize_angle(world_yaw - origin_yaw)
        odom.pose.pose.orientation.z = math.sin(relative_yaw * 0.5)
        odom.pose.pose.orientation.w = math.cos(relative_yaw * 0.5)
        odom.pose.covariance = source.pose.covariance

        # Gazebo's OdometryPublisher derives twist from the physical model
        # pose.  The child frame is unchanged by the world-to-odom transform,
        # so its body-frame twist can be forwarded directly.
        odom.twist = source.twist
        self._odom_publisher.publish(odom)

        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = self._base_frame
        transform.transform.translation.x = odom.pose.pose.position.x
        transform.transform.translation.y = odom.pose.pose.position.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation = odom.pose.pose.orientation
        self._tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboPoseOdom()
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
