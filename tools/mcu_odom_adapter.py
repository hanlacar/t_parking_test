#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class McuOdomAdapter(Node):

    def __init__(self):
        super().__init__('mcu_odom_adapter')

        self.odom_pub = self.create_publisher(
            Odometry,
            '/odom',
            10
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.subscription = self.create_subscription(
            Odometry,
            '/mcu/odom',
            self.odom_callback,
            10
        )

        self.get_logger().info(
            'MCU odom adapter started: '
            '/mcu/odom -> /odom + odom->base_link TF'
        )

    def odom_callback(self, msg):

        # SLAM/Nav2에서 사용할 frame 계약
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        # /odom 재발행
        self.odom_pub.publish(msg)

        # odom -> base_link TF
        tf = TransformStamped()

        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'

        tf.transform.translation.x = msg.pose.pose.position.x
        tf.transform.translation.y = msg.pose.pose.position.y
        tf.transform.translation.z = msg.pose.pose.position.z

        tf.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)

    node = McuOdomAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
