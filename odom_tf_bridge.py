#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTfBridge(Node):
    def __init__(self):
        super().__init__('odom_tf_bridge')

        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            Odometry,
            '/mcu/odom',
            self.odom_callback,
            20
        )

        get_logger = self.get_logger()
        get_logger.info('/mcu/odom -> odom/base_link TF bridge started')

    def odom_callback(self, msg):
        tf = TransformStamped()

        tf.header = msg.header
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'

        tf.transform.translation.x = msg.pose.pose.position.x
        tf.transform.translation.y = msg.pose.pose.position.y
        tf.transform.translation.z = msg.pose.pose.position.z

        tf.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf)


def main():
    rclpy.init()
    node = OdomTfBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
