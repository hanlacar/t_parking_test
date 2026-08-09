#!/usr/bin/env python3
"""Reliable, observable spawn and respawn manager for the parking vehicle."""

import math
import random
import subprocess
import threading
import time

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster


class VehicleSpawnManager(Node):
    OBSTACLE_NAMES = (
        't_parking_random_obstacle', 'parallel_parking_random_obstacle')
    # Slot centres taken from t_parking_exam.sdf.  The T bay spans x=[0, 3],
    # y=[1.75, 6.75] and is halved by the divider at x=1.5.  The parallel
    # spaces span y=[12.0, 13.5] between the walls at x=0 and x=10, halved by
    # the divider at x=5.  Yaw puts the block's 0.90 m axis along each slot's
    # long axis: +y for the T bay, +x for the parallel spaces.  z=0.125 is
    # half the block height, so it rests on the ground plane at z=0.
    SLOT_POSES = {
        'A': (0.75, 4.25, 0.125, 0.0),
        'B': (2.25, 4.25, 0.125, 0.0),
        'C': (2.50, 12.75, 0.125, 1.57079632679),
        'D': (7.50, 12.75, 0.125, 1.57079632679),
    }
    POSES = {
        't_parking': (-3.50, 0.0, 0.0, 0.0),
        'full_course': (-5.0, 0.0, 0.0, 0.0),
        'parallel_parking': (9.75, -0.50, 0.0, 1.57079632679),
        'parallel_ready': (8.75, 10.25, 0.0, 3.14159265359),
    }

    def __init__(self):
        super().__init__('vehicle_spawn_manager')
        self._declare_parameters()
        self.world_name = self.get_parameter('world_name').value
        self.entity_name = self.get_parameter('entity_name').value
        self.pose = self._resolve_spawn_pose()
        self._lock = threading.Lock()
        self._busy = False
        self._clock_samples = []
        self._description = ''
        self._messages = {}
        self._odom = None
        seed = int(self.get_parameter('parking_obstacle_seed').value)
        self._rng = random.SystemRandom() if seed < 0 else random.Random(seed)
        self._obstacle_layout = None
        status_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(
            String, '/parking_practice/spawn_status', status_qos)
        self.ready_pub = self.create_publisher(Bool, '/parking_practice/ready', status_qos)
        self.layout_pub = self.create_publisher(
            String, '/parking_practice/obstacle_layout', status_qos)
        # The vehicle's odometry origin is whatever pose it was created at, so
        # this world pose *is* the odom frame's origin.  Publishing it (and the
        # mode that chose it) lets parking nodes convert the SDF's world slot
        # coordinates into odom at runtime instead of baking in an offset that
        # silently breaks whenever the start pose changes.  The parallel start
        # is rotated (yaw 1.5708), so that conversion is a rotation followed by
        # a translation, never a plain addition.
        self.spawn_pose_pub = self.create_publisher(
            PoseStamped, '/parking_practice/spawn_pose', status_qos)
        self.mode_pub = self.create_publisher(
            String, '/parking_practice/practice_mode', status_qos)
        # The safety mux remains the sole publisher to the Gazebo command topic.
        self.stop_pub = self.create_publisher(
            Twist, '/parking_navigation/cmd_vel_precision', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            Clock, '/clock', self._clock_cb, qos_profile_sensor_data)
        desc_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/robot_description', self._description_cb, desc_qos)
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(
            JointState, '/joint_states', lambda m: self._seen('joint_states'),
            qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, '/scan', lambda m: self._seen('scan'),
            qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, '/scan_rear', lambda m: self._seen('scan_rear'),
            qos_profile_sensor_data)
        self.create_subscription(
            TFMessage, '/tf', self._tf_cb, qos_profile_sensor_data)
        self.create_service(Trigger, '/parking_practice/respawn', self._respawn_service_callback)
        self._publish_spawn_frame()
        self._publish_status('WAITING_FOR_SIM', False)
        threading.Thread(target=self._execute_spawn_cycle, daemon=True).start()

    def _declare_parameters(self):
        defaults = {
            'world_name': 't_parking_exam', 'entity_name': 'turtle_car',
            'practice_mode': 't_parking', 'spawn_retry_count': 5,
            'spawn_retry_interval': 2.0, 'simulation_wait_timeout': 30.0,
            'spawn_verify_timeout': 15.0, 'spawn_settle_timeout': 5.0,
            'randomize_parking_obstacles': True,
            'parking_obstacle_seed': -1,
            'rerandomize_on_respawn': True,
            'x': -5.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _publish_spawn_frame(self):
        x, y, z, yaw = self.pose
        message = PoseStamped()
        message.header.frame_id = 'world'
        message.pose.position.x = float(x)
        message.pose.position.y = float(y)
        message.pose.position.z = float(z)
        message.pose.orientation.z = math.sin(yaw / 2.0)
        message.pose.orientation.w = math.cos(yaw / 2.0)
        self.spawn_pose_pub.publish(message)
        self.mode_pub.publish(String(data=self.practice_mode))
        self.get_logger().info(
            f'practice_mode={self.practice_mode} spawn world pose='
            f'({x}, {y}, {z}, yaw={yaw}); odom originates here')

    def _resolve_spawn_pose(self):
        mode = self.get_parameter('practice_mode').value
        self.practice_mode = mode
        if mode == 'custom':
            return tuple(float(self.get_parameter(k).value) for k in ('x', 'y', 'z', 'yaw'))
        if mode not in self.POSES:
            raise ValueError(f'Invalid practice_mode: {mode}')
        return self.POSES[mode]

    def _clock_cb(self, msg):
        value = msg.clock.sec + msg.clock.nanosec * 1e-9
        self._clock_samples = (self._clock_samples + [value])[-4:]

    def _description_cb(self, msg):
        self._description = msg.data.strip()

    def _seen(self, key):
        self._messages[key] = time.monotonic()

    def _odom_cb(self, msg):
        self._odom = msg
        self._seen('odom')
        transform = TransformStamped()
        transform.header = msg.header
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_footprint'
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def _tf_cb(self, msg):
        if any(t.header.frame_id.lstrip('/') == 'odom' and
               t.child_frame_id.lstrip('/') == 'base_footprint' for t in msg.transforms):
            self._seen('tf')

    def _wait_for_simulation(self):
        timeout = float(self.get_parameter('simulation_wait_timeout').value)
        end = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < end:
            advancing = (
                len(self._clock_samples) >= 2
                and self._clock_samples[-1] > self._clock_samples[0])
            if advancing:
                # ros_gz_sim create itself blocks until the world's create
                # transport service is available; advancing /clock proves the
                # named world and UserCommands system are running.
                return True
            time.sleep(0.1)
        return False

    def _wait_for_robot_description(self):
        end = time.monotonic() + float(self.get_parameter('simulation_wait_timeout').value)
        while rclpy.ok() and time.monotonic() < end:
            if self._description:
                return True
            time.sleep(0.1)
        return False

    def _model_count(self, name):
        try:
            result = subprocess.run(
                ['gz', 'model', '--list'], capture_output=True, text=True, timeout=4, check=False)
            names = [line.strip().removeprefix('-').strip()
                     for line in result.stdout.splitlines()]
            return sum(item == name for item in names)
        except (OSError, subprocess.TimeoutExpired):
            return 0

    def _entity_exists(self):
        return self._model_count(self.entity_name)

    def _remove_named_entity(self, name):
        if self._model_count(name) == 0:
            return True
        try:
            result = subprocess.run(
                ['ros2', 'run', 'ros_gz_sim', 'remove', '--ros-args',
                 '-p', f'world:={self.world_name}', '-p', f'entity_name:={name}'],
                timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return False
        if result.returncode != 0:
            return False
        end = time.monotonic() + 5.0
        while time.monotonic() < end:
            if self._model_count(name) == 0:
                return True
            time.sleep(0.2)
        return False

    def _remove_existing_entity(self):
        return self._remove_named_entity(self.entity_name)

    def _reset_existing_entity_pose(self):
        x, y, z, yaw = self.pose
        request = (
            f'name: "{self.entity_name}", position: {{x: {x}, y: {y}, z: {z}}}, '
            f'orientation: {{z: {math.sin(yaw / 2.0)}, w: {math.cos(yaw / 2.0)}}}')
        command = [
            'gz', 'service', '-s', f'/world/{self.world_name}/set_pose',
            '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000', '--req', request]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=10, check=False)
            return result.returncode == 0 and 'data: true' in result.stdout
        except (OSError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _obstacle_sdf(name):
        return f"""<?xml version="1.0"?>
<sdf version="1.10"><model name="{name}"><static>true</static>
<link name="block_link"><collision name="block_collision"><geometry>
<box><size>0.45 0.90 0.25</size></box></geometry></collision>
<visual name="block_visual"><geometry><box><size>0.45 0.90 0.25</size></box>
</geometry><material><ambient>0.95 0.16 0.03 1</ambient>
<diffuse>1.0 0.22 0.04 1</diffuse></material></visual></link></model></sdf>"""

    def _spawn_obstacle(self, name, slot):
        x, y, z, yaw = self.SLOT_POSES[slot]
        command = [
            'ros2', 'run', 'ros_gz_sim', 'create', '-world', self.world_name,
            '-string', self._obstacle_sdf(name), '-name', name,
            '-allow_renaming', 'false', '-x', str(x), '-y', str(y), '-z', str(z),
            '-Y', str(yaw)]
        try:
            result = subprocess.run(command, timeout=15, check=False)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _prepare_obstacles(self, respawn):
        self._publish_status('REMOVING_OBSTACLES', False)
        if not all(self._remove_named_entity(name) for name in self.OBSTACLE_NAMES):
            return False
        rerandomize = bool(self.get_parameter('rerandomize_on_respawn').value)
        if self._obstacle_layout is None or (respawn and rerandomize):
            self._publish_status('SELECTING_OBSTACLES', False)
            if bool(self.get_parameter('randomize_parking_obstacles').value):
                self._obstacle_layout = f'{self._rng.choice("AB")}-{self._rng.choice("CD")}'
            else:
                self._obstacle_layout = 'A-C'
        self.get_logger().info(f'Selected obstacle layout: {self._obstacle_layout}')
        self.layout_pub.publish(String(data=self._obstacle_layout))
        t_slot, parallel_slot = self._obstacle_layout.split('-')
        self._publish_status('SPAWNING_OBSTACLES', False)
        spawned = (
            self._spawn_obstacle(self.OBSTACLE_NAMES[0], t_slot)
            and self._spawn_obstacle(self.OBSTACLE_NAMES[1], parallel_slot))
        self._publish_status('VERIFYING_OBSTACLES', False)
        return spawned and all(
            self._model_count(name) == 1 for name in self.OBSTACLE_NAMES)

    def _spawn_entity(self):
        x, y, z, yaw = self.pose
        command = [
            'ros2', 'run', 'ros_gz_sim', 'create', '-world', self.world_name,
            '-topic', '/robot_description', '-name', self.entity_name,
            '-allow_renaming', 'false', '-x', str(x), '-y', str(y), '-z', str(z),
            '-Y', str(yaw)]
        try:
            result = subprocess.run(command, timeout=15, check=False)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _verify_spawn(self):
        deadline = time.monotonic() + float(self.get_parameter('spawn_verify_timeout').value)
        required = {'odom', 'joint_states', 'scan', 'scan_rear', 'tf'}
        while rclpy.ok() and time.monotonic() < deadline:
            recent = {k for k, stamp in self._messages.items() if time.monotonic() - stamp < 2.0}
            odom = self._odom
            finite = odom is not None and all(math.isfinite(v) for v in (
                odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z,
                odom.twist.twist.linear.x, odom.twist.twist.angular.z))
            not_fallen = finite and odom.pose.pose.position.z > -0.25
            if required <= recent and finite and not_fallen and self._entity_exists() == 1:
                return True
            time.sleep(0.1)
        return False

    def _wait_until_vehicle_settled(self):
        timeout = float(self.get_parameter('spawn_settle_timeout').value)
        end = time.monotonic() + timeout
        initial = self._odom.pose.pose.position if self._odom else None
        stable_since = None
        while rclpy.ok() and time.monotonic() < end:
            odom = self._odom
            if odom:
                stable = (
                    abs(odom.twist.twist.linear.x) < 0.02
                    and abs(odom.twist.twist.angular.z) < 0.05)
                moved = initial and math.hypot(odom.pose.pose.position.x - initial.x,
                                               odom.pose.pose.position.y - initial.y)
                if moved and moved > 0.25:
                    return False
                stable_since = stable_since or (time.monotonic() if stable else None)
                if not stable:
                    stable_since = None
                if stable_since and time.monotonic() - stable_since >= 1.0:
                    return True
            time.sleep(0.05)
        return False

    def _execute_spawn_cycle(self, respawn=False):
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        try:
            self._publish_status('RESPAWNING' if respawn else 'WAITING_FOR_SIM', False)
            if not self._wait_for_simulation():
                return self._fail('simulation clock or world services not ready')
            self._publish_status('WAITING_FOR_DESCRIPTION', False)
            if not self._wait_for_robot_description():
                return self._fail('robot_description is empty')
            self._publish_status('CHECKING_EXISTING_ENTITY', False)
            reuse_existing = respawn and self._entity_exists() == 1
            if self._entity_exists() and not reuse_existing:
                self._publish_status('REMOVING_EXISTING_ENTITY', False)
                if not self._remove_existing_entity():
                    return self._fail('existing vehicle removal failed')
            if not self._prepare_obstacles(respawn):
                return self._fail('parking obstacle preparation failed')
            if reuse_existing:
                self._publish_status('RESETTING_EXISTING_ENTITY', False)
                if not self._reset_existing_entity_pose():
                    return self._fail('existing vehicle pose reset failed')
                self._publish_status('VERIFYING', False)
                # The already-running sensor plugins remain attached to this model;
                # requiring every bridged topic to rediscover here can create a
                # false timeout even though the entity was never recreated.
                if self._entity_exists() == 1 and self._wait_until_vehicle_settled():
                    self._publish_status('READY', True)
                    return True
                return self._fail('existing vehicle did not become ready after reset')
            attempts = int(self.get_parameter('spawn_retry_count').value)
            interval = float(self.get_parameter('spawn_retry_interval').value)
            for attempt in range(1, attempts + 1):
                self._publish_status('CHECKING_EXISTING_ENTITY', False)
                count = self._entity_exists()
                healthy_existing = (
                    count == 1 and not respawn and self._verify_spawn()
                    and self._wait_until_vehicle_settled())
                if healthy_existing:
                    self._publish_status('READY', True)
                    return True
                if count:
                    self._publish_status('REMOVING_EXISTING_ENTITY', False)
                    if not self._remove_existing_entity():
                        self.get_logger().warning('Could not confirm removal before retry')
                self._messages.clear()
                self._publish_status('SPAWNING', False)
                if self._spawn_entity():
                    self._publish_status('VERIFYING', False)
                    if self._verify_spawn():
                        self._publish_status('SETTLING', False)
                        if self._wait_until_vehicle_settled():
                            self._publish_status('READY', True)
                            return True
                self.get_logger().warning(f'Spawn attempt {attempt}/{attempts} failed')
                if self._entity_exists():
                    self._publish_status('REMOVING_EXISTING_ENTITY', False)
                    self._remove_existing_entity()
                time.sleep(interval)
            return self._fail(f'all {attempts} spawn attempts failed')
        finally:
            self._busy = False

    def _respawn_service_callback(self, request, response):
        del request
        if self._busy:
            response.success = False
            response.message = 'Spawn operation already in progress'
            return response
        self.stop_pub.publish(Twist())
        response.success = self._execute_spawn_cycle(respawn=True)
        response.message = (
            'Vehicle ready' if response.success
            else 'Respawn failed; see spawn_status')
        return response

    def _publish_status(self, status, ready):
        self.get_logger().info(status)
        self.status_pub.publish(String(data=status))
        self.ready_pub.publish(Bool(data=ready))

    def _fail(self, detail):
        self.get_logger().error(detail)
        self._publish_status('FAILED', False)
        return False


def main(args=None):
    rclpy.init(args=args)
    node = VehicleSpawnManager()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
