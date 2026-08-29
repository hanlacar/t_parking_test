#!/usr/bin/env python3
"""Run graph and zero-heartbeat phases of BENCH startup fail-closed."""

import time

from bench_support import (
    bench_motion_allowed,
    duplicate_node_names,
    MCU_NODES,
    preflight_graph_conflicts,
    startup_gate_state,
    StartupSnapshot,
)
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, Float32, Int32, String
from tf2_msgs.msg import TFMessage


class BenchPreflight(Node):
    """Keep Nav2 stopped until graph, zero source, and MCU are all safe."""

    def __init__(self) -> None:
        super().__init__('bench_preflight')
        self.declare_parameter('phase', 'graph')
        self.declare_parameter('bench_mode', False)
        self.declare_parameter('wheels_off_ground', False)
        self.declare_parameter('execute', False)
        self.declare_parameter('observation_sec', 1.5)
        self.declare_parameter('startup_timeout_sec', 15.0)
        self.declare_parameter('zero_samples_required', 3)

        self.bench_anchor_seen = False
        self.bench_vehicle_tf_seen = False
        self.production_map_odom_seen = False
        self.mcu_connected = False
        self.mcu_ready = False
        self.vehicle_mode = ''
        self.safety_state = ''
        self.lidar_drive = None
        self.lidar_wheel = None
        self.drive_zero_samples = 0
        self.wheel_zero_samples = 0
        self.nonzero_command_seen = False

        self.create_subscription(
            Bool, '/mcu/connected', self._connected_callback, 10)
        self.create_subscription(Bool, '/mcu/ready', self._ready_callback, 10)
        self.create_subscription(
            String, '/mcu/current_mode', self._mode_callback, 10)
        self.create_subscription(
            String, '/mcu/safety_state', self._safety_callback, 10)
        self.create_subscription(
            Float32, '/lidar_drive', self._drive_callback, 10)
        self.create_subscription(
            Int32, '/lidar_wheel', self._wheel_callback, 10)
        self.create_subscription(TFMessage, '/tf', self._tf_callback, 100)
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            TFMessage, '/tf_static', self._tf_callback, static_qos)

    @property
    def phase(self) -> str:
        return str(self.get_parameter('phase').value).strip().lower()

    @property
    def actual_motion_requested(self) -> bool:
        return bench_motion_allowed(
            bool(self.get_parameter('bench_mode').value),
            bool(self.get_parameter('wheels_off_ground').value),
            bool(self.get_parameter('execute').value),
        )

    def _connected_callback(self, msg: Bool) -> None:
        self.mcu_connected = bool(msg.data)

    def _ready_callback(self, msg: Bool) -> None:
        self.mcu_ready = bool(msg.data)

    def _mode_callback(self, msg: String) -> None:
        self.vehicle_mode = str(msg.data).strip().upper()

    def _safety_callback(self, msg: String) -> None:
        self.safety_state = str(msg.data).strip().upper()

    def _drive_callback(self, msg: Float32) -> None:
        value = float(msg.data)
        self.lidar_drive = value
        if abs(value) < 1.0e-6:
            self.drive_zero_samples += 1
        else:
            self.drive_zero_samples = 0
            self.nonzero_command_seen = True

    def _wheel_callback(self, msg: Int32) -> None:
        value = int(msg.data)
        self.lidar_wheel = value
        if value == 0:
            self.wheel_zero_samples += 1
        else:
            self.wheel_zero_samples = 0
            self.nonzero_command_seen = True

    def _tf_callback(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            parent = transform.header.frame_id.lstrip('/')
            child = transform.child_frame_id.lstrip('/')
            if parent == 'map' and child == 'bench_odom':
                self.bench_anchor_seen = True
            if parent == 'bench_odom' and child == 'bench_base_link':
                self.bench_vehicle_tf_seen = True
            if parent == 'map' and child == 'odom':
                self.production_map_odom_seen = True

    def _full_node_names(self) -> list:
        return [
            (('/' if namespace == '/' else namespace.rstrip('/') + '/')
             + name.lstrip('/'))
            for name, namespace in self.get_node_names_and_namespaces()
        ]

    def graph_errors(self, converter_started: bool) -> list:
        """Return persistent conflicts for the requested startup phase."""
        errors = []
        bench_mode = bool(self.get_parameter('bench_mode').value)
        wheels = bool(self.get_parameter('wheels_off_ground').value)
        execute = bool(self.get_parameter('execute').value)
        if not bench_mode:
            errors.append('bench_mode must be explicitly true')
        if execute and not bench_motion_allowed(bench_mode, wheels, execute):
            errors.append(
                'BENCH MOTION INHIBITED: bench_mode and wheels_off_ground '
                'must both be true')

        nodes = self.get_node_names_and_namespaces()
        full_names = self._full_node_names()
        conflicts = preflight_graph_conflicts(nodes)
        nav_conflicts = [name for name in conflicts if name != '/amcl']
        if nav_conflicts:
            errors.append(
                'duplicate Nav2 nodes: ' + ', '.join(nav_conflicts))
        if '/amcl' in conflicts:
            errors.append('AMCL is running')
        if self.production_map_odom_seen:
            errors.append('existing map->odom broadcaster detected')
        if self.bench_anchor_seen or self.bench_vehicle_tf_seen:
            errors.append('existing BENCH TF broadcaster detected')

        stale_bench_nodes = [
            name for name in (
                '/bench_map_odom_anchor', '/bench_virtual_vehicle')
            if name in full_names
        ]
        if stale_bench_nodes:
            errors.append(
                'existing BENCH nodes: ' + ', '.join(stale_bench_nodes))

        converter_count = full_names.count('/cmd_vel_to_lidar_cmd')
        if converter_started:
            if converter_count > 1:
                errors.append('duplicate command converter nodes detected')
        elif converter_count != 0:
            errors.append('existing command converter detected')

        input_publishers = self.get_publishers_info_by_topic(
            '/t_parking/cmd_vel_control')
        if input_publishers:
            errors.append(
                'unexpected cmd_vel publisher exists before Nav2 startup')

        duplicate_mcu_nodes = duplicate_node_names(nodes, MCU_NODES)
        if duplicate_mcu_nodes:
            errors.append(
                'duplicate MCU nodes: ' + ', '.join(duplicate_mcu_nodes))
        return errors

    def startup_snapshot(self) -> StartupSnapshot:
        return StartupSnapshot(
            drive_publishers=len(self.get_publishers_info_by_topic(
                '/lidar_drive')),
            wheel_publishers=len(self.get_publishers_info_by_topic(
                '/lidar_wheel')),
            drive_zero_samples=self.drive_zero_samples,
            wheel_zero_samples=self.wheel_zero_samples,
            nonzero_command_seen=self.nonzero_command_seen,
            mcu_connected=self.mcu_connected,
            mcu_ready=self.mcu_ready,
            vehicle_mode=self.vehicle_mode,
            safety_state=self.safety_state,
        )

    def run_graph_phase(self) -> list:
        deadline = time.monotonic() + float(
            self.get_parameter('observation_sec').value)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        errors = self.graph_errors(converter_started=False)
        if self.get_publishers_info_by_topic('/lidar_drive'):
            errors.append('pre-existing /lidar_drive publisher detected')
        if self.get_publishers_info_by_topic('/lidar_wheel'):
            errors.append('pre-existing /lidar_wheel publisher detected')
        if not errors:
            self.get_logger().info('[BENCH PRE-PREFLIGHT]\nPASS')
        return errors

    def run_startup_phase(self) -> list:
        timeout = float(self.get_parameter('startup_timeout_sec').value)
        required = int(self.get_parameter('zero_samples_required').value)
        deadline = time.monotonic() + timeout
        last_state = ''
        snapshot = self.startup_snapshot()
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            errors = self.graph_errors(converter_started=True)
            if errors:
                return errors
            snapshot = self.startup_snapshot()
            state = startup_gate_state(
                snapshot,
                zero_samples_required=required,
            )
            if state != last_state:
                last_state = state
                self.get_logger().info(f'[BENCH STARTUP]\n{state}')
            if state.startswith('FAIL_'):
                return [state]
            if state == 'READY':
                self.get_logger().info(
                    '[BENCH STARTUP]\nZERO_COMMAND_CONFIRMED')
                self.get_logger().info('[BENCH STARTUP]\nMCU_STATUS_OK')
                self.get_logger().info('[BENCH PREFLIGHT]\nPASS')
                return []
        return [
            f'startup timeout in {last_state or "UNKNOWN"}: '
            f'drive_publishers={snapshot.drive_publishers}, '
            f'wheel_publishers={snapshot.wheel_publishers}, '
            f'drive_zero_samples={snapshot.drive_zero_samples}, '
            f'wheel_zero_samples={snapshot.wheel_zero_samples}, '
            f'mcu_connected={snapshot.mcu_connected}, '
            f'mcu_ready_diagnostic={snapshot.mcu_ready}, '
            f'vehicle_mode={snapshot.vehicle_mode or "UNKNOWN"}, '
            f'safety_state={snapshot.safety_state or "UNKNOWN"}'
        ]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BenchPreflight()
    if node.phase == 'graph':
        errors = node.run_graph_phase()
    elif node.phase == 'startup':
        errors = node.run_startup_phase()
    else:
        errors = [f'unknown BENCH preflight phase: {node.phase}']
    if errors:
        for error in errors:
            node.get_logger().error(f'[BENCH STARTUP]\nFAIL: {error}')
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
