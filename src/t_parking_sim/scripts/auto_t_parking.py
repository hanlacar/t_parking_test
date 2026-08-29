#!/usr/bin/env python3
"""Plan and execute a Nav2-only T parking and entrance-return cycle."""

import copy
from dataclasses import dataclass
import math
import threading
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

from action_msgs.msg import GoalStatus
from bench_support import (
    bench_motion_allowed,
    duplicate_nav_nodes,
    duplicate_node_names,
    MCU_NODES,
)
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathThroughPoses, FollowPath
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Odometry, Path
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from slam_toolbox.srv import Pause
from std_msgs.msg import Bool, ColorRGBA, Float32, Int32, Int32MultiArray, String
from std_srvs.srv import Trigger
from t_parking_geometry import (
    assess_parking_pose,
    parking_target_from_end_clearance,
    vehicle_longitudinal_extents,
)
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray


SLOT_UNKNOWN = 'UNKNOWN'
SLOT_FREE = 'FREE'
SLOT_OCCUPIED = 'OCCUPIED'


@dataclass
class Slot:
    name: str
    odom_x: float
    odom_y: float
    yaw: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass
class PathMetrics:
    total_length: float
    forward_length: float
    reverse_length: float
    cusp_count: int
    first_reverse_index: int
    directions: List[int]
    max_curvature: float
    final_position_error: float
    final_yaw_error: float


@dataclass
class PlanCandidate:
    slot: Slot
    setup_offset: float
    entry_depth: float
    named_poses: Dict[str, PoseStamped]
    path: Path
    metrics: PathMetrics


@dataclass
class ForwardExitCandidate:
    """A separately planned, forward-only exit from the parked vehicle pose."""

    forward_clear_distance: float
    right_turn_lead: float
    lane_merge_distance: float
    named_poses: Dict[str, PoseStamped]
    path: Path
    metrics: PathMetrics
    turn_direction: str


@dataclass(frozen=True)
class DirectionSegment:
    """An inclusive pose slice with one physically confirmed direction."""

    first_pose: int
    last_pose: int
    direction: int
    length: float


def yaw_from_quaternion(quaternion) -> float:
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(sin_yaw, cos_yaw)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def set_pose_yaw(pose: Pose, yaw: float) -> None:
    pose.orientation.z = math.sin(0.5 * yaw)
    pose.orientation.w = math.cos(0.5 * yaw)


class AutoTParking(Node):
    """State-machine wrapper around Nav2 planning and FollowPath actions."""

    def __init__(self) -> None:
        super().__init__('t_parking_auto')
        self.callback_group = ReentrantCallbackGroup()
        self._declare_parameters()
        self.rviz_only = bool(self.get_parameter('rviz_only').value)
        self.bench_mode = bool(self.get_parameter('bench_mode').value)
        self.wheels_off_ground = bool(
            self.get_parameter('wheels_off_ground').value)
        self.robot_base_frame = str(
            self.get_parameter('robot_base_frame').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.target_slot = str(self.get_parameter('target_slot').value)
        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.execute_path = (
            bool(self.get_parameter('execute').value) and not self.rviz_only)
        self.slots = [self._load_slot('slot_1'), self._load_slot('slot_2')]

        self.worker_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.cancel_requested = threading.Event()

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_publisher = self.create_publisher(
            String, '/t_parking/status', transient_qos)
        self.path_publisher = self.create_publisher(
            Path, '/t_parking/planned_path', transient_qos)
        self.exit_path_publisher = self.create_publisher(
            Path, '/t_parking/exit_path', transient_qos)
        self.forward_publisher = self.create_publisher(
            Path, '/t_parking/forward_path', transient_qos)
        self.forward_exit_publisher = self.create_publisher(
            Path, '/t_parking/forward_exit_path', transient_qos)
        self.reverse_publisher = self.create_publisher(
            Path, '/t_parking/reverse_path', transient_qos)
        self.marker_publisher = self.create_publisher(
            MarkerArray, '/t_parking/markers', transient_qos)
        self.segment_state_publisher = self.create_publisher(
            Int32MultiArray, '/t_parking/active_segment', transient_qos)
        # A zero-only safety output for cancellation, failure, and settling
        # after Nav2 has completed a FollowPath goal.  This clears the
        # velocity smoother input; the smoother then feeds the sole
        # Twist-to-lidar adapter.  This node must never publish /cmd_vel
        # directly because /lidar_* is the final shared vehicle command.
        self.nav_stop_publisher = None
        # Private request consumed by cmd_vel_to_lidar_cmd.  Only genuine
        # safety faults set it; normal segment/cusp/final stops remain
        # /lidar_drive=0 with /lidar_stop=false.
        self.emergency_stop_request_publisher = None
        if not self.rviz_only:
            self.nav_stop_publisher = self.create_publisher(
                Twist, '/cmd_vel_nav', 10)
            self.emergency_stop_request_publisher = self.create_publisher(
                Bool, '/t_parking/emergency_stop_request', 10)

        self.map_msg: Optional[OccupancyGrid] = None
        self.global_costmap: Optional[Costmap] = None
        self.local_costmap: Optional[Costmap] = None
        self.odom_msg: Optional[Odometry] = None
        self.real_odom_msg: Optional[Odometry] = None
        self.front_scan_msg: Optional[LaserScan] = None
        self.scan_received_at = 0.0
        self.rear_scan_msg: Optional[LaserScan] = None
        self.rear_scan_received_at = 0.0
        self.front_scan_count = 0
        self.rear_scan_count = 0
        self.global_costmap_count = 0
        self.local_costmap_count = 0
        self.global_costmap_received_at = 0.0
        self.local_costmap_received_at = 0.0
        self.obstacle_observation_ready = False
        self.cmd_vel_nonzero_seen = False
        self.cmd_vel_direction: Optional[str] = None
        self.last_cmd_vel = Twist()
        self.last_cmd_vel_nav = Twist()
        self.last_cmd_vel_control = Twist()
        self.last_lidar_drive = 0.0
        self.last_lidar_wheel = 0
        self.bench_zero_drive_samples = 0
        self.bench_consecutive_zero_drive_samples = 0
        self.bench_consecutive_zero_wheel_samples = 0
        self.bench_anchor_ready = False
        self.mcu_connected = False
        self.mcu_ready = False
        self.mcu_current_mode = ''
        self.mcu_safety_state = ''
        self.estop_lock = False
        self.mcu_connected_received_at = 0.0
        self.mcu_ready_received_at = 0.0
        self.mcu_mode_received_at = 0.0
        self.mcu_safety_received_at = 0.0
        self.estop_received_at = 0.0
        self.lidar_drive_received_at = 0.0
        self.lidar_wheel_received_at = 0.0
        self.last_command_trace = 0.0
        self.reverse_motion_start_logged = False
        self.reverse_steering_start_logged = False
        self.minimum_exit_linear_x = float('inf')
        self.active_direction_segment: Optional[DirectionSegment] = None
        self.active_segment_number = 0
        self.active_segment_stats: Optional[Dict[str, float]] = None
        self.segment_execution_stats: List[Dict[str, float]] = []
        self._last_direction_sign: Dict[str, int] = {}
        self.active_segment_path: Optional[Path] = None
        self.last_direction_diagnostic: Dict[str, float] = {}
        self.last_stop_elapsed = 0.0
        self.last_stop_linear_speed = float('inf')
        self.last_stop_angular_speed = float('inf')
        self.data_lock = threading.Lock()
        self.readiness_lock = threading.Lock()
        self.readiness_cache: Dict[str, bool] = {}

        self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, transient_qos,
            callback_group=self.callback_group)
        self.create_subscription(
            Costmap, '/global_costmap/costmap_raw',
            self._global_costmap_callback, transient_qos,
            callback_group=self.callback_group)
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group)
        if self.bench_mode and self.odom_topic != '/odom':
            self.create_subscription(
                Odometry, '/odom', self._real_odom_callback,
                qos_profile_sensor_data,
                callback_group=self.callback_group)
        if not self.rviz_only:
            self.create_subscription(
                Costmap, '/local_costmap/costmap_raw',
                self._local_costmap_callback, transient_qos,
                callback_group=self.callback_group)
            self.create_subscription(
                LaserScan, '/scan_front', self._scan_callback,
                qos_profile_sensor_data, callback_group=self.callback_group)
            self.create_subscription(
                LaserScan, '/scan_rear', self._rear_scan_callback,
                qos_profile_sensor_data, callback_group=self.callback_group)
            self.create_subscription(
                Twist, '/cmd_vel', self._cmd_vel_callback, 10,
                callback_group=self.callback_group)
            self.create_subscription(
                Twist, '/cmd_vel_nav', self._cmd_vel_nav_callback, 10,
                callback_group=self.callback_group)
            self.create_subscription(
                Twist, '/t_parking/cmd_vel_control',
                self._cmd_vel_control_callback, 10,
                callback_group=self.callback_group)
            self.create_subscription(
                Float32, '/lidar_drive', self._lidar_drive_callback, 10,
                callback_group=self.callback_group)
            self.create_subscription(
                Int32, '/lidar_wheel', self._lidar_wheel_callback, 10,
                callback_group=self.callback_group)
            self.create_subscription(
                Bool, '/mcu/connected', self._mcu_connected_callback, 10,
                callback_group=self.callback_group)
            self.create_subscription(
                Bool, '/mcu/ready', self._mcu_ready_callback, 10,
                callback_group=self.callback_group)
            self.create_subscription(
                String, '/mcu/current_mode', self._mcu_mode_callback, 10,
                callback_group=self.callback_group)
            self.create_subscription(
                String, '/mcu/safety_state', self._mcu_safety_callback, 10,
                callback_group=self.callback_group)
            self.create_subscription(
                Bool, '/estop_lock', self._estop_lock_callback, 10,
                callback_group=self.callback_group)
            self.create_subscription(
                Bool, '/bench/map_odom_anchor_ready',
                self._bench_anchor_ready_callback, transient_qos,
                callback_group=self.callback_group)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)
        self.plan_client = ActionClient(
            self, ComputePathThroughPoses, '/compute_path_through_poses',
            callback_group=self.callback_group)
        self.follow_client = None
        if not self.rviz_only:
            self.follow_client = ActionClient(
                self, FollowPath, '/follow_path',
                callback_group=self.callback_group)
        self.slam_pause_client = self.create_client(
            Pause, '/slam_toolbox/pause_new_measurements',
            callback_group=self.callback_group)
        self.global_costmap_clear_client = self.create_client(
            ClearEntireCostmap,
            '/global_costmap/clear_entirely_global_costmap',
            callback_group=self.callback_group)
        self.slam_paused_by_node = False
        self.lifecycle_clients = {
            name: self.create_client(
                GetState, f'/{name}/get_state',
                callback_group=self.callback_group)
            # bt_navigator is intentionally absent: this node drives
            # /compute_path_through_poses and /follow_path directly and never
            # calls /navigate_to_pose, so the BT navigator is not launched.
            # Polling a lifecycle node that does not exist would leave the
            # readiness gate below permanently unsatisfied.
            for name in (
                ('planner_server',) if self.rviz_only
                else ('planner_server', 'controller_server'))
        }
        self.lifecycle_state_lock = threading.Lock()
        self.lifecycle_state: Dict[str, bool] = {
            name: False for name in self.lifecycle_clients}
        self.lifecycle_pending: Dict[str, Optional[float]] = {
            name: None for name in self.lifecycle_clients}

        self.create_service(
            Trigger, '/t_parking/start', self._start_service,
            callback_group=self.callback_group)
        self.create_service(
            Trigger, '/t_parking/cancel', self._cancel_service,
            callback_group=self.callback_group)

        self.active_plan_goal = None
        self.active_follow_goal = None
        self.last_plan_failure_reason = 'no planning request has been made'
        self.last_bench_preflight_failure_reason = ''
        self.last_execution_failure_reason = ''
        self.course_entrance_pose: Optional[PoseStamped] = None
        self.parking_wheels_inside = False
        self.wheel_inside_confirm_count = 0
        self.last_wheel_states: Optional[Tuple[bool, bool, bool, bool]] = None
        self.last_wheel_log = 0.0
        self.wheel_position_source: Optional[str] = None
        self.motion_phase = 'idle'
        self.active_motion_path: Optional[Path] = None
        self.active_motion_metrics: Optional[PathMetrics] = None
        self.active_motion_final: Optional[PoseStamped] = None
        self.forward_exit_poses: Dict[str, PoseStamped] = {}
        self.state = 'WAITING'
        self.selected_candidate: Optional[PlanCandidate] = None
        self.last_feedback_log = 0.0
        self._log_info(
            '[AUTO PARKING PARAMETERS]\n'
            f'rviz_only={str(self.rviz_only).lower()}\n'
            f'bench_mode={str(self.bench_mode).lower()}\n'
            f'wheels_off_ground={str(self.wheels_off_ground).lower()}\n'
            f'robot_base_frame={self.robot_base_frame}\n'
            f'odom_topic={self.odom_topic}\n'
            f'auto_start={str(self.auto_start).lower()}\n'
            f'execute={str(self.execute_path).lower()}\n'
            f'target_slot={self.target_slot}')
        self._log_info(
            '[T-PARK][STATE SEQUENCE] APPROACH/PLAN -> FORWARD -> '
            'CUSP STOP -> REVERSE -> PARKED -> EXIT STOP -> EXIT PLAN -> '
            'FORWARD RIGHT TURN -> RETURN/EXIT -> FINAL STOP')
        if self.bench_mode:
            for _ in range(3):
                self._log_info(
                    '[BENCH ONLY - VEHICLE MUST BE OFF THE GROUND]')
        self._publish_status('WAITING')
        self.readiness_timer = self.create_timer(
            1.0, self._readiness_timer_callback,
            callback_group=self.callback_group)
        if self.auto_start:
            self.create_timer(
                1.0, self._auto_start_once,
                callback_group=self.callback_group)

    def _declare_parameters(self) -> None:
        defaults = {
            'target_slot': 'auto', 'auto_start': True, 'execute': True,
            'rviz_only': False,
            'bench_mode': False,
            'wheels_off_ground': False,
            'bench_plan_exit': False,
            'robot_base_frame': 'base_link',
            'odom_topic': '/odom',
            'return_to_entrance': True,
            'stop_when_all_wheels_inside': True,
            'exit_mode': 'forward_right',
            'exit_planner_id': 'ForwardExit',
            'require_forward_only_exit': True,
            'require_right_turn_exit': True,
            'max_reverse_distance_during_exit': 0.01,
            'road_center_y': 0.0, 'road_min_y': -2.16,
            'road_max_y': 2.16, 'approach_lead': 0.50,
            # RViz-only testing deliberately supports only the canonical
            # east-to-west approach on the saved real map.  These defaults
            # are repeated in rviz_t_parking_test.yaml and the launch file so
            # the planner diagnostics document the intended fixed start.
            'opposite_start_x': 9.70,
            'opposite_start_y': 0.0,
            'opposite_start_yaw': math.pi,
            # Return to the captured start position facing east.  This is
            # intentionally independent of the captured outbound yaw=pi.
            'rviz_return_yaw': 0.0,
            'opposite_staging_lead': 0.50,
            'opposite_setup_inset_candidates': [0.45, 0.30, 0.15, 0.0],
            # R-slot_half_width = 1.52-0.915 = 0.605 m is the geometry-based
            # nominal setup for the expanded real-vehicle bay.
            'setup_offset_candidates': [0.60, 0.75, 0.90, 1.05],
            'wall_clearance': 0.08,
            'entry_depth_candidates': [0.10, 0.20, 0.30],
            'vehicle_length': 1.33, 'vehicle_width': 0.78,
            'vehicle_center_x_offset': 0.020,
            'costmap_footprint_padding': 0.025,
            'footprint_clearance': 0.06,
            # Exact minimum distance from the parked rear bumper to the closed
            # end of the slot.  For canonical slot_1, 3.06 m is derived from
            # (slot_depth 7.45 m - vehicle_length 1.33 m) / 2: the physical
            # envelope is centred longitudinally instead of being pushed to
            # the back curb.  The full footprint is validated before PARKED.
            'parking_end_clearance_m': 3.06,
            # Legacy value retained only so older overlays remain loadable and
            # the before/after geometry can be reported. It no longer selects
            # the final parking target.
            'rear_clearance': 0.70,
            # Lower bound on the rear-bumper-to-curb gap, independent of
            # whatever rear_clearance is configured to. Derived from real
            # geometry rather than picked by hand: global_costmap /
            # local_costmap's inflation_radius (nav2_params.yaml) is the
            # radius around the physical curb where planning/control cost
            # already climbs, and wheel_radius is the extra pad needed so
            # the wheel contact patch -- not just the footprint centre --
            # clears that gradient. Keeps a future rear_clearance edit from
            # silently reintroducing a too-deep stop.
            'curb_inflation_radius': 0.55,
            'final_pose_overshoot': 0.0,
            'wheel_frames': [
                'front_left_wheel_link', 'front_right_wheel_link',
                'rear_left_wheel_link', 'rear_right_wheel_link'],
            'wheel_radius': 0.14,
            'wheel_base': 0.77,
            'wheel_track': 0.67,
            'wheel_inside_margin': 0.01,
            'wheel_inside_confirm_count': 5,
            'wheel_check_period': 0.10,
            'stop_confirm_count': 3,
            'minimum_turning_radius': 1.52,
            'curvature_tolerance_factor': 1.00,
            'minimum_reverse_length': 0.30,
            'maximum_cusps': 2,
            'occupied_threshold': 50,
            'system_wait_timeout': 120.0, 'map_wait_timeout': 120.0,
            'scan_timeout': 2.0,
            'obstacle_observation_frames': 10,
            'costmap_observation_updates': 3,
            'obstacle_observation_settle_time': 1.0,
            'rear_emergency_stop_distance': 0.15,
            'rear_emergency_sector_deg': 30.0,
            'observation_odom_x': 6.50,
            'observation_odom_y': 0.0,
            'observation_odom_yaw': 0.0,
            'map_observation_settle_time': 3.0,
            'action_timeout': 45.0,
            'follow_path_timeout': 180.0,
            'parking_cancel_timeout': 5.0,
            'parking_stop_grace_timeout': 2.0,
            'entrance_pose_sample_count': 5,
            'entrance_pose_sample_interval': 0.10,
            'entrance_pose_capture_timeout': 10.0,
            'localization_stability_duration': 5.0,
            'entrance_pose_position_stability': 0.03,
            'entrance_pose_yaw_stability': 0.05,
            'entrance_staging_distance': 1.0,
            'entrance_remap_timeout': 15.0,
            'entrance_return_position_tolerance': 0.10,
            'entrance_return_yaw_tolerance': 0.12,
            'maximum_exit_cusps': 4,
            'forward_clear_distance_candidates': [0.02, 0.07, 0.12, 0.17],
            'right_turn_lead_candidates': [0.00, 0.05, 0.10],
            'lane_merge_distance_candidates': [0.80],
            'exit_planning_turn_radius': 1.70,
            'final_position_tolerance': 0.08,
            'final_yaw_tolerance': 0.12,
            # Upper bound only: the strict sustained-odometry stop check
            # normally completes quickly, while a loaded simulator still has
            # enough time to settle safely at a direction cusp.
            'stop_wait_timeout': 120.0,
            'planner_id': 'GridBased',
            'controller_id': 'ParkingFollowPath',
            'forward_controller_id': 'ParkingForward',
            'reverse_controller_id': 'ParkingReverse',
            'goal_checker_id': 'parking_goal_checker',
            'progress_checker_id': 'progress_checker',
            'freeze_slam_during_execution': True,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        slot_defaults = {
            'slot_1': (
                5.585, 4.875, -math.pi / 2,
                4.67, 6.50, 2.16, 7.59),
            'slot_2': (
                7.415, 4.875, -math.pi / 2,
                6.50, 8.33, 2.16, 7.59),
        }
        fields = ('odom_x', 'odom_y', 'yaw', 'min_x', 'max_x', 'min_y', 'max_y')
        for slot_name, values in slot_defaults.items():
            for field, value in zip(fields, values):
                self.declare_parameter(f'{slot_name}.{field}', value)

    def _load_slot(self, name: str) -> Slot:
        def value(field: str) -> float:
            return float(self.get_parameter(f'{name}.{field}').value)

        return Slot(
            name, value('odom_x'), value('odom_y'), value('yaw'),
            value('min_x'), value('max_x'), value('min_y'), value('max_y'))

    def _context_ok(self) -> bool:
        try:
            return bool(self.context.ok())
        except Exception:
            return False

    def _runtime_ok(self) -> bool:
        return not self._stop_event.is_set() and self._context_ok()

    def _abort_requested(self) -> bool:
        return (
            self._stop_event.is_set()
            or self.cancel_requested.is_set()
            or not self._context_ok())

    def _safe_publish(self, publisher, message) -> bool:
        if not self._runtime_ok():
            return False
        try:
            publisher.publish(message)
            return True
        except Exception as exc:
            if self._runtime_ok():
                self._log_error(f'publish failed: {exc}')
            return False

    def _safe_log(self, level: str, message: str) -> bool:
        if not self._runtime_ok():
            return False
        try:
            getattr(self.get_logger(), level)(message)
            return True
        except Exception:
            return False

    def _log_info(self, message: str) -> bool:
        return self._safe_log('info', message)

    def _log_warn(self, message: str) -> bool:
        return self._safe_log('warning', message)

    def _log_error(self, message: str) -> bool:
        return self._safe_log('error', message)

    def _log_pose(self, name: str, pose: PoseStamped) -> None:
        self._log_info(
            f'{name}=({pose.pose.position.x:.3f}, '
            f'{pose.pose.position.y:.3f}, '
            f'{yaw_from_quaternion(pose.pose.orientation):.3f})')

    def _interruptible_sleep(
            self, duration: float, stop_on_cancel: bool = True) -> bool:
        deadline = time.monotonic() + max(0.0, duration)
        while time.monotonic() < deadline:
            if not self._runtime_ok():
                return False
            if stop_on_cancel and self.cancel_requested.is_set():
                return False
            remaining = deadline - time.monotonic()
            self._stop_event.wait(min(0.05, max(0.0, remaining)))
        return self._runtime_ok() and (
            not stop_on_cancel or not self.cancel_requested.is_set())

    def _map_callback(self, msg: OccupancyGrid) -> None:
        with self.data_lock:
            self.map_msg = msg

    def _global_costmap_callback(self, msg: Costmap) -> None:
        with self.data_lock:
            self.global_costmap = msg
            self.global_costmap_count += 1
            self.global_costmap_received_at = time.monotonic()

    def _local_costmap_callback(self, msg: Costmap) -> None:
        with self.data_lock:
            self.local_costmap = msg
            self.local_costmap_count += 1
            self.local_costmap_received_at = time.monotonic()

    def _scan_callback(self, msg: LaserScan) -> None:
        with self.data_lock:
            self.front_scan_msg = msg
            self.scan_received_at = time.monotonic()
            self.front_scan_count += 1

    def _rear_scan_callback(self, msg: LaserScan) -> None:
        with self.data_lock:
            self.rear_scan_msg = msg
            self.rear_scan_received_at = time.monotonic()
            self.rear_scan_count += 1

    def _odom_callback(self, msg: Odometry) -> None:
        with self.data_lock:
            self.odom_msg = msg

    def _real_odom_callback(self, msg: Odometry) -> None:
        with self.data_lock:
            self.real_odom_msg = msg

    def _cmd_vel_callback(self, msg: Twist) -> None:
        if not self._runtime_ok():
            return
        with self.data_lock:
            self.last_cmd_vel = msg
            diagnostic_log = self._record_direction_sample_locked(
                'cmd_vel', float(msg.linear.x))
            if self.motion_phase == 'forward_exit':
                self.minimum_exit_linear_x = min(
                    self.minimum_exit_linear_x, float(msg.linear.x))
        if diagnostic_log:
            self._log_info(diagnostic_log)
        moving = (
            abs(msg.linear.x) > 1.0e-4
            or abs(msg.linear.y) > 1.0e-4
            or abs(msg.angular.z) > 1.0e-4)
        if not moving:
            return
        if not self.cmd_vel_nonzero_seen:
            self.cmd_vel_nonzero_seen = True
            self._log_info(
                '[T-PARK] first non-zero /cmd_vel received: '
                f'linear.x={msg.linear.x:.3f}m/s '
                f'angular.z={msg.angular.z:.3f}rad/s')
        direction = 'reverse' if msg.linear.x < 0.0 else 'forward'
        if direction != self.cmd_vel_direction:
            self.cmd_vel_direction = direction
            self._log_info(f'/cmd_vel motion state: {direction}')

    def _cmd_vel_nav_callback(self, msg: Twist) -> None:
        if not self._runtime_ok():
            return
        with self.data_lock:
            self.last_cmd_vel_nav = msg
            diagnostic_log = self._record_direction_sample_locked(
                'cmd_vel_nav', float(msg.linear.x))
        if diagnostic_log:
            self._log_info(diagnostic_log)

    def _cmd_vel_control_callback(self, msg: Twist) -> None:
        if not self._runtime_ok():
            return
        with self.data_lock:
            self.last_cmd_vel_control = msg
            if self.active_segment_stats is not None:
                self.active_segment_stats['minimum_cmd_vel_linear_x'] = min(
                    self.active_segment_stats['minimum_cmd_vel_linear_x'],
                    float(msg.linear.x))
                self.active_segment_stats['maximum_cmd_vel_linear_x'] = max(
                    self.active_segment_stats['maximum_cmd_vel_linear_x'],
                    float(msg.linear.x))
            diagnostic_log = self._record_direction_sample_locked(
                'cmd_vel_control', float(msg.linear.x))
            log_reverse_start = (
                self.active_direction_segment is not None
                and self.active_direction_segment.direction < 0
                and msg.linear.x < -0.01
                and not self.reverse_motion_start_logged)
            if log_reverse_start:
                self.reverse_motion_start_logged = True
        if diagnostic_log:
            self._log_info(diagnostic_log)
        self._trace_command_chain(
            'REVERSE_MOTION_START' if log_reverse_start else '')

    def _lidar_drive_callback(self, msg: Float32) -> None:
        if not self._runtime_ok():
            return
        value = float(msg.data)
        with self.data_lock:
            self.last_lidar_drive = value
            self.lidar_drive_received_at = time.monotonic()
            if abs(value) < 1.0e-6:
                self.bench_zero_drive_samples += 1
                self.bench_consecutive_zero_drive_samples += 1
            else:
                self.bench_consecutive_zero_drive_samples = 0
            diagnostic_log = self._record_direction_sample_locked(
                'lidar_drive', value)
        if diagnostic_log:
            self._log_info(diagnostic_log)

    def _lidar_wheel_callback(self, msg: Int32) -> None:
        if not self._runtime_ok():
            return
        value = int(msg.data)
        with self.data_lock:
            self.last_lidar_wheel = value
            self.lidar_wheel_received_at = time.monotonic()
            if value == 0:
                self.bench_consecutive_zero_wheel_samples += 1
            else:
                self.bench_consecutive_zero_wheel_samples = 0
            stats = self.active_segment_stats
            if stats is not None:
                stats['lidar_wheel_samples'] += 1
                stats['minimum_lidar_wheel'] = min(
                    stats['minimum_lidar_wheel'], value)
                stats['maximum_lidar_wheel'] = max(
                    stats['maximum_lidar_wheel'], value)
                stats['max_abs_lidar_wheel'] = max(
                    stats['max_abs_lidar_wheel'], abs(value))
                if (value != 0
                        and stats['first_nonzero_lidar_wheel'] == 0):
                    stats['first_nonzero_lidar_wheel'] = value
                if (abs(value) >= 2
                        and stats['first_major_lidar_wheel'] == 0):
                    stats['first_major_lidar_wheel'] = value
                saturated = abs(value) >= 27
                if saturated:
                    stats['lidar_wheel_saturation_samples'] += 1
                    if not stats['lidar_wheel_saturation_active']:
                        stats['lidar_wheel_saturation_events'] += 1
                stats['lidar_wheel_saturation_active'] = saturated
                sign = 1 if value > 0 else -1 if value < 0 else 0
                previous = stats['last_lidar_wheel_nonzero_sign']
                if sign and previous and sign != previous:
                    stats['lidar_wheel_sign_reversals'] += 1
                if sign:
                    stats['last_lidar_wheel_nonzero_sign'] = sign
            log_reverse_steering = (
                self.active_direction_segment is not None
                and self.active_direction_segment.direction < 0
                and self.last_cmd_vel_control.linear.x < -0.01
                and value != 0
                and not self.reverse_steering_start_logged)
            if log_reverse_steering:
                self.reverse_steering_start_logged = True
        if log_reverse_steering:
            self._trace_command_chain('REVERSE_STEERING_ACTIVE')

    def _mcu_connected_callback(self, msg: Bool) -> None:
        with self.data_lock:
            self.mcu_connected = bool(msg.data)
            self.mcu_connected_received_at = time.monotonic()

    def _mcu_ready_callback(self, msg: Bool) -> None:
        with self.data_lock:
            self.mcu_ready = bool(msg.data)
            self.mcu_ready_received_at = time.monotonic()

    def _mcu_mode_callback(self, msg: String) -> None:
        with self.data_lock:
            self.mcu_current_mode = str(msg.data).strip().upper()
            self.mcu_mode_received_at = time.monotonic()

    def _mcu_safety_callback(self, msg: String) -> None:
        with self.data_lock:
            self.mcu_safety_state = str(msg.data).strip().upper()
            self.mcu_safety_received_at = time.monotonic()

    def _estop_lock_callback(self, msg: Bool) -> None:
        with self.data_lock:
            self.estop_lock = bool(msg.data)
            self.estop_received_at = time.monotonic()
        if bool(msg.data) and self.bench_mode:
            self.cancel_requested.set()
            self._cancel_active_goals()
            self._emergency_stop(emergency=True)
            self._log_error(
                '[BENCH] /estop_lock=true: FollowPath cancelled and zero '
                'command requested immediately')

    def _bench_anchor_ready_callback(self, msg: Bool) -> None:
        with self.data_lock:
            self.bench_anchor_ready = bool(msg.data)

    def _trace_command_chain(self, force_label: str = '') -> None:
        """Log one synchronized view of every command conversion stage."""
        now = time.monotonic()
        with self.data_lock:
            segment = self.active_direction_segment
            if not force_label and (
                    segment is None or now - self.last_command_trace < 0.5):
                return
            self.last_command_trace = now
            cmd_nav = copy.deepcopy(self.last_cmd_vel_nav)
            cmd_control = copy.deepcopy(self.last_cmd_vel_control)
            drive = self.last_lidar_drive
            wheel = self.last_lidar_wheel
            cmd_vel = copy.deepcopy(self.last_cmd_vel)
            segment_number = self.active_segment_number
            direction = 0 if segment is None else segment.direction
        label = force_label or 'RUNNING'
        self._log_info(
            '[COMMAND TRACE] '
            f'state={label} segment={segment_number} '
            f'direction={"REVERSE" if direction < 0 else "FORWARD" if direction > 0 else "NONE"} '
            f'cmd_vel_nav=({cmd_nav.linear.x:+.4f},{cmd_nav.angular.z:+.4f}) '
            f'cmd_vel_control=({cmd_control.linear.x:+.4f},{cmd_control.angular.z:+.4f}) '
            f'lidar_drive={drive:+.1f} lidar_wheel={wheel:+d} '
            f'cmd_vel=({cmd_vel.linear.x:+.4f},{cmd_vel.angular.z:+.4f})')

    def _record_direction_sample_locked(
            self, source: str, value: float) -> Optional[str]:
        """Count nonzero command samples that oppose the active path run."""
        segment = self.active_direction_segment
        stats = self.active_segment_stats
        if segment is None or stats is None:
            return None
        epsilon = 0.01
        sign = 1 if value > epsilon else -1 if value < -epsilon else 0
        previous_sign = self._last_direction_sign.get(source, 0)
        self._last_direction_sign[source] = sign
        if sign == 0 or sign == segment.direction:
            return None
        sample_key = f'{source}_wrong_samples'
        event_key = f'{source}_wrong_events'
        stats[sample_key] += 1
        new_event = previous_sign != sign
        if new_event:
            stats[event_key] += 1
        event_count = stats[event_key]
        sample_count = stats[sample_key]
        # Log each new wrong-sign run, while still counting every bad sample.
        if not new_event:
            return None
        diagnostic = dict(self.last_direction_diagnostic)
        detail = ''
        if diagnostic:
            detail = (
                f' path_index={int(diagnostic.get("path_index", -1))} '
                f'robot=({diagnostic.get("robot_x", float("nan")):.3f},'
                f'{diagnostic.get("robot_y", float("nan")):.3f},'
                f'{diagnostic.get("robot_yaw", float("nan")):.3f}) '
                f'diagnostic_carrot_base_x='
                f'{diagnostic.get("carrot_base_x", float("nan")):+.3f}')
        return (
            '[T-PARK][DIRECTION ERROR] '
            f'expected={"REVERSE" if segment.direction < 0 else "FORWARD"} '
            f'{source}={value:+.3f} segment={self.active_segment_number} '
            f'event={event_count} bad_samples={sample_count}{detail}')

    def _auto_start_once(self) -> None:
        if not self.auto_start or not self._runtime_ok():
            return
        if self.bench_mode:
            with self.readiness_lock:
                readiness = dict(self.readiness_cache)
            # Nav2's Hybrid-A* plugin initialization can take tens of
            # seconds on the real map.  Keep this BENCH-only timer armed
            # until every plan/control dependency is genuinely ready.
            if not readiness or not all(readiness.values()):
                return
            if self._start_worker():
                self.auto_start = False
            return
        self.auto_start = False
        self._start_worker()

    def _start_service(self, _request, response):
        if not self._runtime_ok():
            response.success = False
            response.message = 'automatic T-parking node is shutting down'
            return response
        if self._start_worker():
            response.success = True
            response.message = 'automatic T-parking started'
        else:
            response.success = False
            response.message = 'automatic T-parking is already running'
        return response

    def _cancel_service(self, _request, response):
        if not self._runtime_ok():
            response.success = False
            response.message = 'automatic T-parking node is shutting down'
            return response
        running = (
            self._worker_thread is not None
            and self._worker_thread.is_alive())
        self.cancel_requested.set()
        self._cancel_active_goals()
        self._emergency_stop()
        stopped = self._wait_until_stopped(float(
            self.get_parameter('stop_wait_timeout').value))
        self._publish_status('CANCELLED')
        response.success = stopped
        if stopped:
            response.message = (
                'active run cancelled; vehicle stopped'
                if running else 'vehicle already stopped; zero command confirmed')
        else:
            response.message = 'cancellation requested, but vehicle stop timed out'
        return response

    def _bench_runtime_graph_ok(self) -> bool:
        if not self.bench_mode:
            return True
        nodes = self.get_node_names_and_namespaces()
        full_names = [
            (('/' if namespace == '/' else namespace.rstrip('/') + '/')
             + name.lstrip('/'))
            for name, namespace in nodes]
        if '/amcl' in full_names:
            self._log_error('BENCH START ABORTED: AMCL is running')
            return False
        invalid = duplicate_nav_nodes(nodes)
        if invalid:
            self._log_error(
                'BENCH START ABORTED: duplicate Nav2 nodes or missing '
                'single owner: ' + ', '.join(invalid))
            return False
        required_bench_nodes = (
            '/bench_map_odom_anchor',
            '/bench_virtual_vehicle',
            '/cmd_vel_to_lidar_cmd',
        )
        invalid_bench = [
            name for name in required_bench_nodes
            if full_names.count(name) != 1]
        if invalid_bench:
            self._log_error(
                'BENCH START ABORTED: BENCH node owner count is not one: '
                + ', '.join(invalid_bench))
            return False
        duplicate_mcu = duplicate_node_names(nodes, MCU_NODES)
        if duplicate_mcu:
            self._log_error(
                'BENCH START ABORTED: duplicate MCU nodes: '
                + ', '.join(duplicate_mcu))
            return False
        return True

    def _bench_motion_preflight(
            self, require_fresh_status: bool = True) -> bool:
        """Check BENCH output gates without rejecting latched MCU state."""
        self.last_bench_preflight_failure_reason = ''
        if not self.bench_mode:
            return True
        if not bench_motion_allowed(
                self.bench_mode, self.wheels_off_ground, self.execute_path):
            self._log_error(
                'BENCH MOTION INHIBITED: bench_mode and wheels_off_ground '
                'must both be true')
            self.last_bench_preflight_failure_reason = (
                'bench_mode_and_wheels_off_ground_not_enabled')
            return False
        if not self._bench_runtime_graph_ok():
            self.last_bench_preflight_failure_reason = (
                'bench_runtime_graph_check_failed')
            return False
        now = time.monotonic()
        with self.data_lock:
            mcu_connected = self.mcu_connected
            mcu_ready = self.mcu_ready
            mode = self.mcu_current_mode
            safety_state = self.mcu_safety_state
            estop = self.estop_lock
            drive = self.last_lidar_drive
            wheel = self.last_lidar_wheel
            connected_fresh = now - self.mcu_connected_received_at < 2.0
            mode_fresh = now - self.mcu_mode_received_at < 2.0
            safety_fresh = now - self.mcu_safety_received_at < 2.0
            drive_fresh = now - self.lidar_drive_received_at < 1.0
            wheel_fresh = now - self.lidar_wheel_received_at < 1.0
            drive_zero_samples = self.bench_consecutive_zero_drive_samples
            wheel_zero_samples = self.bench_consecutive_zero_wheel_samples
        drive_publishers = len(
            self.get_publishers_info_by_topic('/lidar_drive'))
        wheel_publishers = len(
            self.get_publishers_info_by_topic('/lidar_wheel'))
        checks = {
            'mcu_connected': mcu_connected,
            'vehicle_mode_T_PARK': mode == 'T_PARK',
            'mcu_safety_state_OK': safety_state == 'OK',
            'estop_unlocked': not estop,
            'lidar_drive_single_publisher': drive_publishers == 1,
            'lidar_wheel_single_publisher': wheel_publishers == 1,
            'lidar_drive_fresh': drive_fresh,
            'lidar_wheel_fresh': wheel_fresh,
            'lidar_drive_zero': abs(drive) < 1.0e-6,
            'lidar_wheel_zero': wheel == 0,
            'lidar_drive_zero_continuous': drive_zero_samples >= 3,
            'lidar_wheel_zero_continuous': wheel_zero_samples >= 3,
        }
        # Official MCU state topics can be event-driven. Fresh reception is a
        # startup observation requirement; later segments still require the
        # last fail-closed state values plus fresh zero command heartbeats.
        if require_fresh_status:
            checks.update({
                'connected_status_fresh': connected_fresh,
                'mode_status_fresh': mode_fresh,
                'safety_status_fresh': safety_fresh,
            })
        missing = [name for name, ready in checks.items() if not ready]
        if missing:
            self.last_bench_preflight_failure_reason = (
                'bench_motion_preflight_failed:' + ','.join(missing))
            self._log_error(
                'BENCH MOTION INHIBITED: physical-output preflight failed\n'
                + '\n'.join(f'- {name}=false' for name in missing))
            return False
        self._log_info(
            '[BENCH MOTION PREFLIGHT]\n'
            'mcu_connected=true\n'
            'vehicle_mode=T_PARK\n'
            'mcu_safety_state=OK\n'
            'estop_lock=false\n'
            'lidar_drive_publishers=1\n'
            'lidar_wheel_publishers=1\n'
            'lidar_drive=0.0\n'
            'lidar_wheel=0\n'
            f'status_freshness_required={require_fresh_status}\n'
            f'mcu_ready_diagnostic={mcu_ready}')
        return True

    def _start_worker(self) -> bool:
        with self.worker_lock:
            if not self._runtime_ok():
                return False
            if (self._worker_thread is not None
                    and self._worker_thread.is_alive()):
                return False
            # Parameters may be changed between service-triggered runs.  Read
            # them here so plan-only validation can safely precede execution
            # without restarting Gazebo, SLAM, or Nav2 and losing the map.
            self.target_slot = str(self.get_parameter('target_slot').value)
            self.bench_mode = bool(self.get_parameter('bench_mode').value)
            self.wheels_off_ground = bool(
                self.get_parameter('wheels_off_ground').value)
            self.execute_path = (
                bool(self.get_parameter('execute').value)
                and not self.rviz_only)
            if self.bench_mode and not self._bench_runtime_graph_ok():
                return False
            if self.bench_mode and self.execute_path:
                if not self._bench_motion_preflight():
                    return False
            self.cancel_requested.clear()
            self._set_emergency_stop_request(False)
            self.cmd_vel_nonzero_seen = False
            self.cmd_vel_direction = None
            self.active_direction_segment = None
            self.active_segment_number = 0
            self.active_segment_stats = None
            self.active_segment_path = None
            self.segment_execution_stats = []
            self._last_direction_sign = {}
            self.last_direction_diagnostic = {}
            self.obstacle_observation_ready = False
            self.course_entrance_pose = None
            self.parking_wheels_inside = False
            self.wheel_inside_confirm_count = 0
            self.last_wheel_states = None
            self.last_wheel_log = 0.0
            self.motion_phase = 'idle'
            self.active_motion_path = None
            self.active_motion_metrics = None
            self.active_motion_final = None
            self._log_info(
                '[AUTO PARKING PARAMETERS]\n'
                f'rviz_only={str(self.rviz_only).lower()}\n'
                f'bench_mode={str(self.bench_mode).lower()}\n'
                f'wheels_off_ground={str(self.wheels_off_ground).lower()}\n'
                f'auto_start={str(self.auto_start).lower()}\n'
                f'execute={str(self.execute_path).lower()}\n'
                f'target_slot={self.target_slot}')
            self._worker_thread = threading.Thread(
                target=self._run_state_machine,
                name='t_parking_worker', daemon=False)
            self._worker_thread.start()
            return True

    def stop(self, join_timeout: float = 5.0) -> bool:
        """Stop actions and join the non-daemon worker before node teardown."""
        self.cancel_requested.set()
        if self._context_ok():
            self._cancel_active_goals()
            self._emergency_stop()
        self._stop_event.set()
        thread = self._worker_thread
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout=join_timeout)
        return thread is None or not thread.is_alive()

    def _publish_status(self, status: str, detail: str = '') -> None:
        if not self._runtime_ok():
            return
        if status != self.state or detail:
            message = status if not detail else f'{status}: {detail}'
            self._log_info(message)
        self.state = status
        msg = String()
        msg.data = status if not detail else f'{status}: {detail}'
        self._safe_publish(self.status_publisher, msg)

    def _run_state_machine(self) -> None:
        try:
            if self._abort_requested():
                return
            self._publish_status('WAIT_SYSTEM')
            if not self._wait_for_system():
                if not self._abort_requested():
                    self._fail('system readiness timeout')
                return
            if self._abort_requested():
                self._cancelled()
                return

            if not self._capture_course_entrance_pose():
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail('could not capture a stable course entrance pose')
                return
            if self._wheel_centers_map() is None:
                self._fail('wheel TF and base-footprint geometry are unavailable')
                return

            # ComputePathThroughPoses is called directly, so this node does
            # not pass through Nav2 BT recovery nodes that normally clear a
            # world-frame obstacle layer after localization changes.  Clear
            # only the global costmap after the stable entrance pose has been
            # captured.  The mandatory fresh scan/costmap window below then
            # restores all current obstacles before slot classification or
            # planning; no collision or inflation check is bypassed.
            if (not self.rviz_only and not self.bench_mode
                    and not self._clear_global_costmap()):
                self._fail(
                    'could not reset stale global obstacle data after '
                    'localization settled')
                return

            self._publish_status('WAIT_MAP')
            if self.rviz_only:
                # The dry-run deliberately has no LaserScan source.  Slot
                # geometry is still the same deterministic Slot data used by
                # the real planner; only live occupancy classification is
                # bypassed in this explicitly isolated mode.
                with self.data_lock:
                    map_ready = self.map_msg is not None
                    costmap_ready = self.global_costmap is not None
                available_slots = (
                    list(self.slots) if map_ready and costmap_ready else [])
                self._log_info(
                    '[RVIZ ONLY] using configured deterministic slot geometry; '
                    'LaserScan occupancy detection is disabled')
            elif self.bench_mode:
                with self.data_lock:
                    map_ready = self.map_msg is not None
                    costmap_ready = (
                        self.global_costmap is not None
                        and self.local_costmap is not None)
                available_slots = [
                    slot for slot in self.slots
                    if slot.name == 'slot_1' and map_ready and costmap_ready]
                self._log_info(
                    '[BENCH MODE]\n'
                    'LIVE OBSTACLE VALIDATION DISABLED FOR '
                    'WHEELS-OFF-GROUND TEST ONLY')
            else:
                self._publish_status('WAIT_OBSTACLE_OBSERVATION')
                if not self._wait_for_stable_obstacle_observation():
                    if self._abort_requested():
                        self._cancelled()
                    else:
                        self._fail(
                            'front/rear lidar sectors and costmaps '
                            'did not provide a '
                            'stable obstacle observation window')
                    return
                available_slots = self._wait_for_map_and_slots()
            if not available_slots:
                if self._abort_requested():
                    self._cancelled()
                    return
                self._fail(
                    'parking slots are unknown, occupied, or costmaps are unavailable')
                return

            self._publish_status('SELECT_SLOT')
            requested = self.target_slot
            if requested not in ('auto', 'slot_1', 'slot_2'):
                self._fail(f'invalid target_slot={requested!r}')
                return
            if requested != 'auto':
                available_slots = [
                    slot for slot in available_slots if slot.name == requested]
                if not available_slots:
                    self._fail(f'requested {requested} is not safely available')
                    return

            self._publish_status('BUILD_PARKING_POSES')
            self._publish_status('PLANNING_PARKING')
            for slot in available_slots:
                self._log_parking_geometry(slot)
            candidates = self._plan_all_candidates(available_slots)
            if not candidates:
                if self._abort_requested():
                    self._cancelled()
                    return
                self._fail('all setup/depth candidates were rejected')
                return
            # Fewest cusps first, then the smoothest curve. Pulling farther
            # beyond the geometry-derived setup point forces a sharper swing
            # back, so the gentlest valid candidate is the one to execute.
            self.selected_candidate = min(
                candidates,
                key=lambda item: (
                    item.metrics.cusp_count,
                    item.metrics.max_curvature,
                    item.metrics.total_length,
                    -item.metrics.reverse_length,
                ))
            candidate = self.selected_candidate
            self._log_info(f'selected: {candidate.slot.name}')
            self._log_info(
                f'selected {candidate.slot.name}: setup_offset='
                f'{candidate.setup_offset:.2f}, entry_depth={candidate.entry_depth:.2f}')
            for name, pose in candidate.named_poses.items():
                self._log_info(
                    f'{name}_pose map=({pose.pose.position.x:.3f}, '
                    f'{pose.pose.position.y:.3f}, '
                    f'{yaw_from_quaternion(pose.pose.orientation):.3f})')

            self._publish_status('VALIDATE_PARKING_PATH')
            self._publish_plan(candidate)
            metrics = candidate.metrics
            self._log_info(
                f'path poses={len(candidate.path.poses)} '
                f'total={metrics.total_length:.3f}m '
                f'forward={metrics.forward_length:.3f}m '
                f'reverse={metrics.reverse_length:.3f}m '
                f'cusps={metrics.cusp_count} '
                f'first_reverse_index={metrics.first_reverse_index} '
                f'max_curvature={metrics.max_curvature:.3f} 1/m '
                f'final_position_error={metrics.final_position_error:.3f}m '
                f'final_yaw_error={metrics.final_yaw_error:.3f}rad '
                f'controller_id={self.get_parameter("controller_id").value} '
                f'goal_checker_id={self.get_parameter("goal_checker_id").value}')

            if self.bench_mode:
                segments = self._direction_segments(
                    candidate.path, candidate.metrics.directions)
                self._log_info(
                    '[BENCH EXPECTED SEGMENTS]\n' + '\n'.join(
                        f'segment={number + 1} '
                        f'direction={"REVERSE" if segment.direction < 0 else "FORWARD"} '
                        f'length={segment.length:.3f}m'
                        for number, segment in enumerate(segments)))

            self._publish_status('PLAN_VALID')

            if self.rviz_only:
                self._publish_status('PLANNING_RVIZ_RETURN')
                rviz_return = self._plan_rviz_return(
                    candidate.named_poses['final'], candidate.slot)
                if rviz_return is None:
                    self._fail(
                        'could not build a collision-free RViz return path')
                    return
                self._publish_forward_exit_plan(rviz_return)
                self._log_info(
                    f'RViz return path poses={len(rviz_return.path.poses)} '
                    f'total={rviz_return.metrics.total_length:.3f}m '
                    f'forward={rviz_return.metrics.forward_length:.3f}m '
                    f'reverse={rviz_return.metrics.reverse_length:.3f}m '
                    f'cusps={rviz_return.metrics.cusp_count} '
                    f'max_curvature='
                    f'{rviz_return.metrics.max_curvature:.3f} 1/m '
                    f'position_error='
                    f'{rviz_return.metrics.final_position_error:.3f}m '
                    f'yaw_error={rviz_return.metrics.final_yaw_error:.3f}rad')
                self._publish_status('RVIZ_RETURN_PLAN_VALID')
                self._log_info(
                    'rviz_only: parking and planner-generated return Paths '
                    'published; fake vehicle owns visual playback and no '
                    'FollowPath or command output exists')
                return

            if not self.execute_path:
                if (self.bench_mode and not bool(self.get_parameter(
                        'bench_plan_exit').value)):
                    self._log_info(
                        'execute=false: BENCH parking plan validated; '
                        'no FollowPath goal or wheel motion was requested')
                    self._publish_status('FINISHED')
                    return
                parking_stop_pose = self._expected_wheels_inside_stop_pose(
                    candidate.slot)
                if parking_stop_pose is None:
                    self._fail('could not calculate the expected wheel-inside stop pose')
                    return
                self._publish_status('BUILD_FORWARD_EXIT_POSES')
                self._publish_status('PLANNING_FORWARD_EXIT')
                self._log_info(
                    '[T-PARK][EXIT PLAN] mode=plan_only '
                    'start=configured PARKED pose '
                    'goal=captured course entrance planner=ForwardExit(DUBIN)')
                forward_exit = self._plan_forward_right_exit(
                    parking_stop_pose, candidate.slot, use_current_start=False)
                if forward_exit is None:
                    self._fail('could not build a safe forward-right exit path')
                    return
                self._publish_forward_exit_plan(forward_exit)
                self._publish_status('FORWARD_EXIT_PLAN_VALID')
                self._log_info(
                    'execute=false: parking and forward-right exit plans are '
                    'validated and published; vehicle remains stopped')
                self._publish_status('FINISHED')
                return

            if bool(self.get_parameter('freeze_slam_during_execution').value):
                if not self._toggle_slam_measurements():
                    if self._abort_requested():
                        self._cancelled()
                        return
                    self._fail('could not pause SLAM measurements for stable execution TF')
                    return
                self.slam_paused_by_node = True
                self._log_info(
                    'SLAM measurements paused during FollowPath; '
                    'map->odom remains fixed')

            self.parking_wheels_inside = False
            self.wheel_inside_confirm_count = 0
            self.last_wheel_states = None
            self.motion_phase = 'parking'
            self._publish_status('EXECUTING_PARKING')
            self.last_execution_failure_reason = ''
            if not self._execute(
                    candidate, bool(self.get_parameter(
                        'stop_when_all_wheels_inside').value)):
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail(
                        self.last_execution_failure_reason
                        or 'parking path execution failed')
                return
            if (bool(self.get_parameter('stop_when_all_wheels_inside').value)
                    and not self.parking_wheels_inside):
                self._fail(
                    'parking FollowPath ended before all four wheels were '
                    'confirmed inside the selected slot')
                return

            self._publish_status('STOPPING_AFTER_PARKING')
            if not self._wait_until_stopped(float(self.get_parameter(
                    'parking_stop_grace_timeout').value)):
                self._log_warn(
                    'vehicle still moving after parking-goal cancellation; '
                    'sending zero-velocity safety pulse')
                self._emergency_stop(emergency=True)
                if not self._wait_until_stopped(float(
                        self.get_parameter('stop_wait_timeout').value)):
                    self._fail(
                        'vehicle did not stop after parking FollowPath cancel')
                    return
            self._log_info(
                'parking stop confirmed; planning exit without added delay')

            parking_stop_pose = self._current_map_pose_stamped()
            if parking_stop_pose is None:
                self._fail(
                    f'could not capture map->{self.robot_base_frame} '
                    'parking stop pose')
                return
            self._log_pose('parking_stop_pose', parking_stop_pose)
            if not self._validate_and_log_parked(
                    candidate.slot, parking_stop_pose):
                self._fail(
                    'parking stop pose failed full-footprint/end-clearance '
                    'validation')
                return
            self._publish_status('PARKED')
            if not self._confirm_command_zero('EXIT STOP'):
                self._fail(
                    'parked command handoff did not provide three fresh zero '
                    'drive/wheel samples')
                return
            if not bool(self.get_parameter('return_to_entrance').value):
                if not self._confirm_command_zero('FINAL ZERO'):
                    self._fail('final command zero confirmation failed')
                    return
                self._publish_status('SUCCESS', 'return_to_entrance=false')
                return
            if str(self.get_parameter('exit_mode').value) != 'forward_right':
                self._fail('unsupported exit_mode; only forward_right is safe')
                return

            self._publish_status('BUILD_FORWARD_EXIT_POSES')
            self._publish_status('PLANNING_FORWARD_EXIT')
            self._log_info(
                '[T-PARK][EXIT PLAN] start=validated PARKED pose '
                'goal=captured course entrance planner=ForwardExit(DUBIN)')
            forward_exit = self._plan_forward_right_exit(
                parking_stop_pose, candidate.slot, use_current_start=True)
            if forward_exit is None:
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail('could not build a safe forward-right exit path')
                return
            self._publish_forward_exit_plan(forward_exit)
            self._publish_status('FORWARD_EXIT_PLAN_VALID')
            self._log_info(
                f'forward exit path poses={len(forward_exit.path.poses)} '
                f'total={forward_exit.metrics.total_length:.3f}m '
                f'forward={forward_exit.metrics.forward_length:.3f}m '
                f'reverse={forward_exit.metrics.reverse_length:.3f}m '
                f'cusps={forward_exit.metrics.cusp_count} '
                f'max_curvature={forward_exit.metrics.max_curvature:.3f} 1/m '
                f'turn_direction={forward_exit.turn_direction}')

            self.motion_phase = 'forward_exit'
            self.forward_exit_poses = forward_exit.named_poses
            self.minimum_exit_linear_x = float('inf')
            self._publish_status('EXECUTING_FORWARD_FROM_SLOT')
            if not self._execute_forward_exit(forward_exit):
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail('forward-right exit FollowPath failed')
                return
            if not self._confirm_motion_stop('forward exit path'):
                self._fail('vehicle did not stop after forward exit path')
                return

            self._publish_status('VERIFY_ENTRANCE_RETURN')
            position_error, yaw_error = self._verify_entrance_return()
            if position_error > float(self.get_parameter(
                    'entrance_return_position_tolerance').value):
                self._fail(
                    f'entrance return position error {position_error:.3f}m')
                return
            if yaw_error > float(self.get_parameter(
                    'entrance_return_yaw_tolerance').value):
                self._fail(f'entrance return yaw error {yaw_error:.3f}rad')
                return
            self._publish_status(
                'RETURNED_TO_ENTRANCE',
                f'position_error={position_error:.3f}m '
                f'yaw_error={yaw_error:.3f}rad')
            self._log_info(
                '[T-PARK][EXIT COMPLETE] '
                f'position_error={position_error:.3f}m '
                f'yaw_error={yaw_error:.3f}rad')
            if not self._confirm_command_zero('FINAL ZERO'):
                self._fail('final command zero confirmation failed')
                return
            self._publish_status(
                'SUCCESS',
                f'position_error={position_error:.3f}m yaw_error={yaw_error:.3f}rad')
        except Exception as exc:  # Keep a failed parking run fail-safe.
            if self._runtime_ok():
                self._log_error(
                    f'unhandled automatic parking error: {exc}\n'
                    f'{traceback.format_exc()}')
                self._fail(str(exc))
        finally:
            self.active_plan_goal = None
            self.active_follow_goal = None
            self.motion_phase = 'idle'
            self.active_motion_path = None
            self.active_motion_metrics = None
            self.active_motion_final = None
            if self.slam_paused_by_node and self._runtime_ok():
                if self._toggle_slam_measurements():
                    self._log_info('SLAM measurements resumed')
                else:
                    self._log_error(
                        'failed to resume SLAM measurements after parking')
                self.slam_paused_by_node = False

    def _toggle_slam_measurements(self) -> bool:
        if not self._runtime_ok():
            return False
        if not self.slam_pause_client.wait_for_service(timeout_sec=2.0):
            self._log_error(
                '/slam_toolbox/pause_new_measurements is unavailable')
            return False
        if not self._runtime_ok():
            return False
        response = self._wait_future(
            self.slam_pause_client.call_async(Pause.Request()), 2.0)
        return response is not None and bool(response.status)

    def _wait_for_system(self) -> bool:
        deadline = time.monotonic() + float(
            self.get_parameter('system_wait_timeout').value)
        while time.monotonic() < deadline and not self._abort_requested():
            with self.readiness_lock:
                readiness = dict(self.readiness_cache)
            if readiness and all(readiness.values()) and self._vehicle_stopped():
                return True
            if not self._interruptible_sleep(0.25):
                return False
        with self.readiness_lock:
            readiness = dict(self.readiness_cache)
        missing = [name for name, ready in readiness.items() if not ready]
        if not readiness:
            missing = ['readiness_probe']
        if not self._vehicle_stopped():
            missing.append('vehicle_stopped')
        self._log_error(
            'system readiness timeout\nMissing:\n'
            + '\n'.join(f'- {name}' for name in missing))
        return False

    def _clear_global_costmap(self) -> bool:
        """Reset stale map-frame observations before the fresh sensor gate."""
        if not self._vehicle_stopped():
            self._log_error(
                'refusing global costmap reset while vehicle is moving')
            return False
        if not self.global_costmap_clear_client.wait_for_service(
                timeout_sec=2.0):
            self._log_error(
                '/global_costmap/clear_entirely_global_costmap is unavailable')
            return False
        response = self._wait_future(
            self.global_costmap_clear_client.call_async(
                ClearEntireCostmap.Request()),
            3.0)
        if response is None:
            self._log_error('global costmap reset failed or timed out')
            return False
        self._log_info(
            '[GLOBAL COSTMAP RESET]\n'
            'reason=discard map-frame obstacle markings from before the '
            'stable entrance localization\n'
            'next_gate=fresh scans and costmap updates required')
        return True

    def _readiness_timer_callback(self) -> None:
        # A timer makes readiness observable even when auto_start=false and the
        # node is intentionally waiting for /t_parking/start.
        if (not self._runtime_ok()
                or not self.readiness_lock.acquire(blocking=False)):
            return
        try:
            readiness = self._readiness_snapshot()
            self.readiness_cache = readiness
        except Exception as exc:
            # An unhandled exception here would otherwise be swallowed by the
            # executor's callback future without ever being logged, silently
            # ending all future readiness observation for the rest of the run.
            self._log_error(
                f'readiness timer callback failed: {exc}\n'
                f'{traceback.format_exc()}')
            return
        finally:
            self.readiness_lock.release()
        lines = ['[READINESS]']
        lines.extend(
            f'{name}={str(value).lower()}'
            for name, value in readiness.items())
        self._log_info('\n'.join(lines))

    def _readiness_snapshot(self) -> Dict[str, bool]:
        if not self._runtime_ok():
            return {}
        now = time.monotonic()
        scan_timeout = float(self.get_parameter('scan_timeout').value)
        with self.data_lock:
            map_ready = self.map_msg is not None
            global_costmap_ready = self.global_costmap is not None
            local_costmap_ready = self.local_costmap is not None
            robot_spawned = self.odom_msg is not None
            rear_received_at = self.rear_scan_received_at
            bench_anchor_ready = self.bench_anchor_ready
        lifecycle = {
            name: self._poll_lifecycle_state(name, client)
            for name, client in self.lifecycle_clients.items()
        }
        if self.rviz_only:
            return {
                'robot_spawned': robot_spawned,
                'odom': robot_spawned,
                'map': map_ready,
                'map_to_base_tf': bool(self.tf_buffer.can_transform(
                    'map', self.robot_base_frame, Time(),
                    timeout=Duration(seconds=0.05))),
                'planner_active': lifecycle['planner_server'],
                'compute_path_server': self.plan_client.wait_for_server(
                    timeout_sec=0.05),
                'global_costmap': global_costmap_ready,
            }
        if self.bench_mode:
            return {
                'virtual_bench_odom': robot_spawned,
                'saved_map': map_ready,
                'bench_anchor': bench_anchor_ready,
                'map_to_base_tf': bool(self.tf_buffer.can_transform(
                    'map', self.robot_base_frame, Time(),
                    timeout=Duration(seconds=0.05))),
                'planner_active': lifecycle['planner_server'],
                'controller_active': lifecycle['controller_server'],
                'compute_path_server': self.plan_client.wait_for_server(
                    timeout_sec=0.05),
                'follow_path_server': self.follow_client.wait_for_server(
                    timeout_sec=0.05),
                'global_costmap_static_only': global_costmap_ready,
                'local_costmap_static_only': local_costmap_ready,
            }
        return {
            'robot_spawned': robot_spawned,
            'front_scan': now - self.scan_received_at < scan_timeout,
            'rear_scan': now - rear_received_at < scan_timeout,
            'odom': robot_spawned,
            'map': map_ready,
            'map_to_base_tf': bool(self.tf_buffer.can_transform(
                'map', self.robot_base_frame, Time(),
                timeout=Duration(seconds=0.05))),
            'planner_active': lifecycle['planner_server'],
            'controller_active': lifecycle['controller_server'],
            'compute_path_server': self.plan_client.wait_for_server(
                timeout_sec=0.05),
            'follow_path_server': self.follow_client.wait_for_server(
                timeout_sec=0.05),
            'global_costmap': global_costmap_ready,
            'local_costmap': local_costmap_ready,
        }

    def _poll_lifecycle_state(self, name: str, client) -> bool:
        """Return the last known lifecycle state without blocking.

        The readiness timer callback runs on the shared MultiThreadedExecutor.
        Blocking it on a service future forces that future's response to be
        delivered by one of the same pool's threads; under load, overlapping
        1 Hz timer ticks can pile up until every thread is stuck waiting and
        none is left to deliver any response, permanently starving the timer.
        Firing the request asynchronously and caching the result on arrival
        keeps this callback non-blocking so it can never exhaust the pool.

        A request whose response never arrives (dropped under load) must
        not leave lifecycle_pending stuck forever -- that would freeze this
        entry's cached state and stop it from ever being retried.  Any
        pending request older than pending_timeout is treated as lost and
        retried on the next call.
        """
        pending_timeout = 2.0
        now = time.monotonic()
        with self.lifecycle_state_lock:
            pending_since = self.lifecycle_pending.get(name)
            if pending_since is not None:
                if now - pending_since < pending_timeout:
                    return self.lifecycle_state[name]
                self.lifecycle_pending[name] = None
        if not self._runtime_ok() or not client.service_is_ready():
            with self.lifecycle_state_lock:
                return self.lifecycle_state[name]

        def _on_response(completed_future, name=name) -> None:
            try:
                result = completed_future.result()
                active = result is not None and result.current_state.id == 3
            except Exception:
                active = False
            with self.lifecycle_state_lock:
                self.lifecycle_state[name] = active
                self.lifecycle_pending[name] = None

        with self.lifecycle_state_lock:
            self.lifecycle_pending[name] = now
            state = self.lifecycle_state[name]
        client.call_async(GetState.Request()).add_done_callback(_on_response)
        return state

    def _wait_future(self, future, timeout: float):
        done = threading.Event()

        def mark_done(completed_future):
            # Retrieve an exception even when the caller timed out so rclpy
            # does not report an unhandled Destroyable exception at shutdown.
            try:
                completed_future.exception()
            except Exception:
                pass
            done.set()
        future.add_done_callback(mark_done)
        deadline = time.monotonic() + timeout
        while not done.is_set():
            if self._abort_requested() or time.monotonic() >= deadline:
                future.cancel()
                return None
            self._stop_event.wait(min(0.05, deadline - time.monotonic()))
        try:
            return future.result()
        except Exception:
            return None

    def _vehicle_stopped(self) -> bool:
        with self.data_lock:
            odom = self.odom_msg
        if odom is None:
            return False
        twist = odom.twist.twist
        return (
            math.hypot(twist.linear.x, twist.linear.y) < 0.02
            and abs(twist.angular.z) < 0.03)

    def _capture_course_entrance_pose(self) -> bool:
        required = max(1, int(self.get_parameter(
            'entrance_pose_sample_count').value))
        interval = max(0.02, float(self.get_parameter(
            'entrance_pose_sample_interval').value))
        stability_duration = max(interval, float(self.get_parameter(
            'localization_stability_duration').value))
        required = max(required, int(math.ceil(
            stability_duration / interval)) + 1)
        deadline = time.monotonic() + float(self.get_parameter(
            'entrance_pose_capture_timeout').value)
        position_limit = float(self.get_parameter(
            'entrance_pose_position_stability').value)
        yaw_limit = float(self.get_parameter(
            'entrance_pose_yaw_stability').value)
        samples: List[Tuple[float, float, float, float]] = []
        while time.monotonic() < deadline and not self._abort_requested():
            pose = self._current_map_pose()
            if pose is not None:
                samples.append((time.monotonic(), *pose))
                samples = samples[-required:]
                observed_duration = samples[-1][0] - samples[0][0]
                if (len(samples) == required
                        and observed_duration >= stability_duration):
                    average_x = sum(item[1] for item in samples) / required
                    average_y = sum(item[2] for item in samples) / required
                    average_yaw = math.atan2(
                        sum(math.sin(item[3]) for item in samples),
                        sum(math.cos(item[3]) for item in samples))
                    maximum_position_delta = max(
                        math.hypot(
                            item[1] - average_x, item[2] - average_y)
                        for item in samples)
                    maximum_yaw_delta = max(
                        abs(normalize_angle(item[3] - average_yaw))
                        for item in samples)
                    if (maximum_position_delta <= position_limit
                            and maximum_yaw_delta <= yaw_limit):
                        captured = PoseStamped()
                        captured.header.frame_id = 'map'
                        captured.header.stamp = self.get_clock().now().to_msg()
                        captured.pose.position.x = average_x
                        captured.pose.position.y = average_y
                        set_pose_yaw(captured.pose, average_yaw)
                        self.course_entrance_pose = captured
                        self._log_info(
                            '[ENTRANCE POSE CAPTURED]\n'
                            f'x={average_x:.3f}\n'
                            f'y={average_y:.3f}\n'
                            f'yaw={average_yaw:.3f}\n'
                            f'samples={required}\n'
                            f'stability_duration={observed_duration:.2f}s\n'
                            f'max_position_delta={maximum_position_delta:.6f}m\n'
                            f'max_yaw_delta={maximum_yaw_delta:.6f}rad')
                        return True
            if not self._interruptible_sleep(interval):
                return False
        return False

    def _wheel_centers_map(
            self) -> Optional[Dict[str, Tuple[float, float]]]:
        labels = ('front_left', 'front_right', 'rear_left', 'rear_right')
        frames = [str(value) for value in self.get_parameter(
            'wheel_frames').value]
        if len(frames) == 4:
            positions: Dict[str, Tuple[float, float]] = {}
            try:
                for label, frame in zip(labels, frames):
                    transform = self.tf_buffer.lookup_transform(
                        'map', frame, Time(),
                        timeout=Duration(seconds=0.03))
                    positions[label] = (
                        transform.transform.translation.x,
                        transform.transform.translation.y)
                if self.wheel_position_source != 'tf':
                    self._log_info(
                        'wheel centres sourced from TF: '
                        + ', '.join(frames))
                    self.wheel_position_source = 'tf'
                return positions
            except tf2_ros.TransformException:
                pass

        current = self._current_map_pose()
        if current is None:
            return None
        wheel_base = float(self.get_parameter('wheel_base').value)
        wheel_track = float(self.get_parameter('wheel_track').value)
        if wheel_base <= 0.0 or wheel_track <= 0.0:
            return None
        if self.wheel_position_source != 'geometry':
            self._log_warn(
                'wheel-link TF unavailable; using URDF wheel-base/track '
                f'geometry relative to map->{self.robot_base_frame}')
            self.wheel_position_source = 'geometry'
        half_base = 0.5 * wheel_base
        half_track = 0.5 * wheel_track
        local = {
            'front_left': (half_base, half_track),
            'front_right': (half_base, -half_track),
            'rear_left': (-half_base, half_track),
            'rear_right': (-half_base, -half_track),
        }
        cosine = math.cos(current[2])
        sine = math.sin(current[2])
        return {
            label: (
                current[0] + cosine * point[0] - sine * point[1],
                current[1] + sine * point[0] + cosine * point[1])
            for label, point in local.items()
        }

    def _parking_target_depth(
            self, slot: Slot) -> Tuple[float, float]:
        """Return the base pose for the configured physical end clearance.

        This is a footprint calculation, not a path-length adjustment.  The
        slot yaw points from its closed end toward its mouth; the rear bumper
        is placed exactly ``parking_end_clearance_m`` from that closed end.
        """
        return parking_target_from_end_clearance(
            slot_center_x=slot.odom_x,
            slot_center_y=slot.odom_y,
            slot_yaw=slot.yaw,
            min_x=slot.min_x,
            max_x=slot.max_x,
            min_y=slot.min_y,
            max_y=slot.max_y,
            vehicle_length=float(self.get_parameter('vehicle_length').value),
            vehicle_center_x_offset=float(self.get_parameter(
                'vehicle_center_x_offset').value),
            parking_end_clearance_m=float(self.get_parameter(
                'parking_end_clearance_m').value),
        )

    def _parking_pose_assessment(
            self, slot: Slot, pose: PoseStamped):
        return assess_parking_pose(
            base_x=pose.pose.position.x,
            base_y=pose.pose.position.y,
            vehicle_yaw=yaw_from_quaternion(pose.pose.orientation),
            slot_yaw=slot.yaw,
            min_x=slot.min_x,
            max_x=slot.max_x,
            min_y=slot.min_y,
            max_y=slot.max_y,
            vehicle_length=float(self.get_parameter('vehicle_length').value),
            vehicle_width=float(self.get_parameter('vehicle_width').value),
            vehicle_center_x_offset=float(self.get_parameter(
                'vehicle_center_x_offset').value),
        )

    def _log_parking_geometry(self, slot: Slot) -> None:
        target_x, target_y = self._parking_target_depth(slot)
        target = self._map_pose(target_x, target_y, slot.yaw)
        assessment = self._parking_pose_assessment(slot, target)
        front_extent, rear_extent = vehicle_longitudinal_extents(
            float(self.get_parameter('vehicle_length').value),
            float(self.get_parameter('vehicle_center_x_offset').value))
        self._log_info(
            '[T-PARK][PARKING GEOMETRY]\n'
            f'slot_id={slot.name}\n'
            f'slot_bounds_x={slot.min_x:.3f}..{slot.max_x:.3f}\n'
            f'slot_bounds_y={slot.min_y:.3f}..{slot.max_y:.3f}\n'
            f'slot_yaw={slot.yaw:.6f}\n'
            f'slot_depth={assessment.slot_depth:.3f}m\n'
            f'pose_reference={self.robot_base_frame}\n'
            f'vehicle_length={float(self.get_parameter("vehicle_length").value):.3f}m\n'
            f'vehicle_center_x_offset='
            f'{float(self.get_parameter("vehicle_center_x_offset").value):.3f}m\n'
            f'front_extent={front_extent:.3f}m\n'
            f'rear_extent={rear_extent:.3f}m\n'
            f'legacy_rear_clearance='
            f'{float(self.get_parameter("rear_clearance").value):.3f}m\n'
            f'parking_end_clearance_m='
            f'{float(self.get_parameter("parking_end_clearance_m").value):.3f}m\n'
            f'final_parking_target=({target_x:.3f},{target_y:.3f},'
            f'{slot.yaw:.6f})\n'
            f'calculated_end_clearance={assessment.end_clearance:.3f}m\n'
            f'calculated_front_clearance={assessment.front_clearance:.3f}m\n'
            f'calculated_side_clearance={assessment.side_clearance:.3f}m\n'
            f'footprint_inside_slot={str(assessment.footprint_inside).lower()}\n'
            'candidate_depth_coordinate=distance inward from slot mouth along '
            'negative slot-forward axis')

    def _update_wheels_inside(self, slot: Slot) -> bool:
        # Wheel TF positions and slot bounds are both in map.  Transforming
        # the wheel centres to odom here made the real AMCL run compare two
        # different coordinate systems, so the wheels-inside stop could never
        # become true even after a geometrically correct parking manoeuvre.
        positions = self._wheel_centers_map()
        if positions is None:
            self.wheel_inside_confirm_count = 0
            return False
        current = self._current_map_pose_stamped()
        if current is None:
            self.wheel_inside_confirm_count = 0
            return False
        inset = (
            float(self.get_parameter('wheel_radius').value)
            + float(self.get_parameter('wheel_inside_margin').value))
        target_x, target_y = self._parking_target_depth(slot)
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        min_x, max_x = slot.min_x + inset, slot.max_x - inset
        min_y, max_y = slot.min_y + inset, slot.max_y - inset
        labels = ('front_left', 'front_right', 'rear_left', 'rear_right')
        states = tuple(
            min_x <= positions[label][0] <= max_x
            and min_y <= positions[label][1] <= max_y
            for label in labels)
        assessment = self._parking_pose_assessment(slot, current)
        current_projection = (
            current.pose.position.x * forward_x
            + current.pose.position.y * forward_y)
        target_projection = target_x * forward_x + target_y * forward_y
        depth_error = current_projection - target_projection
        depth_reached = depth_error <= float(self.get_parameter(
            'final_position_tolerance').value)
        end_clearance_valid = (
            assessment.end_clearance + 1.0e-6
            >= float(self.get_parameter('parking_end_clearance_m').value))
        accepted = (
            all(states)
            and assessment.footprint_inside
            and depth_reached
            and end_clearance_valid)
        if accepted:
            self.wheel_inside_confirm_count += 1
        else:
            self.wheel_inside_confirm_count = 0
        required = max(1, int(self.get_parameter(
            'wheel_inside_confirm_count').value))
        now = time.monotonic()
        if (states != self.last_wheel_states or all(states)
                or now - self.last_wheel_log >= 0.5):
            self._log_info(
                '[WHEEL CHECK]\n'
                + '\n'.join(
                    f'{label}={str(state).lower()}'
                    for label, state in zip(labels, states))
                + f'\nfootprint_inside_slot='
                f'{str(assessment.footprint_inside).lower()}'
                f'\ndepth_error={depth_error:+.3f}m'
                f'\ndepth_reached={str(depth_reached).lower()}'
                f'\nend_clearance={assessment.end_clearance:.3f}m'
                f'\nfront_clearance={assessment.front_clearance:.3f}m'
                + f'\nconfirm={self.wheel_inside_confirm_count}/{required}')
            self.last_wheel_states = states
            self.last_wheel_log = now
        if self.wheel_inside_confirm_count < required:
            return False
        self._log_info('PARKING POSITION ACCEPTED')
        return True

    def _wait_for_map_and_slots(self) -> List[Slot]:
        deadline = time.monotonic() + float(
            self.get_parameter('map_wait_timeout').value)
        observation_attempted = False
        last_states = None
        while time.monotonic() < deadline and not self._abort_requested():
            states = {slot.name: self._slot_state(slot) for slot in self.slots}
            state_tuple = tuple(states[slot.name] for slot in self.slots)
            if state_tuple != last_states:
                self._log_info(
                    '[SLOT CHECK]\n' + '\n'.join(
                        f'{slot.name}: {states[slot.name]}'
                        for slot in self.slots))
                last_states = state_tuple
            available = [
                slot for slot in self.slots
                if states[slot.name] == SLOT_FREE]
            if available:
                return available
            # If a saved map still leaves either bay unknown, an executing run
            # may make one common, forward Nav2 observation pass before slot
            # selection. Plan-only mode never enters this branch and remains
            # motionless.
            if (self.execute_path and not observation_attempted
                    and SLOT_UNKNOWN in states.values()):
                observation_attempted = True
                self._publish_status('MAP_PARKING_BAY')
                if not self._observe_parking_bay():
                    return []
                if not self._interruptible_sleep(float(self.get_parameter(
                        'map_observation_settle_time').value)):
                    return []
                self.obstacle_observation_ready = False
                if not self._wait_for_stable_obstacle_observation():
                    return []
                continue
            if not self._interruptible_sleep(0.5):
                return []
        return []

    def _observe_parking_bay(self) -> bool:
        pose = self._odom_pose_to_map(
            float(self.get_parameter('observation_odom_x').value),
            float(self.get_parameter('observation_odom_y').value),
            float(self.get_parameter('observation_odom_yaw').value))
        if pose is None:
            self._log_error(
                'cannot transform parking-bay observation pose into map')
            return False
        path = self._request_plan([pose])
        if path is None:
            self._log_error(
                'Nav2 could not plan the parking-bay observation pass')
            return False
        self._log_info(
            'fresh map has no fully known slot; executing a common forward '
            f'observation pass with {len(path.poses)} poses')
        if not self._execute_path_action(path, 1, 1, 1):
            self._log_error('parking-bay observation pass failed')
            return False
        self._emergency_stop()
        if not self._wait_until_stopped(float(
                self.get_parameter('stop_wait_timeout').value)):
            self._log_error(
                'vehicle did not settle after parking-bay observation pass')
            return False
        self._log_info(
            'parking-bay observation pass complete; vehicle stopped')
        return True

    def _wait_for_stable_obstacle_observation(self) -> bool:
        """Require independent fresh lidar and costmap observations."""
        required_scans = max(1, int(self.get_parameter(
            'obstacle_observation_frames').value))
        required_costmaps = max(1, int(self.get_parameter(
            'costmap_observation_updates').value))
        settle_time = max(0.0, float(self.get_parameter(
            'obstacle_observation_settle_time').value))
        timeout = float(self.get_parameter('map_wait_timeout').value)
        started = time.monotonic()
        deadline = started + timeout
        with self.data_lock:
            first_front = self.front_scan_count
            first_rear = self.rear_scan_count
            first_global = self.global_costmap_count
            first_local = self.local_costmap_count
        front_delta = rear_delta = global_delta = local_delta = 0
        blocking_conditions: List[str] = []
        next_debug_log = started
        while time.monotonic() < deadline and not self._abort_requested():
            now = time.monotonic()
            with self.data_lock:
                front_count = self.front_scan_count
                rear_count = self.rear_scan_count
                global_count = self.global_costmap_count
                local_count = self.local_costmap_count
                front_delta = front_count - first_front
                rear_delta = rear_count - first_rear
                global_delta = global_count - first_global
                local_delta = local_count - first_local
                front_age = now - self.scan_received_at
                rear_age = now - self.rear_scan_received_at
                global_age = now - self.global_costmap_received_at
                local_age = now - self.local_costmap_received_at
                front_scan = self.front_scan_msg
                rear_scan = self.rear_scan_msg
            scan_timeout = float(self.get_parameter('scan_timeout').value)
            front_frame = (
                front_scan.header.frame_id.lstrip('/')
                if front_scan is not None else '')
            rear_frame = (
                rear_scan.header.frame_id.lstrip('/')
                if rear_scan is not None else '')
            front_tf_ready = bool(front_frame) and self.tf_buffer.can_transform(
                'map', front_frame, Time(),
                timeout=Duration(seconds=0.05))
            rear_tf_ready = bool(rear_frame) and self.tf_buffer.can_transform(
                'map', rear_frame, Time(),
                timeout=Duration(seconds=0.05))
            sector_half_width = math.radians(float(
                self.get_parameter('rear_emergency_sector_deg').value))
            front_sector_valid, front_sector_detail = (
                self._zero_angle_scan_sector_valid(
                    front_scan, sector_half_width))
            rear_sector_valid, rear_sector_detail = (
                self._zero_angle_scan_sector_valid(
                    rear_scan, sector_half_width))

            # Each producer is evaluated against its own post-reset baseline
            # and receive time.  No front/rear or global/local counter/stamp
            # equality is required: the devices publish at different rates.
            front_fresh = (
                front_delta >= required_scans and front_age < scan_timeout)
            rear_fresh = (
                rear_delta >= required_scans and rear_age < scan_timeout)
            global_fresh = (
                global_delta >= required_costmaps
                and global_age < scan_timeout)
            local_fresh = (
                local_delta >= required_costmaps
                and local_age < scan_timeout)
            settled = now - started >= settle_time
            conditions = {
                'FRONT_SCAN': front_fresh,
                'REAR_SCAN': rear_fresh,
                'GLOBAL_COSTMAP': global_fresh,
                'LOCAL_COSTMAP': local_fresh,
                'FRONT_SECTOR': front_sector_valid,
                'REAR_SECTOR': rear_sector_valid,
                'FRONT_TF': front_tf_ready,
                'REAR_TF': rear_tf_ready,
                'SETTLE_TIME': settled,
            }
            blocking_conditions = [
                name for name, ready in conditions.items() if not ready]

            if now >= next_debug_log:
                front_stamp = self._scan_stamp_text(front_scan)
                rear_stamp = self._scan_stamp_text(rear_scan)
                self._log_info(
                    '[FRESH OBSERVATION DEBUG]\n'
                    f'elapsed={now - started:.3f}s\n'
                    f'front_scan_count={front_count} '
                    f'front_required_count={required_scans} '
                    f'front_delta={front_delta} '
                    f'front_age={front_age:.3f}s '
                    f'front_fresh={front_fresh}\n'
                    f'rear_scan_count={rear_count} '
                    f'rear_required_count={required_scans} '
                    f'rear_delta={rear_delta} '
                    f'rear_age={rear_age:.3f}s '
                    f'rear_fresh={rear_fresh}\n'
                    f'global_costmap_count={global_count} '
                    f'global_required_count={required_costmaps} '
                    f'global_delta={global_delta} '
                    f'global_age={global_age:.3f}s '
                    f'global_costmap_fresh={global_fresh}\n'
                    f'local_costmap_count={local_count} '
                    f'local_required_count={required_costmaps} '
                    f'local_delta={local_delta} '
                    f'local_age={local_age:.3f}s '
                    f'local_costmap_fresh={local_fresh}\n'
                    f'front_frame={front_frame or "<none>"} '
                    f'front_stamp={front_stamp} '
                    f'front_tf_ready={front_tf_ready} '
                    f'front_sector_valid={front_sector_valid} '
                    f'front_sector_detail={front_sector_detail}\n'
                    f'rear_frame={rear_frame or "<none>"} '
                    f'rear_stamp={rear_stamp} '
                    f'rear_tf_ready={rear_tf_ready} '
                    f'rear_sector_valid={rear_sector_valid} '
                    f'rear_sector_detail={rear_sector_detail}\n'
                    f'settle={now - started:.3f}/{settle_time:.3f}s '
                    f'blocking_condition='
                    f'{blocking_conditions if blocking_conditions else "NONE"}')
                next_debug_log = now + 0.75

            if not blocking_conditions:
                self.obstacle_observation_ready = True
                self._log_info(
                    '[OBSTACLE OBSERVATION STABLE]\n'
                    f'front_scan_frames={front_delta}\n'
                    f'front_age={front_age:.3f}s\n'
                    f'rear_scan_frames={rear_delta}\n'
                    f'rear_age={rear_age:.3f}s\n'
                    f'global_costmap_updates={global_delta}\n'
                    f'global_costmap_age={global_age:.3f}s\n'
                    f'local_costmap_updates={local_delta}\n'
                    f'local_costmap_age={local_age:.3f}s\n'
                    f'front_frame={front_frame}\n'
                    f'rear_frame={rear_frame}\n'
                    f'settle_time={now - started:.2f}s')
                return True
            if not self._interruptible_sleep(0.05):
                return False
        self._log_error(
            '[FRESH OBSERVATION TIMEOUT] '
            f'blocking_conditions={blocking_conditions} '
            f'front={front_delta}/{required_scans} '
            f'rear={rear_delta}/{required_scans} '
            f'global_costmap={global_delta}/{required_costmaps} '
            f'local_costmap={local_delta}/{required_costmaps}')
        return False

    @staticmethod
    def _scan_stamp_text(scan: Optional[LaserScan]) -> str:
        if scan is None:
            return '<none>'
        stamp = scan.header.stamp
        return f'{stamp.sec}.{stamp.nanosec:09d}'

    @staticmethod
    def _zero_angle_scan_sector_valid(
            scan: Optional[LaserScan], half_width: float) -> Tuple[bool, str]:
        """Validate the physical zero-radian sector of one lidar message."""
        if scan is None:
            return False, 'no_message'
        if not scan.ranges or scan.angle_increment == 0.0:
            return False, 'empty_scan'
        sector_values = [
            float(value) for index, value in enumerate(scan.ranges)
            if abs(scan.angle_min + index * scan.angle_increment) <= half_width]
        if not sector_values:
            return False, 'zero_rad_not_covered'
        valid_count = sum(
            1 for value in sector_values
            if ((math.isfinite(value)
                 and scan.range_min <= value <= scan.range_max)
                or (math.isinf(value) and value > 0.0)))
        return (
            valid_count > 0,
            f'valid_samples={valid_count}/{len(sector_values)}')

    def _slot_state(self, slot: Slot) -> str:
        if not self.obstacle_observation_ready:
            return SLOT_UNKNOWN
        final_pose = self._map_pose(
            slot.odom_x, slot.odom_y, slot.yaw)
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        clearance = float(self.get_parameter('footprint_clearance').value)

        # First complete vehicle footprint just inside the actual bay
        # entrance.  The entrance boundary depends on slot orientation.
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        entrance_x = slot.odom_x
        entrance_y = slot.odom_y

        if abs(forward_x) >= abs(forward_y):
            if forward_x > 0.0:
                entrance_x = slot.min_x + 0.5 * length + clearance
            else:
                entrance_x = slot.max_x - 0.5 * length - clearance
        else:
            if forward_y > 0.0:
                entrance_y = slot.max_y - 0.5 * length - clearance
            else:
                entrance_y = slot.min_y + 0.5 * length + clearance

        entrance_pose = self._map_pose(
            entrance_x, entrance_y, slot.yaw)
        samples = self._footprint_samples(
            final_pose, length + 2.0 * clearance, width + 2.0 * clearance)
        samples.extend(self._footprint_samples(
            entrance_pose,
            length + 2.0 * clearance,
            width + 2.0 * clearance))
        # Also require known free space immediately behind the parked vehicle.
        rear = self._offset_pose(final_pose, -0.70, 0.0)
        samples.extend(self._footprint_samples(rear, 0.50, width + 2.0 * clearance))
        with self.data_lock:
            map_msg = self.map_msg
            costmap = self.global_costmap
        if map_msg is None or costmap is None:
            return SLOT_UNKNOWN
        # Inflation costs describe whether the *robot centre* can occupy a
        # cell.  Applying them again at every footprint sample double-counts
        # the footprint and incorrectly rejects the otherwise clear 1.5 m
        # slot.  Check the relevant centre poses against inflation, then use
        # raw SLAM occupancy over the complete footprint rectangles.
        for pose in (entrance_pose, final_pose, rear):
            cost_value = self._cost_value(
                costmap, pose.pose.position.x, pose.pose.position.y)
            if cost_value is None or cost_value == 255:
                return SLOT_UNKNOWN
            if cost_value >= 253:
                self._log_info(
                    f'[SLOT DEBUG] {slot.name} OCCUPIED: '
                    f'center costmap lethal at '
                    f'({pose.pose.position.x:.3f}, '
                    f'{pose.pose.position.y:.3f}) '
                    f'cost={cost_value}')
                return SLOT_OCCUPIED
        # A spawned box is 0.45 m wide, so a lidar return on its front or side
        # face need not land on the slot centre cell. Inspect lethal obstacle
        # cells over the complete candidate footprint while deliberately
        # ignoring non-lethal inflation costs (those would double-count the
        # vehicle footprint and reject a geometrically open 1.5 m bay).
        for x, y in samples:
            cost_value = self._cost_value(costmap, x, y)
            if cost_value is None or cost_value == 255:
                return SLOT_UNKNOWN
            if cost_value >= 254:
                self._log_info(
                    f'[SLOT DEBUG] {slot.name} OCCUPIED: '
                    f'footprint costmap lethal at '
                    f'({x:.3f}, {y:.3f}) cost={cost_value}')
                return SLOT_OCCUPIED
        occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        for x, y in samples:
            map_value = self._occupancy_value(map_msg, x, y)
            if map_value is None or map_value < 0:
                return SLOT_UNKNOWN
            if map_value >= occupied_threshold:
                self._log_info(
                    f'[SLOT DEBUG] {slot.name} OCCUPIED: '
                    f'map occupied at ({x:.3f}, {y:.3f}) '
                    f'value={map_value} threshold={occupied_threshold}')
                return SLOT_OCCUPIED
        if not self._final_footprint_inside_slot(slot, final_pose):
            self._log_info(
                f'[SLOT DEBUG] {slot.name} OCCUPIED: '
                'final footprint is outside slot bounds')
            return SLOT_OCCUPIED
        return SLOT_FREE

    def _map_pose(
            self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        set_pose_yaw(pose.pose, float(yaw))
        return pose

    def _odom_pose_to_map(
            self, x: float, y: float, yaw: float) -> Optional[PoseStamped]:
        if self._abort_requested():
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'odom', Time(), timeout=Duration(seconds=0.5))
        except tf2_ros.TransformException:
            return None
        tf_yaw = yaw_from_quaternion(transform.transform.rotation)
        cosine = math.cos(tf_yaw)
        sine = math.sin(tf_yaw)
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = (
            transform.transform.translation.x + cosine * x - sine * y)
        pose.pose.position.y = (
            transform.transform.translation.y + sine * x + cosine * y)
        set_pose_yaw(pose.pose, normalize_angle(tf_yaw + yaw))
        return pose

    @staticmethod
    def _offset_pose(pose: PoseStamped, longitudinal: float, lateral: float):
        yaw = yaw_from_quaternion(pose.pose.orientation)
        result = PoseStamped()
        result.header = pose.header
        result.pose.position.x = (
            pose.pose.position.x + longitudinal * math.cos(yaw)
            - lateral * math.sin(yaw))
        result.pose.position.y = (
            pose.pose.position.y + longitudinal * math.sin(yaw)
            + lateral * math.cos(yaw))
        set_pose_yaw(result.pose, yaw)
        return result

    def _build_named_poses(
        self, slot: Slot, setup_offset: float,
            entry_depth: float) -> Optional[Dict[str, PoseStamped]]:
        road_y = float(self.get_parameter('road_center_y').value)
        overshoot = float(self.get_parameter('final_pose_overshoot').value)
        if abs(overshoot) > 1.0e-9:
            self._log_error(
                'final_pose_overshoot must remain 0.0; the final waypoint is '
                'defined only by physical footprint clearance')
            return None
        target_x, target_y = self._parking_target_depth(slot)
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        entry_x = slot.odom_x
        entry_y = slot.odom_y
        if abs(forward_x) >= abs(forward_y):
            entrance_x = slot.max_x if forward_x > 0.0 else slot.min_x
            entry_x = entrance_x - forward_x * entry_depth
        else:
            entrance_y = slot.max_y if forward_y > 0.0 else slot.min_y
            entry_y = entrance_y - forward_y * entry_depth
        # Canonical approach for both RViz and the real vehicle:
        # start at the east end of the road facing west, pass staging,
        # reach the west-side setup, then reverse into the T branch.
        staging_lead = float(
            self.get_parameter('opposite_staging_lead').value)
        odom_poses = {
            'staging': (
                slot.max_x + staging_lead, road_y, math.pi),
            'setup': (
                slot.min_x + setup_offset, road_y, math.pi),
            'entry': (entry_x, entry_y, slot.yaw),
            'final': (target_x, target_y, slot.yaw),
        }
        # These parking geometry coordinates are defined directly in the
        # saved-map frame.  Do not transform them through map->odom again.
        # RViz-only previously hid this bug because map->odom was identity.
        result = {}
        for name, values in odom_poses.items():
            x, y, yaw = values
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            set_pose_yaw(pose.pose, float(yaw))
            result[name] = pose
        return result

    def _plan_all_candidates(self, slots: Sequence[Slot]) -> List[PlanCandidate]:
        results = []
        self._log_live_start_costmap_snapshot()
        offset_parameter = 'opposite_setup_inset_candidates'
        offsets = [float(value) for value in self.get_parameter(
            offset_parameter).value]
        depths = [float(value) for value in self.get_parameter(
            'entry_depth_candidates').value]
        for slot in slots:
            for setup_offset in offsets:
                for entry_depth in depths:
                    if self._abort_requested():
                        return results
                    named = self._build_named_poses(
                        slot, setup_offset, entry_depth)
                    if named is None:
                        self._log_error(
                            f'reject {slot.name} offset={setup_offset:.2f} '
                            f'depth={entry_depth:.2f}: could not build map-frame '
                            'parking waypoints')
                        continue
                    first_waypoint = named['staging']
                    path = self._request_plan([
                        first_waypoint, named['setup'],
                        named['entry'], named['final']])
                    if path is None:
                        self._log_error(
                            f'reject {slot.name} offset={setup_offset:.2f} '
                            f'depth={entry_depth:.2f}: '
                            f'{self.last_plan_failure_reason}')
                        continue
                    self._log_info(
                        f'[DEBUG CANDIDATE] path received: '
                        f'slot={slot.name} offset={setup_offset:.2f} '
                        f'depth={entry_depth:.2f} poses={len(path.poses)}')

                    metrics = self._analyze_path(path, named['final'])

                    self._log_info(
                        f'[DEBUG CANDIDATE] analyzed: '
                        f'total={metrics.total_length:.3f} '
                        f'reverse={metrics.reverse_length:.3f} '
                        f'cusps={metrics.cusp_count} '
                        f'curvature={metrics.max_curvature:.3f}')

                    minimum_reverse = float(
                        self.get_parameter('minimum_reverse_length').value)
                    reverse_after_setup = self._reverse_length_after_pose(
                        path, metrics, named['setup'])

                    self._log_info(
                        f'[DEBUG CANDIDATE] reverse_after_setup='
                        f'{reverse_after_setup:.3f} '
                        f'minimum={minimum_reverse:.3f}')
                    if reverse_after_setup < minimum_reverse:
                        self._log_info(
                            f'reject {slot.name} offset={setup_offset:.2f} '
                            f'depth={entry_depth:.2f}: reverse after setup '
                            f'{reverse_after_setup:.3f}m < {minimum_reverse:.3f}m')
                        continue
                    valid, reason = self._validate_path(
                        slot, named, path, metrics)
                    if not valid:
                        self._log_info(
                            f'reject {slot.name} offset={setup_offset:.2f} '
                            f'depth={entry_depth:.2f}: {reason}')
                        continue
                    self._log_info(
                        f'accept {slot.name} offset={setup_offset:.2f} '
                        f'depth={entry_depth:.2f}: poses={len(path.poses)} '
                        f'total={metrics.total_length:.3f}m '
                        f'reverse={metrics.reverse_length:.3f}m '
                        f'reverse_after_setup={reverse_after_setup:.3f}m '
                        f'cusps={metrics.cusp_count} '
                        f'max_curvature={metrics.max_curvature:.3f} 1/m')
                    results.append(PlanCandidate(
                        slot, setup_offset, entry_depth, named, path, metrics))
        return results

    def _log_live_start_costmap_snapshot(self) -> None:
        """Explain a planner START_OCCUPIED result without changing safety.

        The static saved map, live obstacle layer, and Hybrid-A* footprint
        collision checker are distinct inputs.  Report their values over the
        actual padded footprint so a real-vehicle failure identifies whether
        the start is outside/unknown, on a mapped wall, or marked by live
        lidar.  This method is diagnostic only; it clears no cells and changes
        no planner or collision-checking decision.
        """
        current = self._current_map_pose_stamped()
        with self.data_lock:
            map_msg = self.map_msg
            costmap = self.global_costmap
        if current is None or map_msg is None or costmap is None:
            self._log_error(
                '[LIVE START COSTMAP] unavailable: map pose, saved map, or '
                'global costmap is missing')
            return

        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        padding = float(self.get_parameter(
            'costmap_footprint_padding').value)
        occupied_threshold = int(self.get_parameter(
            'occupied_threshold').value)
        samples = self._footprint_samples(
            current, length + 2.0 * padding, width + 2.0 * padding)
        counts = {
            'cost_unknown': 0,
            'cost_lethal': 0,
            'cost_inscribed': 0,
            'map_unknown': 0,
            'map_occupied': 0,
        }
        examples = []
        for x, y in samples:
            map_value = self._occupancy_value(map_msg, x, y)
            cost_value = self._cost_value(costmap, x, y)
            if map_value is None or map_value < 0:
                counts['map_unknown'] += 1
            elif map_value >= occupied_threshold:
                counts['map_occupied'] += 1
            if cost_value is None or cost_value == 255:
                counts['cost_unknown'] += 1
            elif cost_value == 254:
                counts['cost_lethal'] += 1
            elif cost_value == 253:
                counts['cost_inscribed'] += 1
            if (len(examples) < 6
                    and (map_value is None or map_value < 0
                         or map_value >= occupied_threshold
                         or cost_value is None or cost_value >= 253)):
                examples.append(
                    f'({x:.3f},{y:.3f}):map={map_value},cost={cost_value}')

        center_cost = self._cost_value(
            costmap, current.pose.position.x, current.pose.position.y)
        center_map = self._occupancy_value(
            map_msg, current.pose.position.x, current.pose.position.y)
        expected_x = float(self.get_parameter('opposite_start_x').value)
        expected_y = float(self.get_parameter('opposite_start_y').value)
        expected_yaw = float(self.get_parameter('opposite_start_yaw').value)
        position_delta = math.hypot(
            current.pose.position.x - expected_x,
            current.pose.position.y - expected_y)
        yaw_delta = abs(normalize_angle(
            yaw_from_quaternion(current.pose.orientation) - expected_yaw))
        self._log_info(
            '[LIVE START COSTMAP]\n'
            f'pose=({current.pose.position.x:.3f},'
            f'{current.pose.position.y:.3f},'
            f'{yaw_from_quaternion(current.pose.orientation):.3f})\n'
            f'canonical_delta_position={position_delta:.3f}m\n'
            f'canonical_delta_yaw={yaw_delta:.3f}rad\n'
            f'center_map={center_map}\n'
            f'center_cost={center_cost}\n'
            f'padded_footprint_samples={len(samples)}\n'
            + '\n'.join(f'{name}={value}' for name, value in counts.items())
            + '\nproblem_samples=' + (
                '; '.join(examples) if examples else 'none'))

    def _request_plan(
            self, goals: Sequence[PoseStamped], planner_id: Optional[str] = None,
            start_pose: Optional[PoseStamped] = None) -> Optional[Path]:
        """Request a Nav2 path, using the live TF pose unless a test start is set.

        Real execution deliberately leaves ``use_start`` false, so Nav2
        starts from the current map-to-robot-base pose at request time. This is
        also true for the initial parking plan: the first staging pose is a
        goal, not a fabricated start pose.  Real exit execution starts from
        the live pose captured after the parking
        FollowPath cancellation.  Plan-only mode supplies its expected
        wheel-inside pose because the robot correctly remains at the course
        entrance and no live TF transform is changed in that mode.
        """
        self.last_plan_failure_reason = 'planning request did not complete'
        if self._abort_requested():
            self.last_plan_failure_reason = 'planning request was aborted'
            return None
        goal = ComputePathThroughPoses.Goal()
        goal.goals = list(goals)
        goal.planner_id = str(
            planner_id if planner_id is not None
            else self.get_parameter('planner_id').value)
        goal.use_start = start_pose is not None
        if start_pose is not None:
            goal.start = self._fresh_pose(start_pose)
        self._log_info(
            'ComputePathThroughPoses request: '
            f'planner_id={goal.planner_id} use_start={str(goal.use_start).lower()} '
            f'goals=' + ', '.join(
                f'({pose.pose.position.x:.3f},{pose.pose.position.y:.3f})'
                for pose in goal.goals))
        if not self._runtime_ok():
            self.last_plan_failure_reason = 'ROS context stopped before planning'
            return None
        send_future = self.plan_client.send_goal_async(goal)
        goal_handle = self._wait_future(send_future, 5.0)
        if goal_handle is None or not goal_handle.accepted:
            self.last_plan_failure_reason = (
                'ComputePathThroughPoses goal was not accepted')
            self._log_error(self.last_plan_failure_reason)
            return None
        self.active_plan_goal = goal_handle
        result_future = goal_handle.get_result_async()
        wrapped = self._wait_future(
            result_future, float(self.get_parameter('action_timeout').value))
        self.active_plan_goal = None
        if wrapped is None:
            self.last_plan_failure_reason = (
                'ComputePathThroughPoses result wait failed/timed out')
            self._log_error(self.last_plan_failure_reason)
            return None
        result = wrapped.result
        error_names = (
            'NONE', 'UNKNOWN', 'INVALID_PLANNER', 'TF_ERROR',
            'START_OUTSIDE_MAP', 'GOAL_OUTSIDE_MAP', 'START_OCCUPIED',
            'GOAL_OCCUPIED', 'TIMEOUT', 'NO_VALID_PATH')
        error_name = next(
            (name for name in error_names
             if result.error_code == getattr(
                 ComputePathThroughPoses.Result, name, -1)),
            'UNRECOGNIZED')
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            self.last_plan_failure_reason = (
                'ComputePathThroughPoses failed: '
                f'action_status={wrapped.status} '
                f'error={error_name}({result.error_code}) '
                f'message={result.error_msg or "<empty>"}')
            self._log_error(self.last_plan_failure_reason)
            return None
        if result.error_code != ComputePathThroughPoses.Result.NONE:
            self.last_plan_failure_reason = (
                'ComputePathThroughPoses failed: '
                f'error={error_name}({result.error_code}) '
                f'message={result.error_msg or "<empty>"}')
            self._log_error(self.last_plan_failure_reason)
            return None
        if not result.path.poses:
            self.last_plan_failure_reason = (
                'ComputePathThroughPoses succeeded but returned an empty path')
            self._log_error(self.last_plan_failure_reason)
            return None
        self.last_plan_failure_reason = ''
        return result.path

    def _current_map_pose_stamped(self) -> Optional[PoseStamped]:
        current = self._current_map_pose()
        if current is None:
            return None
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = current[0]
        pose.pose.position.y = current[1]
        set_pose_yaw(pose.pose, current[2])
        return pose

    def _expected_wheels_inside_stop_pose(
            self, slot: Slot) -> Optional[PoseStamped]:
        """Construct the exact footprint-derived PARKED pose for plan-only."""
        yaw = slot.yaw
        target_x, target_y = self._parking_target_depth(slot)
        pose = self._map_pose(target_x, target_y, yaw)
        self._log_pose('expected_wheels_inside_stop_pose', pose)
        return pose

    def _rear_wheels_clear_slot(self, slot: Slot, pose: PoseStamped) -> bool:
        """Require both rear wheels to have crossed the open slot boundary."""
        wheels = self._wheel_positions_for_pose(pose)
        inset = (float(self.get_parameter('wheel_radius').value)
                 + float(self.get_parameter('wheel_inside_margin').value))
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        if abs(forward_x) >= abs(forward_y):
            entrance_projection = (
                slot.max_x * forward_x if forward_x > 0.0
                else slot.min_x * forward_x)
        else:
            entrance_projection = (
                slot.max_y * forward_y if forward_y > 0.0
                else slot.min_y * forward_y)
        return all(
            x * forward_x + y * forward_y
            >= entrance_projection + inset
            for x, y in (wheels['rear_left'], wheels['rear_right']))

    def _wheel_positions_for_pose(
            self, pose: PoseStamped) -> Dict[str, Tuple[float, float]]:
        """Return ideal wheel centres in the pose's existing map frame."""
        centre_x = pose.pose.position.x
        centre_y = pose.pose.position.y
        yaw = yaw_from_quaternion(pose.pose.orientation)
        half_base = 0.5 * float(self.get_parameter('wheel_base').value)
        half_track = 0.5 * float(self.get_parameter('wheel_track').value)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        result = {}
        for axle, longitudinal in (
                ('front', half_base), ('rear', -half_base)):
            for side, lateral in (
                    ('left', half_track), ('right', -half_track)):
                result[f'{axle}_{side}'] = (
                    centre_x + longitudinal * cosine - lateral * sine,
                    centre_y + longitudinal * sine + lateral * cosine)
        return result

    def _minimum_forward_clear(
            self, slot: Slot, parking_stop_pose: PoseStamped
            ) -> Optional[float]:
        """Straight distance needed for both rear tyres to clear the slot.

        This is a diagnostic bound, not the turn start.  Requiring this full
        distance before steering is an overconstraint: the finite-radius
        right arc must begin inside a wide T mouth and its complete swept
        footprint is validated instead.
        """
        # The actual stop pose comes from map to the configured robot base; the slot is
        # configured in map.  A second map->odom transform here made the exit
        # clearance depend on AMCL's non-identity transform.
        centre_x = parking_stop_pose.pose.position.x
        centre_y = parking_stop_pose.pose.position.y
        yaw = yaw_from_quaternion(parking_stop_pose.pose.orientation)
        rear_offset = -0.5 * float(self.get_parameter('wheel_base').value)
        rear_x = centre_x + rear_offset * math.cos(yaw)
        rear_y = centre_y + rear_offset * math.sin(yaw)
        inset = (float(self.get_parameter('wheel_radius').value)
                 + float(self.get_parameter('wheel_inside_margin').value))
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        if abs(forward_x) >= abs(forward_y):
            entrance = slot.max_x if forward_x > 0.0 else slot.min_x
            distance = (entrance - rear_x) * forward_x + inset
        else:
            entrance = slot.max_y if forward_y > 0.0 else slot.min_y
            distance = (entrance - rear_y) * forward_y + inset
        return max(0.0, distance)

    def _build_forward_exit_poses(
            self, parking_stop_pose: PoseStamped, forward_clear: float,
            right_turn_lead: float, lane_merge: float
            ) -> Tuple[Optional[Dict[str, PoseStamped]], Dict[str, object]]:
        geometry: Dict[str, object] = {
            'forward_clear': forward_clear,
            'right_turn_lead': right_turn_lead,
            'lane_merge_distance': lane_merge,
            'turn_direction': 'RIGHT',
            'arc_angle': math.pi / 2.0,
        }
        if self.course_entrance_pose is None:
            geometry['reason'] = 'course entrance pose is unavailable'
            return None, geometry
        stop = self._fresh_pose(parking_stop_pose)
        clear = self._offset_pose(stop, forward_clear, 0.0)
        clear.header.stamp = self.get_clock().now().to_msg()
        turn_entry = self._offset_pose(clear, right_turn_lead, 0.0)
        turn_entry.header.stamp = self.get_clock().now().to_msg()

        heading = yaw_from_quaternion(stop.pose.orientation)
        forward = (math.cos(heading), math.sin(heading))
        right = (math.sin(heading), -math.cos(heading))
        minimum_radius = float(
            self.get_parameter('minimum_turning_radius').value)
        planning_radius = float(
            self.get_parameter('exit_planning_turn_radius').value)
        target = self._fresh_pose(self.course_entrance_pose)
        target_dx = target.pose.position.x - turn_entry.pose.position.x
        target_dy = target.pose.position.y - turn_entry.pose.position.y
        # For a 90-degree right arc followed by a straight, expressed in the
        # vehicle's frame at turn_entry:
        #   target_delta = R * forward + (R + straight) * right.
        # Solve R from the real entrance coordinate instead of forcing a
        # fixed chamfer radius that leaves the merge line offset from the
        # entrance and makes Dubins loop in the opposite direction.
        turn_offset = target_dx * forward[0] + target_dy * forward[1]
        rightward_distance = target_dx * right[0] + target_dy * right[1]
        straight_after_turn = rightward_distance - turn_offset
        geometry.update({
            'turn_radius': turn_offset,
            'straight_after_turn': straight_after_turn,
            'turn_center': (
                turn_entry.pose.position.x + right[0] * turn_offset,
                turn_entry.pose.position.y + right[1] * turn_offset),
            'curvature': (
                1.0 / turn_offset if turn_offset > 0.0 else float('inf')),
        })
        merge = self._fresh_pose(turn_entry)
        merge.pose.position.x += turn_offset * (forward[0] + right[0])
        merge.pose.position.y += turn_offset * (forward[1] + right[1])
        approach_yaw = normalize_angle(heading - math.pi / 2.0)
        set_pose_yaw(merge.pose, approach_yaw)
        merge.header.stamp = self.get_clock().now().to_msg()
        set_pose_yaw(target.pose, approach_yaw)
        named = {
            'parking_stop': stop,
            'slot_forward_clear': clear,
            'right_turn_entry': turn_entry,
            'lane_merge': merge,
            'course_entrance': target,
        }
        if turn_offset < max(minimum_radius, planning_radius):
            geometry['reason'] = (
                f'turn radius {turn_offset:.3f}m is below required '
                f'{max(minimum_radius, planning_radius):.3f}m')
        elif straight_after_turn < lane_merge:
            geometry['reason'] = (
                f'post-turn straight {straight_after_turn:.3f}m is below '
                f'required {lane_merge:.3f}m')
        else:
            geometry['reason'] = 'valid'
        return named, geometry

    def _build_rviz_return_poses(
            self, parking_stop_pose: PoseStamped
            ) -> Optional[Dict[str, PoseStamped]]:
        """Build a forward right-turn return using the existing exit geometry.

        The saved-map T branch is wide enough to begin the turn before the
        rear wheels have completely crossed its mouth.  Place a quarter-circle
        arc of the configured ForwardExit radius so it ends on the y=0 road,
        then continue straight to the captured start position.  Smac still
        generates the actual Dubins path and the normal exit validator checks
        every footprint sample; these poses only constrain the desired curve.
        """
        if self.course_entrance_pose is None:
            return None
        stop = self._fresh_pose(parking_stop_pose)
        target = self._fresh_pose(self.course_entrance_pose)
        return_yaw = float(self.get_parameter('rviz_return_yaw').value)
        set_pose_yaw(target.pose, return_yaw)

        heading = yaw_from_quaternion(stop.pose.orientation)
        expected_yaw = normalize_angle(heading - math.pi / 2.0)
        if abs(normalize_angle(return_yaw - expected_yaw)) > 0.05:
            self._log_error(
                'RViz return yaw is not the natural heading after the '
                'required forward right turn')
            return None

        forward = (math.cos(heading), math.sin(heading))
        right = (math.sin(heading), -math.cos(heading))
        dx = target.pose.position.x - stop.pose.position.x
        dy = target.pose.position.y - stop.pose.position.y
        forward_distance = dx * forward[0] + dy * forward[1]
        rightward_distance = dx * right[0] + dy * right[1]
        radius = float(self.get_parameter('exit_planning_turn_radius').value)
        straight_before_turn = forward_distance - radius
        straight_after_turn = rightward_distance - radius
        minimum_straight = min(float(value) for value in self.get_parameter(
            'lane_merge_distance_candidates').value)
        if straight_before_turn < 0.0 or straight_after_turn < minimum_straight:
            self._log_error(
                'captured RViz start position cannot be reached by the '
                f'forward right-turn geometry: before={straight_before_turn:.3f}m '
                f'after={straight_after_turn:.3f}m radius={radius:.3f}m')
            return None

        turn_entry = self._offset_pose(stop, straight_before_turn, 0.0)
        turn_entry.header.stamp = self.get_clock().now().to_msg()
        lane_merge = self._fresh_pose(turn_entry)
        lane_merge.pose.position.x += radius * (forward[0] + right[0])
        lane_merge.pose.position.y += radius * (forward[1] + right[1])
        set_pose_yaw(lane_merge.pose, return_yaw)
        lane_merge.header.stamp = self.get_clock().now().to_msg()
        return {
            'parking_stop': stop,
            'right_turn_entry': turn_entry,
            'lane_merge': lane_merge,
            'course_entrance': target,
        }

    def _plan_rviz_return(
            self, parking_stop_pose: PoseStamped, slot: Slot
            ) -> Optional[ForwardExitCandidate]:
        """Plan and validate the RViz-only full return with ForwardExit."""
        named = self._build_rviz_return_poses(parking_stop_pose)
        if named is None:
            return None
        self._log_info(
            '[RVIZ RETURN WAYPOINTS]\n'
            + '\n'.join(
                f'{name}=({pose.pose.position.x:.3f},'
                f'{pose.pose.position.y:.3f},'
                f'{yaw_from_quaternion(pose.pose.orientation):.3f})'
                for name, pose in named.items()))
        path = self._request_plan(
            [named['right_turn_entry'], named['lane_merge'],
             named['course_entrance']],
            planner_id=str(self.get_parameter('exit_planner_id').value),
            start_pose=named['parking_stop'])
        if path is None:
            return None
        path = self._dedupe_stationary_poses(path)
        metrics = self._analyze_path(path, named['course_entrance'])
        valid, turn_direction, reason = self._validate_forward_exit_path(
            path, metrics, named, slot)
        if not valid:
            self._log_error(f'RViz return path rejected: {reason}')
            return None
        straight_before = math.hypot(
            named['right_turn_entry'].pose.position.x
            - named['parking_stop'].pose.position.x,
            named['right_turn_entry'].pose.position.y
            - named['parking_stop'].pose.position.y)
        straight_after = math.hypot(
            named['course_entrance'].pose.position.x
            - named['lane_merge'].pose.position.x,
            named['course_entrance'].pose.position.y
            - named['lane_merge'].pose.position.y)
        return ForwardExitCandidate(
            straight_before, 0.0, straight_after, named, path, metrics,
            turn_direction)

    def _plan_forward_right_exit(
            self, parking_stop_pose: PoseStamped, slot: Slot,
            use_current_start: bool) -> Optional[ForwardExitCandidate]:
        planner_id = str(self.get_parameter('exit_planner_id').value)
        minimum_clear = self._minimum_forward_clear(slot, parking_stop_pose)
        if minimum_clear is None:
            self._log_error(
                'could not compute the minimum forward-clear distance')
            return None
        stop_yaw = yaw_from_quaternion(parking_stop_pose.pose.orientation)
        forward = (math.cos(stop_yaw), math.sin(stop_yaw))
        target = self.course_entrance_pose
        if target is None:
            self._log_error('course entrance pose is unavailable')
            return None
        target_forward_distance = (
            (target.pose.position.x - parking_stop_pose.pose.position.x)
            * forward[0]
            + (target.pose.position.y - parking_stop_pose.pose.position.y)
            * forward[1])
        required_radius = max(
            float(self.get_parameter('minimum_turning_radius').value),
            float(self.get_parameter('exit_planning_turn_radius').value))
        latest_turn_start = target_forward_distance - required_radius
        turn_start_margins = [float(value) for value in self.get_parameter(
            'forward_clear_distance_candidates').value]
        clear_candidates = [
            latest_turn_start - margin for margin in turn_start_margins
            if latest_turn_start - margin >= 0.0]
        self._log_info(
            f'straight-only rear-wheel full-clear={minimum_clear:.3f}m; '
            f'latest legal turn start={latest_turn_start:.3f}m; '
            'turn-start candidates='
            + ', '.join(f'{value:.3f}' for value in clear_candidates))
        lead_candidates = [float(value) for value in self.get_parameter(
            'right_turn_lead_candidates').value]
        merge_candidates = [float(value) for value in self.get_parameter(
            'lane_merge_distance_candidates').value]
        for forward_clear in clear_candidates:
            for right_turn_lead in lead_candidates:
                for lane_merge in merge_candidates:
                    if self._abort_requested():
                        return None
                    named, geometry = self._build_forward_exit_poses(
                        parking_stop_pose, forward_clear, right_turn_lead,
                        lane_merge)
                    if named is None:
                        self._log_exit_candidate_debug(
                            slot, None, geometry, None, 'REJECT',
                            str(geometry.get('reason', 'geometry unavailable')))
                        continue
                    if geometry['reason'] != 'valid':
                        self._log_exit_candidate_debug(
                            slot, named, geometry, None, 'REJECT',
                            str(geometry['reason']))
                        continue
                    # A wide T mouth permits the legal Ackermann arc to start
                    # while the rear wheels are still inside.  Requiring full
                    # straight-line clearance first overconstrains the turn.
                    # The lane-merge pose must nevertheless have both rear
                    # tyre envelopes fully beyond the mouth; the complete
                    # swept body is checked after planning below.
                    if not self._rear_wheels_clear_slot(
                            slot, named['lane_merge']):
                        reason = 'rear tyres are not clear at lane merge'
                        self._log_exit_candidate_debug(
                            slot, named, geometry, None, 'REJECT', reason)
                        continue
                    self._log_info(
                        '[FORWARD EXIT WAYPOINTS]\n'
                        + '\n'.join(
                            f'{name}=({pose.pose.position.x:.3f},'
                            f'{pose.pose.position.y:.3f},'
                            f'{yaw_from_quaternion(pose.pose.orientation):.3f})'
                            for name, pose in named.items()))
                    path = self._request_plan(
                        [named['slot_forward_clear'],
                         named['right_turn_entry'],
                         named['lane_merge'],
                         named['course_entrance']],
                        planner_id=planner_id,
                        start_pose=None if use_current_start else named['parking_stop'])
                    if path is None:
                        self._log_exit_candidate_debug(
                            slot, named, geometry, None, 'REJECT',
                            self.last_plan_failure_reason)
                        continue
                    path = self._dedupe_stationary_poses(path)
                    metrics = self._analyze_path(
                        path, named['course_entrance'])
                    valid, turn_direction, reason = (
                        self._validate_forward_exit_path(
                            path, metrics, named, slot))
                    if not valid:
                        self._log_exit_candidate_debug(
                            slot, named, geometry, path, 'REJECT', reason,
                            metrics)
                        continue
                    self._log_exit_candidate_debug(
                        slot, named, geometry, path, 'ACCEPT', 'valid', metrics)
                    self._log_info(
                        '[FORWARD EXIT PATH]\n'
                        f'poses={len(path.poses)}\n'
                        f'total_length={metrics.total_length:.3f}\n'
                        f'forward_length={metrics.forward_length:.3f}\n'
                        f'reverse_length={metrics.reverse_length:.3f}\n'
                        f'reverse_segments={self._reverse_segment_count(metrics)}\n'
                        f'cusps={metrics.cusp_count}\n'
                        'first_motion=FORWARD\n'
                        f'turn_direction={turn_direction}')
                    return ForwardExitCandidate(
                        forward_clear, right_turn_lead, lane_merge, named,
                        path, metrics, turn_direction)
        return None

    def _ideal_exit_geometry_path(
            self, named: Dict[str, PoseStamped], geometry: Dict[str, object]
            ) -> Path:
        """Sample the intended straight/right-arc/straight geometry at 5 cm."""
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()

        def append_linear(start: PoseStamped, end: PoseStamped) -> None:
            distance = math.hypot(
                end.pose.position.x - start.pose.position.x,
                end.pose.position.y - start.pose.position.y)
            count = max(1, int(math.ceil(distance / 0.05)))
            start_yaw = yaw_from_quaternion(start.pose.orientation)
            end_yaw = yaw_from_quaternion(end.pose.orientation)
            yaw_delta = normalize_angle(end_yaw - start_yaw)
            for index in range(count + 1):
                if path.poses and index == 0:
                    continue
                ratio = index / count
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = (
                    start.pose.position.x
                    + ratio * (end.pose.position.x - start.pose.position.x))
                pose.pose.position.y = (
                    start.pose.position.y
                    + ratio * (end.pose.position.y - start.pose.position.y))
                set_pose_yaw(pose.pose, start_yaw + ratio * yaw_delta)
                path.poses.append(pose)

        append_linear(named['parking_stop'], named['right_turn_entry'])
        turn_entry = named['right_turn_entry']
        radius = float(geometry['turn_radius'])
        heading = yaw_from_quaternion(turn_entry.pose.orientation)
        forward = (math.cos(heading), math.sin(heading))
        right = (math.sin(heading), -math.cos(heading))
        arc_length = abs(radius) * math.pi / 2.0
        arc_count = max(1, int(math.ceil(arc_length / 0.05)))
        for index in range(1, arc_count + 1):
            angle = (math.pi / 2.0) * index / arc_count
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = (
                turn_entry.pose.position.x
                + radius * math.sin(angle) * forward[0]
                + radius * (1.0 - math.cos(angle)) * right[0])
            pose.pose.position.y = (
                turn_entry.pose.position.y
                + radius * math.sin(angle) * forward[1]
                + radius * (1.0 - math.cos(angle)) * right[1])
            set_pose_yaw(pose.pose, heading - angle)
            path.poses.append(pose)
        append_linear(named['lane_merge'], named['course_entrance'])
        return path

    def _exit_collision_flags(self, path: Path) -> Tuple[bool, bool]:
        """Return static-map and live-costmap collision flags."""
        with self.data_lock:
            costmap = self.global_costmap
            map_msg = self.map_msg
        if costmap is None or map_msg is None:
            return True, True
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        occupied_threshold = int(
            self.get_parameter('occupied_threshold').value)
        map_collision = False
        costmap_collision = False
        for pose in path.poses:
            center_cost = self._cost_value(
                costmap, pose.pose.position.x, pose.pose.position.y)
            if center_cost is None or 253 <= center_cost <= 254:
                costmap_collision = True
            for x, y in self._footprint_samples(pose, length, width):
                map_value = self._occupancy_value(map_msg, x, y)
                cost_value = self._cost_value(costmap, x, y)
                if map_value is None or map_value >= occupied_threshold:
                    map_collision = True
                if cost_value is None or cost_value == 254:
                    costmap_collision = True
        return map_collision, costmap_collision

    def _path_side_wall_clearance(
            self, slot: Slot, path: Path) -> Optional[float]:
        """Return minimum body clearance to the two slot side walls."""
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        longitudinal_is_x = abs(forward_x) >= abs(forward_y)
        minimum = float('inf')
        for pose in path.poses:
            for x, y in self._footprint_samples(
                    pose, length, width, edge_only=True):
                longitudinal = x if longitudinal_is_x else y
                longitudinal_min = (
                    slot.min_x if longitudinal_is_x else slot.min_y)
                longitudinal_max = (
                    slot.max_x if longitudinal_is_x else slot.max_y)
                if not longitudinal_min <= longitudinal <= longitudinal_max:
                    continue
                side = y if longitudinal_is_x else x
                side_min = slot.min_y if longitudinal_is_x else slot.min_x
                side_max = slot.max_y if longitudinal_is_x else slot.max_x
                minimum = min(minimum, side - side_min, side_max - side)
        return minimum if math.isfinite(minimum) else None

    def _log_exit_candidate_debug(
            self, slot: Slot, named: Optional[Dict[str, PoseStamped]],
            geometry: Dict[str, object], path: Optional[Path],
            decision: str, reason: str,
            metrics: Optional[PathMetrics] = None) -> None:
        """Emit one complete diagnostic block for every exit candidate."""
        def pose_text(pose: Optional[PoseStamped]) -> str:
            if pose is None:
                return 'unavailable'
            return (
                f'({pose.pose.position.x:.6f},'
                f'{pose.pose.position.y:.6f},'
                f'{yaw_from_quaternion(pose.pose.orientation):.6f})')

        if named is None:
            self._log_info(
                '[EXIT CANDIDATE DEBUG]\n'
                f'forward_clear={float(geometry.get("forward_clear", 0.0)):.6f}\n'
                'start_pose=unavailable\nstraight_end_pose=unavailable\n'
                'turn_center=unavailable\n'
                f'turn_radius={float(geometry.get("turn_radius", 0.0)):.6f}\n'
                'turn_direction=RIGHT\narc_angle=1.570796\n'
                'arc_end_pose=unavailable\nvehicle_heading=unavailable\n'
                'rear_axle_pose=unavailable\nfront_left=unavailable\n'
                'front_right=unavailable\nrear_left=unavailable\n'
                'rear_right=unavailable\nslot_clearance=unavailable\n'
                'wall_clearance=unavailable\nmap_collision=unknown\n'
                'costmap_collision=unknown\ncurvature=unavailable\n'
                f'{decision}: {reason}')
            return

        diagnostic_path = (
            path if path is not None
            else self._ideal_exit_geometry_path(named, geometry))
        map_collision, costmap_collision = self._exit_collision_flags(
            diagnostic_path)
        slot_clearance = self._path_wall_clearance(slot, diagnostic_path)
        wall_clearance = self._path_side_wall_clearance(slot, diagnostic_path)
        straight_end = named['slot_forward_clear']
        wheels = self._wheel_positions_for_pose(straight_end)
        rear_axle = (
            0.5 * (wheels['rear_left'][0] + wheels['rear_right'][0]),
            0.5 * (wheels['rear_left'][1] + wheels['rear_right'][1]))
        turn_center = geometry.get('turn_center')
        curvature = (
            metrics.max_curvature if metrics is not None
            else float(geometry.get('curvature', float('inf'))))
        self._log_info(
            '[EXIT CANDIDATE DEBUG]\n'
            f'forward_clear={float(geometry["forward_clear"]):.6f}\n'
            f'start_pose={pose_text(named["parking_stop"])}\n'
            f'straight_end_pose={pose_text(straight_end)}\n'
            f'turn_center=({turn_center[0]:.6f},{turn_center[1]:.6f})\n'
            f'turn_radius={float(geometry["turn_radius"]):.6f}\n'
            f'turn_direction={geometry["turn_direction"]}\n'
            f'arc_angle={float(geometry["arc_angle"]):.6f}\n'
            f'arc_end_pose={pose_text(named["lane_merge"])}\n'
            f'vehicle_heading='
            f'{yaw_from_quaternion(straight_end.pose.orientation):.6f}\n'
            f'rear_axle_pose=({rear_axle[0]:.6f},{rear_axle[1]:.6f})\n'
            f'front_left=({wheels["front_left"][0]:.6f},'
            f'{wheels["front_left"][1]:.6f})\n'
            f'front_right=({wheels["front_right"][0]:.6f},'
            f'{wheels["front_right"][1]:.6f})\n'
            f'rear_left=({wheels["rear_left"][0]:.6f},'
            f'{wheels["rear_left"][1]:.6f})\n'
            f'rear_right=({wheels["rear_right"][0]:.6f},'
            f'{wheels["rear_right"][1]:.6f})\n'
            f'slot_clearance='
            f'{slot_clearance if slot_clearance is not None else "unavailable"}\n'
            f'wall_clearance='
            f'{wall_clearance if wall_clearance is not None else "unavailable"}\n'
            f'map_collision={str(map_collision).lower()}\n'
            f'costmap_collision={str(costmap_collision).lower()}\n'
            f'curvature={curvature:.6f}\n'
            f'{decision}: {reason}')

    def _fresh_pose(self, source: PoseStamped) -> PoseStamped:
        result = PoseStamped()
        result.header.frame_id = 'map'
        result.header.stamp = self.get_clock().now().to_msg()
        result.pose.position.x = source.pose.position.x
        result.pose.position.y = source.pose.position.y
        result.pose.position.z = source.pose.position.z
        set_pose_yaw(result.pose, yaw_from_quaternion(source.pose.orientation))
        return result

    def _validate_forward_exit_path(
            self, path: Path, metrics: PathMetrics,
            named: Dict[str, PoseStamped], slot: Slot
            ) -> Tuple[bool, str, str]:
        if not path.poses:
            return False, 'NONE', 'path is empty'
        nonzero_directions = [direction for direction in metrics.directions
                              if direction != 0]
        if not nonzero_directions:
            return False, 'NONE', 'path has no meaningful motion segment'
        if nonzero_directions[0] < 0 or nonzero_directions[-1] < 0:
            return False, 'NONE', 'first or last meaningful motion is reverse'
        reverse_segments = self._reverse_segment_count(metrics)
        reverse_limit = float(self.get_parameter(
            'max_reverse_distance_during_exit').value)
        if bool(self.get_parameter('require_forward_only_exit').value) and (
                metrics.reverse_length > reverse_limit or reverse_segments > 0):
            return False, 'NONE', (
                f'reverse_length={metrics.reverse_length:.3f}m '
                f'reverse_segments={reverse_segments}')
        if metrics.cusp_count != 0:
            return False, 'NONE', f'cusp count {metrics.cusp_count} is not zero'
        minimum_radius = float(
            self.get_parameter('minimum_turning_radius').value)
        curvature_factor = float(
            self.get_parameter('curvature_tolerance_factor').value)
        if metrics.max_curvature > curvature_factor / minimum_radius:
            return False, 'NONE', (
                f'curvature {metrics.max_curvature:.3f} exceeds '
                f'{curvature_factor / minimum_radius:.3f} 1/m')
        position_tolerance = float(self.get_parameter(
            'entrance_return_position_tolerance').value)
        if metrics.final_position_error > position_tolerance:
            return False, 'NONE', (
                f'planned endpoint position error '
                f'{metrics.final_position_error:.3f}m')
        yaw_tolerance = float(self.get_parameter(
            'entrance_return_yaw_tolerance').value)
        if metrics.final_yaw_error > yaw_tolerance:
            return False, 'NONE', (
                f'planned endpoint yaw error {metrics.final_yaw_error:.3f}rad')
        # ComputePathThroughPoses concatenates independently-planned
        # SmacPlannerHybrid segments at each intermediate waypoint. Each
        # segment's shared endpoint is quantized to the nearest of
        # angle_quantization_bins (72 => 5 deg / 0.0873 rad) independently,
        # so a genuine zero-distance seam routinely differs by exactly one
        # bin. That is stitching noise, not a real in-place spin, so the
        # threshold must clear a single bin with margin.
        in_place_yaw_threshold = 0.20
        for previous, current in zip(path.poses, path.poses[1:]):
            distance = math.hypot(
                current.pose.position.x - previous.pose.position.x,
                current.pose.position.y - previous.pose.position.y)
            yaw_change = abs(normalize_angle(
                yaw_from_quaternion(current.pose.orientation)
                - yaw_from_quaternion(previous.pose.orientation)))
            if distance < 0.002 and yaw_change > in_place_yaw_threshold:
                return False, 'NONE', 'in-place rotation segment detected'
        turn_reference = named.get(
            'slot_forward_clear', named['right_turn_entry'])
        turn_direction = self._first_major_turn_direction(
            path, turn_reference)
        if (bool(self.get_parameter('require_right_turn_exit').value)
                and turn_direction != 'RIGHT'):
            return False, turn_direction, 'first major turn is not vehicle-relative RIGHT'
        # A single right turn onto the main lane needs roughly 90 degrees of
        # cumulative heading change. Summing unsigned per-step yaw deltas
        # (rather than comparing only start/end heading) catches a DUBIN
        # solution that loops most of the way around before settling on the
        # right heading -- start and end yaw alone can look identical to a
        # clean turn even when the path circled to get there.
        total_heading_change = sum(
            abs(normalize_angle(
                yaw_from_quaternion(current.pose.orientation)
                - yaw_from_quaternion(previous.pose.orientation)))
            for previous, current in zip(path.poses, path.poses[1:]))
        max_heading_change = math.radians(150.0)
        if total_heading_change > max_heading_change:
            return False, turn_direction, (
                f'cumulative heading change {math.degrees(total_heading_change):.1f}'
                f' deg exceeds {math.degrees(max_heading_change):.1f} deg '
                '(path loops instead of a single turn)')
        footprint_valid, footprint_reason = self._validate_exit_footprint(path)
        if not footprint_valid:
            return False, turn_direction, footprint_reason
        wall_valid, wall_reason = self._path_clears_slot_walls(slot, path)
        if not wall_valid:
            return False, turn_direction, wall_reason
        if not self._rear_wheels_clear_slot(slot, named['lane_merge']):
            return False, turn_direction, 'rear tyres are not clear at lane merge'
        return True, turn_direction, 'valid'

    @staticmethod
    def _reverse_segment_count(metrics: PathMetrics) -> int:
        return sum(1 for direction in metrics.directions if direction < 0)

    def _first_major_turn_direction(
            self, path: Path, forward_clear_pose: PoseStamped) -> str:
        start_yaw = yaw_from_quaternion(path.poses[0].pose.orientation)
        clear_index = min(
            range(len(path.poses)),
            key=lambda index: math.hypot(
                path.poses[index].pose.position.x
                - forward_clear_pose.pose.position.x,
                path.poses[index].pose.position.y
                - forward_clear_pose.pose.position.y))
        for pose in path.poses[clear_index:]:
            heading_change = normalize_angle(
                yaw_from_quaternion(pose.pose.orientation) - start_yaw)
            if abs(heading_change) >= math.radians(12.0):
                # The sign is measured against the initial vehicle heading,
                # not a world-axis assumption: negative is vehicle-right.
                return 'RIGHT' if heading_change < 0.0 else 'LEFT'
        return 'NONE'

    def _validate_exit_footprint(self, path: Path) -> Tuple[bool, str]:
        with self.data_lock:
            costmap = self.global_costmap
            map_msg = self.map_msg
        if costmap is None or map_msg is None:
            return False, 'global costmap or SLAM map unavailable'
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        occupied_threshold = int(
            self.get_parameter('occupied_threshold').value)
        # NO_INFORMATION (cost 255 / occupancy -1) means unobserved, not
        # unsafe: the course-entrance area is self-occluded at spawn and
        # never re-observed afterwards even though it is the open lane the
        # vehicle already drove through once (confirmed against the static
        # world geometry). Only reject genuine out-of-bounds samples and
        # actual occupied/lethal cells.
        for index, pose in enumerate(path.poses):
            center_cost = self._cost_value(
                costmap, pose.pose.position.x, pose.pose.position.y)
            if center_cost is None:
                return False, f'path center index {index} is outside the costmap'
            if 253 <= center_cost <= 254:
                return False, f'path center index {index} is lethal/inscribed'
            for x, y in self._footprint_samples(pose, length, width):
                map_value = self._occupancy_value(map_msg, x, y)
                cost_value = self._cost_value(costmap, x, y)
                if map_value is None or cost_value is None:
                    return False, f'footprint index {index} is outside the map bounds'
                if map_value >= occupied_threshold or cost_value == 254:
                    return False, f'footprint index {index} overlaps obstacle'
        return True, 'valid'

    @staticmethod
    def _dedupe_stationary_poses(path: Path) -> Path:
        """Drop zero-distance duplicate poses left by waypoint stitching.

        ComputePathThroughPoses concatenates independently-planned
        SmacPlannerHybrid segments at each intermediate waypoint, and each
        segment's shared endpoint is quantized to the nearest heading bin
        independently. The result is a poses[i]/poses[i+1] pair at the exact
        same position but a few degrees apart in heading -- geometrically
        harmless, but RegulatedPurePursuitController's live lookahead/
        curvature tracking can momentarily treat that discontinuity as
        behind the vehicle and emit a brief negative-velocity correction.
        Collapse those duplicates before the path is handed to FollowPath.
        """
        if len(path.poses) < 2:
            return path
        cleaned = Path()
        cleaned.header = path.header
        cleaned.poses = [path.poses[0]]
        for pose in path.poses[1:]:
            previous = cleaned.poses[-1]
            distance = math.hypot(
                pose.pose.position.x - previous.pose.position.x,
                pose.pose.position.y - previous.pose.position.y)
            if distance < 0.002:
                cleaned.poses[-1] = pose
                continue
            cleaned.poses.append(pose)
        return cleaned

    def _analyze_path(self, path: Path, final_pose: PoseStamped) -> PathMetrics:
        total = 0.0
        raw_directions = []
        edge_lengths = []
        for previous, current in zip(path.poses, path.poses[1:]):
            dx = current.pose.position.x - previous.pose.position.x
            dy = current.pose.position.y - previous.pose.position.y
            length = math.hypot(dx, dy)
            edge_lengths.append(length)
            if length < 1e-6:
                raw_directions.append(0)
                continue
            yaw = yaw_from_quaternion(previous.pose.orientation)
            dot = dx * math.cos(yaw) + dy * math.sin(yaw)
            # A direction score very close to zero is inconclusive.  Do not
            # turn one nearly lateral / quantized pose pair into a cusp.
            direction = 0 if abs(dot) < 1.0e-4 else 1 if dot > 0.0 else -1
            raw_directions.append(direction)
            total += length
        directions = self._stabilize_directions(
            raw_directions, edge_lengths)
        forward = sum(
            length for length, direction in zip(edge_lengths, directions)
            if direction > 0)
        reverse = sum(
            length for length, direction in zip(edge_lengths, directions)
            if direction < 0)
        first_reverse = next(
            (index + 1 for index, direction in enumerate(directions)
             if direction < 0), -1)
        nonzero = [value for value in directions if value]
        cusps = sum(a != b for a, b in zip(nonzero, nonzero[1:]))

        # Estimate curvature over a physical window instead of adjacent grid
        # samples.  Three nearly coincident Hybrid-A* poses amplify map-grid
        # quantization, and a Reeds-Shepp cusp is a stop/direction change where
        # curvature is undefined.  A 0.30 m window on each side remains well
        # below this vehicle's 1.52 m minimum turning radius while rejecting
        # real over-curvature.  The old 0.15 m window was only three 0.05 m
        # map cells and
        # reported a legal Dubins arc as 0.681 instead of 0.658 1/m from
        # Hybrid-A* grid quantization.  The acceptance limit itself remains
        # exactly 1 / minimum_turning_radius.
        max_curvature = 0.0
        poses = path.poses
        curvature_window = 0.30
        for middle_index in range(1, len(poses) - 1):
            previous_direction = self._nearest_direction(
                directions, middle_index - 1, -1)
            next_direction = self._nearest_direction(
                directions, middle_index, 1)
            if previous_direction == 0 or previous_direction != next_direction:
                continue
            first_index, previous_length = self._window_endpoint(
                poses, directions, middle_index, -1,
                previous_direction, curvature_window)
            last_index, next_length = self._window_endpoint(
                poses, directions, middle_index, 1,
                next_direction, curvature_window)
            if (previous_length < curvature_window
                    or next_length < curvature_window):
                continue
            first = poses[first_index]
            middle = poses[middle_index]
            last = poses[last_index]
            a = math.hypot(
                middle.pose.position.x - first.pose.position.x,
                middle.pose.position.y - first.pose.position.y)
            b = math.hypot(
                last.pose.position.x - middle.pose.position.x,
                last.pose.position.y - middle.pose.position.y)
            c = math.hypot(
                last.pose.position.x - first.pose.position.x,
                last.pose.position.y - first.pose.position.y)
            denominator = a * b * c
            if denominator < 1e-7:
                continue
            twice_area = abs(
                (middle.pose.position.x - first.pose.position.x)
                * (last.pose.position.y - first.pose.position.y)
                - (middle.pose.position.y - first.pose.position.y)
                * (last.pose.position.x - first.pose.position.x))
            max_curvature = max(max_curvature, 2.0 * twice_area / denominator)

        end = path.poses[-1].pose
        position_error = math.hypot(
            end.position.x - final_pose.pose.position.x,
            end.position.y - final_pose.pose.position.y)
        yaw_error = abs(normalize_angle(
            yaw_from_quaternion(end.orientation)
            - yaw_from_quaternion(final_pose.pose.orientation)))
        return PathMetrics(
            total, forward, reverse, cusps, first_reverse,
            directions, max_curvature, position_error, yaw_error)

    @staticmethod
    def _stabilize_directions(
            raw: Sequence[int], edge_lengths: Sequence[float]) -> List[int]:
        """Suppress a one-pose direction glitch using path-spacing hysteresis."""
        if not raw:
            return []
        directions = list(raw)
        first_nonzero = next((value for value in directions if value), 0)
        if first_nonzero == 0:
            return directions
        previous = first_nonzero
        for index, value in enumerate(directions):
            if value == 0:
                directions[index] = previous
            else:
                previous = value

        nonzero_lengths = sorted(
            length for length in edge_lengths if length > 1.0e-6)
        middle = len(nonzero_lengths) // 2
        median_spacing = (
            nonzero_lengths[middle]
            if len(nonzero_lengths) % 2
            else 0.5 * (
                nonzero_lengths[middle - 1] + nonzero_lengths[middle]))
        confirmation_length = max(0.10, 3.0 * median_spacing)

        while True:
            runs = []
            start = 0
            for index in range(1, len(directions) + 1):
                if (index < len(directions)
                        and directions[index] == directions[start]):
                    continue
                runs.append((start, index, directions[start]))
                start = index
            merged = False
            for run_index, (first, last, _direction) in enumerate(runs):
                run_length = sum(edge_lengths[first:last])
                if last - first >= 3 and run_length >= confirmation_length:
                    continue
                # Only collapse an internal pulse bracketed by the same real
                # direction.  A genuinely short terminal run is retained and
                # rejected explicitly by the segment minimum-length check.
                if (0 < run_index < len(runs) - 1
                        and runs[run_index - 1][2] == runs[run_index + 1][2]):
                    directions[first:last] = [runs[run_index - 1][2]] * (
                        last - first)
                    merged = True
                    break
            if not merged:
                return directions

    @staticmethod
    def _nearest_direction(
            directions: Sequence[int], start: int, step: int) -> int:
        index = start
        while 0 <= index < len(directions):
            if directions[index] != 0:
                return directions[index]
            index += step
        return 0

    @staticmethod
    def _window_endpoint(
            poses: Sequence[PoseStamped], directions: Sequence[int],
            middle_index: int, step: int, direction: int,
            target_length: float) -> Tuple[int, float]:
        index = middle_index
        accumulated = 0.0
        while 0 <= index + step < len(poses):
            segment_index = index if step > 0 else index - 1
            segment_direction = directions[segment_index]
            if segment_direction not in (0, direction):
                break
            next_index = index + step
            accumulated += math.hypot(
                poses[next_index].pose.position.x
                - poses[index].pose.position.x,
                poses[next_index].pose.position.y
                - poses[index].pose.position.y)
            index = next_index
            if accumulated >= target_length:
                break
        return index, accumulated

    @staticmethod
    def _reverse_length_after_pose(
            path: Path, metrics: PathMetrics, pose: PoseStamped) -> float:
        setup_index = min(
            range(len(path.poses)),
            key=lambda index: math.hypot(
                path.poses[index].pose.position.x - pose.pose.position.x,
                path.poses[index].pose.position.y - pose.pose.position.y))
        reverse_length = 0.0
        for index in range(setup_index, len(path.poses) - 1):
            if metrics.directions[index] >= 0:
                continue
            reverse_length += math.hypot(
                path.poses[index + 1].pose.position.x
                - path.poses[index].pose.position.x,
                path.poses[index + 1].pose.position.y
                - path.poses[index].pose.position.y)
        return reverse_length

    def _validate_path(
            self, slot: Slot, named: Dict[str, PoseStamped], path: Path,
            metrics: PathMetrics) -> Tuple[bool, str]:
        if metrics.first_reverse_index < 0:
            return False, 'no reverse segment'
        maximum_cusps = int(self.get_parameter('maximum_cusps').value)
        if metrics.cusp_count > maximum_cusps:
            return False, (
                f'cusp count {metrics.cusp_count} exceeds safe maximum '
                f'{maximum_cusps}')
        minimum_radius = float(
            self.get_parameter('minimum_turning_radius').value)
        curvature_factor = float(
            self.get_parameter('curvature_tolerance_factor').value)
        if metrics.max_curvature > curvature_factor / minimum_radius:
            return False, (
                f'curvature {metrics.max_curvature:.3f} exceeds '
                f'{curvature_factor / minimum_radius:.3f} 1/m')
        if not self._setup_inside_road(named['setup']):
            return False, 'setup footprint crosses main-road boundary'
        if not self._final_footprint_inside_slot(slot, named['final']):
            return False, 'final footprint is outside selected slot'
        wall_ok, wall_reason = self._path_clears_slot_walls(slot, path)
        if not wall_ok:
            return False, wall_reason
        with self.data_lock:
            costmap = self.global_costmap
            map_msg = self.map_msg
        if costmap is None or map_msg is None:
            return False, 'global costmap or SLAM map unavailable'
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        center_x = float(self.get_parameter(
            'vehicle_center_x_offset').value)
        occupied_threshold = int(
            self.get_parameter('occupied_threshold').value)
        start_pose = path.poses[0]
        start_position = start_pose.pose.position
        start_yaw = yaw_from_quaternion(start_pose.pose.orientation)
        start_cos = math.cos(start_yaw)
        start_sin = math.sin(start_yaw)
        start_shadow_margin = 0.05

        def inside_start_footprint(x: float, y: float) -> bool:
            """Return true only for cells hidden by the stationary vehicle."""
            dx = x - start_position.x
            dy = y - start_position.y
            local_x = start_cos * dx + start_sin * dy
            local_y = -start_sin * dx + start_cos * dy
            return (
                center_x - 0.5 * length - start_shadow_margin
                <= local_x
                <= center_x + 0.5 * length + start_shadow_margin
                and abs(local_y) <= 0.5 * width + start_shadow_margin)

        for index, pose in enumerate(path.poses):
            center_cost = self._cost_value(
                costmap, pose.pose.position.x, pose.pose.position.y)
            if center_cost is None:
                return False, f'path center index {index} is outside costmap'
            # The stationary vehicle self-occludes cells immediately under
            # its current footprint. Nav2 has already accepted this live pose
            # as the path start, so allow UNKNOWN only at a sample point that
            # is physically under that initial footprint (plus one map cell).
            # Every other point on the path still requires observed free
            # space.  A static-map UNKNOWN cell is acceptable only after the
            # live global obstacle layer has ray-cleared it to a known cost;
            # a costmap UNKNOWN cell is never treated as traversable.
            if (center_cost == 255
                    and not inside_start_footprint(
                        pose.pose.position.x, pose.pose.position.y)):
                return False, f'path center index {index} is unknown'
            if 253 <= center_cost <= 254:
                return False, f'path center index {index} is lethal/inscribed'
            for x, y in self._footprint_samples(pose, length, width):
                map_value = self._occupancy_value(map_msg, x, y)
                cost_value = self._cost_value(costmap, x, y)
                if map_value is None or cost_value is None:
                    return False, f'footprint index {index} is outside map bounds'
                if (cost_value == 255
                        and not inside_start_footprint(x, y)):
                    return False, (
                        f'footprint index {index} reaches unknown at '
                        f'({x:.3f},{y:.3f}); map={map_value} '
                        f'cost={cost_value}')
                if map_value >= occupied_threshold or cost_value == 254:
                    return False, f'footprint index {index} overlaps obstacle'
        return True, 'valid'

    def _path_clears_slot_walls(
            self, slot: Slot, path: Path) -> Tuple[bool, str]:
        """Require the whole entry path to keep clear of the slot side walls.

        The costmap sweep in _validate_path only rejects a footprint that has
        already reached an obstacle or inflated cell, and
        _final_footprint_inside_slot only looks at the parked pose.  Neither
        catches the failure this guards against: a reverse arc that is legal
        cell-by-cell but swings the body across the entrance corner on its way
        in.  Any footprint sample that is inside the bay longitudinally must
        stay wall_clearance away from both side walls.
        """
        if self._abort_requested():
            return False, 'aborted'
        clearance = float(self.get_parameter('wall_clearance').value)
        if clearance <= 0.0:
            return True, 'valid'
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        longitudinal_is_x = abs(forward_x) >= abs(forward_y)
        minimum_side = (
            slot.min_y + clearance if longitudinal_is_x
            else slot.min_x + clearance)
        maximum_side = (
            slot.max_y - clearance if longitudinal_is_x
            else slot.max_x - clearance)
        for index, pose in enumerate(path.poses):
            for x, y in self._footprint_samples(
                    pose, length, width, edge_only=True):
                # Only samples level with the bay can touch its side walls;
                # the approach and setup legs run along the road outside it.
                longitudinal = x if longitudinal_is_x else y
                longitudinal_min = slot.min_x if longitudinal_is_x else slot.min_y
                longitudinal_max = slot.max_x if longitudinal_is_x else slot.max_y
                if not longitudinal_min <= longitudinal <= longitudinal_max:
                    continue
                side = y if longitudinal_is_x else x
                if not minimum_side <= side <= maximum_side:
                    return False, (
                        f'path index {index} comes within {clearance:.3f}m of '
                        f'the {slot.name} side wall (map side={side:.3f}, '
                        f'allowed {minimum_side:.3f}..{maximum_side:.3f})')
        return True, 'valid'

    def _pose_wall_clearance(
            self, slot: Slot, pose: PoseStamped) -> Optional[float]:
        """Return footprint clearance to the two bay sides and back curb."""
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        longitudinal_is_x = abs(forward_x) >= abs(forward_y)
        minimum = float('inf')
        for x, y in self._footprint_samples(
                pose, length, width, edge_only=True):
            longitudinal = x if longitudinal_is_x else y
            longitudinal_min = slot.min_x if longitudinal_is_x else slot.min_y
            longitudinal_max = slot.max_x if longitudinal_is_x else slot.max_y
            if not longitudinal_min <= longitudinal <= longitudinal_max:
                continue
            side = y if longitudinal_is_x else x
            side_min = slot.min_y if longitudinal_is_x else slot.min_x
            side_max = slot.max_y if longitudinal_is_x else slot.max_x
            forward_sign = forward_x if longitudinal_is_x else forward_y
            back_clearance = (
                longitudinal - longitudinal_min if forward_sign > 0.0
                else longitudinal_max - longitudinal)
            minimum = min(
                minimum,
                side - side_min,
                side_max - side,
                back_clearance,
            )
        return minimum if math.isfinite(minimum) else None

    def _path_wall_clearance(
            self, slot: Slot, path: Path) -> Optional[float]:
        clearances = [
            value for pose in path.poses
            if (value := self._pose_wall_clearance(slot, pose)) is not None
        ]
        return min(clearances) if clearances else None

    def _update_actual_wall_clearance(self, slot: Slot) -> None:
        current = self._current_map_pose()
        if current is None:
            return
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = current[0]
        pose.pose.position.y = current[1]
        set_pose_yaw(pose.pose, current[2])
        clearance = self._pose_wall_clearance(slot, pose)
        if clearance is None:
            return
        with self.data_lock:
            if self.active_segment_stats is not None:
                self.active_segment_stats['minimum_actual_wall_clearance'] = min(
                    self.active_segment_stats['minimum_actual_wall_clearance'],
                    clearance)

    def _setup_inside_road(self, pose: PoseStamped) -> bool:
        if self._abort_requested():
            return False
        road_min = float(self.get_parameter('road_min_y').value)
        road_max = float(self.get_parameter('road_max_y').value)
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        for x, y in self._footprint_samples(pose, length, width, edge_only=True):
            del x
            if not road_min < y < road_max:
                return False
        return True

    def _final_footprint_inside_slot(
            self, slot: Slot, pose: PoseStamped) -> bool:
        if self._abort_requested():
            return False

        # pose and slot bounds are both defined directly in the map frame.
        # Do not transform the footprint through odom again.
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)

        for x, y in self._footprint_samples(
                pose, length, width, edge_only=True):
            if not (
                    slot.min_x < x < slot.max_x
                    and slot.min_y < y < slot.max_y):
                return False
        return True

    @staticmethod
    def _grid_coordinates(origin: Pose, resolution: float, x: float, y: float):
        yaw = yaw_from_quaternion(origin.orientation)
        dx = x - origin.position.x
        dy = y - origin.position.y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return math.floor(local_x / resolution), math.floor(local_y / resolution)

    def _occupancy_value(
            self, grid: OccupancyGrid, x: float, y: float) -> Optional[int]:
        col, row = self._grid_coordinates(
            grid.info.origin, grid.info.resolution, x, y)
        if not (0 <= col < grid.info.width and 0 <= row < grid.info.height):
            return None
        return int(grid.data[row * grid.info.width + col])

    def _cost_value(self, grid: Costmap, x: float, y: float) -> Optional[int]:
        col, row = self._grid_coordinates(
            grid.metadata.origin, grid.metadata.resolution, x, y)
        if not (0 <= col < grid.metadata.size_x and 0 <= row < grid.metadata.size_y):
            return None
        return int(grid.data[row * grid.metadata.size_x + col])

    def _footprint_samples(
            self, pose: PoseStamped, length: float, width: float,
            edge_only: bool = False) -> List[Tuple[float, float]]:
        resolution = 0.05
        half_length = 0.5 * length
        half_width = 0.5 * width
        center_x = float(self.get_parameter(
            'vehicle_center_x_offset').value)
        nx = max(2, math.ceil(length / resolution))
        ny = max(2, math.ceil(width / resolution))
        local_points = []
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                if edge_only and ix not in (0, nx) and iy not in (0, ny):
                    continue
                local_points.append((
                    center_x - half_length + length * ix / nx,
                    -half_width + width * iy / ny))
        yaw = yaw_from_quaternion(pose.pose.orientation)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return [(
            pose.pose.position.x + cosine * x - sine * y,
            pose.pose.position.y + sine * x + cosine * y)
            for x, y in local_points]

    def _execute(
            self, candidate: PlanCandidate, monitor_wheels_inside: bool) -> bool:
        return self._execute_segmented_path(
            candidate.path, candidate.metrics,
            monitor_slot=candidate.slot if monitor_wheels_inside else None)

    def _execute_forward_exit(self, candidate: ForwardExitCandidate) -> bool:
        """Execute exactly one all-forward FollowPath action for the exit."""
        directions = [value for value in candidate.metrics.directions if value]
        if not directions or any(value < 0 for value in directions):
            self._log_error('refusing to execute a non-forward exit path')
            return False
        self._log_info(
            'FollowPath forward exit: one DUBIN path, no direction cusp; '
            'direction-locked ParkingForward controller')
        succeeded = self._execute_segmented_path(
            candidate.path, candidate.metrics, monitor_slot=None,
            forward_exit=True)
        if not succeeded:
            return False
        stats = self.segment_execution_stats[-1]
        first_wheel = int(stats['first_major_lidar_wheel'])
        if first_wheel == 0:
            first_wheel = int(stats['first_nonzero_lidar_wheel'])
        if bool(self.get_parameter('require_right_turn_exit').value):
            if first_wheel <= 0:
                self._log_error(
                    '[T-PARK][EXIT STEERING SIGN] FAIL: expected positive '
                    '/lidar_wheel for a forward right turn, got '
                    f'{first_wheel:+d}')
                return False
        self._log_info(
            '[T-PARK][EXIT STEERING SIGN] result=PASS '
            'geometry=RIGHT ros_angular_z=negative '
            'ros_steering=negative lidar_wheel=positive '
            f'first_major_lidar_wheel={first_wheel:+d} '
            f'max_abs_lidar_wheel={stats["max_abs_lidar_wheel"]:.0f}deg '
            'limit=27deg')
        return stats['max_abs_lidar_wheel'] <= 27.0

    @staticmethod
    def _direction_segments(
            path: Path, directions: Sequence[int]) -> List[DirectionSegment]:
        if not directions or len(path.poses) != len(directions) + 1:
            return []
        segments = []
        run_first = 0
        run_direction = directions[0]
        for edge_index in range(1, len(directions) + 1):
            if (edge_index < len(directions)
                    and directions[edge_index] == run_direction):
                continue
            length = sum(
                math.hypot(
                    path.poses[index + 1].pose.position.x
                    - path.poses[index].pose.position.x,
                    path.poses[index + 1].pose.position.y
                    - path.poses[index].pose.position.y)
                for index in range(run_first, edge_index))
            segments.append(DirectionSegment(
                run_first, edge_index, run_direction, length))
            if edge_index < len(directions):
                run_first = edge_index
                run_direction = directions[edge_index]
        return segments

    def _validate_segment_geometry(
            self, path: Path, segments: Sequence[DirectionSegment]) -> bool:
        if not segments:
            self._log_error('[T-PARK][SEGMENT] no executable segments')
            return False
        reconstructed = []
        for segment_number, segment in enumerate(segments):
            poses = path.poses[segment.first_pose:segment.last_pose + 1]
            reconstructed.extend(poses if segment_number == 0 else poses[1:])
        geometry_equal = (
            len(reconstructed) == len(path.poses)
            and all(actual.pose == expected.pose for actual, expected in zip(
                reconstructed, path.poses)))
        segment_length = sum(segment.length for segment in segments)
        full_length = sum(
            math.hypot(
                current.pose.position.x - previous.pose.position.x,
                current.pose.position.y - previous.pose.position.y)
            for previous, current in zip(path.poses, path.poses[1:]))
        length_difference = segment_length - full_length
        self._log_info(
            '[T-PARK][SEGMENT GEOMETRY]\n'
            f'full_pose_count={len(path.poses)}\n'
            f'reconstructed_pose_count={len(reconstructed)}\n'
            f'full_length={full_length:.6f}m\n'
            f'segment_length_sum={segment_length:.6f}m\n'
            f'length_difference={length_difference:+.12f}m\n'
            f'pose_geometry_equal={str(geometry_equal).lower()}')
        if not geometry_equal or abs(length_difference) > 1.0e-9:
            self._log_error(
                '[T-PARK][SEGMENT] slicing changed the original path geometry')
            return False

        spacings = sorted(
            math.hypot(
                current.pose.position.x - previous.pose.position.x,
                current.pose.position.y - previous.pose.position.y)
            for previous, current in zip(path.poses, path.poses[1:])
            if math.hypot(
                current.pose.position.x - previous.pose.position.x,
                current.pose.position.y - previous.pose.position.y) > 1.0e-6)
        median_spacing = spacings[len(spacings) // 2] if spacings else 0.0
        minimum_length = max(0.10, 3.0 * median_spacing)
        for segment in segments:
            if (segment.last_pose - segment.first_pose < 3
                    or segment.length < minimum_length):
                self._log_error(
                    '[T-PARK][SEGMENT] fake/unsafe short segment: '
                    f'poses={segment.first_pose}..{segment.last_pose} '
                    f'length={segment.length:.3f}m '
                    f'minimum={minimum_length:.3f}m')
                return False
        return True

    def _fresh_segment_path(
            self, path: Path, segment: DirectionSegment) -> Path:
        """Copy one inclusive pose slice and refresh only ROS timestamps."""
        result = Path()
        result.header = copy.deepcopy(path.header)
        stamp = self.get_clock().now().to_msg()
        result.header.stamp = stamp
        result.poses = copy.deepcopy(
            path.poses[segment.first_pose:segment.last_pose + 1])
        for pose in result.poses:
            if not pose.header.frame_id:
                pose.header.frame_id = result.header.frame_id
            pose.header.stamp = stamp
        return result

    def _log_segment_start_relation(
            self, path: Path, segment: DirectionSegment) -> None:
        current = self._current_map_pose()
        if current is None:
            self._log_error(
                '[T-PARK][SEGMENT] current map pose unavailable at start')
            return
        cosine = math.cos(current[2])
        sine = math.sin(current[2])
        points = []
        for local_index, pose in enumerate(path.poses):
            dx = pose.pose.position.x - current[0]
            dy = pose.pose.position.y - current[1]
            distance = math.hypot(dx, dy)
            if distance < 0.02:
                continue
            base_x = cosine * dx + sine * dy
            base_y = -sine * dx + cosine * dy
            points.append((segment.first_pose + local_index, base_x, base_y))
            if len(points) == 3:
                break
        relation = ', '.join(
            f'index={index} base=({base_x:+.3f},{base_y:+.3f})'
            for index, base_x, base_y in points)
        self._log_info(
            f'[T-PARK][SEGMENT {self.active_segment_number}] start relation: '
            f'expected={"REVERSE" if segment.direction < 0 else "FORWARD"}; '
            f'{relation or "no point beyond 0.02m"}')

    def _begin_segment_monitor(
            self, segment_number: int, segment: DirectionSegment,
            path: Path, monitor_slot: Optional[Slot]) -> None:
        expected_clearance = (
            self._path_wall_clearance(monitor_slot, path)
            if monitor_slot is not None and segment.direction < 0
            else None)
        with self.data_lock:
            start_odom = copy.deepcopy(self.odom_msg)
            start_real_odom = copy.deepcopy(self.real_odom_msg)
        stats = {
            'segment': segment_number,
            'direction': segment.direction,
            'cmd_vel_nav_wrong_samples': 0,
            'cmd_vel_nav_wrong_events': 0,
            'cmd_vel_wrong_samples': 0,
            'cmd_vel_wrong_events': 0,
            'cmd_vel_control_wrong_samples': 0,
            'cmd_vel_control_wrong_events': 0,
            'lidar_drive_wrong_samples': 0,
            'lidar_drive_wrong_events': 0,
            'max_abs_lidar_wheel': 0.0,
            'minimum_lidar_wheel': float('inf'),
            'maximum_lidar_wheel': -float('inf'),
            'first_nonzero_lidar_wheel': 0,
            'first_major_lidar_wheel': 0,
            'lidar_wheel_samples': 0,
            'minimum_cmd_vel_linear_x': float('inf'),
            'maximum_cmd_vel_linear_x': -float('inf'),
            'start_odom_x': (
                float(start_odom.pose.pose.position.x)
                if start_odom is not None else float('nan')),
            'start_odom_y': (
                float(start_odom.pose.pose.position.y)
                if start_odom is not None else float('nan')),
            'start_real_odom_x': (
                float(start_real_odom.pose.pose.position.x)
                if start_real_odom is not None else float('nan')),
            'start_real_odom_y': (
                float(start_real_odom.pose.pose.position.y)
                if start_real_odom is not None else float('nan')),
            'lidar_wheel_saturation_samples': 0,
            'lidar_wheel_saturation_events': 0,
            'lidar_wheel_saturation_active': False,
            'lidar_wheel_sign_reversals': 0,
            'last_lidar_wheel_nonzero_sign': 0,
            'minimum_expected_wall_clearance': (
                float('inf') if expected_clearance is None
                else expected_clearance),
            'minimum_actual_wall_clearance': float('inf'),
        }
        with self.data_lock:
            self.active_direction_segment = segment
            self.active_segment_number = segment_number
            self.active_segment_stats = stats
            self.active_segment_path = path
            self._last_direction_sign = {}
            self.last_direction_diagnostic = {}
            self.reverse_motion_start_logged = False
            self.reverse_steering_start_logged = False
        if expected_clearance is not None:
            self._log_info(
                '[WALL CLEARANCE] '
                f'expected_path_minimum={expected_clearance:.4f}m')

    def _finish_segment_monitor(self) -> None:
        with self.data_lock:
            stats = self.active_segment_stats
            finish_odom = copy.deepcopy(self.odom_msg)
            finish_real_odom = copy.deepcopy(self.real_odom_msg)
            self.active_direction_segment = None
            self.active_segment_stats = None
            self.active_segment_path = None
            self._last_direction_sign = {}
        if stats is None:
            return
        self.segment_execution_stats.append(dict(stats))
        wheel_samples = max(1, int(stats['lidar_wheel_samples']))
        saturation_ratio = (
            100.0 * stats['lidar_wheel_saturation_samples'] / wheel_samples)
        expected_clearance = stats['minimum_expected_wall_clearance']
        actual_clearance = stats['minimum_actual_wall_clearance']
        expected_text = (
            f'{expected_clearance:.4f}m'
            if math.isfinite(expected_clearance) else 'unavailable')
        actual_text = (
            f'{actual_clearance:.4f}m'
            if math.isfinite(actual_clearance) else 'unavailable')
        finish_x = (
            float(finish_odom.pose.pose.position.x)
            if finish_odom is not None else float('nan'))
        finish_y = (
            float(finish_odom.pose.pose.position.y)
            if finish_odom is not None else float('nan'))
        odom_delta = math.hypot(
            finish_x - stats['start_odom_x'],
            finish_y - stats['start_odom_y'])
        real_finish_x = (
            float(finish_real_odom.pose.pose.position.x)
            if finish_real_odom is not None else float('nan'))
        real_finish_y = (
            float(finish_real_odom.pose.pose.position.y)
            if finish_real_odom is not None else float('nan'))
        real_odom_delta = math.hypot(
            real_finish_x - stats['start_real_odom_x'],
            real_finish_y - stats['start_real_odom_y'])
        minimum_wheel = stats['minimum_lidar_wheel']
        maximum_wheel = stats['maximum_lidar_wheel']
        minimum_cmd = stats['minimum_cmd_vel_linear_x']
        maximum_cmd = stats['maximum_cmd_vel_linear_x']
        if self.bench_mode:
            self._log_info(
                '[BENCH SEGMENT]\n'
                f'segment={int(stats["segment"]) + 1}\n'
                f'direction={"REVERSE" if stats["direction"] < 0 else "FORWARD"}\n'
                f'cmd_vel.linear.x_min={minimum_cmd:+.4f}\n'
                f'cmd_vel.linear.x_max={maximum_cmd:+.4f}\n'
                f'lidar_drive_expected={stats["direction"]:+.1f}\n'
                f'lidar_wheel_min={minimum_wheel:+.0f}\n'
                f'lidar_wheel_max={maximum_wheel:+.0f}\n'
                f'virtual_odom_delta={odom_delta:.4f}m\n'
                f'real_odom_delta={real_odom_delta:.4f}m\n'
                'wheel_mapping=negative:left, positive:right')
        self._log_info(
            f'[T-PARK][SEGMENT {stats["segment"]}] direction summary: '
            f'cmd_vel_nav_wrong_samples={stats["cmd_vel_nav_wrong_samples"]} '
            f'cmd_vel_nav_wrong_events={stats["cmd_vel_nav_wrong_events"]} '
            f'cmd_vel_wrong_samples={stats["cmd_vel_wrong_samples"]} '
            f'cmd_vel_wrong_events={stats["cmd_vel_wrong_events"]} '
            f'cmd_vel_control_wrong_samples='
            f'{stats["cmd_vel_control_wrong_samples"]} '
            f'cmd_vel_control_wrong_events='
            f'{stats["cmd_vel_control_wrong_events"]} '
            f'lidar_drive_wrong_samples={stats["lidar_drive_wrong_samples"]} '
            f'lidar_drive_wrong_events={stats["lidar_drive_wrong_events"]} '
            f'max_abs_lidar_wheel={stats["max_abs_lidar_wheel"]:.3f} '
            f'lidar_wheel_saturation_samples='
            f'{stats["lidar_wheel_saturation_samples"]}/'
            f'{stats["lidar_wheel_samples"]} '
            f'saturation_ratio={saturation_ratio:.3f}% '
            f'saturation_events={stats["lidar_wheel_saturation_events"]} '
            f'steering_sign_reversals='
            f'{stats["lidar_wheel_sign_reversals"]} '
            f'minimum_expected_wall_clearance={expected_text} '
            f'minimum_actual_wall_clearance={actual_text}')

    def _execute_segmented_path(
            self, path: Path, metrics: PathMetrics,
            monitor_slot: Optional[Slot], forward_exit: bool = False) -> bool:
        self.last_execution_failure_reason = ''
        directions = metrics.directions
        if not directions:
            self._log_error('validated path has no motion segments')
            self.last_execution_failure_reason = (
                'validated path has no motion segments')
            return False
        self.active_motion_path = path
        self.active_motion_metrics = metrics
        self.active_motion_final = path.poses[-1]
        segments = self._direction_segments(path, directions)
        if not self._validate_segment_geometry(path, segments):
            self.last_execution_failure_reason = (
                'segment geometry validation failed before FollowPath')
            return False
        self._log_info(
            f'[T-PARK][SEGMENT]\ntotal poses: {len(path.poses)}\n'
            f'segment count: {len(segments)}\n'
            f'cusps: {max(0, len(segments) - 1)}\n'
            f'max curvature: {metrics.max_curvature:.6f} 1/m')
        for number, segment in enumerate(segments):
            first_pose = path.poses[segment.first_pose]
            last_pose = path.poses[segment.last_pose]
            segment_path = self._fresh_segment_path(path, segment)
            segment_metrics = self._analyze_path(segment_path, last_pose)
            if number:
                self._log_info(
                    f'cusp {number}: pose index={segment.first_pose} '
                    f'pose=({first_pose.pose.position.x:.3f},'
                    f'{first_pose.pose.position.y:.3f},'
                    f'{yaw_from_quaternion(first_pose.pose.orientation):.3f})')
            self._log_info(
                f'segment {number}: '
                f'direction={"REVERSE" if segment.direction < 0 else "FORWARD"} '
                f'poses={segment.first_pose}..{segment.last_pose} '
                f'count={segment.last_pose - segment.first_pose + 1} '
                f'length={segment.length:.3f}m '
                f'start_yaw={yaw_from_quaternion(first_pose.pose.orientation):.3f} '
                f'end_yaw={yaw_from_quaternion(last_pose.pose.orientation):.3f} '
                f'max_curvature={segment_metrics.max_curvature:.6f} 1/m')

        for run_number, segment in enumerate(segments, start=1):
            segment_path = self._fresh_segment_path(path, segment)
            self._begin_segment_monitor(
                run_number - 1, segment, segment_path, monitor_slot)
            self._log_info(
                f'[T-PARK][SEGMENT {run_number - 1}]\n'
                f'direction = '
                f'{"REVERSE" if segment.direction < 0 else "FORWARD"}\n'
                f'poses = {segment.first_pose}..{segment.last_pose}\n'
                f'length = {segment.length:.3f}m')
            self._log_segment_start_relation(segment_path, segment)
            succeeded = self._execute_path_action(
                segment_path, run_number, len(segments), segment.direction,
                monitor_slot=monitor_slot, forward_exit=forward_exit)
            self._finish_segment_monitor()
            if not succeeded:
                return False
            if monitor_slot is not None and self.parking_wheels_inside:
                return True
            if run_number < len(segments):
                # End one Nav2 action completely before giving RPP the next
                # direction.  Do not publish a competing zero command here;
                # controller_server and velocity_smoother own that chain.
                next_segment = segments[run_number]
                self._log_info(
                    '[T-PARK] cusp reached; waiting for vehicle stop...')
                if not self._wait_until_stopped(float(
                        self.get_parameter('stop_wait_timeout').value)):
                    self._log_pre_segment_failure(
                        run_number, next_segment.direction, forward_exit,
                        'vehicle_did_not_settle_at_direction_cusp')
                    return False
                self._log_info(
                    '[T-PARK] vehicle stopped at cusp: '
                    f'linear_speed={self.last_stop_linear_speed:.4f}m/s '
                    f'angular_speed={self.last_stop_angular_speed:.4f}rad/s '
                    f'elapsed={self.last_stop_elapsed:.3f}s')
                if self.bench_mode:
                    if not self._confirm_cusp_zero_handoff(
                            segment.direction, next_segment.direction,
                            run_number):
                        return False
                self._trace_command_chain('CUSP_REACHED')
        return True

    def _confirm_cusp_zero_handoff(
            self, from_direction: int, to_direction: int,
            next_segment_number: int, timeout: float = 2.0) -> bool:
        """Require three fresh paired zero commands before reversing."""
        from_name = 'REVERSE' if from_direction < 0 else 'FORWARD'
        to_name = 'REVERSE' if to_direction < 0 else 'FORWARD'
        self._log_info(
            '[T-PARK][CUSP HANDOFF] '
            f'from={from_name} to={to_name} stage=WAIT_ZERO')
        with self.data_lock:
            self.bench_consecutive_zero_drive_samples = 0
            self.bench_consecutive_zero_wheel_samples = 0
        deadline = time.monotonic() + timeout
        reported_samples = 0
        drive = float('inf')
        wheel = 999
        drive_samples = 0
        wheel_samples = 0
        while time.monotonic() < deadline and not self._abort_requested():
            now = time.monotonic()
            with self.data_lock:
                drive = self.last_lidar_drive
                wheel = self.last_lidar_wheel
                drive_samples = self.bench_consecutive_zero_drive_samples
                wheel_samples = self.bench_consecutive_zero_wheel_samples
                fresh = (
                    now - self.lidar_drive_received_at < 1.0
                    and now - self.lidar_wheel_received_at < 1.0)
            paired_samples = min(drive_samples, wheel_samples, 3)
            if fresh and abs(drive) < 1.0e-6 and wheel == 0:
                while reported_samples < paired_samples:
                    reported_samples += 1
                    self._log_info(
                        '[T-PARK][CUSP ZERO] '
                        f'sample={reported_samples}/3')
                if paired_samples >= 3:
                    self._log_info(
                        '[T-PARK][CUSP STOP] result=PASS '
                        f'drive_zero_samples={drive_samples} '
                        f'wheel_zero_samples={wheel_samples}')
                    return True
            if not self._interruptible_sleep(0.05, stop_on_cancel=False):
                break
        reason = (
            'paired_zero_handoff_failed:'
            f'drive={drive:+.1f},wheel={wheel:+d},'
            f'drive_zero_samples={drive_samples},'
            f'wheel_zero_samples={wheel_samples}')
        self._log_pre_segment_failure(
            next_segment_number, to_direction, False, reason)
        return False

    def _log_pre_segment_failure(
            self, segment_number: int, direction: int,
            forward_exit: bool, reason: str) -> None:
        """Preserve the exact reason when no FollowPath goal was sent."""
        phase = 'EXIT' if forward_exit else 'PARKING'
        direction_name = 'REVERSE' if direction < 0 else 'FORWARD'
        self.last_execution_failure_reason = (
            f'pre-segment failure: phase={phase} segment={segment_number} '
            f'direction={direction_name} reason={reason}')
        self._log_error(
            '[T-PARK][PRE-SEGMENT FAILURE]\n'
            f'phase={phase}\n'
            f'segment={segment_number}\n'
            f'direction={direction_name}\n'
            f'reason={reason}')

    def _execute_path_action(
            self, path: Path, run_number: int, run_count: int,
            direction: int, monitor_slot: Optional[Slot] = None,
            forward_exit: bool = False) -> bool:
        if self.bench_mode and not self._bench_motion_preflight(
                require_fresh_status=False):
            self._log_pre_segment_failure(
                run_number - 1, direction, forward_exit,
                self.last_bench_preflight_failure_reason)
            self._emergency_stop()
            return False
        require_physical_rear_lidar = not (
            self.bench_mode and self.wheels_off_ground)
        if direction < 0 and require_physical_rear_lidar:
            healthy, rear_distance, reason = self._rear_scan_state()
            if not healthy:
                self._log_pre_segment_failure(
                    run_number - 1, direction, forward_exit,
                    f'rear_lidar_sector_{reason.replace(" ", "_")}')
                self._emergency_stop(emergency=True)
                return False
            self._log_info(
                f'rear safety armed: nearest={rear_distance:.3f}m '
                'threshold='
                f'{float(self.get_parameter("rear_emergency_stop_distance").value):.3f}m')
        elif direction < 0:
            self._log_info(
                '[BENCH SAFETY]\n'
                'physical rear LiDAR availability check skipped:\n'
                'bench_mode=true wheels_off_ground=true')
        if self._abort_requested():
            self._log_pre_segment_failure(
                run_number - 1, direction, forward_exit,
                'abort_requested_before_goal_send')
            return False
        goal = FollowPath.Goal()
        goal.path = path
        with self.data_lock:
            active_segment = self.active_direction_segment
            active_segment_number = self.active_segment_number
        if active_segment is None:
            goal.controller_id = str(
                self.get_parameter('controller_id').value)
        else:
            controller_parameter = (
                'reverse_controller_id' if direction < 0
                else 'forward_controller_id')
            goal.controller_id = str(
                self.get_parameter(controller_parameter).value)
            segment_state = Int32MultiArray()
            segment_state.data = [
                int(active_segment_number), int(direction),
                int(active_segment.first_pose), int(active_segment.last_pose)]
            self._safe_publish(self.segment_state_publisher, segment_state)
            self._log_info(
                '[T-PARK][SEGMENT STATE] '
                f'segment={active_segment_number} '
                f'direction={"REVERSE" if direction < 0 else "FORWARD"} '
                f'full_range={active_segment.first_pose}..'
                f'{active_segment.last_pose} '
                f'controller_id={goal.controller_id}')
        goal.goal_checker_id = (
            str(self.get_parameter('goal_checker_id').value)
            if run_number == run_count else 'cusp_goal_checker')
        goal.progress_checker_id = str(
            self.get_parameter('progress_checker_id').value)
        self._log_info(
            '[T-PARK][SEGMENT START] '
            f'phase={"EXIT" if forward_exit else "PARKING"} '
            f'segment={run_number - 1} '
            f'direction={"REVERSE" if direction < 0 else "FORWARD"} '
            f'controller_id={goal.controller_id}')
        self._log_info(
            f'[T-PARK] sending FollowPath goal {run_number}/{run_count}: '
            f'controller_id={goal.controller_id} '
            f'goal_checker_id={goal.goal_checker_id} '
            f'progress_checker_id={goal.progress_checker_id}')
        if not self._runtime_ok():
            self._log_pre_segment_failure(
                run_number - 1, direction, forward_exit,
                'runtime_not_ok_before_goal_send')
            return False
        send_future = self.follow_client.send_goal_async(
            goal, feedback_callback=self._follow_feedback)
        goal_handle = self._wait_future(send_future, 5.0)
        if goal_handle is None or not goal_handle.accepted:
            self.last_execution_failure_reason = (
                f'FollowPath goal rejected: controller_id={goal.controller_id}')
            self._log_info(
                f'[T-PARK][SEGMENT {run_number - 1}] REJECTED: '
                f'controller_id={goal.controller_id} '
                f'goal_checker_id={goal.goal_checker_id}')
            return False
        self._log_info(
            '[T-PARK] FollowPath goal accepted: '
            f'controller_id={goal.controller_id} '
            f'goal_checker_id={goal.goal_checker_id}')
        self.active_follow_goal = goal_handle
        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + float(
            self.get_parameter('follow_path_timeout').value)
        next_wheel_check = time.monotonic()
        while not result_future.done():
            if self._abort_requested():
                if self._context_ok():
                    goal_handle.cancel_goal_async()
                return False
            if time.monotonic() >= deadline:
                self._log_info(
                    f'[T-PARK][SEGMENT {run_number - 1}] TIMEOUT; '
                    'cancelling FollowPath goal')
                if self._context_ok():
                    goal_handle.cancel_goal_async()
                self._emergency_stop(emergency=True)
                return False
            if direction < 0 and require_physical_rear_lidar:
                healthy, rear_distance, reason = self._rear_scan_state()
                threshold = float(self.get_parameter(
                    'rear_emergency_stop_distance').value)
                if not healthy or rear_distance <= threshold:
                    if healthy:
                        detail = (
                            f'obstacle at {rear_distance:.3f}m <= '
                            f'{threshold:.3f}m')
                    else:
                        detail = reason
                    self._log_error(
                        f'REAR EMERGENCY STOP: {detail}; cancelling FollowPath')
                    if self._context_ok():
                        goal_handle.cancel_goal_async()
                    self._emergency_stop(emergency=True)
                    return False
            if forward_exit:
                with self.data_lock:
                    commanded_linear_x = float(self.last_cmd_vel.linear.x)
                if commanded_linear_x < -0.01:
                    self._log_error(
                        'FORWARD EXIT SAFETY STOP: /cmd_vel.linear.x='
                        f'{commanded_linear_x:.3f}m/s is reverse; '
                        'cancelling FollowPath')
                    if self._context_ok():
                        goal_handle.cancel_goal_async()
                    self._emergency_stop(emergency=True)
                    return False
            now = time.monotonic()
            if monitor_slot is not None and now >= next_wheel_check:
                next_wheel_check = now + float(self.get_parameter(
                    'wheel_check_period').value)
                self._update_actual_wall_clearance(monitor_slot)
                if self._update_wheels_inside(monitor_slot):
                    self.parking_wheels_inside = True
                    self._publish_status('WHEELS_INSIDE')
                    if not self._cancel_parking_follow_path(
                            goal_handle, result_future):
                        return False
                    self._log_info(
                        '[T-PARK][SEGMENT RESULT] '
                        f'segment={run_number - 1} '
                        f'direction={"REVERSE" if direction < 0 else "FORWARD"} '
                        'result=SUCCEEDED reason=parked_footprint_confirmed')
                    return True
            if not self._interruptible_sleep(0.05):
                return False
        try:
            wrapped = result_future.result()
        except Exception as exc:
            self._log_error(f'FollowPath result retrieval failed: {exc}')
            return False
        self.active_follow_goal = None
        result = wrapped.result
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            names = {
                GoalStatus.STATUS_ABORTED: 'ABORTED',
                GoalStatus.STATUS_CANCELED: 'CANCELED',
            }
            self._log_info(
                f'[T-PARK][SEGMENT {run_number - 1}] '
                f'{names.get(wrapped.status, f"STATUS_{wrapped.status}")}: '
                f'error_code={result.error_code} '
                f'error_msg={result.error_msg!r}')
            self._log_info(
                '[T-PARK][SEGMENT RESULT] '
                f'segment={run_number - 1} '
                f'direction={"REVERSE" if direction < 0 else "FORWARD"} '
                f'result={names.get(wrapped.status, "FAILED")} '
                f'error_code={result.error_code}')
            return False
        if monitor_slot is not None and run_number == run_count:
            # At parking speed the controller can report goal success on the
            # same cycle that begins the wheel-containment confirmation
            # window.  Keep applying the existing geometric test while the
            # now-stationary vehicle settles; do not reject a valid final pose
            # merely because five timer samples had not elapsed in motion.
            period = float(self.get_parameter('wheel_check_period').value)
            required = int(self.get_parameter(
                'wheel_inside_confirm_count').value)
            confirmation_deadline = time.monotonic() + period * (required + 2)
            self._log_info(
                'parking FollowPath reached endpoint; completing existing '
                f'all-wheel-inside confirmation ({required} samples)')
            while time.monotonic() < confirmation_deadline:
                self._update_actual_wall_clearance(monitor_slot)
                if self._update_wheels_inside(monitor_slot):
                    self.parking_wheels_inside = True
                    self._publish_status('WHEELS_INSIDE')
                    self._log_info(
                        'parking endpoint accepted after stationary '
                        'all-wheel-inside confirmation')
                    self._log_info(
                        '[T-PARK][SEGMENT RESULT] '
                        f'segment={run_number - 1} '
                        f'direction={"REVERSE" if direction < 0 else "FORWARD"} '
                        'result=SUCCEEDED reason=parked_footprint_confirmed')
                    return True
                if not self._interruptible_sleep(period):
                    return False
            self._log_error(
                'parking FollowPath reached its planned endpoint but the '
                'existing all-wheel-inside check did not confirm containment')
            return False
        self._log_info(
            f'[T-PARK][SEGMENT {run_number - 1}] SUCCEEDED: '
            f'error_code={result.error_code} '
            f'error_msg={result.error_msg!r}')
        self._log_info(
            '[T-PARK][SEGMENT RESULT] '
            f'segment={run_number - 1} '
            f'direction={"REVERSE" if direction < 0 else "FORWARD"} '
            f'result={"SUCCEEDED" if result.error_code == FollowPath.Result.NONE else "FAILED"} '
            f'error_code={result.error_code}')
        return result.error_code == FollowPath.Result.NONE

    def _cancel_parking_follow_path(self, goal_handle, result_future) -> bool:
        self._log_info(
            'all wheels confirmed inside; cancelling active parking FollowPath')

        # The wheel-confirmation timer and controller goal checker can finish
        # on the same control cycle.  Reaching the planned goal after all four
        # wheels were already confirmed inside is success, not a failed
        # cancellation.  Check this race before sending a redundant request.
        if result_future.done():
            wrapped = result_future.result()
            self.active_follow_goal = None
            if wrapped.status in (
                    GoalStatus.STATUS_SUCCEEDED,
                    GoalStatus.STATUS_CANCELED):
                self._log_info(
                    'parking FollowPath already terminated after wheels-inside '
                    f'confirmation with status={wrapped.status}; accepted')
                return True
            self._log_error(
                'parking FollowPath terminated unexpectedly after '
                f'wheels-inside confirmation: status={wrapped.status}')
            return False

        cancel_future = goal_handle.cancel_goal_async()
        timeout = float(self.get_parameter('parking_cancel_timeout').value)
        response = self._wait_future(cancel_future, timeout)
        if response is None or not response.goals_canceling:
            # A successful result may win the race while the cancel request is
            # in flight.  Accept it because wheel containment was confirmed
            # before entering this function.
            wrapped = self._wait_future(result_future, timeout)
            self.active_follow_goal = None
            if (wrapped is not None
                    and wrapped.status == GoalStatus.STATUS_SUCCEEDED):
                self._log_info(
                    'parking FollowPath reached goal while cancel was in '
                    'flight; wheels-inside confirmation already passed')
                return True
            self._log_error(
                'parking FollowPath cancel request was not accepted')
            return False
        self._log_info(
            'parking FollowPath cancel response accepted; waiting for '
            'controller goal termination')
        wrapped = self._wait_future(result_future, timeout)
        self.active_follow_goal = None
        if wrapped is None:
            self._log_error(
                'parking FollowPath did not terminate after cancel response')
            return False
        if wrapped.status not in (
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_SUCCEEDED):
            self._log_error(
                f'parking FollowPath cancel ended with status={wrapped.status}')
            return False
        self._log_info(
            'parking FollowPath result confirmed '
            f'status={wrapped.status} after wheels-inside confirmation')
        return True

    def _rear_scan_state(self) -> Tuple[bool, float, str]:
        with self.data_lock:
            scan = self.rear_scan_msg
            received_at = self.rear_scan_received_at
            local_costmap_ready = self.local_costmap is not None
        if scan is None:
            return False, float('inf'), 'has no message'
        age = time.monotonic() - received_at
        timeout = float(self.get_parameter('scan_timeout').value)
        if age > timeout:
            return False, float('inf'), f'is stale (age={age:.2f}s)'
        if not local_costmap_ready:
            return False, float('inf'), 'local costmap is unavailable'
        sector = math.radians(float(
            self.get_parameter('rear_emergency_sector_deg').value))
        distances = []
        for index, value in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            rear_angle_error = abs(angle)
            if (rear_angle_error <= sector and math.isfinite(value)
                    and scan.range_min <= value <= scan.range_max):
                distances.append(float(value))
        if not distances:
            # Positive infinity is the standard LaserScan representation for
            # no return inside range_max.  That is a valid clear-space safety
            # result, unlike NaN, a stale scan, or an empty message.
            sector_values = [
                value for index, value in enumerate(scan.ranges)
                if abs(scan.angle_min
                       + index * scan.angle_increment) <= sector]
            if sector_values and all(math.isinf(value) and value > 0.0
                                     for value in sector_values):
                return True, float(scan.range_max), 'clear to range_max'
            return False, float('inf'), 'rear-facing sector has no valid range'
        return True, min(distances), 'valid'

    def _follow_feedback(self, feedback_message) -> None:
        if not self._runtime_ok():
            return
        feedback = feedback_message.feedback
        now = time.monotonic()
        if now - self.last_feedback_log < 0.5:
            return
        self.last_feedback_log = now
        current_pose = self._current_map_pose()
        path = self.active_motion_path
        metrics = self.active_motion_metrics
        final_pose = self.active_motion_final
        if (path is None or metrics is None or final_pose is None
                or current_pose is None):
            return
        self._update_direction_diagnostic(current_pose)
        if self.motion_phase == 'parking' and not self.parking_wheels_inside:
            self._publish_status('EXECUTING_PARKING')
        positions = path.poses
        nearest = min(
            range(len(positions)),
            key=lambda index: math.hypot(
                positions[index].pose.position.x - current_pose[0],
                positions[index].pose.position.y - current_pose[1]))
        segment_index = min(nearest, len(metrics.directions) - 1)
        direction = (
            metrics.directions[segment_index]
            if metrics.directions else 1)
        if self.motion_phase == 'forward_exit':
            clear_pose = self.forward_exit_poses.get('slot_forward_clear')
            merge_pose = self.forward_exit_poses.get('lane_merge')
            if clear_pose is not None and merge_pose is not None:
                clear_index = min(
                    range(len(positions)),
                    key=lambda index: math.hypot(
                        positions[index].pose.position.x
                        - clear_pose.pose.position.x,
                        positions[index].pose.position.y
                        - clear_pose.pose.position.y))
                merge_index = min(
                    range(len(positions)),
                    key=lambda index: math.hypot(
                        positions[index].pose.position.x
                        - merge_pose.pose.position.x,
                        positions[index].pose.position.y
                        - merge_pose.pose.position.y))
                if nearest <= clear_index:
                    self._publish_status('EXECUTING_FORWARD_FROM_SLOT')
                elif nearest <= merge_index:
                    self._publish_status('EXECUTING_RIGHT_TURN')
                else:
                    self._publish_status('DRIVING_TO_ENTRANCE')
        final = final_pose.pose.position
        final_distance = math.hypot(
            final.x - current_pose[0], final.y - current_pose[1])
        self._log_info(
            f'pose=({current_pose[0]:.3f},{current_pose[1]:.3f},'
            f'{current_pose[2]:.3f}) remaining={feedback.distance_to_goal:.3f}m '
            f'speed={feedback.speed:.3f}m/s path_index={nearest} '
            f'direction={"reverse" if direction < 0 else "forward"} '
            f'final_distance={final_distance:.3f}m')

    def _update_direction_diagnostic(
            self, current_pose: Tuple[float, float, float]) -> None:
        """Approximate RPP's first distance-qualified transformed carrot."""
        with self.data_lock:
            path = self.active_segment_path
            segment = self.active_direction_segment
            odom = self.odom_msg
        if path is None or segment is None or not path.poses:
            return
        nearest = min(
            range(len(path.poses)),
            key=lambda index: math.hypot(
                path.poses[index].pose.position.x - current_pose[0],
                path.poses[index].pose.position.y - current_pose[1]))
        speed = 0.0 if odom is None else abs(float(
            odom.twist.twist.linear.x))
        # Mirrors the unchanged ParkingFollowPath limits/time in
        # nav2_params.yaml solely for diagnostics; it does not affect control.
        lookahead = max(0.15, min(0.35, speed * 1.0))
        carrot_index = len(path.poses) - 1
        for index in range(nearest, len(path.poses)):
            pose = path.poses[index].pose.position
            if math.hypot(
                    pose.x - current_pose[0],
                    pose.y - current_pose[1]) >= lookahead:
                carrot_index = index
                break
        carrot = path.poses[carrot_index].pose.position
        dx = carrot.x - current_pose[0]
        dy = carrot.y - current_pose[1]
        carrot_base_x = (
            math.cos(current_pose[2]) * dx
            + math.sin(current_pose[2]) * dy)
        with self.data_lock:
            self.last_direction_diagnostic = {
                'path_index': float(segment.first_pose + nearest),
                'robot_x': current_pose[0],
                'robot_y': current_pose[1],
                'robot_yaw': current_pose[2],
                'carrot_base_x': carrot_base_x,
            }

    def _current_map_pose(self) -> Optional[Tuple[float, float, float]]:
        if self._abort_requested():
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', self.robot_base_frame, Time(),
                timeout=Duration(seconds=0.5))
        except tf2_ros.TransformException:
            return None
        return (
            transform.transform.translation.x,
            transform.transform.translation.y,
            yaw_from_quaternion(transform.transform.rotation),
        )

    def _verify_final_pose(self, candidate: PlanCandidate) -> Tuple[float, float]:
        current = self._current_map_pose()
        if current is None:
            return float('inf'), float('inf')
        final = candidate.named_poses['final'].pose
        return (
            math.hypot(
                current[0] - final.position.x,
                current[1] - final.position.y),
            abs(normalize_angle(current[2] - yaw_from_quaternion(final.orientation))),
        )

    def _validate_and_log_parked(
            self, slot: Slot, pose: PoseStamped) -> bool:
        """Require the actual stopped physical footprint to be safely parked."""
        assessment = self._parking_pose_assessment(slot, pose)
        required_end = float(self.get_parameter(
            'parking_end_clearance_m').value)
        end_valid = assessment.end_clearance + 1.0e-6 >= required_end
        front_valid = assessment.front_clearance >= -1.0e-6
        valid = assessment.footprint_inside and end_valid and front_valid
        self._log_info(
            '[T-PARK][PARKED]\n'
            f'final_pose=({pose.pose.position.x:.3f},'
            f'{pose.pose.position.y:.3f},'
            f'{yaw_from_quaternion(pose.pose.orientation):.6f})\n'
            f'slot_id={slot.name}\n'
            f'rear/end_clearance={assessment.end_clearance:.3f}m\n'
            f'front_clearance={assessment.front_clearance:.3f}m\n'
            f'side_clearance={assessment.side_clearance:.3f}m\n'
            f'configured_end_clearance={required_end:.3f}m\n'
            f'vehicle_footprint_inside_slot='
            f'{str(assessment.footprint_inside).lower()}\n'
            f'result={"PASS" if valid else "FAIL"}')
        return valid

    def _verify_entrance_return(self) -> Tuple[float, float]:
        current = self._current_map_pose()
        if current is None or self.course_entrance_pose is None:
            return float('inf'), float('inf')
        entrance = self.course_entrance_pose.pose
        # Position must match the real, originally-captured entrance point.
        # Heading must not: a forward-only vehicle returning along the same
        # single lane necessarily arrives facing opposite to its original
        # outbound heading, so compare against the planned approach heading
        # for this exit (direction of travel into the entrance) instead of
        # the outbound spawn yaw.
        expected_pose = self.forward_exit_poses.get('course_entrance')
        expected_orientation = (
            expected_pose.pose.orientation if expected_pose is not None
            else entrance.orientation)
        return (
            math.hypot(
                current[0] - entrance.position.x,
                current[1] - entrance.position.y),
            abs(normalize_angle(
                current[2] - yaw_from_quaternion(expected_orientation))),
        )

    def _confirm_motion_stop(self, context: str) -> bool:
        if self._wait_until_stopped(float(self.get_parameter(
                'parking_stop_grace_timeout').value)):
            return True
        self._log_warn(
            f'vehicle still moving after {context}; '
            'sending zero-velocity safety pulse')
        self._emergency_stop(
            final=context == 'final entrance leg', emergency=True)
        return self._wait_until_stopped(float(
            self.get_parameter('stop_wait_timeout').value))

    def _wait_until_stopped(self, timeout: float) -> bool:
        started = time.monotonic()
        deadline = time.monotonic() + timeout
        stable_count = 0
        required = max(1, int(self.get_parameter('stop_confirm_count').value))
        next_log = time.monotonic()
        while time.monotonic() < deadline:
            if not self._runtime_ok():
                return False
            if self._vehicle_stopped():
                stable_count += 1
                if stable_count >= required:
                    with self.data_lock:
                        odom = self.odom_msg
                    if odom is not None:
                        twist = odom.twist.twist
                        self.last_stop_linear_speed = math.hypot(
                            twist.linear.x, twist.linear.y)
                        self.last_stop_angular_speed = abs(twist.angular.z)
                    self.last_stop_elapsed = time.monotonic() - started
                    return True
            else:
                stable_count = 0
            if time.monotonic() >= next_log:
                with self.data_lock:
                    odom = self.odom_msg
                if odom is not None:
                    twist = odom.twist.twist
                    self._log_info(
                        'waiting for stop: linear_speed='
                        f'{math.hypot(twist.linear.x, twist.linear.y):.4f}m/s '
                        f'angular_speed={abs(twist.angular.z):.4f}rad/s '
                        f'confirm={stable_count}/{required}')
                next_log = time.monotonic() + 1.0
            if not self._interruptible_sleep(0.1, stop_on_cancel=False):
                return False
        return False

    def _confirm_command_zero(self, label: str, timeout: float = 2.0) -> bool:
        """Publish a zero handoff and require three fresh paired MCU samples."""
        with self.data_lock:
            self.bench_consecutive_zero_drive_samples = 0
            self.bench_consecutive_zero_wheel_samples = 0
        self._emergency_stop(final=False, emergency=False)
        deadline = time.monotonic() + timeout
        drive = float('inf')
        wheel = 999
        drive_samples = 0
        wheel_samples = 0
        while time.monotonic() < deadline and not self._abort_requested():
            now = time.monotonic()
            with self.data_lock:
                drive = self.last_lidar_drive
                wheel = self.last_lidar_wheel
                drive_samples = self.bench_consecutive_zero_drive_samples
                wheel_samples = self.bench_consecutive_zero_wheel_samples
                fresh = (
                    now - self.lidar_drive_received_at < 1.0
                    and now - self.lidar_wheel_received_at < 1.0)
            if (fresh and abs(drive) < 1.0e-6 and wheel == 0
                    and drive_samples >= 3 and wheel_samples >= 3):
                self._log_info(
                    f'[T-PARK][{label}] result=PASS '
                    f'drive={drive:+.1f} wheel={wheel:+d} '
                    f'drive_zero_samples={drive_samples} '
                    f'wheel_zero_samples={wheel_samples}')
                return True
            if not self._interruptible_sleep(0.05, stop_on_cancel=False):
                return False
        self._log_error(
            f'[T-PARK][{label}] result=FAIL '
            f'drive={drive:+.1f} wheel={wheel:+d} '
            f'drive_zero_samples={drive_samples} '
            f'wheel_zero_samples={wheel_samples}')
        return False

    def _publish_plan(self, candidate: PlanCandidate) -> None:
        if not self._runtime_ok():
            return
        candidate.path.header.stamp = self.get_clock().now().to_msg()
        self._safe_publish(self.path_publisher, candidate.path)
        self._log_info(
            f'[T-PARK] path generated: slot={candidate.slot.name} '
            f'poses={len(candidate.path.poses)}')
        forward = Path()
        reverse = Path()
        forward.header = candidate.path.header
        reverse.header = candidate.path.header
        for index, direction in enumerate(candidate.metrics.directions):
            target = forward if direction >= 0 else reverse
            target.poses.extend([
                candidate.path.poses[index], candidate.path.poses[index + 1]])
        self._safe_publish(self.forward_publisher, forward)
        self._safe_publish(self.reverse_publisher, reverse)
        self._safe_publish(self.marker_publisher, self._make_markers(candidate))

    def _publish_forward_exit_plan(
            self, candidate: ForwardExitCandidate) -> None:
        if not self._runtime_ok():
            return
        candidate.path.header.stamp = self.get_clock().now().to_msg()
        self._safe_publish(self.exit_path_publisher, candidate.path)
        self._safe_publish(self.forward_exit_publisher, candidate.path)
        markers = MarkerArray()
        colors = {
            'parking_stop': ColorRGBA(r=0.2, g=0.8, b=0.2, a=0.9),
            'slot_forward_clear': ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.9),
            'right_turn_entry': ColorRGBA(r=1.0, g=0.7, b=0.0, a=0.9),
            'lane_merge': ColorRGBA(r=1.0, g=0.25, b=0.25, a=0.9),
            'course_entrance': ColorRGBA(r=0.8, g=0.2, b=1.0, a=0.9),
        }
        stamp = self.get_clock().now().to_msg()
        for marker_id, (name, pose) in enumerate(candidate.named_poses.items()):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'forward_exit_poses'
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose.pose
            marker.scale.x = 0.38
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            marker.color = colors[name]
            markers.markers.append(marker)
        self._safe_publish(self.marker_publisher, markers)

    def _make_markers(self, candidate: PlanCandidate) -> MarkerArray:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0
        colors = {
            'approach': ColorRGBA(r=0.1, g=0.5, b=1.0, a=0.9),
            'staging': ColorRGBA(r=0.1, g=0.5, b=1.0, a=0.9),
            'setup': ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9),
            'entry': ColorRGBA(r=0.8, g=0.2, b=1.0, a=0.9),
            'final': ColorRGBA(r=0.1, g=1.0, b=0.2, a=0.9),
        }
        for name, pose in candidate.named_poses.items():
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'parking_poses'
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose.pose
            marker.scale.x = 0.45
            marker.scale.y = 0.09
            marker.scale.z = 0.09
            marker.color = colors[name]
            marker.text = name
            markers.markers.append(marker)

        final = candidate.named_poses['final']
        footprint = Marker()
        footprint.header.frame_id = 'map'
        footprint.header.stamp = stamp
        footprint.ns = 'final_footprint'
        footprint.id = marker_id
        footprint.type = Marker.LINE_STRIP
        footprint.action = Marker.ADD
        footprint.scale.x = 0.035
        footprint.color = ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0)
        corners = self._footprint_samples(
            final,
            float(self.get_parameter('vehicle_length').value),
            float(self.get_parameter('vehicle_width').value),
            edge_only=True)
        # Explicit ordered rectangle for an unambiguous final footprint marker.
        yaw = yaw_from_quaternion(final.pose.orientation)
        half_l = 0.5 * float(self.get_parameter('vehicle_length').value)
        half_w = 0.5 * float(self.get_parameter('vehicle_width').value)
        center_x = float(self.get_parameter(
            'vehicle_center_x_offset').value)
        ordered = [
            (center_x + half_l, half_w),
            (center_x + half_l, -half_w),
            (center_x - half_l, -half_w),
            (center_x - half_l, half_w),
            (center_x + half_l, half_w),
        ]
        del corners
        for local_x, local_y in ordered:
            point = Point()
            point.x = (
                final.pose.position.x + math.cos(yaw) * local_x
                - math.sin(yaw) * local_y)
            point.y = (
                final.pose.position.y + math.sin(yaw) * local_x
                + math.cos(yaw) * local_y)
            footprint.points.append(point)
        markers.markers.append(footprint)
        return markers

    def _cancel_active_goals(self) -> None:
        if not self._context_ok():
            return
        for goal_handle in (self.active_plan_goal, self.active_follow_goal):
            if goal_handle is not None:
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass

    def _set_emergency_stop_request(self, active: bool) -> None:
        if self.emergency_stop_request_publisher is None:
            return
        msg = Bool()
        msg.data = active
        self._safe_publish(self.emergency_stop_request_publisher, msg)

    def _emergency_stop(
            self, final: bool = False, emergency: bool = False) -> None:
        if not self._runtime_ok():
            return
        if self.nav_stop_publisher is None:
            return
        if emergency:
            self._set_emergency_stop_request(True)
        zero = Twist()
        for _ in range(5):
            if not self._safe_publish(self.nav_stop_publisher, zero):
                return
            if not self._interruptible_sleep(0.05, stop_on_cancel=False):
                return
        if final:
            self._log_info('final zero /cmd_vel_nav published')

    def _fail(self, reason: str) -> None:
        self._cancel_active_goals()
        if self.execute_path:
            self._emergency_stop()
        self._publish_status('FAILED', reason)

    def _cancelled(self) -> None:
        self._cancel_active_goals()
        self._emergency_stop()
        self._publish_status('CANCELLED')


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = AutoTParking()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        worker_joined = node.stop(join_timeout=5.0)
        if not worker_joined:
            print('t_parking_worker did not stop before teardown')
        executor.remove_node(node)
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
