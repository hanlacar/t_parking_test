#!/usr/bin/env python3
"""Plan and execute a Nav2-only T parking and entrance-return cycle."""

from dataclasses import dataclass
import math
import threading
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathThroughPoses, FollowPath
from nav2_msgs.msg import Costmap
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
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray


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
        self.target_slot = str(self.get_parameter('target_slot').value)
        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.execute_path = bool(self.get_parameter('execute').value)
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
        # A zero-only safety output for cancellation, failure, and settling
        # after Nav2 has completed a FollowPath goal.  Nav2's Jazzy bringup
        # remaps the controller output to /cmd_vel_nav, so clear both the
        # upstream smoother input and the final Gazebo bridge input.  Clearing
        # only /cmd_vel is temporary because the velocity smoother continues
        # republishing its last non-zero sample until its timeout expires.
        self.nav_stop_publisher = self.create_publisher(
            Twist, '/cmd_vel_nav', 10)
        self.stop_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.map_msg: Optional[OccupancyGrid] = None
        self.global_costmap: Optional[Costmap] = None
        self.local_costmap: Optional[Costmap] = None
        self.odom_msg: Optional[Odometry] = None
        self.scan_received_at = 0.0
        self.rear_scan_msg: Optional[LaserScan] = None
        self.rear_scan_received_at = 0.0
        self.cmd_vel_nonzero_seen = False
        self.cmd_vel_direction: Optional[str] = None
        self.last_cmd_vel = Twist()
        self.minimum_exit_linear_x = float('inf')
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
            Costmap, '/local_costmap/costmap_raw',
            self._local_costmap_callback, transient_qos,
            callback_group=self.callback_group)
        self.create_subscription(
            LaserScan, '/scan', self._scan_callback, qos_profile_sensor_data,
            callback_group=self.callback_group)
        self.create_subscription(
            LaserScan, '/scan_rear', self._rear_scan_callback,
            qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos_profile_sensor_data,
            callback_group=self.callback_group)
        self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, 10,
            callback_group=self.callback_group)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)
        self.plan_client = ActionClient(
            self, ComputePathThroughPoses, '/compute_path_through_poses',
            callback_group=self.callback_group)
        self.follow_client = ActionClient(
            self, FollowPath, '/follow_path',
            callback_group=self.callback_group)
        self.slam_pause_client = self.create_client(
            Pause, '/slam_toolbox/pause_new_measurements',
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
            for name in ('planner_server', 'controller_server')
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
            f'auto_start={str(self.auto_start).lower()}\n'
            f'execute={str(self.execute_path).lower()}\n'
            f'target_slot={self.target_slot}')
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
            'target_slot': 'auto', 'auto_start': False, 'execute': False,
            'return_to_entrance': True,
            'stop_when_all_wheels_inside': True,
            'exit_mode': 'forward_right',
            'exit_planner_id': 'ForwardExit',
            'require_forward_only_exit': True,
            'require_right_turn_exit': True,
            'max_reverse_distance_during_exit': 0.01,
            'road_center_y': 0.0, 'road_min_y': -1.75,
            'road_max_y': 1.75, 'approach_lead': 0.50,
            # Must stay at or above vehicle_length: pulling up less than one
            # car length past the slot forces a reverse arc too sharp to clear
            # the slot's side wall.
            'setup_offset_candidates': [0.90, 1.15, 1.40, 1.65],
            'wall_clearance': 0.08,
            'entry_depth_candidates': [0.10, 0.20, 0.30],
            'vehicle_length': 0.90, 'vehicle_width': 0.45,
            'footprint_clearance': 0.06,
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
            'final_pose_overshoot': 0.10,
            'wheel_frames': [
                'front_left_wheel_link', 'front_right_wheel_link',
                'rear_left_wheel_link', 'rear_right_wheel_link'],
            'wheel_radius': 0.095,
            'wheel_base': 0.58,
            'wheel_track': 0.38,
            'wheel_inside_margin': 0.01,
            'wheel_inside_confirm_count': 5,
            'wheel_check_period': 0.10,
            'stop_confirm_count': 3,
            'minimum_turning_radius': 0.946,
            'curvature_tolerance_factor': 1.20,
            'minimum_reverse_length': 0.30,
            'maximum_cusps': 2,
            'occupied_threshold': 50,
            'system_wait_timeout': 120.0, 'map_wait_timeout': 120.0,
            'scan_timeout': 2.0,
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
            'entrance_pose_position_stability': 0.03,
            'entrance_pose_yaw_stability': 0.05,
            'entrance_staging_distance': 1.0,
            'entrance_remap_timeout': 15.0,
            'entrance_return_position_tolerance': 0.10,
            'entrance_return_yaw_tolerance': 0.12,
            'maximum_exit_cusps': 4,
            'forward_clear_distance_candidates': [0.45, 0.55, 0.65, 0.75],
            'right_turn_lead_candidates': [0.15, 0.25, 0.35],
            'lane_merge_distance_candidates': [0.80, 1.00, 1.20],
            'final_position_tolerance': 0.08,
            'final_yaw_tolerance': 0.12,
            # Upper bound only: the strict sustained-odometry stop check
            # normally completes quickly, while a loaded simulator still has
            # enough time to settle safely at a direction cusp.
            'stop_wait_timeout': 120.0,
            'planner_id': 'GridBased',
            'controller_id': 'ParkingFollowPath',
            'goal_checker_id': 'parking_goal_checker',
            'progress_checker_id': 'progress_checker',
            'freeze_slam_during_execution': True,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        slot_defaults = {
            'slot_1': (5.75, 4.25, -math.pi / 2, 5.0, 6.5, 1.75, 6.75),
            'slot_2': (7.25, 4.25, -math.pi / 2, 6.5, 8.0, 1.75, 6.75),
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
        return self._safe_log('warn', message)

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

    def _local_costmap_callback(self, msg: Costmap) -> None:
        with self.data_lock:
            self.local_costmap = msg

    def _scan_callback(self, _msg: LaserScan) -> None:
        self.scan_received_at = time.monotonic()

    def _rear_scan_callback(self, msg: LaserScan) -> None:
        with self.data_lock:
            self.rear_scan_msg = msg
            self.rear_scan_received_at = time.monotonic()

    def _odom_callback(self, msg: Odometry) -> None:
        with self.data_lock:
            self.odom_msg = msg

    def _cmd_vel_callback(self, msg: Twist) -> None:
        if not self._runtime_ok():
            return
        with self.data_lock:
            self.last_cmd_vel = msg
            if self.motion_phase == 'forward_exit':
                self.minimum_exit_linear_x = min(
                    self.minimum_exit_linear_x, float(msg.linear.x))
        moving = (
            abs(msg.linear.x) > 1.0e-4
            or abs(msg.linear.y) > 1.0e-4
            or abs(msg.angular.z) > 1.0e-4)
        if not moving:
            return
        if not self.cmd_vel_nonzero_seen:
            self.cmd_vel_nonzero_seen = True
            self._log_info(
                'first non-zero /cmd_vel received: '
                f'linear.x={msg.linear.x:.3f}m/s '
                f'angular.z={msg.angular.z:.3f}rad/s')
        direction = 'reverse' if msg.linear.x < 0.0 else 'forward'
        if direction != self.cmd_vel_direction:
            self.cmd_vel_direction = direction
            self._log_info(f'/cmd_vel motion state: {direction}')

    def _auto_start_once(self) -> None:
        if not self.auto_start or not self._runtime_ok():
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
            self.execute_path = bool(self.get_parameter('execute').value)
            self.cancel_requested.clear()
            self.cmd_vel_nonzero_seen = False
            self.cmd_vel_direction = None
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

            self._publish_status('WAIT_MAP')
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
            candidates = self._plan_all_candidates(available_slots)
            if not candidates:
                if self._abort_requested():
                    self._cancelled()
                    return
                self._fail('all setup/depth candidates were rejected')
                return
            # Fewest cusps first, then the smoothest curve.  Preferring the
            # largest setup_offset instead was tried and is wrong for this
            # bay: measured max_curvature rises with the offset (0.90 -> 0.542,
            # 1.15 -> 0.605, 1.40 -> 1.176, 1.65 -> 1.181 1/m), because pulling
            # further past the slot forces a sharper swing back to line up with
            # it.  The 1.65 path validated but sat at 93% of the curvature
            # limit, and RegulatedPurePursuitController could not track it --
            # it drifted onto the slot's west wall and stopped on its own
            # collision check.  The real fix for turning in too early is the
            # raised setup_offset_candidates floor (>= one vehicle length), not
            # the selection order; with every candidate now at or above that
            # floor, the gentlest arc is also the one that gets driven.
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

            self._publish_status('PLAN_VALID')

            if not self.execute_path:
                parking_stop_pose = self._expected_wheels_inside_stop_pose(
                    candidate.slot)
                if parking_stop_pose is None:
                    self._fail('could not calculate the expected wheel-inside stop pose')
                    return
                self._publish_status('BUILD_FORWARD_EXIT_POSES')
                self._publish_status('PLANNING_FORWARD_EXIT')
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
            if not self._execute(
                    candidate, bool(self.get_parameter(
                        'stop_when_all_wheels_inside').value)):
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail('FollowPath failed')
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
                self._emergency_stop()
                if not self._wait_until_stopped(float(
                        self.get_parameter('stop_wait_timeout').value)):
                    self._fail(
                        'vehicle did not stop after parking FollowPath cancel')
                    return
            self._log_info(
                'parking stop confirmed; planning exit without added delay')

            parking_stop_pose = self._current_map_pose_stamped()
            if parking_stop_pose is None:
                self._fail('could not capture map->base_footprint parking stop pose')
                return
            self._log_pose('parking_stop_pose', parking_stop_pose)
            if not bool(self.get_parameter('return_to_entrance').value):
                self._publish_status('SUCCESS', 'return_to_entrance=false')
                return
            if str(self.get_parameter('exit_mode').value) != 'forward_right':
                self._fail('unsupported exit_mode; only forward_right is safe')
                return

            self._publish_status('BUILD_FORWARD_EXIT_POSES')
            self._publish_status('PLANNING_FORWARD_EXIT')
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
        lifecycle = {
            name: self._poll_lifecycle_state(name, client)
            for name, client in self.lifecycle_clients.items()
        }
        return {
            'robot_spawned': robot_spawned,
            'front_scan': now - self.scan_received_at < scan_timeout,
            'rear_scan': now - rear_received_at < scan_timeout,
            'odom': robot_spawned,
            'map': map_ready,
            'map_to_base_tf': bool(self.tf_buffer.can_transform(
                'map', 'base_footprint', Time(),
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
        deadline = time.monotonic() + float(self.get_parameter(
            'entrance_pose_capture_timeout').value)
        position_limit = float(self.get_parameter(
            'entrance_pose_position_stability').value)
        yaw_limit = float(self.get_parameter(
            'entrance_pose_yaw_stability').value)
        samples: List[Tuple[float, float, float]] = []
        while time.monotonic() < deadline and not self._abort_requested():
            pose = self._current_map_pose()
            if pose is not None:
                samples.append(pose)
                samples = samples[-required:]
                if len(samples) == required:
                    average_x = sum(item[0] for item in samples) / required
                    average_y = sum(item[1] for item in samples) / required
                    average_yaw = math.atan2(
                        sum(math.sin(item[2]) for item in samples),
                        sum(math.cos(item[2]) for item in samples))
                    maximum_position_delta = max(
                        math.hypot(
                            item[0] - average_x, item[1] - average_y)
                        for item in samples)
                    maximum_yaw_delta = max(
                        abs(normalize_angle(item[2] - average_yaw))
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
                            f'samples={required}')
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
                'geometry relative to map->base_footprint')
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

    def _wheel_centers_odom(
            self) -> Optional[Dict[str, Tuple[float, float]]]:
        positions = self._wheel_centers_map()
        if positions is None:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'map', Time(), timeout=Duration(seconds=0.05))
        except tf2_ros.TransformException:
            return None
        yaw = yaw_from_quaternion(transform.transform.rotation)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return {
            label: (
                transform.transform.translation.x + cosine * point[0]
                - sine * point[1],
                transform.transform.translation.y + sine * point[0]
                + cosine * point[1])
            for label, point in positions.items()
        }

    def _parking_target_depth(
            self, slot: Slot, extra_depth: float = 0.0) -> Tuple[float, float]:
        """Return the deepest safe parking centre (odom frame) for a slot.

        The configured slot.odom_x/odom_y is the geometric centre of the
        whole bay rectangle, which spans metres from the entrance to the
        back curb.  Parking there leaves the vehicle far short of the
        curb.  Instead, place the centre so the rear bumper stops
        rear_clearance short of the far (curb) edge -- but never less than
        curb_inflation_radius + wheel_radius, regardless of how
        rear_clearance is configured.  That floor is the real, physical
        stay-outside-the-inflation-gradient distance (see curb_inflation_radius
        above); a rear_clearance smaller than it would aim the stop at or
        past the curb's cost gradient, which is what "too deep" means in
        practice, so it is clamped here rather than trusted as-is.

        ``extra_depth`` pushes the centre further still, past the safe
        stopping point.  It is used only for the FollowPath waypoint
        (never for the wheel-inside check) so the literal path endpoint
        stays deeper than where the vehicle is actually meant to stop.
        Without it, the FollowPath goal checker (0.05m tolerance) can
        declare the goal reached at almost the same instant the
        wheel-inside monitor starts counting confirmations, and Nav2 can
        win that race before the monitor reaches wheel_inside_confirm_count
        and cancels -- which this node treats as a failure, since it means
        the stop was not verified by the wheel check.
        """
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        half_length = 0.5 * float(self.get_parameter('vehicle_length').value)
        minimum_clearance = (
            float(self.get_parameter('curb_inflation_radius').value)
            + float(self.get_parameter('wheel_radius').value))
        configured_clearance = float(self.get_parameter('rear_clearance').value)
        clearance = max(configured_clearance, minimum_clearance) - extra_depth
        target_x, target_y = slot.odom_x, slot.odom_y
        if abs(forward_x) >= abs(forward_y):
            far_edge = slot.min_x if forward_x > 0.0 else slot.max_x
            target_x = far_edge + forward_x * (half_length + clearance)
        else:
            far_edge = slot.min_y if forward_y > 0.0 else slot.max_y
            target_y = far_edge + forward_y * (half_length + clearance)
        return target_x, target_y

    def _update_wheels_inside(self, slot: Slot) -> bool:
        positions = self._wheel_centers_odom()
        if positions is None:
            self.wheel_inside_confirm_count = 0
            return False
        inset = (
            float(self.get_parameter('wheel_radius').value)
            + float(self.get_parameter('wheel_inside_margin').value))
        axle_offset = 0.5 * float(self.get_parameter('wheel_base').value)
        target_x, target_y = self._parking_target_depth(slot)
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        min_x, max_x = slot.min_x + inset, slot.max_x - inset
        min_y, max_y = slot.min_y + inset, slot.max_y - inset
        # The slot's own far edge is the physical curb, metres away from
        # the entrance, so bounding wheels against the full slot box means
        # "just past the entrance" already reads as fully parked.  Anchor
        # the entrance-side bound to the actual parking target instead, so
        # this only accepts the vehicle once it has reached that depth.
        # The far (curb) side keeps the original slot-edge bound as a
        # collision safety check.
        if abs(forward_x) >= abs(forward_y):
            if forward_x > 0.0:
                max_x = target_x + axle_offset + inset
            else:
                min_x = target_x - axle_offset - inset
        else:
            if forward_y > 0.0:
                max_y = target_y + axle_offset + inset
            else:
                min_y = target_y - axle_offset - inset
        labels = ('front_left', 'front_right', 'rear_left', 'rear_right')
        states = tuple(
            min_x <= positions[label][0] <= max_x
            and min_y <= positions[label][1] <= max_y
            for label in labels)
        if all(states):
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
        while time.monotonic() < deadline and not self._abort_requested():
            availability = {
                slot.name: self._slot_is_available(slot) for slot in self.slots}
            available = [
                slot for slot in self.slots if availability[slot.name]]
            if available:
                lines = ['[SLOT CHECK]']
                lines.extend(
                    f'{slot.name}: '
                    f'{"FREE" if availability[slot.name] else "OCCUPIED"}'
                    for slot in self.slots)
                self._log_info('\n'.join(lines))
                return available
            # From the protected initial pose, the low entrance curb occludes
            # the deep half of both slots from the horizontal lidar beam.  An
            # executing run may therefore make one common, forward Nav2
            # observation pass before slot selection.  Plan-only mode never
            # enters this branch and remains motionless.
            if self.execute_path and not observation_attempted:
                observation_attempted = True
                self._publish_status('MAP_PARKING_BAY')
                if not self._observe_parking_bay():
                    return []
                if not self._interruptible_sleep(float(self.get_parameter(
                        'map_observation_settle_time').value)):
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

    def _slot_is_available(self, slot: Slot) -> bool:
        final_pose = self._odom_pose_to_map(slot.odom_x, slot.odom_y, slot.yaw)
        if final_pose is None:
            return False
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        clearance = float(self.get_parameter('footprint_clearance').value)
        # Check the first complete vehicle footprint inside the bay as well as
        # the final parking footprint.  An entrance-side obstruction can make
        # a slot unusable even though the final pose itself remains clear.
        entrance_pose = self._odom_pose_to_map(
            slot.odom_x,
            slot.min_y + 0.5 * length + clearance,
            slot.yaw)
        if entrance_pose is None:
            return False
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
            return False
        # Inflation costs describe whether the *robot centre* can occupy a
        # cell.  Applying them again at every footprint sample double-counts
        # the footprint and incorrectly rejects the otherwise clear 1.5 m
        # slot.  Check the relevant centre poses against inflation, then use
        # raw SLAM occupancy over the complete footprint rectangles.
        for pose in (entrance_pose, final_pose, rear):
            cost_value = self._cost_value(
                costmap, pose.pose.position.x, pose.pose.position.y)
            if cost_value is None or cost_value == 255 or cost_value >= 253:
                return False
        occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        for x, y in samples:
            map_value = self._occupancy_value(map_msg, x, y)
            if map_value is None or map_value < 0 or map_value >= occupied_threshold:
                return False
        return self._final_footprint_inside_slot(slot, final_pose)

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
        approach_lead = float(self.get_parameter('approach_lead').value)
        overshoot = float(self.get_parameter('final_pose_overshoot').value)
        target_x, target_y = self._parking_target_depth(slot, overshoot)
        odom_poses = {
            'approach': (slot.min_x - approach_lead, road_y, 0.0),
            # Offset is measured beyond the far edge of the selected slot.
            'setup': (slot.max_x + setup_offset, road_y, 0.0),
            'entry': (slot.odom_x, slot.min_y + entry_depth, slot.yaw),
            'final': (target_x, target_y, slot.yaw),
        }
        result = {}
        for name, values in odom_poses.items():
            pose = self._odom_pose_to_map(*values)
            if pose is None:
                return None
            result[name] = pose
        return result

    def _plan_all_candidates(self, slots: Sequence[Slot]) -> List[PlanCandidate]:
        results = []
        offsets = [float(value) for value in self.get_parameter(
            'setup_offset_candidates').value]
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
                        continue
                    path = self._request_plan([
                        named['approach'], named['setup'],
                        named['entry'], named['final']])
                    if path is None:
                        continue
                    metrics = self._analyze_path(path, named['final'])
                    minimum_reverse = float(
                        self.get_parameter('minimum_reverse_length').value)
                    reverse_after_setup = self._reverse_length_after_pose(
                        path, metrics, named['setup'])
                    if reverse_after_setup < minimum_reverse:
                        self._log_warn(
                            f'reject {slot.name} offset={setup_offset:.2f} '
                            f'depth={entry_depth:.2f}: reverse after setup '
                            f'{reverse_after_setup:.3f}m < {minimum_reverse:.3f}m')
                        continue
                    valid, reason = self._validate_path(
                        slot, named, path, metrics)
                    if not valid:
                        self._log_warn(
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

    def _request_plan(
            self, goals: Sequence[PoseStamped], planner_id: Optional[str] = None,
            start_pose: Optional[PoseStamped] = None) -> Optional[Path]:
        """Request a Nav2 path, using the live TF pose unless a test start is set.

        Real exit execution deliberately leaves ``use_start`` false, so Nav2
        starts from the map->base_footprint pose captured after the parking
        FollowPath cancellation.  Plan-only mode supplies its expected
        wheel-inside pose because the robot correctly remains at the course
        entrance and no live TF transform is changed in that mode.
        """
        if self._abort_requested():
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
            return None
        send_future = self.plan_client.send_goal_async(goal)
        goal_handle = self._wait_future(send_future, 5.0)
        if goal_handle is None or not goal_handle.accepted:
            self._log_warn('ComputePathThroughPoses goal not accepted')
            return None
        self.active_plan_goal = goal_handle
        result_future = goal_handle.get_result_async()
        wrapped = self._wait_future(
            result_future, float(self.get_parameter('action_timeout').value))
        self.active_plan_goal = None
        if wrapped is None:
            self._log_warn('ComputePathThroughPoses result wait failed/timed out')
            return None
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            self._log_warn(
                f'ComputePathThroughPoses non-success status={wrapped.status}')
            return None
        result = wrapped.result
        if result.error_code != ComputePathThroughPoses.Result.NONE:
            self._log_warn(
                f'ComputePathThroughPoses error={result.error_code}: '
                f'{result.error_msg}')
            return None
        if not result.path.poses:
            self._log_warn(
                'ComputePathThroughPoses succeeded but returned an empty path')
            return None
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
        """Construct the first all-wheel-inside pose for execute=false checks.

        Mirrors the target-anchored bounds in _update_wheels_inside: the
        wheel check now trips once the vehicle is within wheel_inset of the
        deep parking target, not once it has merely crossed the entrance.
        """
        yaw = slot.yaw
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        wheel_inset = (
            float(self.get_parameter('wheel_radius').value)
            + float(self.get_parameter('wheel_inside_margin').value))
        target_x, target_y = self._parking_target_depth(slot)
        centre_x, centre_y = target_x, target_y
        if abs(forward_x) >= abs(forward_y):
            centre_x = (
                target_x + wheel_inset if forward_x > 0.0
                else target_x - wheel_inset)
        else:
            centre_y = (
                target_y + wheel_inset if forward_y > 0.0
                else target_y - wheel_inset)
        pose = self._odom_pose_to_map(centre_x, centre_y, yaw)
        if pose is not None:
            self._log_pose('expected_wheels_inside_stop_pose', pose)
        return pose

    def _rear_wheels_clear_slot(self, slot: Slot, pose: PoseStamped) -> bool:
        """Require both rear wheels to have crossed the open slot boundary."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'map', Time(), timeout=Duration(seconds=0.5))
        except tf2_ros.TransformException:
            return False
        transform_yaw = yaw_from_quaternion(transform.transform.rotation)
        cosine = math.cos(transform_yaw)
        sine = math.sin(transform_yaw)
        dx = pose.pose.position.x
        dy = pose.pose.position.y
        centre_x = transform.transform.translation.x + cosine * dx - sine * dy
        centre_y = transform.transform.translation.y + sine * dx + cosine * dy
        yaw = normalize_angle(
            transform_yaw + yaw_from_quaternion(pose.pose.orientation))
        rear_offset = -0.5 * float(self.get_parameter('wheel_base').value)
        half_track = 0.5 * float(self.get_parameter('wheel_track').value)
        wheels = []
        for lateral in (-half_track, half_track):
            wheels.append((
                centre_x + rear_offset * math.cos(yaw)
                - lateral * math.sin(yaw),
                centre_y + rear_offset * math.sin(yaw)
                + lateral * math.cos(yaw)))
        inset = (float(self.get_parameter('wheel_radius').value)
                 + float(self.get_parameter('wheel_inside_margin').value))
        return all(
            not (slot.min_x + inset <= x <= slot.max_x - inset
                 and slot.min_y + inset <= y <= slot.max_y - inset)
            for x, y in wheels)

    def _minimum_forward_clear(
            self, slot: Slot, parking_stop_pose: PoseStamped
            ) -> Optional[float]:
        """Distance the rear axle must travel forward to clear the slot.

        forward_clear_distance_candidates were sized for a parking stop
        close to the entrance.  Now that the vehicle parks much deeper,
        near the back curb, that fixed distance can fall far short of
        what is actually needed to move the rear wheels past the
        entrance line, rejecting every exit candidate before a path is
        ever planned.  Compute the real distance from the actual parking
        depth instead of assuming a fixed one; the configured candidates
        are then added on top of this as extra safety buffer.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'map', Time(), timeout=Duration(seconds=0.5))
        except tf2_ros.TransformException:
            return None
        transform_yaw = yaw_from_quaternion(transform.transform.rotation)
        cosine = math.cos(transform_yaw)
        sine = math.sin(transform_yaw)
        dx = parking_stop_pose.pose.position.x
        dy = parking_stop_pose.pose.position.y
        centre_x = transform.transform.translation.x + cosine * dx - sine * dy
        centre_y = transform.transform.translation.y + sine * dx + cosine * dy
        yaw = normalize_angle(
            transform_yaw
            + yaw_from_quaternion(parking_stop_pose.pose.orientation))
        rear_offset = -0.5 * float(self.get_parameter('wheel_base').value)
        rear_x = centre_x + rear_offset * math.cos(yaw)
        rear_y = centre_y + rear_offset * math.sin(yaw)
        inset = (float(self.get_parameter('wheel_radius').value)
                 + float(self.get_parameter('wheel_inside_margin').value))
        forward_x = math.cos(slot.yaw)
        forward_y = math.sin(slot.yaw)
        if abs(forward_x) >= abs(forward_y):
            entrance = slot.max_x if forward_x > 0.0 else slot.min_x
            distance = (entrance - rear_x) * forward_x - inset
        else:
            entrance = slot.max_y if forward_y > 0.0 else slot.min_y
            distance = (entrance - rear_y) * forward_y - inset
        return max(0.0, distance)

    def _build_forward_exit_poses(
            self, parking_stop_pose: PoseStamped, forward_clear: float,
            right_turn_lead: float, lane_merge: float
            ) -> Optional[Dict[str, PoseStamped]]:
        if self.course_entrance_pose is None:
            return None
        stop = self._fresh_pose(parking_stop_pose)
        clear = self._offset_pose(stop, forward_clear, 0.0)
        clear.header.stamp = self.get_clock().now().to_msg()
        turn_entry = self._offset_pose(clear, right_turn_lead, 0.0)
        turn_entry.header.stamp = self.get_clock().now().to_msg()

        heading = yaw_from_quaternion(stop.pose.orientation)
        forward = (math.cos(heading), math.sin(heading))
        right = (math.sin(heading), -math.cos(heading))
        # This offset is a 45-degree chamfer standing in for a quarter-circle
        # right turn: forward_offset = right_offset = radius traces the same
        # start/end tangents as an arc of that radius.  If the offset is
        # smaller than the vehicle's own minimum_turning_radius, the chamfer
        # sits *inside* the smallest circle the vehicle can actually drive,
        # so no forward-only arc connects turn_entry to merge on the correct
        # side -- DUBIN then has to loop the long way around to satisfy both
        # the position and the heading, which is exactly the oversized loop
        # this exit path must not have.  Flooring the offset at
        # minimum_turning_radius guarantees every candidate stays a
        # physically reachable single turn.
        minimum_radius = float(
            self.get_parameter('minimum_turning_radius').value)
        turn_offset = max(lane_merge, minimum_radius)
        merge = self._fresh_pose(turn_entry)
        merge.pose.position.x += turn_offset * (forward[0] + right[0])
        merge.pose.position.y += turn_offset * (forward[1] + right[1])
        target = self._fresh_pose(self.course_entrance_pose)
        dx = target.pose.position.x - merge.pose.position.x
        dy = target.pose.position.y - merge.pose.position.y
        if math.hypot(dx, dy) < 0.05:
            return None
        # A forward-only, no-reverse vehicle returning along the same single
        # lane necessarily arrives heading opposite to its original outbound
        # heading; forcing the originally-captured (outbound) yaw here would
        # make every candidate a physically impossible 180-degree flip.
        # Orient the final approach to match the actual direction of travel.
        approach_yaw = math.atan2(dy, dx)
        set_pose_yaw(merge.pose, approach_yaw)
        merge.header.stamp = self.get_clock().now().to_msg()
        set_pose_yaw(target.pose, approach_yaw)
        return {
            'parking_stop': stop,
            'slot_forward_clear': clear,
            'right_turn_entry': turn_entry,
            'lane_merge': merge,
            'course_entrance': target,
        }

    def _plan_forward_right_exit(
            self, parking_stop_pose: PoseStamped, slot: Slot,
            use_current_start: bool) -> Optional[ForwardExitCandidate]:
        planner_id = str(self.get_parameter('exit_planner_id').value)
        minimum_clear = self._minimum_forward_clear(slot, parking_stop_pose)
        if minimum_clear is None:
            self._log_error(
                'could not compute the minimum forward-clear distance '
                '(map->odom transform unavailable)')
            return None
        clear_candidates = [
            minimum_clear + float(value) for value in self.get_parameter(
                'forward_clear_distance_candidates').value]
        self._log_info(
            f'minimum forward-clear to leave the slot={minimum_clear:.3f}m; '
            'candidates=' + ', '.join(f'{value:.3f}' for value in clear_candidates))
        lead_candidates = [float(value) for value in self.get_parameter(
            'right_turn_lead_candidates').value]
        merge_candidates = [float(value) for value in self.get_parameter(
            'lane_merge_distance_candidates').value]
        for forward_clear in clear_candidates:
            for right_turn_lead in lead_candidates:
                for lane_merge in merge_candidates:
                    if self._abort_requested():
                        return None
                    named = self._build_forward_exit_poses(
                        parking_stop_pose, forward_clear, right_turn_lead,
                        lane_merge)
                    if named is None:
                        continue
                    if not self._rear_wheels_clear_slot(
                            slot, named['slot_forward_clear']):
                        self._log_info(
                            f'reject forward exit clear={forward_clear:.2f}: '
                            'rear wheels have not cleared the slot entrance')
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
                        self._log_info(
                            f'reject forward exit clear={forward_clear:.2f} '
                            f'lead={right_turn_lead:.2f} merge={lane_merge:.2f}: '
                            'ComputePathThroughPoses returned no path')
                        continue
                    path = self._dedupe_stationary_poses(path)
                    metrics = self._analyze_path(
                        path, named['course_entrance'])
                    valid, turn_direction, reason = (
                        self._validate_forward_exit_path(path, metrics, named))
                    if not valid:
                        self._log_info(
                            'reject forward exit '
                            f'clear={forward_clear:.2f} lead={right_turn_lead:.2f} '
                            f'merge={lane_merge:.2f}: {reason}')
                        continue
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
            named: Dict[str, PoseStamped]) -> Tuple[bool, str, str]:
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
        turn_direction = self._first_major_turn_direction(
            path, named['slot_forward_clear'])
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
        total = forward = reverse = 0.0
        directions = []
        first_reverse = -1
        for index, (previous, current) in enumerate(
                zip(path.poses, path.poses[1:])):
            dx = current.pose.position.x - previous.pose.position.x
            dy = current.pose.position.y - previous.pose.position.y
            length = math.hypot(dx, dy)
            if length < 1e-6:
                directions.append(0)
                continue
            yaw = yaw_from_quaternion(previous.pose.orientation)
            dot = dx * math.cos(yaw) + dy * math.sin(yaw)
            direction = 1 if dot >= 0.0 else -1
            directions.append(direction)
            total += length
            if direction > 0:
                forward += length
            else:
                reverse += length
                if first_reverse < 0:
                    first_reverse = index + 1
        nonzero = [value for value in directions if value]
        cusps = sum(a != b for a, b in zip(nonzero, nonzero[1:]))

        # Estimate curvature over a physical window instead of adjacent grid
        # samples.  Three nearly coincident Hybrid-A* poses amplify map-grid
        # quantization, and a Reeds-Shepp cusp is a stop/direction change where
        # curvature is undefined.  A 0.15 m window on each side remains well
        # below this vehicle's 0.946 m minimum turning radius while rejecting
        # real over-curvature.
        max_curvature = 0.0
        poses = path.poses
        curvature_window = 0.15
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
        occupied_threshold = int(
            self.get_parameter('occupied_threshold').value)
        for index, pose in enumerate(path.poses):
            center_cost = self._cost_value(
                costmap, pose.pose.position.x, pose.pose.position.y)
            if center_cost is None or center_cost == 255:
                return False, f'path center index {index} is unknown'
            if center_cost >= 253:
                return False, f'path center index {index} is lethal/inscribed'
            for x, y in self._footprint_samples(pose, length, width):
                map_value = self._occupancy_value(map_msg, x, y)
                cost_value = self._cost_value(costmap, x, y)
                if (map_value is None or map_value < 0
                        or cost_value is None or cost_value == 255):
                    return False, f'footprint index {index} reaches unknown'
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
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'map', Time(), timeout=Duration(seconds=0.5))
        except tf2_ros.TransformException:
            return False, 'odom->map unavailable for the wall clearance check'
        tf_yaw = yaw_from_quaternion(transform.transform.rotation)
        cosine = math.cos(tf_yaw)
        sine = math.sin(tf_yaw)
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        minimum_x = slot.min_x + clearance
        maximum_x = slot.max_x - clearance
        for index, pose in enumerate(path.poses):
            for x, y in self._footprint_samples(
                    pose, length, width, edge_only=True):
                ox = transform.transform.translation.x + cosine * x - sine * y
                oy = transform.transform.translation.y + sine * x + cosine * y
                # Only samples level with the bay can touch its side walls;
                # the approach and setup legs run along the road outside it.
                if not slot.min_y <= oy <= slot.max_y:
                    continue
                if not minimum_x <= ox <= maximum_x:
                    return False, (
                        f'path index {index} comes within {clearance:.3f}m of '
                        f'the {slot.name} side wall (odom x={ox:.3f}, allowed '
                        f'{minimum_x:.3f}..{maximum_x:.3f})')
        return True, 'valid'

    def _setup_inside_road(self, pose: PoseStamped) -> bool:
        if self._abort_requested():
            return False
        road_min = float(self.get_parameter('road_min_y').value)
        road_max = float(self.get_parameter('road_max_y').value)
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'map', Time(), timeout=Duration(seconds=0.5))
        except tf2_ros.TransformException:
            return False
        tf_yaw = yaw_from_quaternion(transform.transform.rotation)
        cosine = math.cos(tf_yaw)
        sine = math.sin(tf_yaw)
        for x, y in self._footprint_samples(pose, length, width, edge_only=True):
            ox = transform.transform.translation.x + cosine * x - sine * y
            oy = transform.transform.translation.y + sine * x + cosine * y
            del ox
            if not road_min < oy < road_max:
                return False
        return True

    def _final_footprint_inside_slot(
            self, slot: Slot, pose: PoseStamped) -> bool:
        if self._abort_requested():
            return False
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'map', Time(), timeout=Duration(seconds=0.5))
        except tf2_ros.TransformException:
            return False
        tf_yaw = yaw_from_quaternion(transform.transform.rotation)
        cosine = math.cos(tf_yaw)
        sine = math.sin(tf_yaw)
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        for x, y in self._footprint_samples(pose, length, width, edge_only=True):
            ox = transform.transform.translation.x + cosine * x - sine * y
            oy = transform.transform.translation.y + sine * x + cosine * y
            if not (slot.min_x < ox < slot.max_x and slot.min_y < oy < slot.max_y):
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
        nx = max(2, math.ceil(length / resolution))
        ny = max(2, math.ceil(width / resolution))
        local_points = []
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                if edge_only and ix not in (0, nx) and iy not in (0, ny):
                    continue
                local_points.append((
                    -half_length + length * ix / nx,
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
        self.active_motion_path = candidate.path
        self.active_motion_metrics = candidate.metrics
        self.active_motion_final = candidate.path.poses[-1]
        self._log_info(
            'FollowPath forward exit: one DUBIN path, no direction cusp')
        return self._execute_path_action(
            candidate.path, 1, 1, 1, monitor_slot=None, forward_exit=True)

    def _execute_segmented_path(
            self, path: Path, metrics: PathMetrics,
            monitor_slot: Optional[Slot]) -> bool:
        directions = metrics.directions
        if not directions:
            self._log_error('validated path has no motion segments')
            return False
        self.active_motion_path = path
        self.active_motion_metrics = metrics
        self.active_motion_final = path.poses[-1]
        runs = []
        run_start = 0
        run_direction = directions[0] if directions[0] else 1
        for index, direction in enumerate(directions):
            effective = direction if direction else run_direction
            if effective == run_direction:
                continue
            runs.append((run_start, index, run_direction))
            run_start = index
            run_direction = effective
        runs.append((run_start, len(directions), run_direction))

        for run_number, (first, last, direction) in enumerate(runs, start=1):
            segment = Path()
            segment.header = path.header
            segment.poses = path.poses[first:last + 1]
            self._log_info(
                f'FollowPath segment {run_number}/{len(runs)} '
                f'direction={"reverse" if direction < 0 else "forward"} '
                f'poses={len(segment.poses)} path_indices={first}..{last}')
            if not self._execute_path_action(
                    segment, run_number, len(runs), direction,
                    monitor_slot=monitor_slot):
                return False
            if monitor_slot is not None and self.parking_wheels_inside:
                return True
            if run_number < len(runs):
                # End one Nav2 action completely before giving RPP the next
                # direction.  A single path whose forward and reverse runs
                # overlap near a cusp can otherwise make pure pursuit choose
                # alternating velocity signs at the same physical location.
                self._emergency_stop()
                if not self._wait_until_stopped(float(
                        self.get_parameter('stop_wait_timeout').value)):
                    self._log_error(
                        'vehicle did not settle at FollowPath direction cusp')
                    return False
        return True

    def _execute_path_action(
            self, path: Path, run_number: int, run_count: int,
            direction: int, monitor_slot: Optional[Slot] = None,
            forward_exit: bool = False) -> bool:
        if direction < 0:
            healthy, rear_distance, reason = self._rear_scan_state()
            if not healthy:
                self._log_error(
                    f'reverse FollowPath inhibited: rear lidar {reason}')
                self._emergency_stop()
                return False
            self._log_info(
                f'rear safety armed: nearest={rear_distance:.3f}m '
                'threshold='
                f'{float(self.get_parameter("rear_emergency_stop_distance").value):.3f}m')
        if self._abort_requested():
            return False
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = str(self.get_parameter('controller_id').value)
        goal.goal_checker_id = (
            str(self.get_parameter('goal_checker_id').value)
            if run_number == run_count else 'cusp_goal_checker')
        goal.progress_checker_id = str(
            self.get_parameter('progress_checker_id').value)
        self._log_info(
            f'FollowPath request {run_number}/{run_count}: '
            f'controller_id={goal.controller_id} '
            f'goal_checker_id={goal.goal_checker_id} '
            f'progress_checker_id={goal.progress_checker_id}')
        if not self._runtime_ok():
            return False
        send_future = self.follow_client.send_goal_async(
            goal, feedback_callback=self._follow_feedback)
        goal_handle = self._wait_future(send_future, 5.0)
        if goal_handle is None or not goal_handle.accepted:
            self._log_error(
                'FollowPath goal rejected: '
                f'controller_id={goal.controller_id} '
                f'goal_checker_id={goal.goal_checker_id}')
            return False
        self._log_info(
            'FollowPath goal accepted: '
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
                self._log_error(
                    'FollowPath execution timeout; cancelling goal')
                if self._context_ok():
                    goal_handle.cancel_goal_async()
                self._emergency_stop()
                return False
            if direction < 0:
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
                    self._emergency_stop()
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
                    self._emergency_stop()
                    return False
            now = time.monotonic()
            if monitor_slot is not None and now >= next_wheel_check:
                next_wheel_check = now + float(self.get_parameter(
                    'wheel_check_period').value)
                if self._update_wheels_inside(monitor_slot):
                    self.parking_wheels_inside = True
                    self._publish_status('WHEELS_INSIDE')
                    if not self._cancel_parking_follow_path(
                            goal_handle, result_future):
                        return False
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
        if monitor_slot is not None and run_number == run_count:
            self._log_error(
                'parking FollowPath reached its planned endpoint without '
                'five consecutive all-wheel-inside confirmations')
            return False
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            self._log_error(
                f'FollowPath status={wrapped.status} '
                f'error_code={result.error_code} '
                f'error_msg={result.error_msg!r}')
            return False
        self._log_info(
            f'FollowPath segment {run_number}/{run_count} result '
            f'error_code={result.error_code} '
            f'error_msg={result.error_msg!r}')
        return result.error_code == FollowPath.Result.NONE

    def _cancel_parking_follow_path(self, goal_handle, result_future) -> bool:
        self._log_info(
            'all wheels confirmed inside; cancelling active parking FollowPath')
        cancel_future = goal_handle.cancel_goal_async()
        timeout = float(self.get_parameter('parking_cancel_timeout').value)
        response = self._wait_future(cancel_future, timeout)
        if response is None or not response.goals_canceling:
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
        if wrapped.status != GoalStatus.STATUS_CANCELED:
            self._log_error(
                f'parking FollowPath cancel ended with status={wrapped.status}')
            return False
        self._log_info(
            'parking FollowPath result confirmed STATUS_CANCELED')
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
            if (abs(angle) <= sector and math.isfinite(value)
                    and scan.range_min <= value <= scan.range_max):
                distances.append(float(value))
        if not distances:
            # Positive infinity is the standard LaserScan representation for
            # no return inside range_max.  That is a valid clear-space safety
            # result, unlike NaN, a stale scan, or an empty message.
            sector_values = [
                value for index, value in enumerate(scan.ranges)
                if abs(scan.angle_min + index * scan.angle_increment) <= sector]
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

    def _current_map_pose(self) -> Optional[Tuple[float, float, float]]:
        if self._abort_requested():
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', Time(), timeout=Duration(seconds=0.5))
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
        self._emergency_stop(final=context == 'final entrance leg')
        return self._wait_until_stopped(float(
            self.get_parameter('stop_wait_timeout').value))

    def _wait_until_stopped(self, timeout: float) -> bool:
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

    def _publish_plan(self, candidate: PlanCandidate) -> None:
        if not self._runtime_ok():
            return
        candidate.path.header.stamp = self.get_clock().now().to_msg()
        self._safe_publish(self.path_publisher, candidate.path)
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
        ordered = [(half_l, half_w), (half_l, -half_w),
                   (-half_l, -half_w), (-half_l, half_w), (half_l, half_w)]
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

    def _emergency_stop(self, final: bool = False) -> None:
        if not self._runtime_ok():
            return
        zero = Twist()
        for _ in range(5):
            if not self._safe_publish(self.nav_stop_publisher, zero):
                return
            if not self._safe_publish(self.stop_publisher, zero):
                return
            if not self._interruptible_sleep(0.05, stop_on_cancel=False):
                return
        if final:
            self._log_info('final zero /cmd_vel and /cmd_vel_nav published')

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
