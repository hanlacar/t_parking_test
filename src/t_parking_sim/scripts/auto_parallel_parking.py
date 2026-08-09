#!/usr/bin/env python3
"""Plan and execute an automatic parallel-parking entry and forward exit.

This node is deliberately independent of ``auto_t_parking.py``.  The Nav2
plumbing (action wrapping, lifecycle readiness, wheel-inside confirmation,
SLAM freeze, stop confirmation) is copied rather than shared, because every
one of those helpers is bound to node attributes created in the T-parking
node's ``__init__`` and several of them bake in the perpendicular-parking
assumption that the vehicle's heading axis is also the slot's depth axis.
Parallel parking violates exactly that assumption -- here the heading is
parallel to the slot's long axis and "depth" is lateral -- so sharing the
code would mean editing a node that already works.

Frames
------
The vehicle spawns at world (9.75, -0.50, yaw=1.5708) and the odometry
origin is that pose, so odom is the world frame both translated and rotated
by 90 degrees::

    odom_x = world_y + 0.50
    odom_y = 9.75 - world_x        (note the sign flip)
    odom_yaw = world_yaw - 1.5708

All slot parameters are stored in odom (see parallel_parking_auto.yaml) and
converted into ``map`` at plan time, because Nav2 plans in ``map``.

In odom, the slot's depth axis is x (min_x is the mouth, max_x the inner
wall) and its long axis is y, which is also the driving direction: yaw
1.5708 means travelling west in world, i.e. +odom_y, with the slots on the
vehicle's right.
"""

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
    """A parallel slot in the odom frame.

    ``min_x``/``max_x`` bound the depth (mouth to inner wall) and
    ``min_y``/``max_y`` bound the length along the lane.  This is the
    opposite axis assignment from the T bay.
    """

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
class EntryCandidate:
    slot: Slot
    lane_offset: float
    approach_offset: float
    named_poses: Dict[str, PoseStamped]
    path: Path
    metrics: PathMetrics
    reverse_after_approach: float


@dataclass
class ExitCandidate:
    exit_lead: float
    named_poses: Dict[str, PoseStamped]
    path: Path
    metrics: PathMetrics


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


class AutoParallelParking(Node):
    """State machine for parallel parking with a strictly forward exit."""

    # Which slot each obstacle_layout suffix blocks.
    LAYOUT_BLOCKS = {'C': 'slot_1', 'D': 'slot_2'}

    def __init__(self) -> None:
        super().__init__('parallel_parking_auto')
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
            String, '/parallel_parking/status', transient_qos)
        self.path_publisher = self.create_publisher(
            Path, '/parallel_parking/planned_path', transient_qos)
        self.exit_path_publisher = self.create_publisher(
            Path, '/parallel_parking/exit_path', transient_qos)
        self.forward_publisher = self.create_publisher(
            Path, '/parallel_parking/forward_path', transient_qos)
        self.reverse_publisher = self.create_publisher(
            Path, '/parallel_parking/reverse_path', transient_qos)
        self.marker_publisher = self.create_publisher(
            MarkerArray, '/parallel_parking/markers', transient_qos)
        # Zero-only safety outputs.  Nav2's bringup remaps the controller
        # output to /cmd_vel_nav, so both the smoother input and the final
        # bridge input have to be cleared or the smoother keeps republishing
        # its last non-zero sample.
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
        self.obstacle_layout: Optional[str] = None
        self.last_cmd_vel = Twist()
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
        # vehicle_spawn_manager latches this, so a late subscription still
        # receives the layout that was chosen at spawn time.
        self.create_subscription(
            String, '/parking_practice/obstacle_layout',
            self._layout_callback, transient_qos,
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
            # /compute_path_through_poses and /follow_path directly.
            for name in ('planner_server', 'controller_server')
        }
        self.lifecycle_state_lock = threading.Lock()
        self.lifecycle_state: Dict[str, bool] = {
            name: False for name in self.lifecycle_clients}
        self.lifecycle_pending: Dict[str, Optional[float]] = {
            name: None for name in self.lifecycle_clients}

        self.create_service(
            Trigger, '/parallel_parking/start', self._start_service,
            callback_group=self.callback_group)
        self.create_service(
            Trigger, '/parallel_parking/cancel', self._cancel_service,
            callback_group=self.callback_group)

        self.active_plan_goal = None
        self.active_follow_goal = None
        self.parking_wheels_inside = False
        self.wheel_inside_confirm_count = 0
        self.last_wheel_states: Optional[Tuple[bool, bool, bool, bool]] = None
        self.last_wheel_log = 0.0
        self.wheel_position_source: Optional[str] = None
        self.motion_phase = 'idle'
        self.active_motion_path: Optional[Path] = None
        self.active_motion_metrics: Optional[PathMetrics] = None
        self.active_motion_final: Optional[PoseStamped] = None
        self.state = 'WAITING'
        self.last_feedback_log = 0.0
        self._publish_status('WAITING')
        self.readiness_timer = self.create_timer(
            1.0, self._readiness_timer_callback,
            callback_group=self.callback_group)
        if self.auto_start:
            self.create_timer(
                1.0, self._auto_start_once,
                callback_group=self.callback_group)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        defaults = {
            'target_slot': 'auto', 'auto_start': False, 'execute': True,
            'lane_min_x': 9.00, 'lane_max_x': 12.52,
            'staging_via_odom_y': 0.0,
            'staging_hop_distance': 3.0,
            'corner_entry_odom_x': 10.00,
            'corner_exit_odom_y': 3.00,
            'corner_arc_radius': 1.30,
            'transit_controller_id': 'FollowPath',
            'transit_goal_checker_id': 'goal_checker',
            'lane_offset_candidates': [0.80],
            'staging_final_gap': 0.0,
            'approach_offset_candidates': [1.80, 2.20, 2.60],
            'slot_end_clearance': 0.35,
            'entry_angle': 0.55,
            'entry_depth': 0.35,
            'entry_lead': 1.00,
            'entry_loop_length_factor': 2.2,
            'exit_lead_candidates': [1.20, 1.80, 2.50],
            'slot_depth_clearance': 0.35,
            'vehicle_length': 0.90, 'vehicle_width': 0.45,
            'footprint_clearance': 0.06,
            'minimum_reverse_length': 0.30,
            'maximum_cusps': 2,
            'minimum_turning_radius': 0.946,
            'curvature_tolerance_factor': 1.20,
            'occupied_threshold': 50,
            'exit_mode': 'forward',
            'exit_planner_id': 'ForwardExit',
            'require_forward_only_exit': True,
            'max_reverse_distance_during_exit': 0.01,
            'maximum_exit_cusps': 0,
            'exit_position_tolerance': 0.25,
            'exit_yaw_tolerance': 0.25,
            'wheel_frames': [
                'front_left_wheel_link', 'front_right_wheel_link',
                'rear_left_wheel_link', 'rear_right_wheel_link'],
            'wheel_radius': 0.095,
            'wheel_base': 0.58,
            'wheel_track': 0.38,
            'wheel_inside_margin': 0.01,
            'parked_yaw_tolerance': 0.20,
            'wheel_inside_confirm_count': 5,
            'wheel_check_period': 0.10,
            'stop_confirm_count': 3,
            'system_wait_timeout': 120.0,
            'map_wait_timeout': 120.0,
            'layout_wait_timeout': 20.0,
            'scan_timeout': 2.0,
            'rear_emergency_stop_distance': 0.15,
            'rear_emergency_sector_deg': 30.0,
            'map_observation_settle_time': 3.0,
            'action_timeout': 45.0,
            'follow_path_timeout': 180.0,
            'parking_cancel_timeout': 5.0,
            'parking_stop_grace_timeout': 2.0,
            'stop_wait_timeout': 120.0,
            'freeze_slam_during_execution': True,
            'planner_id': 'GridBased',
            'controller_id': 'ParkingFollowPath',
            'goal_checker_id': 'parking_goal_checker',
            'progress_checker_id': 'progress_checker',
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        # Defaults match parallel_parking_auto.yaml; see that file for the
        # world->odom derivation behind every number.
        slot_defaults = {
            'slot_1': (13.26, 7.18, 1.57079632679, 12.52, 14.00, 4.79, 9.57),
            'slot_2': (13.26, 2.32, 1.57079632679, 12.52, 14.00, -0.07, 4.71),
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

    # ------------------------------------------------------------------
    # Lifecycle plumbing
    # ------------------------------------------------------------------

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

    # rclpy caches a logger's severity per *call site*: dispatching every
    # level from one shared `getattr(logger, level)(message)` line makes the
    # second severity used at that line raise "Logger severity cannot be
    # changed between calls", which a blanket except then swallows -- so
    # warnings and errors disappear entirely.  Each severity therefore needs
    # its own physical call site.
    def _log_info(self, message: str) -> bool:
        if not self._runtime_ok():
            return False
        try:
            self.get_logger().info(message)
            return True
        except Exception:
            return False

    def _log_warn(self, message: str) -> bool:
        if not self._runtime_ok():
            return False
        try:
            self.get_logger().warning(message)
            return True
        except Exception:
            return False

    def _log_error(self, message: str) -> bool:
        if not self._runtime_ok():
            return False
        try:
            self.get_logger().error(message)
            return True
        except Exception:
            return False

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

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

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

    def _layout_callback(self, msg: String) -> None:
        layout = msg.data.strip().upper()
        with self.data_lock:
            changed = layout != self.obstacle_layout
            self.obstacle_layout = layout
        if changed:
            self._log_info(f'obstacle layout received: {layout}')

    # ------------------------------------------------------------------
    # Services and worker
    # ------------------------------------------------------------------

    def _auto_start_once(self) -> None:
        if not self.auto_start or not self._runtime_ok():
            return
        self.auto_start = False
        self._start_worker()

    def _start_service(self, _request, response):
        if not self._runtime_ok():
            response.success = False
            response.message = 'automatic parallel-parking node is shutting down'
            return response
        if self._start_worker():
            response.success = True
            response.message = 'automatic parallel parking started'
        else:
            response.success = False
            response.message = 'automatic parallel parking is already running'
        return response

    def _cancel_service(self, _request, response):
        if not self._runtime_ok():
            response.success = False
            response.message = 'automatic parallel-parking node is shutting down'
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
            self.target_slot = str(self.get_parameter('target_slot').value)
            self.execute_path = bool(self.get_parameter('execute').value)
            self.cancel_requested.clear()
            self.parking_wheels_inside = False
            self.wheel_inside_confirm_count = 0
            self.last_wheel_states = None
            self.last_wheel_log = 0.0
            self.motion_phase = 'idle'
            self.active_motion_path = None
            self.active_motion_metrics = None
            self.active_motion_final = None
            self._log_info(
                '[AUTO PARALLEL PARKING PARAMETERS]\n'
                f'auto_start={str(self.auto_start).lower()}\n'
                f'execute={str(self.execute_path).lower()}\n'
                f'target_slot={self.target_slot}\n'
                f'exit_mode={self.get_parameter("exit_mode").value}\n'
                f'exit_planner_id={self.get_parameter("exit_planner_id").value}')
            self._worker_thread = threading.Thread(
                target=self._run_state_machine,
                name='parallel_parking_worker', daemon=False)
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

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _run_state_machine(self) -> None:
        try:
            if self._abort_requested():
                return
            self._publish_status('WAIT_SYSTEM')
            if not self._wait_for_system():
                if not self._abort_requested():
                    self._fail('system readiness timeout')
                return
            if self._wheel_centers_map() is None:
                self._fail('wheel TF and base-footprint geometry are unavailable')
                return

            self._publish_status('WAIT_MAP')
            if not self._wait_for_map():
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail('SLAM map or Nav2 costmaps never became available')
                return

            self._publish_status('SELECT_SLOT')
            slot = self._select_slot()
            if slot is None:
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail('no free parallel slot could be selected')
                return
            self._publish_status('SLOT_SELECTED', slot.name)

            self._publish_status('BUILD_APPROACH')
            if self.execute_path:
                if not self._drive_to_approach(slot):
                    if self._abort_requested():
                        self._cancelled()
                    else:
                        self._fail('could not drive to the parallel approach pose')
                    return
                if not self._interruptible_sleep(float(self.get_parameter(
                        'map_observation_settle_time').value)):
                    self._cancelled()
                    return

            if (self.execute_path
                    and bool(self.get_parameter(
                        'freeze_slam_during_execution').value)):
                # Freeze only after the staging drive: the parking area has to
                # be mapped first, and map->odom must then stay fixed for the
                # whole manoeuvre.
                if not self._toggle_slam_measurements():
                    if self._abort_requested():
                        self._cancelled()
                        return
                    self._fail('could not pause SLAM measurements')
                    return
                self.slam_paused_by_node = True
                self._log_info('SLAM measurements paused; map->odom is fixed')

            self._publish_status('PLAN_ENTRY')
            candidate = self._plan_entry(slot)
            if candidate is None:
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail(
                        'no entry path started with a reverse segment; refusing '
                        'to settle for a forward slide-in')
                return

            self._publish_status('VALIDATE_ENTRY')
            self._publish_entry_plan(candidate)
            metrics = candidate.metrics
            self._log_info(
                f'[ENTRY PLAN] slot={candidate.slot.name} '
                f'lane_offset={candidate.lane_offset:.2f} '
                f'approach_offset={candidate.approach_offset:.2f} '
                f'poses={len(candidate.path.poses)} '
                f'total={metrics.total_length:.3f}m '
                f'forward={metrics.forward_length:.3f}m '
                f'reverse={metrics.reverse_length:.3f}m '
                f'reverse_after_approach={candidate.reverse_after_approach:.3f}m '
                f'cusps={metrics.cusp_count} '
                f'first_reverse_index={metrics.first_reverse_index} '
                f'max_curvature={metrics.max_curvature:.3f} 1/m')
            self._publish_status('ENTRY_PLAN_VALID')

            if not self.execute_path:
                self._log_info(
                    'execute=false: the entry plan is validated and published; '
                    'the vehicle remains stopped')
                self._publish_status('FINISHED')
                return

            # The validated plan starts at the approach pose.  Put the vehicle
            # there, then re-plan from where it actually stopped so the path
            # handed to FollowPath begins at the vehicle, and re-check that
            # its first segment is still a reverse.
            executable = self._prepare_executable_entry(candidate)
            if executable is None:
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail(
                        'the entry path re-planned from the approach pose no '
                        'longer starts with a reverse segment')
                return
            entry_path, entry_metrics = executable

            self.parking_wheels_inside = False
            self.wheel_inside_confirm_count = 0
            self.last_wheel_states = None
            self.motion_phase = 'parking'
            self._publish_status('EXECUTE_ENTRY')
            if not self._execute_segmented_path(
                    entry_path, entry_metrics, monitor_slot=slot):
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail('entry FollowPath failed')
                return

            self._publish_status('CONFIRM_PARKED')
            candidate.path = entry_path
            candidate.metrics = entry_metrics
            if not self.parking_wheels_inside:
                self._fail(
                    'entry FollowPath ended before all four wheels were '
                    'confirmed inside the selected slot')
                return

            self._publish_status('STOP')
            if not self._confirm_motion_stop('parallel entry'):
                self._fail('vehicle did not stop after the entry path')
                return
            parked_pose = self._current_map_pose_stamped()
            if parked_pose is None:
                self._fail('could not capture the parked map pose')
                return
            self._log_pose('parked_pose', parked_pose)

            if str(self.get_parameter('exit_mode').value) != 'forward':
                self._publish_status('SUCCESS', 'exit_mode is not forward')
                return

            self._publish_status('PLAN_EXIT')
            exit_candidate = self._plan_forward_exit(slot, candidate.lane_offset)
            if exit_candidate is None:
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail(
                        'ForwardExit (DUBIN) produced no valid forward-only '
                        'exit path; refusing to mix in a reverse segment')
                return
            self._publish_exit_plan(exit_candidate)
            self._log_info(
                f'[EXIT PLAN] exit_lead={exit_candidate.exit_lead:.2f} '
                f'poses={len(exit_candidate.path.poses)} '
                f'total={exit_candidate.metrics.total_length:.3f}m '
                f'forward={exit_candidate.metrics.forward_length:.3f}m '
                f'reverse={exit_candidate.metrics.reverse_length:.3f}m '
                f'cusps={exit_candidate.metrics.cusp_count}')
            self._publish_status('EXIT_PLAN_VALID')

            self.motion_phase = 'forward_exit'
            self._publish_status('EXECUTE_EXIT')
            if not self._execute_forward_exit(exit_candidate):
                if self._abort_requested():
                    self._cancelled()
                else:
                    self._fail('forward exit FollowPath failed')
                return
            if not self._confirm_motion_stop('forward exit'):
                self._fail('vehicle did not stop after the forward exit path')
                return
            self._publish_status(
                'SUCCESS',
                f'slot={slot.name} '
                f'entry_reverse={candidate.metrics.reverse_length:.3f}m '
                f'exit_reverse={exit_candidate.metrics.reverse_length:.3f}m')
        except Exception as exc:  # Keep a failed run fail-safe.
            if self._runtime_ok():
                self._log_error(
                    f'unhandled parallel parking error: {exc}\n'
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
                    self._log_error('failed to resume SLAM measurements')
                self.slam_paused_by_node = False

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

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
        if (not self._runtime_ok()
                or not self.readiness_lock.acquire(blocking=False)):
            return
        try:
            readiness = self._readiness_snapshot()
            self.readiness_cache = readiness
        except Exception as exc:
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

        The readiness timer shares the MultiThreadedExecutor, so blocking it
        on a service future would let overlapping ticks consume every thread
        in the pool and starve the responses they are waiting for.  Fire the
        request asynchronously and cache the answer instead, and treat a
        request older than pending_timeout as lost so a dropped response can
        never freeze this entry permanently.
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

    def _wait_for_map(self) -> bool:
        deadline = time.monotonic() + float(
            self.get_parameter('map_wait_timeout').value)
        while time.monotonic() < deadline and not self._abort_requested():
            with self.data_lock:
                ready = (
                    self.map_msg is not None
                    and self.global_costmap is not None
                    and self.local_costmap is not None)
            if ready:
                return True
            if not self._interruptible_sleep(0.25):
                return False
        return False

    # ------------------------------------------------------------------
    # Slot selection
    # ------------------------------------------------------------------

    def _select_slot(self) -> Optional[Slot]:
        by_name = {slot.name: slot for slot in self.slots}
        requested = self.target_slot
        if requested not in ('auto', 'slot_1', 'slot_2'):
            self._log_error(f'invalid target_slot={requested!r}')
            return None
        if requested != 'auto':
            self._log_info(f'target_slot parameter forces {requested}')
            return by_name[requested]

        layout = self._wait_for_layout()
        if layout is not None:
            suffix = layout.split('-')[-1].strip()
            blocked = self.LAYOUT_BLOCKS.get(suffix)
            if blocked is None:
                self._log_warn(
                    f'obstacle layout {layout!r} has an unknown parallel '
                    f'suffix {suffix!r}; falling back to a map-based check')
            else:
                free = 'slot_2' if blocked == 'slot_1' else 'slot_1'
                self._log_info(
                    f'[SLOT CHECK] layout={layout} blocks={blocked} '
                    f'-> target={free}')
                return by_name[free]
        else:
            self._log_warn(
                '/parking_practice/obstacle_layout was not received; '
                'falling back to a map-based free-slot check')

        for slot in self.slots:
            if self._slot_is_free(slot):
                self._log_info(f'[SLOT CHECK] map fallback selected {slot.name}')
                return slot
        self._log_error('map fallback found no free parallel slot')
        return None

    def _wait_for_layout(self) -> Optional[str]:
        deadline = time.monotonic() + float(
            self.get_parameter('layout_wait_timeout').value)
        while time.monotonic() < deadline and not self._abort_requested():
            with self.data_lock:
                layout = self.obstacle_layout
            if layout:
                return layout
            if not self._interruptible_sleep(0.2):
                return None
        return None

    def _slot_is_free(self, slot: Slot) -> bool:
        """Fallback occupancy test over the slot rectangle in the SLAM map."""
        with self.data_lock:
            map_msg = self.map_msg
        if map_msg is None:
            return False
        occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        step = 0.10
        inset = 0.10
        x = slot.min_x + inset
        while x <= slot.max_x - inset:
            y = slot.min_y + inset
            while y <= slot.max_y - inset:
                pose = self._odom_pose_to_map(x, y, slot.yaw)
                if pose is None:
                    return False
                value = self._occupancy_value(
                    map_msg, pose.pose.position.x, pose.pose.position.y)
                # Unknown cells make the fallback unusable rather than free;
                # this path only runs when the layout topic was missing.
                if value is None or value < 0 or value >= occupied_threshold:
                    return False
                y += step
            x += step
        return True

    # ------------------------------------------------------------------
    # Pose construction
    # ------------------------------------------------------------------

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

    def _parking_depth(self, slot: Slot) -> float:
        """Return the odom-x (depth) coordinate of the parked centre.

        Depth is lateral for parallel parking, so this is *not* the
        heading-axis calculation the T-bay node performs.  The vehicle sits
        on the slot's depth centre, which is validated to leave at least
        slot_depth_clearance between each vehicle side and both the inner
        wall and the mouth line.
        """
        return 0.5 * (slot.min_x + slot.max_x)

    def _depth_clearance_ok(self, slot: Slot) -> bool:
        half_width = 0.5 * float(self.get_parameter('vehicle_width').value)
        required = float(self.get_parameter('slot_depth_clearance').value)
        depth = self._parking_depth(slot)
        to_wall = slot.max_x - depth - half_width
        to_mouth = depth - half_width - slot.min_x
        if min(to_wall, to_mouth) < required:
            self._log_error(
                f'{slot.name} depth clearance too small: wall={to_wall:.3f}m '
                f'mouth={to_mouth:.3f}m required={required:.3f}m')
            return False
        self._log_info(
            f'{slot.name} parked depth={depth:.3f} '
            f'wall_clearance={to_wall:.3f}m mouth_clearance={to_mouth:.3f}m')
        return True

    def _parking_longitudinal(self, slot: Slot) -> float:
        """Return the odom-y (along-lane) coordinate of the parked centre.

        The vehicle reverses in from the high-odom_y end, so it parks just
        inside that end rather than in the middle of the slot.  This slot is
        4.78 m long for a 0.90 m vehicle: aiming at its longitudinal centre
        would demand a ~5 m reverse, which Smac can only satisfy with a
        three-cusp snake that pure pursuit then stalls on.  Parking where the
        vehicle actually arrives is both the shorter manoeuvre and what a
        driver does -- pull in behind the car ahead.
        """
        end_clearance = float(self.get_parameter('slot_end_clearance').value)
        half_length = 0.5 * float(self.get_parameter('vehicle_length').value)
        # Never past the slot's longitudinal centre: parking deeper than the
        # centre is what forced the ~5 m reverse that Smac could only answer
        # with a stalling three-cusp snake, and parking shallower than
        # end_clearance would leave the nose hanging out of the slot.
        return min(slot.odom_y, slot.max_y - (half_length + end_clearance))

    def _build_entry_poses(
            self, slot: Slot, lane_offset: float,
            approach_offset: float) -> Optional[Dict[str, PoseStamped]]:
        """Build the approach and parked poses for one candidate.

        The approach pose sits in the lane, ``approach_offset`` metres *past*
        the parked pose in the driving direction.  From there the parked pose
        is behind and to the right, so reversing is the short way in and a
        forward entry would have to loop around -- exactly the asymmetry that
        makes the manoeuvre a parallel park rather than a slide-in.
        """
        lane_x = slot.min_x - lane_offset
        park_y = self._parking_longitudinal(slot)
        entry_angle = float(self.get_parameter('entry_angle').value)
        entry_depth = float(self.get_parameter('entry_depth').value)
        entry_lead = float(self.get_parameter('entry_lead').value)
        odom_poses = {
            'approach': (lane_x, park_y + approach_offset, slot.yaw),
            # Gate pose: nose still in the lane, tail already swung into the
            # slot mouth.  Planning approach -> parked directly lets Smac
            # answer "reverse 0.5 m then loop 7 m forward into the open bay",
            # which passes a first-segment test but is not a parallel park.
            'entry': (
                slot.min_x + entry_depth, park_y + entry_lead,
                normalize_angle(slot.yaw - entry_angle)),
            'parked': (self._parking_depth(slot), park_y, slot.yaw),
        }
        result = {}
        for name, values in odom_poses.items():
            pose = self._odom_pose_to_map(*values)
            if pose is None:
                return None
            result[name] = pose
        return result

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

    def _fresh_pose(self, source: PoseStamped) -> PoseStamped:
        result = PoseStamped()
        result.header.frame_id = 'map'
        result.header.stamp = self.get_clock().now().to_msg()
        result.pose.position.x = source.pose.position.x
        result.pose.position.y = source.pose.position.y
        result.pose.position.z = source.pose.position.z
        set_pose_yaw(result.pose, yaw_from_quaternion(source.pose.orientation))
        return result

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

    # ------------------------------------------------------------------
    # Staging drive
    # ------------------------------------------------------------------

    def _current_odom_pose(self) -> Optional[Tuple[float, float, float]]:
        with self.data_lock:
            odom = self.odom_msg
        if odom is None:
            return None
        return (
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            yaw_from_quaternion(odom.pose.pose.orientation),
        )

    def _staging_waypoints(
            self, slot: Slot, lane_offset: float,
            approach_offset: float
    ) -> Optional[List[List[Tuple[float, float, float]]]]:
        """Return the staging route as a list of hops, each a list of goals.

        Most hops are a single goal.  The corner is one hop carrying two
        goals, so ComputePathThroughPoses shapes the whole turn in one path
        instead of stopping the vehicle half way round it.

        The route is an L: up the start corridor along +odom_x, then west
        along the lane in +odom_y.  It is emitted as short hops because the
        SLAM map only reaches a few metres ahead (scan_filter truncates
        /scan_static at 10 m and the corridor self-occludes), while
        GridBased runs with allow_unknown:false -- so a single long plan to
        the corner is rejected until the vehicle has driven far enough to
        actually see it.
        """
        current = self._current_odom_pose()
        if current is None:
            return None
        hop = max(0.5, float(self.get_parameter('staging_hop_distance').value))
        corridor_y = float(self.get_parameter('staging_via_odom_y').value)
        corner_entry_x = float(self.get_parameter('corner_entry_odom_x').value)
        corner_exit_y = float(self.get_parameter('corner_exit_odom_y').value)
        lane_x = slot.min_x - lane_offset
        # Stop the staging route short of every candidate approach pose, so
        # the later alignment onto the selected one is always a plain forward
        # move along the same lane line.  A sideways or backward nudge of a
        # few tens of centimetres is what pure pursuit reports as "Failed to
        # make progress" on an Ackermann vehicle.
        approach_y = (
            self._parking_longitudinal(slot) + approach_offset
            - float(self.get_parameter('staging_final_gap').value))
        waypoints: List[List[Tuple[float, float, float]]] = []

        # Leg A: up the start corridor, heading +odom_x (odom yaw 0).  It
        # deliberately stops short of the lane line: the corner is a 90 deg
        # turn and this vehicle needs 0.946 m of turning radius, so trying to
        # pivot at the lane line itself wedges it against the slot mouth.
        x = current[0] + hop
        while x < corner_entry_x - 0.5 * hop:
            waypoints.append([(x, corridor_y, 0.0)])
            x += hop
        waypoints.append([(corner_entry_x, corridor_y, 0.0)])

        # Leg B: the corner.  A single goal on the far side of it leaves Smac
        # free to answer with a cusped manoeuvre -- measured cusps=2 with
        # 0.665 m of reverse -- and RegulatedPurePursuitController stalls on
        # that ("Failed to make progress") right at the slot mouth.  Pinning
        # the half-way point of a constant-radius left arc fixes the shape, so
        # the corner comes out as one smooth forward turn.  The mid pose is
        # the 45 deg point of a circle of radius corner_arc_radius whose
        # centre sits at (corner_entry_x, corridor_y + r).
        radius = max(
            float(self.get_parameter('minimum_turning_radius').value),
            float(self.get_parameter('corner_arc_radius').value))
        half = math.sqrt(0.5)
        waypoints.append([
            (corner_entry_x + half * radius,
             corridor_y + (1.0 - half) * radius,
             0.25 * math.pi),
            (lane_x, corner_exit_y, slot.yaw),
        ])

        # Leg C: west along the lane to the approach pose.
        y = corner_exit_y + hop
        while y < approach_y - 0.5 * hop:
            waypoints.append([(lane_x, y, slot.yaw)])
            y += hop
        if approach_y > corner_exit_y + 0.35:
            waypoints.append([(lane_x, approach_y, slot.yaw)])
        return waypoints

    def _drive_to_approach(self, slot: Slot) -> bool:
        """Drive from the course start into the lane beside the target slot.

        Nav2's GridBased planner runs with allow_unknown:false, so the entry
        manoeuvre cannot be planned until the vehicle has actually observed
        the slot strip.  This leg does that, hopping along the route and
        letting SLAM extend the map at every stop.
        """
        lane_offsets = [float(value) for value in self.get_parameter(
            'lane_offset_candidates').value]
        approach_offsets = [float(value) for value in self.get_parameter(
            'approach_offset_candidates').value]
        waypoints = self._staging_waypoints(
            slot, lane_offsets[0], min(approach_offsets))
        if not waypoints:
            self._log_error('could not build the staging route')
            return False
        self._publish_status('DRIVING_TO_APPROACH')
        self._log_info(
            '[STAGING ROUTE] ' + ' -> '.join(
                ' & '.join(f'({x:.2f},{y:.2f},{yaw:.2f})' for x, y, yaw in hop)
                for hop in waypoints))

        for index, hop_goals in enumerate(waypoints, start=1):
            if self._abort_requested():
                return False
            targets = [self._odom_pose_to_map(*goal) for goal in hop_goals]
            if any(target is None for target in targets):
                self._log_error('could not transform a staging waypoint')
                return False
            # Skip hops the vehicle has already passed: an earlier plan often
            # overshoots a waypoint on its way to the one after it.
            x, y, _ = hop_goals[-1]
            current = self._current_odom_pose()
            if current is not None and math.hypot(
                    x - current[0], y - current[1]) < 0.35:
                self._log_info(
                    f'staging hop {index}/{len(waypoints)} already reached')
                continue
            # One retry: an aborted FollowPath leaves the vehicle part-way
            # along the hop, and re-planning from where it actually stopped
            # usually succeeds where the original path no longer applies.
            for attempt in (1, 2):
                path = self._request_plan(targets)
                if path is None:
                    self._log_error(
                        f'staging hop {index}/{len(waypoints)} to odom '
                        f'({x:.2f},{y:.2f}) could not be planned; the map '
                        'does not reach that far yet')
                    return False
                if self._drive_path(
                        path, f'staging hop {index}/{len(waypoints)}'):
                    break
                if attempt == 2 or self._abort_requested():
                    self._log_error(
                        f'staging hop {index}/{len(waypoints)} failed twice')
                    return False
                self._log_warn(
                    f'staging hop {index}/{len(waypoints)} aborted; '
                    're-planning from the current pose')
                self._emergency_stop()
                if not self._wait_until_stopped(float(
                        self.get_parameter('stop_wait_timeout').value)):
                    return False
        return True

    def _drive_path(self, path: Path, context: str) -> bool:
        metrics = self._analyze_path(path, path.poses[-1])
        self._log_info(
            f'[{context.upper()}] poses={len(path.poses)} '
            f'total={metrics.total_length:.3f}m '
            f'reverse={metrics.reverse_length:.3f}m '
            f'cusps={metrics.cusp_count}')
        self.motion_phase = 'staging'
        if not self._execute_segmented_path(
                path, metrics, monitor_slot=None,
                controller_id=str(self.get_parameter(
                    'transit_controller_id').value),
                goal_checker_id=str(self.get_parameter(
                    'transit_goal_checker_id').value)):
            return False
        return self._confirm_motion_stop(context)

    # ------------------------------------------------------------------
    # Entry planning
    # ------------------------------------------------------------------

    def _plan_entry(self, slot: Slot) -> Optional[EntryCandidate]:
        if not self._depth_clearance_ok(slot):
            return None
        lane_offsets = [float(value) for value in self.get_parameter(
            'lane_offset_candidates').value]
        approach_offsets = [float(value) for value in self.get_parameter(
            'approach_offset_candidates').value]
        minimum_reverse = float(
            self.get_parameter('minimum_reverse_length').value)
        candidates: List[EntryCandidate] = []
        for lane_offset in lane_offsets:
            for approach_offset in approach_offsets:
                if self._abort_requested():
                    return None
                named = self._build_entry_poses(
                    slot, lane_offset, approach_offset)
                if named is None:
                    continue
                if not self._approach_inside_lane(named['approach']):
                    self._log_warn(
                        f'reject lane={lane_offset:.2f} '
                        f'approach={approach_offset:.2f}: approach footprint '
                        'leaves the driving lane')
                    continue
                # Plan *from* the approach pose rather than from wherever the
                # vehicle happens to be standing.  Planning from the current
                # pose prepends the transit onto the approach pose, and that
                # leading forward segment -- sometimes only a few centimetres
                # of waypoint-stitching jitter -- is indistinguishable from a
                # genuine forward entry, which made this test answer a
                # question about the transit instead of about the manoeuvre.
                # Anchor the plan at the approach pose instead of wherever
                # the vehicle happens to stand: planning from the current
                # pose prepends the transit, and that leading forward segment
                # made this test answer a question about the transit rather
                # than about the manoeuvre -- so the verdict flipped with the
                # centimetres of where the staging drive happened to stop.
                path = self._request_plan(
                    [named['entry'], named['parked']],
                    start_pose=named['approach'])
                if path is None:
                    continue
                path = self._dedupe_stationary_poses(path)
                metrics = self._analyze_path(path, named['parked'])
                first_direction = self._first_drive_direction(metrics)
                # THE acceptance test for parallel parking: the first thing
                # the vehicle does once it is alongside the slot must be to
                # reverse.  A forward-only entry into this oversized bay is a
                # legal Reeds-Shepp path but it is not a parallel park.
                if first_direction >= 0:
                    self._log_warn(
                        f'reject lane={lane_offset:.2f} '
                        f'approach={approach_offset:.2f}: first move from '
                        'the approach pose is forward, not reverse')
                    continue
                reverse_after = metrics.reverse_length
                if reverse_after < minimum_reverse:
                    self._log_warn(
                        f'reject lane={lane_offset:.2f} '
                        f'approach={approach_offset:.2f}: reverse after '
                        f'approach {reverse_after:.3f}m < '
                        f'{minimum_reverse:.3f}m')
                    continue
                valid, reason = self._validate_entry_path(
                    slot, named, path, metrics)
                if not valid:
                    self._log_warn(
                        f'reject lane={lane_offset:.2f} '
                        f'approach={approach_offset:.2f}: {reason}')
                    continue
                self._log_info(
                    f'accept lane={lane_offset:.2f} '
                    f'approach={approach_offset:.2f}: '
                    f'reverse_after_approach={reverse_after:.3f}m '
                    f'cusps={metrics.cusp_count} '
                    f'max_curvature={metrics.max_curvature:.3f} 1/m')
                candidates.append(EntryCandidate(
                    slot, lane_offset, approach_offset, named, path, metrics,
                    reverse_after))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item.metrics.cusp_count,
                item.metrics.max_curvature,
                item.metrics.total_length,
            ))

    def _first_direction_after_pose(
            self, path: Path, metrics: PathMetrics, pose: PoseStamped) -> int:
        """Return the first non-zero drive direction at or after ``pose``.

        The planned path starts at the vehicle's current pose, so its leading
        segments are the transit onto the approach pose.  What decides whether
        this is a parallel park is the first direction taken *after* the
        vehicle is alongside the slot.
        """
        if not path.poses or not metrics.directions:
            return 0
        start_index = min(
            range(len(path.poses)),
            key=lambda index: math.hypot(
                path.poses[index].pose.position.x - pose.pose.position.x,
                path.poses[index].pose.position.y - pose.pose.position.y))
        for index in range(start_index, len(metrics.directions)):
            if metrics.directions[index]:
                return metrics.directions[index]
        return 0

    @staticmethod
    def _reverse_length_after_pose(
            path: Path, metrics: PathMetrics, pose: PoseStamped) -> float:
        start_index = min(
            range(len(path.poses)),
            key=lambda index: math.hypot(
                path.poses[index].pose.position.x - pose.pose.position.x,
                path.poses[index].pose.position.y - pose.pose.position.y))
        reverse_length = 0.0
        for index in range(start_index, len(path.poses) - 1):
            if metrics.directions[index] >= 0:
                continue
            reverse_length += math.hypot(
                path.poses[index + 1].pose.position.x
                - path.poses[index].pose.position.x,
                path.poses[index + 1].pose.position.y
                - path.poses[index].pose.position.y)
        return reverse_length

    @staticmethod
    def _first_drive_direction(metrics: PathMetrics) -> int:
        """Return the direction of the path's first real motion segment.

        Equivalent to asking whether ``first_reverse_index`` is 1, but it also
        distinguishes "starts forward" from "never reverses at all".
        """
        for direction in metrics.directions:
            if direction:
                return direction
        return 0

    def _prepare_executable_entry(
            self, candidate: EntryCandidate
    ) -> Optional[Tuple[Path, PathMetrics]]:
        """Line the vehicle up with the approach pose and re-plan from it."""
        approach = candidate.named_poses['approach']
        current = self._current_map_pose()
        if current is None:
            self._log_error('cannot read the current pose to align on approach')
            return None
        distance = math.hypot(
            current[0] - approach.pose.position.x,
            current[1] - approach.pose.position.y)
        yaw_error = abs(normalize_angle(
            current[2] - yaw_from_quaternion(approach.pose.orientation)))
        self._log_info(
            f'approach alignment: distance={distance:.3f}m '
            f'yaw_error={yaw_error:.3f}rad')
        if distance > 0.15 or yaw_error > 0.10:
            self._publish_status('ALIGN_ON_APPROACH')
            path = self._request_plan([approach])
            if path is None:
                self._log_error('could not plan the approach alignment leg')
                return None
            if not self._drive_path(path, 'approach alignment'):
                return None

        path = self._request_plan([
            candidate.named_poses['entry'], candidate.named_poses['parked']])
        if path is None:
            self._log_error('could not re-plan the entry from the approach pose')
            return None
        path = self._dedupe_stationary_poses(path)
        metrics = self._analyze_path(path, candidate.named_poses['parked'])
        direction = self._first_drive_direction(metrics)
        self._log_info(
            f'[ENTRY REPLAN] first_segment='
            f'{"reverse" if direction < 0 else "forward"} '
            f'first_reverse_index={metrics.first_reverse_index} '
            f'reverse={metrics.reverse_length:.3f}m '
            f'forward={metrics.forward_length:.3f}m '
            f'cusps={metrics.cusp_count}')
        if direction >= 0:
            return None
        if metrics.reverse_length < float(
                self.get_parameter('minimum_reverse_length').value):
            self._log_error(
                f're-planned entry reverses only {metrics.reverse_length:.3f}m')
            return None
        valid, reason = self._validate_entry_path(
            candidate.slot, candidate.named_poses, path, metrics)
        if not valid:
            self._log_error(f're-planned entry rejected: {reason}')
            return None
        return path, metrics

    def _validate_entry_path(
            self, slot: Slot, named: Dict[str, PoseStamped], path: Path,
            metrics: PathMetrics) -> Tuple[bool, str]:
        # Loop guard.  A path that backs up a token 0.5 m and then swings 7 m
        # forward into the open bay satisfies every per-segment test while
        # being a wide forward entry in disguise.  A real parallel park stays
        # close to the straight line between the approach and parked poses.
        direct = math.hypot(
            named['parked'].pose.position.x - named['approach'].pose.position.x,
            named['parked'].pose.position.y - named['approach'].pose.position.y)
        factor = float(self.get_parameter('entry_loop_length_factor').value)
        limit = factor * max(direct, 1.0)
        if metrics.total_length > limit:
            return False, (
                f'path length {metrics.total_length:.3f}m exceeds '
                f'{limit:.3f}m for a {direct:.3f}m approach-to-parked span '
                '(loops instead of backing in)')
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
        if not self._footprint_inside_slot(slot, named['parked']):
            return False, 'parked footprint is outside the selected slot'
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

    def _approach_inside_lane(self, pose: PoseStamped) -> bool:
        """Check the approach footprint against the lane's depth bounds.

        Only the odom-x (depth) axis is bounded: the lane runs the whole
        length of the odom-y axis, and the slot strip and the far kerb are
        what the vehicle must stay between.
        """
        if self._abort_requested():
            return False
        lane_min = float(self.get_parameter('lane_min_x').value)
        lane_max = float(self.get_parameter('lane_max_x').value)
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
            if not lane_min < ox < lane_max:
                return False
        return True

    def _footprint_inside_slot(self, slot: Slot, pose: PoseStamped) -> bool:
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
            if not (slot.min_x < ox < slot.max_x
                    and slot.min_y < oy < slot.max_y):
                return False
        return True

    # ------------------------------------------------------------------
    # Exit planning
    # ------------------------------------------------------------------

    def _plan_forward_exit(
            self, slot: Slot, lane_offset: float) -> Optional[ExitCandidate]:
        """Plan a strictly forward exit with the DUBIN ForwardExit planner.

        DUBIN has no reverse primitive, so the planner cannot return a path
        containing a reverse segment.  ``_validate_exit_path`` still measures
        the reverse length afterwards as an independent second layer.
        """
        planner_id = str(self.get_parameter('exit_planner_id').value)
        leads = [float(value) for value in self.get_parameter(
            'exit_lead_candidates').value]
        lane_x = slot.min_x - lane_offset
        for lead in leads:
            if self._abort_requested():
                return None
            target = self._odom_pose_to_map(
                lane_x, slot.max_y + lead, slot.yaw)
            if target is None:
                continue
            self._log_pose(f'exit_target(lead={lead:.2f})', target)
            path = self._request_plan([target], planner_id=planner_id)
            if path is None:
                self._log_warn(
                    f'ForwardExit found no path for exit_lead={lead:.2f}')
                continue
            path = self._dedupe_stationary_poses(path)
            metrics = self._analyze_path(path, target)
            valid, reason = self._validate_exit_path(path, metrics)
            if not valid:
                self._log_warn(f'reject exit_lead={lead:.2f}: {reason}')
                continue
            return ExitCandidate(
                lead, {'parked': path.poses[0], 'exit_target': target},
                path, metrics)
        return None

    def _validate_exit_path(
            self, path: Path, metrics: PathMetrics) -> Tuple[bool, str]:
        if not path.poses:
            return False, 'path is empty'
        directions = [value for value in metrics.directions if value]
        if not directions:
            return False, 'path has no meaningful motion segment'
        # Second layer over the DUBIN guarantee: refuse anything reversing.
        if bool(self.get_parameter('require_forward_only_exit').value):
            reverse_limit = float(self.get_parameter(
                'max_reverse_distance_during_exit').value)
            reverse_segments = sum(1 for value in directions if value < 0)
            if metrics.reverse_length > reverse_limit or reverse_segments > 0:
                return False, (
                    f'reverse_length={metrics.reverse_length:.3f}m '
                    f'reverse_segments={reverse_segments} exceeds the '
                    f'{reverse_limit:.3f}m forward-only allowance')
        maximum_cusps = int(self.get_parameter('maximum_exit_cusps').value)
        if metrics.cusp_count > maximum_cusps:
            return False, (
                f'cusp count {metrics.cusp_count} exceeds {maximum_cusps}')
        minimum_radius = float(
            self.get_parameter('minimum_turning_radius').value)
        curvature_factor = float(
            self.get_parameter('curvature_tolerance_factor').value)
        if metrics.max_curvature > curvature_factor / minimum_radius:
            return False, (
                f'curvature {metrics.max_curvature:.3f} exceeds '
                f'{curvature_factor / minimum_radius:.3f} 1/m '
                '(minimum turning radius cannot make this exit)')
        if metrics.final_position_error > float(self.get_parameter(
                'exit_position_tolerance').value):
            return False, (
                f'planned endpoint position error '
                f'{metrics.final_position_error:.3f}m')
        if metrics.final_yaw_error > float(self.get_parameter(
                'exit_yaw_tolerance').value):
            return False, (
                f'planned endpoint yaw error {metrics.final_yaw_error:.3f}rad')
        # A genuine zero-distance seam between stitched Smac segments differs
        # by about one heading bin; anything larger is a real in-place spin.
        for previous, current in zip(path.poses, path.poses[1:]):
            distance = math.hypot(
                current.pose.position.x - previous.pose.position.x,
                current.pose.position.y - previous.pose.position.y)
            yaw_change = abs(normalize_angle(
                yaw_from_quaternion(current.pose.orientation)
                - yaw_from_quaternion(previous.pose.orientation)))
            if distance < 0.002 and yaw_change > 0.20:
                return False, 'in-place rotation segment detected'
        return self._validate_exit_footprint(path)

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
        # ForwardExit runs with allow_unknown:true because the lane behind the
        # parked vehicle is self-occluded.  NO_INFORMATION therefore means
        # unobserved, not unsafe; only real occupied/lethal cells and
        # out-of-bounds samples are rejected here.
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
                    return False, f'footprint index {index} is outside the map'
                if map_value >= occupied_threshold or cost_value == 254:
                    return False, f'footprint index {index} overlaps obstacle'
        return True, 'valid'

    # ------------------------------------------------------------------
    # Nav2 requests
    # ------------------------------------------------------------------

    def _request_plan(
            self, goals: Sequence[PoseStamped], planner_id: Optional[str] = None,
            start_pose: Optional[PoseStamped] = None) -> Optional[Path]:
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
            f'planner_id={goal.planner_id} '
            f'use_start={str(goal.use_start).lower()} goals='
            + ', '.join(
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

    # ------------------------------------------------------------------
    # Path analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe_stationary_poses(path: Path) -> Path:
        """Drop zero-distance duplicate poses left by waypoint stitching.

        ComputePathThroughPoses concatenates independently-planned Smac
        segments, and each segment quantizes its shared endpoint heading on
        its own.  The resulting same-position, few-degrees-apart pair is
        geometrically harmless but can make pure pursuit emit a brief
        negative-velocity correction, which the forward-exit guard would
        (correctly, but unhelpfully) treat as reversing.
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

        # Measure curvature over a physical window rather than adjacent grid
        # samples: three nearly coincident Hybrid-A* poses amplify map-grid
        # quantization, and a Reeds-Shepp cusp has undefined curvature.  A
        # 0.15 m window stays well inside the 0.946 m minimum turning radius.
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

    # ------------------------------------------------------------------
    # Costmap sampling
    # ------------------------------------------------------------------

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
        if not (0 <= col < grid.metadata.size_x
                and 0 <= row < grid.metadata.size_y):
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

    # ------------------------------------------------------------------
    # Wheel-inside confirmation
    # ------------------------------------------------------------------

    def _wheel_centers_map(self) -> Optional[Dict[str, Tuple[float, float]]]:
        labels = ('front_left', 'front_right', 'rear_left', 'rear_right')
        frames = [str(value) for value in self.get_parameter(
            'wheel_frames').value]
        if len(frames) == 4:
            positions: Dict[str, Tuple[float, float]] = {}
            try:
                for label, frame in zip(labels, frames):
                    transform = self.tf_buffer.lookup_transform(
                        'map', frame, Time(), timeout=Duration(seconds=0.03))
                    positions[label] = (
                        transform.transform.translation.x,
                        transform.transform.translation.y)
                if self.wheel_position_source != 'tf':
                    self._log_info(
                        'wheel centres sourced from TF: ' + ', '.join(frames))
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
        cosine = math.cos(current[2])
        sine = math.sin(current[2])
        offsets = {
            'front_left': (half_base, half_track),
            'front_right': (half_base, -half_track),
            'rear_left': (-half_base, half_track),
            'rear_right': (-half_base, -half_track),
        }
        return {
            label: (
                current[0] + cosine * dx - sine * dy,
                current[1] + sine * dx + cosine * dy)
            for label, (dx, dy) in offsets.items()
        }

    def _wheel_centers_odom(self) -> Optional[Dict[str, Tuple[float, float]]]:
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

    def _update_wheels_inside(self, slot: Slot) -> bool:
        """Confirm all four wheel centres are inside the slot rectangle.

        Unlike the T bay there is no entrance-side anchor here: the whole
        rectangle is legal parking space, so containment plus alignment is the
        criterion.  The inset puts the required clearance at 0.295 m from the
        vehicle centreline, which is already stricter than the 0.225 m
        footprint half-width, so passing the wheel test also means the whole
        footprint is inside the slot.

        Alignment has to be checked separately.  The slot is 4.78 m long and
        1.48 m deep for a 0.90 x 0.45 m vehicle, so a car still 25 deg across
        the slot mid-manoeuvre already has all four wheels inside the
        rectangle -- containment alone would stop it parked crooked.
        """
        positions = self._wheel_centers_odom()
        if positions is None:
            self.wheel_inside_confirm_count = 0
            return False
        current = self._current_odom_pose()
        if current is None:
            self.wheel_inside_confirm_count = 0
            return False
        yaw_error = abs(normalize_angle(current[2] - slot.yaw))
        aligned = yaw_error <= float(
            self.get_parameter('parked_yaw_tolerance').value)
        inset = (
            float(self.get_parameter('wheel_radius').value)
            + float(self.get_parameter('wheel_inside_margin').value))
        min_x, max_x = slot.min_x + inset, slot.max_x - inset
        min_y, max_y = slot.min_y + inset, slot.max_y - inset
        labels = ('front_left', 'front_right', 'rear_left', 'rear_right')
        states = tuple(
            min_x <= positions[label][0] <= max_x
            and min_y <= positions[label][1] <= max_y
            for label in labels)
        if all(states) and aligned:
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
                + f'\naligned={str(aligned).lower()} '
                f'yaw_error={yaw_error:.3f}rad'
                + f'\nconfirm={self.wheel_inside_confirm_count}/{required}')
            self.last_wheel_states = states
            self.last_wheel_log = now
        if self.wheel_inside_confirm_count < required:
            return False
        self._log_info('PARALLEL PARKING POSITION ACCEPTED')
        return True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_forward_exit(self, candidate: ExitCandidate) -> bool:
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
            monitor_slot: Optional[Slot],
            controller_id: Optional[str] = None,
            goal_checker_id: Optional[str] = None) -> bool:
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
                    monitor_slot=monitor_slot, controller_id=controller_id,
                    goal_checker_id=goal_checker_id):
                return False
            if monitor_slot is not None and self.parking_wheels_inside:
                return True
            if run_number < len(runs):
                # Finish one Nav2 action completely before handing pure
                # pursuit the opposite direction; overlapping runs near a cusp
                # otherwise produce alternating velocity signs in one place.
                self._emergency_stop()
                if not self._wait_until_stopped(float(
                        self.get_parameter('stop_wait_timeout').value)):
                    self._log_error(
                        'vehicle did not settle at a FollowPath direction cusp')
                    return False
        return True

    def _execute_path_action(
            self, path: Path, run_number: int, run_count: int,
            direction: int, monitor_slot: Optional[Slot] = None,
            forward_exit: bool = False, controller_id: Optional[str] = None,
            goal_checker_id: Optional[str] = None) -> bool:
        if direction < 0:
            healthy, rear_distance, reason = self._rear_scan_state()
            if not healthy:
                self._log_error(
                    f'reverse FollowPath inhibited: rear lidar {reason}')
                self._emergency_stop()
                return False
            self._log_info(
                f'rear safety armed: nearest={rear_distance:.3f}m threshold='
                f'{float(self.get_parameter("rear_emergency_stop_distance").value):.3f}m')
        if self._abort_requested():
            return False
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = str(
            controller_id if controller_id is not None
            else self.get_parameter('controller_id').value)
        # Transit hops pass a looser checker; the parking run keeps the strict
        # one, and intermediate cusps only need to stop cleanly.
        final_checker = (
            goal_checker_id if goal_checker_id is not None
            else str(self.get_parameter('goal_checker_id').value))
        goal.goal_checker_id = (
            final_checker if run_number == run_count else 'cusp_goal_checker')
        goal.progress_checker_id = str(
            self.get_parameter('progress_checker_id').value)
        self._log_info(
            f'FollowPath request {run_number}/{run_count}: '
            f'controller_id={goal.controller_id} '
            f'goal_checker_id={goal.goal_checker_id}')
        if not self._runtime_ok():
            return False
        send_future = self.follow_client.send_goal_async(
            goal, feedback_callback=self._follow_feedback)
        goal_handle = self._wait_future(send_future, 5.0)
        if goal_handle is None or not goal_handle.accepted:
            self._log_error('FollowPath goal rejected')
            return False
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
                self._log_error('FollowPath execution timeout; cancelling goal')
                if self._context_ok():
                    goal_handle.cancel_goal_async()
                self._emergency_stop()
                return False
            if direction < 0:
                healthy, rear_distance, reason = self._rear_scan_state()
                threshold = float(self.get_parameter(
                    'rear_emergency_stop_distance').value)
                if not healthy or rear_distance <= threshold:
                    detail = (
                        f'obstacle at {rear_distance:.3f}m <= {threshold:.3f}m'
                        if healthy else reason)
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
                'entry FollowPath reached its planned endpoint without '
                'consecutive all-wheel-inside confirmations')
            return False
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            self._log_error(
                f'FollowPath status={wrapped.status} '
                f'error_code={result.error_code} '
                f'error_msg={result.error_msg!r}')
            return False
        self._log_info(
            f'FollowPath segment {run_number}/{run_count} result '
            f'error_code={result.error_code}')
        return result.error_code == FollowPath.Result.NONE

    def _cancel_parking_follow_path(self, goal_handle, result_future) -> bool:
        self._log_info(
            'all wheels confirmed inside; cancelling the active FollowPath')
        cancel_future = goal_handle.cancel_goal_async()
        timeout = float(self.get_parameter('parking_cancel_timeout').value)
        response = self._wait_future(cancel_future, timeout)
        if response is None or not response.goals_canceling:
            self._log_error('FollowPath cancel request was not accepted')
            return False
        wrapped = self._wait_future(result_future, timeout)
        self.active_follow_goal = None
        if wrapped is None:
            self._log_error('FollowPath did not terminate after cancel')
            return False
        if wrapped.status != GoalStatus.STATUS_CANCELED:
            self._log_error(
                f'FollowPath cancel ended with status={wrapped.status}')
            return False
        self._log_info('FollowPath result confirmed STATUS_CANCELED')
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
            # +inf is the standard LaserScan "no return inside range_max",
            # which is valid clear space -- unlike NaN or a stale scan.
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
        positions = path.poses
        nearest = min(
            range(len(positions)),
            key=lambda index: math.hypot(
                positions[index].pose.position.x - current_pose[0],
                positions[index].pose.position.y - current_pose[1]))
        segment_index = min(nearest, len(metrics.directions) - 1)
        direction = (
            metrics.directions[segment_index] if metrics.directions else 1)
        final = final_pose.pose.position
        final_distance = math.hypot(
            final.x - current_pose[0], final.y - current_pose[1])
        self._log_info(
            f'pose=({current_pose[0]:.3f},{current_pose[1]:.3f},'
            f'{current_pose[2]:.3f}) remaining={feedback.distance_to_goal:.3f}m '
            f'speed={feedback.speed:.3f}m/s path_index={nearest} '
            f'direction={"reverse" if direction < 0 else "forward"} '
            f'final_distance={final_distance:.3f}m')

    def _confirm_motion_stop(self, context: str) -> bool:
        if self._wait_until_stopped(float(self.get_parameter(
                'parking_stop_grace_timeout').value)):
            return True
        self._log_warn(
            f'vehicle still moving after {context}; '
            'sending zero-velocity safety pulse')
        self._emergency_stop()
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

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _publish_entry_plan(self, candidate: EntryCandidate) -> None:
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
        self._safe_publish(
            self.marker_publisher,
            self._make_markers(candidate.named_poses, 'entry_poses', {
                'approach': ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9),
                'parked': ColorRGBA(r=0.1, g=1.0, b=0.2, a=0.9),
            }, candidate.named_poses['parked']))

    def _publish_exit_plan(self, candidate: ExitCandidate) -> None:
        if not self._runtime_ok():
            return
        candidate.path.header.stamp = self.get_clock().now().to_msg()
        self._safe_publish(self.exit_path_publisher, candidate.path)
        self._safe_publish(
            self.marker_publisher,
            self._make_markers(candidate.named_poses, 'exit_poses', {
                'parked': ColorRGBA(r=0.2, g=0.8, b=0.2, a=0.9),
                'exit_target': ColorRGBA(r=1.0, g=0.25, b=0.25, a=0.9),
            }, None))

    def _make_markers(
            self, named_poses: Dict[str, PoseStamped], namespace: str,
            colors: Dict[str, ColorRGBA],
            footprint_pose: Optional[PoseStamped]) -> MarkerArray:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0
        for name, pose in named_poses.items():
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = namespace
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose.pose
            marker.scale.x = 0.45
            marker.scale.y = 0.09
            marker.scale.z = 0.09
            marker.color = colors.get(
                name, ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.9))
            marker.text = name
            markers.markers.append(marker)
        if footprint_pose is None:
            return markers
        footprint = Marker()
        footprint.header.frame_id = 'map'
        footprint.header.stamp = stamp
        footprint.ns = f'{namespace}_footprint'
        footprint.id = marker_id
        footprint.type = Marker.LINE_STRIP
        footprint.action = Marker.ADD
        footprint.scale.x = 0.035
        footprint.color = ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0)
        yaw = yaw_from_quaternion(footprint_pose.pose.orientation)
        half_l = 0.5 * float(self.get_parameter('vehicle_length').value)
        half_w = 0.5 * float(self.get_parameter('vehicle_width').value)
        ordered = [(half_l, half_w), (half_l, -half_w),
                   (-half_l, -half_w), (-half_l, half_w), (half_l, half_w)]
        for local_x, local_y in ordered:
            point = Point()
            point.x = (
                footprint_pose.pose.position.x + math.cos(yaw) * local_x
                - math.sin(yaw) * local_y)
            point.y = (
                footprint_pose.pose.position.y + math.sin(yaw) * local_x
                + math.cos(yaw) * local_y)
            footprint.points.append(point)
        markers.markers.append(footprint)
        return markers

    # ------------------------------------------------------------------
    # Shutdown paths
    # ------------------------------------------------------------------

    def _cancel_active_goals(self) -> None:
        if not self._context_ok():
            return
        for goal_handle in (self.active_plan_goal, self.active_follow_goal):
            if goal_handle is not None:
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass

    def _emergency_stop(self) -> None:
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
    node = AutoParallelParking()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        worker_joined = node.stop(join_timeout=5.0)
        if not worker_joined:
            print('parallel_parking_worker did not stop before teardown')
        executor.remove_node(node)
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
