#!/usr/bin/env python3
"""
Park parallel to the road inside either half of the existing T bay.

The proven parallel-parking executor is reused without changing its normal
configuration or interface.  This subclass changes only the geometry: the T
bay's long/depth axes are swapped, and the east half is approached westbound
so the complete reverse S-curve crosses the bay mouth instead of its curb.
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple

from auto_parallel_parking import (
    AutoParallelParking,
    EntryCandidate,
    normalize_angle,
    PathMetrics,
    Slot,
    yaw_from_quaternion,
)
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from parallel_entry_reference import (
    rear_axle_to_vehicle_center,
    vehicle_center_to_rear_axle,
)
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class ParallelInTSlot(AutoParallelParking):
    """T-slot-specific candidate generator and collision validator."""

    LAYOUT_BLOCKS = {'A': 'slot_1', 'B': 'slot_2'}

    def __init__(self) -> None:
        self.rejected_goals: List[Tuple[PoseStamped, str]] = []
        self.selected_candidate: Optional[EntryCandidate] = None
        self.selected_clearance = 0.0
        self._clearance_grid_identity = None
        self._clearance_field: Optional[List[float]] = None
        self.preview_staging_path: Optional[Path] = None
        self.full_plan_metrics: Optional[PathMetrics] = None
        self.executed_staging_metrics: List[PathMetrics] = []
        super().__init__()
        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.staging_path_publisher = self.create_publisher(
            Path, '/parallel_in_t_slot/staging_path', transient_qos)

    def _declare_parameters(self) -> None:
        super()._declare_parameters()
        additions = {
            'goal_longitudinal_offsets': [0.0, -0.05, 0.05, -0.10, 0.10],
            'goal_depth_offsets': [0.0, -0.05, 0.05, -0.10, 0.10],
            'safety_margin': 0.03,
            'steering_limit_deg': 22.0,
            'road_min_y': -2.10,
            'road_max_y': 2.10,
            'road_center_y': 0.0,
            'east_turnaround_x': 10.50,
            'staging_turnaround_cusps': 3,
        }
        for name, default in additions.items():
            self.declare_parameter(name, default)

    # ------------------------------------------------------------------
    # Existing-T-slot selection
    # ------------------------------------------------------------------

    def _select_slot(self) -> Optional[Slot]:
        by_name = {slot.name: slot for slot in self.slots}
        if self.target_slot not in ('auto', 'slot_1', 'slot_2'):
            self._log_error(f'invalid target_slot={self.target_slot!r}')
            return None
        if self.target_slot != 'auto':
            self._log_info(f'target_slot parameter forces {self.target_slot}')
            return by_name[self.target_slot]

        layout = self._wait_for_layout()
        if layout and layout != 'NONE':
            t_obstacle = layout.split('-')[0].strip()
            blocked = self.LAYOUT_BLOCKS.get(t_obstacle)
            if blocked is not None:
                free = 'slot_2' if blocked == 'slot_1' else 'slot_1'
                self._log_info(
                    f'[T SLOT CHECK] layout={layout} blocks={blocked} '
                    f'-> target={free}')
                return by_name[free]
        for slot in self.slots:
            if self._slot_is_free(slot):
                self._log_info(f'[T SLOT CHECK] map selected {slot.name}')
                return slot
        self._log_error('no obstacle-free half of the existing T bay was found')
        return None

    def _analyze_entry_path(
            self, path: Path, final_pose: PoseStamped) -> PathMetrics:
        """Measure steering curvature at the rear axle of a centre path."""
        metrics = super()._analyze_path(path, final_pose)
        if not path.poses:
            return metrics

        wheel_base = float(self.get_parameter('wheel_base').value)
        rear_path = Path()
        rear_path.header = path.header
        for centre in path.poses:
            yaw = yaw_from_quaternion(centre.pose.orientation)
            rear_x, rear_y, _ = vehicle_center_to_rear_axle(
                (centre.pose.position.x, centre.pose.position.y, yaw),
                wheel_base)
            rear = PoseStamped()
            rear.header = centre.header
            rear.pose.position.x = rear_x
            rear.pose.position.y = rear_y
            rear.pose.position.z = centre.pose.position.z
            rear.pose.orientation = centre.pose.orientation
            rear_path.poses.append(rear)

        # The centre locus has a tangent jump when steering changes
        # instantaneously.  That is not rear-axle curvature and must not be
        # compared with tan(max_steering) / wheel_base.
        rear_metrics = super()._analyze_path(
            rear_path, rear_path.poses[-1])
        return PathMetrics(
            metrics.total_length,
            metrics.forward_length,
            metrics.reverse_length,
            metrics.cusp_count,
            metrics.first_reverse_index,
            metrics.directions,
            rear_metrics.max_curvature,
            metrics.final_position_error,
            metrics.final_yaw_error)

    def _validate_entry_path(
            self, slot: Slot, named: Dict[str, PoseStamped], path: Path,
            metrics: PathMetrics,
            maximum_cusps: Optional[int] = None) -> Tuple[bool, str]:
        """Validate a centre-reference entry with rear steering curvature."""
        entry_metrics = self._analyze_entry_path(path, named['parked'])
        return super()._validate_entry_path(
            slot, named, path, entry_metrics, maximum_cusps)

    def _prepare_executable_entry(
            self, candidate: EntryCandidate
    ) -> Optional[Tuple[Path, PathMetrics]]:
        """Keep executable entry metrics in the bicycle reference contract."""
        executable = super()._prepare_executable_entry(candidate)
        if executable is None:
            return None
        path, _ = executable
        return path, self._analyze_entry_path(
            path, candidate.named_poses['parked'])

    # ------------------------------------------------------------------
    # Swapped-axis geometry: slot length is odom X, depth is odom Y.
    # ------------------------------------------------------------------

    def _depth_clearance_ok(self, slot: Slot) -> bool:
        usable_length = slot.max_x - slot.min_x
        usable_depth = slot.max_y - slot.min_y
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        margin = float(self.get_parameter('safety_margin').value)
        ok = (
            usable_length + 1.0e-9 >= length + 2.0 * margin
            and usable_depth + 1.0e-9 >= width + 2.0 * margin)
        self._log_info(
            f'[PARALLEL_T_SLOT] {slot.name} bounds '
            f'{usable_length:.3f}m x {usable_depth:.3f}m; '
            f'vehicle={length:.3f}m x {width:.3f}m; '
            f'margin={margin:.3f}m; final_fit={str(ok).lower()}')
        if not ok:
            self._log_error(
                '[PARALLEL_T_SLOT] infeasible\n'
                'reason: final footprint does not fit')
        return ok

    def _build_t_entry(
            self, slot: Slot, lane_offset: float, approach_offset: float,
            longitudinal_offset: float, depth_offset: float
    ) -> Optional[Tuple[Dict[str, PoseStamped], Path]]:
        park_x = 0.5 * (slot.min_x + slot.max_x) + longitudinal_offset
        park_y = 0.5 * (slot.min_y + slot.max_y) + depth_offset
        lane_y = slot.min_y - lane_offset
        yaw = slot.yaw
        forward = (math.cos(yaw), math.sin(yaw))
        left = (-forward[1], forward[0])
        lateral = (park_y - lane_y) * left[1]
        if abs(lateral) < 1.0e-6:
            return None

        radius = max(
            float(self.get_parameter('minimum_turning_radius').value),
            float(self.get_parameter('entry_turning_radius').value))
        cosine = 1.0 - abs(lateral) / (2.0 * radius)
        if not -1.0 <= cosine <= 1.0:
            return None
        arc_angle = math.acos(cosine)
        required = 2.0 * radius * math.sin(arc_angle)
        straight_lead = approach_offset - required
        if straight_lead < -1.0e-6:
            return None

        lane_x = park_x
        approach_x = lane_x + approach_offset * forward[0]
        approach_y = lane_y + approach_offset * forward[1]
        spacing = max(
            0.02, float(self.get_parameter('entry_pose_spacing').value))
        wheel_base = float(self.get_parameter('wheel_base').value)
        center_approach = (approach_x, approach_y, yaw)
        center_goal = (park_x, park_y, yaw)
        rear_goal = vehicle_center_to_rear_axle(center_goal, wheel_base)
        rear_samples = [vehicle_center_to_rear_axle(
            center_approach, wheel_base)]
        if straight_lead > 1.0e-6:
            self._integrate_bicycle_motion(
                rear_samples, -straight_lead, 0.0, spacing)
        turn_sign = math.copysign(1.0, lateral)
        self._integrate_bicycle_motion(
            rear_samples, -radius * arc_angle, turn_sign / radius, spacing)
        transition_index = len(rear_samples) - 1
        self._integrate_bicycle_motion(
            rear_samples, -radius * arc_angle, -turn_sign / radius, spacing)

        rear_endpoint = rear_samples[-1]
        if (math.hypot(
                rear_endpoint[0] - rear_goal[0],
                rear_endpoint[1] - rear_goal[1]) > 1.0e-4
                or abs(normalize_angle(
                    rear_endpoint[2] - rear_goal[2])) > 1.0e-4):
            self._log_error('rear-axle entry integration endpoint mismatch')
            return None

        center_samples = [
            rear_axle_to_vehicle_center(sample, wheel_base)
            for sample in rear_samples
        ]
        odom_named = {
            'staging': center_approach,
            'approach': center_approach,
            'transition': center_samples[transition_index],
            'parked': center_goal,
        }
        named: Dict[str, PoseStamped] = {}
        for name, values in odom_named.items():
            pose = self._odom_pose_to_map(*values)
            if pose is None:
                return None
            named[name] = pose

        path = Path()
        for x, y, sample_yaw in center_samples:
            pose = self._odom_pose_to_map(x, y, sample_yaw)
            if pose is None:
                return None
            path.poses.append(pose)
        path.header = path.poses[0].header
        endpoint = path.poses[-1]
        if (math.hypot(
                endpoint.pose.position.x - named['parked'].pose.position.x,
                endpoint.pose.position.y - named['parked'].pose.position.y)
                > 1.0e-4):
            return None
        return named, path

    def _approach_inside_lane(self, pose: PoseStamped) -> bool:
        road_min = float(self.get_parameter('road_min_y').value)
        road_max = float(self.get_parameter('road_max_y').value)
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'map', Time(), timeout=Duration(seconds=0.5))
        except Exception:
            return False
        tf_yaw = yaw_from_quaternion(transform.transform.rotation)
        cosine = math.cos(tf_yaw)
        sine = math.sin(tf_yaw)
        for x, y in self._footprint_samples(
                pose, length, width, edge_only=True):
            odom_y = transform.transform.translation.y + sine * x + cosine * y
            if not road_min < odom_y < road_max:
                return False
        return True

    def _footprint_inside_slot(self, slot: Slot, pose: PoseStamped) -> bool:
        if self._abort_requested():
            return False
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'map', Time(), timeout=Duration(seconds=0.5))
        except Exception:
            return False
        tf_yaw = yaw_from_quaternion(transform.transform.rotation)
        cosine = math.cos(tf_yaw)
        sine = math.sin(tf_yaw)
        margin = float(self.get_parameter('safety_margin').value)
        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        for x, y in self._footprint_samples(
                pose, length, width, edge_only=True):
            ox = transform.transform.translation.x + cosine * x - sine * y
            oy = transform.transform.translation.y + sine * x + cosine * y
            if not (
                    slot.min_x + margin <= ox <= slot.max_x - margin
                    and slot.min_y + margin <= oy <= slot.max_y - margin):
                return False
        return True

    # ------------------------------------------------------------------
    # Candidate search and full swept-footprint acceptance
    # ------------------------------------------------------------------

    def _plan_entry(self, slot: Slot) -> Optional[EntryCandidate]:
        if not self._depth_clearance_ok(slot):
            return None
        wheelbase = float(self.get_parameter('wheel_base').value)
        steering_limit = math.radians(float(
            self.get_parameter('steering_limit_deg').value))
        physical_radius = wheelbase / math.tan(steering_limit)
        configured_radius = float(
            self.get_parameter('minimum_turning_radius').value)
        if configured_radius + 1.0e-9 < physical_radius:
            self._log_error(
                '[PARALLEL_T_SLOT] infeasible\n'
                'reason: required steering exceeds 22 deg')
            return None
        self._log_info(
            f'[PARALLEL_T_SLOT] bicycle minimum radius={physical_radius:.3f}m '
            f'(wheelbase={wheelbase:.3f}m, steering=22.0deg); '
            f'planner radius={configured_radius:.3f}m')

        lane_offsets = [float(value) for value in self.get_parameter(
            'lane_offset_candidates').value]
        approach_offsets = [float(value) for value in self.get_parameter(
            'approach_offset_candidates').value]
        longitudinal_offsets = [float(value) for value in self.get_parameter(
            'goal_longitudinal_offsets').value]
        depth_offsets = [float(value) for value in self.get_parameter(
            'goal_depth_offsets').value]
        minimum_reverse = float(
            self.get_parameter('minimum_reverse_length').value)
        candidates: List[Tuple[EntryCandidate, float]] = []
        self.rejected_goals = []

        for longitudinal in longitudinal_offsets:
            for depth in depth_offsets:
                # Centre plus independent +/- X and +/- Y probes matches the
                # requested candidate pattern without multiplying it into a
                # redundant 5 x 5 Cartesian grid.
                if abs(longitudinal) > 1.0e-9 and abs(depth) > 1.0e-9:
                    continue
                goal_for_rejection: Optional[PoseStamped] = None
                accepted_this_goal = False
                last_reason = 'no kinematically valid reverse S-curve'
                for lane_offset in lane_offsets:
                    for approach_offset in approach_offsets:
                        built = self._build_t_entry(
                            slot, lane_offset, approach_offset,
                            longitudinal, depth)
                        if built is None:
                            continue
                        named, path = built
                        goal_for_rejection = named['parked']
                        if not self._approach_inside_lane(named['approach']):
                            last_reason = 'staging footprint leaves the road'
                            continue
                        path = self._dedupe_stationary_poses(path)
                        metrics = self._analyze_entry_path(
                            path, named['parked'])
                        if self._first_drive_direction(metrics) >= 0:
                            last_reason = 'entry does not start in reverse'
                            continue
                        if metrics.reverse_length < minimum_reverse:
                            last_reason = 'reverse segment is too short'
                            continue
                        valid, reason = self._validate_entry_path(
                            slot, named, path, metrics)
                        if not valid:
                            last_reason = reason
                            continue
                        clearance = self._minimum_obstacle_clearance(path)
                        candidate = EntryCandidate(
                            slot, lane_offset, approach_offset, named, path,
                            metrics, metrics.reverse_length)
                        candidates.append((candidate, clearance))
                        accepted_this_goal = True
                        self._log_info(
                            f'[GOAL ACCEPT] offset=({longitudinal:+.2f},'
                            f'{depth:+.2f}) goal=('
                            f'{named["parked"].pose.position.x:.3f},'
                            f'{named["parked"].pose.position.y:.3f},'
                            f'{yaw_from_quaternion(named["parked"].pose.orientation):.3f}) '
                            f'clearance={clearance:.3f}m '
                            f'length={metrics.total_length:.3f}m')
                if not accepted_this_goal and goal_for_rejection is not None:
                    self.rejected_goals.append((goal_for_rejection, last_reason))
                    self._log_warn(
                        f'[GOAL REJECT] offset=({longitudinal:+.2f},'
                        f'{depth:+.2f}) reason={last_reason}')

        if not candidates:
            self._log_error(
                '[PARALLEL_T_SLOT] infeasible\n'
                'reason: no collision-free reverse S-curve path')
            return None

        # The staging route ends at the nominal approach built from the first
        # lane candidate and the shortest approach candidate.  Correcting the
        # footprint reference changes clearance scores, but must not select an
        # entry whose start no longer matches the pose we just staged at.
        waypoint_groups = self._staging_waypoints(
            slot, lane_offsets[0], min(approach_offsets))
        if not waypoint_groups:
            return None
        nominal_approach = self._odom_pose_to_map(
            *waypoint_groups[-1][-1])
        if nominal_approach is None:
            return None

        def candidate_key(item: Tuple[EntryCandidate, float]):
            candidate, candidate_clearance = item
            approach = candidate.named_poses['approach']
            capture_error = math.hypot(
                approach.pose.position.x
                - nominal_approach.pose.position.x,
                approach.pose.position.y
                - nominal_approach.pose.position.y)
            return (
                round(capture_error, 2),
                -round(candidate_clearance, 2),
                candidate.metrics.total_length,
                candidate.metrics.max_curvature)

        # Within the staged-pose contract, prefer clearance; path length and
        # curvature break near ties.
        chosen, clearance = min(
            candidates, key=candidate_key)
        self.selected_candidate = chosen
        self.selected_clearance = clearance
        if not self.execute_path and not self._plan_staging_preview(chosen):
            self._log_error(
                '[PARALLEL_T_SLOT] infeasible\n'
                'reason: no collision-free path from the original start '
                'to staging')
            return None
        return chosen

    def _plan_staging_preview(self, candidate: EntryCandidate) -> bool:
        """Plan and validate original-start to staging without moving."""
        waypoint_groups = self._staging_waypoints(
            candidate.slot, candidate.lane_offset,
            candidate.approach_offset)
        if not waypoint_groups:
            return False
        targets = []
        flattened = [goal for group in waypoint_groups for goal in group]
        for index, values in enumerate(flattened):
            if index == len(flattened) - 1:
                targets.append(candidate.named_poses['staging'])
                continue
            pose = self._odom_pose_to_map(*values)
            if pose is None:
                return False
            targets.append(pose)
        preview = self._request_plan(
            targets, planner_id=str(self.get_parameter(
                'transit_planner_id').value))
        if preview is None:
            return False
        preview = self._dedupe_stationary_poses(preview)
        metrics = self._analyze_path(preview, targets[-1])
        curvature_limit = 1.0 / float(
            self.get_parameter('minimum_turning_radius').value)
        if metrics.max_curvature > curvature_limit + 1.0e-3:
            return False
        if metrics.cusp_count > int(
                self.get_parameter('staging_turnaround_cusps').value):
            return False
        valid, reason = self._validate_exit_footprint(preview)
        if not valid:
            self._log_warn(f'staging preview rejected: {reason}')
            return False
        self.preview_staging_path = preview
        self._safe_publish(self.staging_path_publisher, preview)
        return True

    def _minimum_obstacle_clearance(self, path: Path) -> float:
        """Conservative map-cell clearance from every swept body edge."""
        with self.data_lock:
            grid = self.map_msg
        if grid is None:
            return 0.0
        threshold = int(self.get_parameter('occupied_threshold').value)
        resolution = float(grid.info.resolution)
        identity = (id(grid), grid.info.width, grid.info.height)
        if (self._clearance_grid_identity != identity
                or self._clearance_field is None):
            size = grid.info.width * grid.info.height
            distances = [float('inf')] * size
            queue = []
            for index, value in enumerate(grid.data):
                # Unknown is a clearance boundary too: accepted paths may not
                # touch it, and "far from occupied but beside unknown" must
                # never win the safest-candidate ranking.
                if int(value) < 0 or int(value) >= threshold:
                    distances[index] = 0.0
                    heapq.heappush(queue, (0.0, index))
            neighbours = (
                (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
                (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)))
            while queue:
                distance, index = heapq.heappop(queue)
                if distance != distances[index]:
                    continue
                row, col = divmod(index, grid.info.width)
                for dr, dc, cost in neighbours:
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < grid.info.height
                            and 0 <= nc < grid.info.width):
                        continue
                    target = nr * grid.info.width + nc
                    candidate = distance + cost
                    if candidate < distances[target]:
                        distances[target] = candidate
                        heapq.heappush(queue, (candidate, target))
            self._clearance_grid_identity = identity
            self._clearance_field = distances

        length = float(self.get_parameter('vehicle_length').value)
        width = float(self.get_parameter('vehicle_width').value)
        best = float('inf')
        stride = max(1, len(path.poses) // 100)
        for pose in path.poses[::stride]:
            for x, y in self._footprint_samples(
                    pose, length, width, edge_only=True):
                col, row = self._grid_coordinates(
                    grid.info.origin, resolution, x, y)
                if not (0 <= col < grid.info.width
                        and 0 <= row < grid.info.height):
                    return 0.0
                cell_distance = self._clearance_field[
                    row * grid.info.width + col]
                best = min(best, cell_distance * resolution)
        return max(0.0, best)

    # ------------------------------------------------------------------
    # Original start -> staging. Slot 2 includes a map-planned turnaround.
    # ------------------------------------------------------------------

    def _staging_waypoints(
            self, slot: Slot, lane_offset: float,
            approach_offset: float
    ) -> Optional[List[List[Tuple[float, float, float]]]]:
        current = self._current_odom_pose()
        if current is None:
            return None
        lane_y = slot.min_y - lane_offset
        park_x = 0.5 * (slot.min_x + slot.max_x)
        approach_x = park_x + approach_offset * math.cos(slot.yaw)
        approach_y = lane_y + approach_offset * math.sin(slot.yaw)
        road_y = float(self.get_parameter('road_center_y').value)
        hop = max(0.5, float(
            self.get_parameter('staging_hop_distance').value))
        waypoints: List[List[Tuple[float, float, float]]] = []

        destination_before_turn = approach_x - 2.5
        if abs(normalize_angle(slot.yaw)) > 1.0:
            destination_before_turn = float(
                self.get_parameter('east_turnaround_x').value)
        x = current[0] + hop
        while x < destination_before_turn - 0.5 * hop:
            waypoints.append([(x, road_y, 0.0)])
            x += hop
        if destination_before_turn > current[0] + 0.35:
            waypoints.append([(destination_before_turn, road_y, 0.0)])
        waypoints.append([(approach_x, approach_y, slot.yaw)])
        return waypoints

    def _drive_path(self, path: Path, context: str) -> bool:
        path = self._dedupe_stationary_poses(path)
        metrics = self._analyze_path(path, path.poses[-1])
        curvature_limit = 1.0 / float(
            self.get_parameter('minimum_turning_radius').value)
        maximum_cusps = int(
            self.get_parameter('staging_turnaround_cusps').value)
        if metrics.max_curvature > curvature_limit + 1.0e-3:
            steering_limit = float(
                self.get_parameter('steering_limit_deg').value)
            self._log_error(
                f'{context} exceeds the {steering_limit:.1f} deg '
                'steering constraint')
            return False
        if metrics.cusp_count > maximum_cusps:
            self._log_error(
                f'{context} needs {metrics.cusp_count} cusps; '
                f'limit is {maximum_cusps}')
            return False
        valid, reason = self._validate_exit_footprint(path)
        if not valid:
            self._log_error(f'{context} swept-footprint collision: {reason}')
            return False
        self.motion_phase = 'staging'
        if not self._execute_segmented_path(
                path, metrics, monitor_slot=None,
                controller_id=str(self.get_parameter(
                    'controller_id').value),
                goal_checker_id=str(self.get_parameter(
                    'transit_goal_checker_id').value)):
            return False
        if not self._confirm_motion_stop(context):
            return False
        self.executed_staging_metrics.append(metrics)
        return True

    def _run_state_machine(self) -> None:
        self.executed_staging_metrics = []
        self.preview_staging_path = None
        self.full_plan_metrics = None
        super()._run_state_machine()

    # ------------------------------------------------------------------
    # Diagnostics and RViz
    # ------------------------------------------------------------------

    def _publish_entry_plan(self, candidate: EntryCandidate) -> None:
        super()._publish_entry_plan(candidate)
        display_path = candidate.path
        if self.preview_staging_path is not None:
            display_path = Path()
            display_path.header = self.preview_staging_path.header
            display_path.poses = (
                list(self.preview_staging_path.poses)
                + list(candidate.path.poses))
            display_path = self._dedupe_stationary_poses(display_path)
            self._safe_publish(self.path_publisher, display_path)
            display_metrics = self._analyze_path(
                display_path, candidate.named_poses['parked'])
            forward = Path()
            reverse = Path()
            forward.header = display_path.header
            reverse.header = display_path.header
            for index, direction in enumerate(display_metrics.directions):
                target = forward if direction >= 0 else reverse
                target.poses.extend([
                    display_path.poses[index], display_path.poses[index + 1]])
            self._safe_publish(self.forward_publisher, forward)
            self._safe_publish(self.reverse_publisher, reverse)
            self.full_plan_metrics = display_metrics
        else:
            self.full_plan_metrics = candidate.metrics
        combined = self._make_markers(candidate.named_poses, 'entry_poses', {
            'staging': ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9),
            'approach': ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9),
            'transition': ColorRGBA(r=0.8, g=0.3, b=1.0, a=0.9),
            'parked': ColorRGBA(r=0.1, g=1.0, b=0.2, a=0.9),
        }, candidate.named_poses['parked'])
        combined.markers.extend(self._diagnostic_markers(candidate).markers)
        self._safe_publish(self.marker_publisher, combined)
        metrics = self.full_plan_metrics
        steering = math.degrees(math.atan(
            float(self.get_parameter('wheel_base').value)
            * candidate.metrics.max_curvature))
        goal = candidate.named_poses['parked']
        staging = candidate.named_poses['staging']
        self._log_info(
            '[PARALLEL_T_SLOT] feasible\n'
            f'goal=({goal.pose.position.x:.3f}, '
            f'{goal.pose.position.y:.3f}, '
            f'{yaw_from_quaternion(goal.pose.orientation):.3f})\n'
            f'staging=({staging.pose.position.x:.3f}, '
            f'{staging.pose.position.y:.3f}, '
            f'{yaw_from_quaternion(staging.pose.orientation):.3f})\n'
            f'path_length={metrics.total_length:.3f}m '
            f'forward={metrics.forward_length:.3f}m '
            f'reverse={metrics.reverse_length:.3f}m '
            f'cusps={metrics.cusp_count} '
            f'max_steering={steering:.2f}deg '
            f'minimum_obstacle_clearance={self.selected_clearance:.3f}m')

    def _diagnostic_markers(self, candidate: EntryCandidate) -> MarkerArray:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 1000

        boundary = Marker()
        boundary.header.frame_id = 'map'
        boundary.header.stamp = stamp
        boundary.ns = 'parallel_t_slot_boundary'
        boundary.id = marker_id
        marker_id += 1
        boundary.type = Marker.LINE_STRIP
        boundary.action = Marker.ADD
        boundary.scale.x = 0.04
        boundary.color = ColorRGBA(r=0.1, g=0.8, b=1.0, a=1.0)
        slot = candidate.slot
        corners = [
            (slot.min_x, slot.min_y), (slot.max_x, slot.min_y),
            (slot.max_x, slot.max_y), (slot.min_x, slot.max_y),
            (slot.min_x, slot.min_y)]
        for x, y in corners:
            pose = self._odom_pose_to_map(x, y, 0.0)
            if pose is None:
                continue
            boundary.points.append(Point(
                x=pose.pose.position.x, y=pose.pose.position.y, z=0.05))
        markers.markers.append(boundary)

        current = self._current_map_pose_stamped()
        if current is not None:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'parallel_t_slot_current_pose'
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = current.pose
            marker.scale.x = 0.55
            marker.scale.y = 0.12
            marker.scale.z = 0.12
            marker.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=1.0)
            markers.markers.append(marker)

        for rejected, _reason in self.rejected_goals:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'parallel_t_slot_rejected_goals'
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = rejected.pose
            marker.scale.x = 0.30
            marker.scale.y = 0.07
            marker.scale.z = 0.07
            marker.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.75)
            markers.markers.append(marker)

        result = Marker()
        result.header.frame_id = 'map'
        result.header.stamp = stamp
        result.ns = 'parallel_t_slot_collision_result'
        result.id = marker_id
        result.type = Marker.TEXT_VIEW_FACING
        result.action = Marker.ADD
        result.pose = candidate.named_poses['parked'].pose
        result.pose.position.z = 0.65
        result.scale.z = 0.22
        result.color = ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0)
        result.text = (
            f'SELECTED: collision free\n'
            f'clearance {self.selected_clearance:.2f} m | '
            f'cusps {candidate.metrics.cusp_count}')
        markers.markers.append(result)
        return markers

    def _publish_status(self, status: str, detail: str = '') -> None:
        super()._publish_status(status, detail)
        if status != 'PARKING_SUCCESS' or self.selected_candidate is None:
            return
        candidate = self.selected_candidate
        current = self._current_map_pose()
        goal = candidate.named_poses['parked']
        if current is None:
            position_error = float('nan')
            yaw_error = float('nan')
        else:
            position_error = math.hypot(
                current[0] - goal.pose.position.x,
                current[1] - goal.pose.position.y)
            yaw_error = abs(normalize_angle(
                current[2] - yaw_from_quaternion(goal.pose.orientation)))
        metrics = candidate.metrics
        staging_forward = sum(
            item.forward_length for item in self.executed_staging_metrics)
        staging_reverse = sum(
            item.reverse_length for item in self.executed_staging_metrics)
        staging_cusps = sum(
            item.cusp_count for item in self.executed_staging_metrics)
        staging_length = sum(
            item.total_length for item in self.executed_staging_metrics)
        staging_curvature = max(
            (item.max_curvature for item in self.executed_staging_metrics),
            default=0.0)
        steering = math.degrees(math.atan(
            float(self.get_parameter('wheel_base').value)
            * max(metrics.max_curvature, staging_curvature)))
        self._log_info(
            '[PARALLEL_T_SLOT] PARKED\n'
            f'position_error={position_error:.3f}m\n'
            f'yaw_error={yaw_error:.3f}rad\n'
            f'minimum_obstacle_clearance={self.selected_clearance:.3f}m\n'
            f'max_steering_angle_used={steering:.2f}deg\n'
            f'total_path_length={metrics.total_length + staging_length:.3f}m\n'
            f'forward_distance={metrics.forward_length + staging_forward:.3f}m\n'
            f'reverse_distance={metrics.reverse_length + staging_reverse:.3f}m\n'
            f'cusp_count={metrics.cusp_count + staging_cusps}')

    def _fail(self, reason: str) -> None:
        self._log_error(f'[PARALLEL_T_SLOT] infeasible\nreason: {reason}')
        super()._fail(reason)


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ParallelInTSlot()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.cancel_requested.set()
        node._stop_event.set()
        node._cancel_active_goals()
        if node.execute_path:
            node._emergency_stop()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
