#!/usr/bin/env python3
"""Stationary diagnostics for the live T-parking start-to-staging failure.

This node intentionally has no vehicle-command publishers and no FollowPath
client.  It only reads map/costmap/lidar/TF data, clears the global costmap
after localization is stable, and sends ComputePathToPose requests.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped
from nav2_msgs.action import ComputePathThroughPoses, ComputePathToPose
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
import tf2_ros


START_TO_STAGING_MAX_STEP_M = 0.05
STAGING_X = 7.45
STAGING_Y = 0.0
ROAD_YAW = math.pi

# nav2_params.yaml footprint plus its configured 0.025 m padding.  Coordinates
# are relative to base_link (the rear-axle reference), not the body centre.
PADDED_FOOTPRINT = (
    (0.710, 0.415),
    (0.710, -0.415),
    (-0.670, -0.415),
    (-0.670, 0.415),
)


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def yaw_from_quaternion(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z
               + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y
                     + quaternion.z * quaternion.z),
    )


def set_pose_yaw(pose: Pose, yaw: float) -> None:
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = math.sin(0.5 * yaw)
    pose.orientation.w = math.cos(0.5 * yaw)


def path_length(poses: Sequence[PoseStamped]) -> float:
    return sum(
        math.hypot(
            current.pose.position.x - previous.pose.position.x,
            current.pose.position.y - previous.pose.position.y,
        )
        for previous, current in zip(poses, poses[1:])
    )


@dataclass(frozen=True)
class Endpoint:
    x: float
    y: float
    angle: float
    distance: float


@dataclass
class FootprintResult:
    center_cost: Optional[int]
    max_cost: Optional[int]
    counts: Dict[int, int]
    static_occupied: int
    collision: bool
    collision_cells: List[Tuple[int, int, float, float, int, Optional[int]]]


class StartStagingDiagnostics(Node):

    def __init__(self) -> None:
        super().__init__('start_staging_diagnostics')
        transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_msg: Optional[OccupancyGrid] = None
        self.global_costmap: Optional[Costmap] = None
        self.local_costmap: Optional[Costmap] = None
        self.front_scan: Optional[LaserScan] = None
        self.rear_scan: Optional[LaserScan] = None
        self.front_count = 0
        self.rear_count = 0
        self.global_count = 0
        self.local_count = 0
        self.global_received_at = 0.0
        self.local_received_at = 0.0
        self.front_received_at = 0.0
        self.rear_received_at = 0.0

        self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, transient)
        self.create_subscription(
            Costmap, '/global_costmap/costmap_raw',
            self._global_costmap_cb, transient)
        self.create_subscription(
            Costmap, '/local_costmap/costmap_raw',
            self._local_costmap_cb, transient)
        self.create_subscription(
            LaserScan, '/scan_front', self._front_cb,
            qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, '/scan_rear', self._rear_cb,
            qos_profile_sensor_data)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)
        self.clear_client = self.create_client(
            ClearEntireCostmap,
            '/global_costmap/clear_entirely_global_costmap')
        self.plan_client = ActionClient(
            self, ComputePathToPose, '/compute_path_to_pose')
        self.through_plan_client = ActionClient(
            self, ComputePathThroughPoses, '/compute_path_through_poses')

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg

    def _global_costmap_cb(self, msg: Costmap) -> None:
        self.global_costmap = msg
        self.global_count += 1
        self.global_received_at = time.monotonic()

    def _local_costmap_cb(self, msg: Costmap) -> None:
        self.local_costmap = msg
        self.local_count += 1
        self.local_received_at = time.monotonic()

    def _front_cb(self, msg: LaserScan) -> None:
        self.front_scan = msg
        self.front_count += 1
        self.front_received_at = time.monotonic()

    def _rear_cb(self, msg: LaserScan) -> None:
        self.rear_scan = msg
        self.rear_count += 1
        self.rear_received_at = time.monotonic()

    def spin_for(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(
                self, timeout_sec=min(0.05, deadline - time.monotonic()))

    def wait_for_inputs(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.map_msg is not None
                    and self.global_costmap is not None
                    and self.local_costmap is not None
                    and self.front_scan is not None
                    and self.rear_scan is not None
                    and self.tf_buffer.can_transform(
                        'map', 'base_link', Time(),
                        timeout=Duration(seconds=0.02))):
                return
        raise RuntimeError('map, costmaps, scans, or map->base_link TF unavailable')

    def current_pose(self) -> Tuple[float, float, float]:
        transform = self.tf_buffer.lookup_transform(
            'map', 'base_link', Time(), timeout=Duration(seconds=0.5))
        return (
            transform.transform.translation.x,
            transform.transform.translation.y,
            yaw_from_quaternion(transform.transform.rotation),
        )

    def wait_for_localization_stability(self) -> Tuple[float, float, float]:
        duration = 5.0
        interval = 0.10
        samples: List[Tuple[float, float, float]] = []
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=interval)
            try:
                samples.append(self.current_pose())
            except tf2_ros.TransformException:
                continue
        if len(samples) < 20:
            raise RuntimeError('insufficient map->base_link samples')
        mean_x = sum(item[0] for item in samples) / len(samples)
        mean_y = sum(item[1] for item in samples) / len(samples)
        mean_yaw = math.atan2(
            sum(math.sin(item[2]) for item in samples),
            sum(math.cos(item[2]) for item in samples),
        )
        max_position_delta = max(
            math.hypot(item[0] - mean_x, item[1] - mean_y)
            for item in samples)
        max_yaw_delta = max(
            abs(normalize_angle(item[2] - mean_yaw)) for item in samples)
        stable = max_position_delta <= 0.03 and max_yaw_delta <= 0.05
        print(
            '[AMCL STABILITY]\n'
            f'samples={len(samples)}\n'
            f'duration={duration:.2f}s\n'
            f'pose=({mean_x:.6f},{mean_y:.6f},{mean_yaw:.6f})\n'
            f'max_position_delta={max_position_delta:.6f}m\n'
            f'max_yaw_delta={max_yaw_delta:.6f}rad\n'
            f'stable={str(stable).lower()}', flush=True)
        if not stable:
            raise RuntimeError('AMCL did not satisfy the configured stability limits')
        return mean_x, mean_y, mean_yaw

    def clear_after_stability_and_wait_fresh(self) -> None:
        if not self.clear_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('global costmap clear service unavailable')
        request_started = time.monotonic()
        future = self.clear_client.call_async(ClearEntireCostmap.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            raise RuntimeError('global costmap clear timed out')
        clear_completed = time.monotonic()
        baseline = (
            self.front_count, self.rear_count,
            self.global_count, self.local_count)
        deadline = clear_completed + 15.0
        passed_at = None
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.monotonic()
            delta = (
                self.front_count - baseline[0],
                self.rear_count - baseline[1],
                self.global_count - baseline[2],
                self.local_count - baseline[3],
            )
            fresh = (
                now - self.front_received_at < 2.0
                and now - self.rear_received_at < 2.0
                and now - self.global_received_at < 2.0
                and now - self.local_received_at < 2.0)
            if (delta[0] >= 10 and delta[1] >= 10
                    and delta[2] >= 3 and delta[3] >= 3
                    and now - clear_completed >= 1.0 and fresh):
                passed_at = now
                break
        if passed_at is None:
            raise RuntimeError('fresh observation gate timed out after clear')
        delta = (
            self.front_count - baseline[0],
            self.rear_count - baseline[1],
            self.global_count - baseline[2],
            self.local_count - baseline[3],
        )
        print(
            '[CLEAR AND FRESH GATE]\n'
            f'clear_call_duration={clear_completed - request_started:.6f}s\n'
            f'clear_completed_monotonic={clear_completed:.6f}\n'
            f'front={delta[0]}/10 age={passed_at - self.front_received_at:.3f}s\n'
            f'rear={delta[1]}/10 age={passed_at - self.rear_received_at:.3f}s\n'
            f'global_costmap={delta[2]}/3 age='
            f'{passed_at - self.global_received_at:.3f}s\n'
            f'local_costmap={delta[3]}/3 age='
            f'{passed_at - self.local_received_at:.3f}s\n'
            f'settle_time={passed_at - clear_completed:.3f}s\n'
            'stable_obstacle_observation=PASS', flush=True)

    @staticmethod
    def grid_coordinates(origin: Pose, resolution: float,
                         x: float, y: float) -> Tuple[int, int]:
        yaw = yaw_from_quaternion(origin.orientation)
        dx = x - origin.position.x
        dy = y - origin.position.y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return math.floor(local_x / resolution), math.floor(local_y / resolution)

    @staticmethod
    def grid_world(origin: Pose, resolution: float,
                   col: int, row: int) -> Tuple[float, float]:
        yaw = yaw_from_quaternion(origin.orientation)
        local_x = (col + 0.5) * resolution
        local_y = (row + 0.5) * resolution
        return (
            origin.position.x
            + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
            origin.position.y
            + math.sin(yaw) * local_x + math.cos(yaw) * local_y,
        )

    def cost_value(self, grid: Costmap, col: int, row: int) -> Optional[int]:
        if not (0 <= col < grid.metadata.size_x
                and 0 <= row < grid.metadata.size_y):
            return None
        return int(grid.data[row * grid.metadata.size_x + col])

    def occupancy_at(self, x: float, y: float) -> Optional[int]:
        if self.map_msg is None:
            return None
        grid = self.map_msg
        col, row = self.grid_coordinates(
            grid.info.origin, grid.info.resolution, x, y)
        if not (0 <= col < grid.info.width and 0 <= row < grid.info.height):
            return None
        return int(grid.data[row * grid.info.width + col])

    @staticmethod
    def footprint_polygon(x: float, y: float, yaw: float
                          ) -> List[Tuple[float, float]]:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return [
            (x + cosine * px - sine * py,
             y + sine * px + cosine * py)
            for px, py in PADDED_FOOTPRINT
        ]

    @staticmethod
    def point_in_convex_polygon(
            point: Tuple[float, float], polygon: Sequence[Tuple[float, float]],
            tolerance: float = 1.0e-9) -> bool:
        signs = []
        for first, second in zip(polygon, polygon[1:] + polygon[:1]):
            cross = (
                (second[0] - first[0]) * (point[1] - first[1])
                - (second[1] - first[1]) * (point[0] - first[0]))
            if abs(cross) > tolerance:
                signs.append(cross > 0.0)
        return not signs or all(value == signs[0] for value in signs)

    def footprint_cells(self, grid: Costmap, x: float, y: float,
                        yaw: float) -> Set[Tuple[int, int]]:
        polygon = self.footprint_polygon(x, y, yaw)
        resolution = grid.metadata.resolution
        corners = [
            self.grid_coordinates(grid.metadata.origin, resolution, *point)
            for point in polygon
        ]
        min_col = min(item[0] for item in corners) - 1
        max_col = max(item[0] for item in corners) + 1
        min_row = min(item[1] for item in corners) - 1
        max_row = max(item[1] for item in corners) + 1
        cells: Set[Tuple[int, int]] = set()
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                wx, wy = self.grid_world(
                    grid.metadata.origin, resolution, col, row)
                if self.point_in_convex_polygon((wx, wy), polygon):
                    cells.add((col, row))
        # Include every grid cell crossed by the exact footprint boundary;
        # centre-only rasterization can otherwise miss a corner contact.
        edge_step = max(0.005, resolution * 0.20)
        for first, second in zip(polygon, polygon[1:] + polygon[:1]):
            length = math.hypot(second[0] - first[0], second[1] - first[1])
            count = max(1, math.ceil(length / edge_step))
            for index in range(count + 1):
                ratio = index / count
                wx = first[0] + ratio * (second[0] - first[0])
                wy = first[1] + ratio * (second[1] - first[1])
                cells.add(self.grid_coordinates(
                    grid.metadata.origin, resolution, wx, wy))
        return cells

    def inspect_footprint(self, grid: Costmap, x: float, y: float,
                          yaw: float) -> FootprintResult:
        center_col, center_row = self.grid_coordinates(
            grid.metadata.origin, grid.metadata.resolution, x, y)
        center_cost = self.cost_value(grid, center_col, center_row)
        counts = {253: 0, 254: 0, 255: 0}
        max_cost = None
        static_occupied = 0
        collision_cells = []
        for col, row in sorted(self.footprint_cells(grid, x, y, yaw)):
            cost = self.cost_value(grid, col, row)
            if cost is None:
                continue
            max_cost = cost if max_cost is None else max(max_cost, cost)
            if cost in counts:
                counts[cost] += 1
            wx, wy = self.grid_world(
                grid.metadata.origin, grid.metadata.resolution, col, row)
            static = self.occupancy_at(wx, wy)
            if static is not None and static >= 50:
                static_occupied += 1
            # Lethal live/static cost or a static occupied map cell is an
            # actual footprint collision.  Cost 253 and unknown are reported
            # independently and never relabelled as lethal.
            if cost == 254 or (static is not None and static >= 50):
                collision_cells.append((col, row, wx, wy, cost, static))
        return FootprintResult(
            center_cost=center_cost,
            max_cost=max_cost,
            counts=counts,
            static_occupied=static_occupied,
            collision=bool(collision_cells),
            collision_cells=collision_cells,
        )

    @staticmethod
    def interpolated_poses(start: Tuple[float, float, float]
                           ) -> List[Tuple[float, float, float]]:
        distance = math.hypot(STAGING_X - start[0], STAGING_Y - start[1])
        segments = max(1, math.ceil(distance / START_TO_STAGING_MAX_STEP_M))
        yaw_delta = normalize_angle(ROAD_YAW - start[2])
        return [
            (
                start[0] + index / segments * (STAGING_X - start[0]),
                start[1] + index / segments * (STAGING_Y - start[1]),
                normalize_angle(start[2] + index / segments * yaw_delta),
            )
            for index in range(segments + 1)
        ]

    def swept_cells(self, grid: Costmap, start: Tuple[float, float, float]
                    ) -> Set[Tuple[int, int]]:
        result: Set[Tuple[int, int]] = set()
        for x, y, yaw in self.interpolated_poses(start):
            result.update(self.footprint_cells(grid, x, y, yaw))
        return result

    def snapshot_persistence(self, start: Tuple[float, float, float]) -> None:
        records = []
        for index in range(1, 11):
            if self.global_costmap is None:
                raise RuntimeError('global costmap disappeared')
            grid = self.global_costmap
            cells = self.swept_cells(grid, start)
            counts = {253: 0, 254: 0}
            lethal_coordinates = []
            for col, row in cells:
                cost = self.cost_value(grid, col, row)
                if cost in counts:
                    counts[cost] += 1
                if cost == 254:
                    lethal_coordinates.append(self.grid_world(
                        grid.metadata.origin, grid.metadata.resolution,
                        col, row))
            records.append((counts[254], counts[253], lethal_coordinates))
            print(
                f'snapshot {index}: lethal={counts[254]} '
                f'inscribed={counts[253]} '
                f'global_costmap_sequence={self.global_count}', flush=True)
            if index < 10:
                self.spin_for(0.5)
        first = records[0][0]
        last = records[-1][0]
        conclusion = (
            'STALE_MARKING_DECAY' if first > 0 and last == 0
            else 'PERSISTENT_CURRENT_MARKING' if last > 0
            else 'NO_LETHAL_MARKING')
        print(
            '[COSTMAP PERSISTENCE]\n'
            f'lethal_sequence={[item[0] for item in records]}\n'
            f'inscribed_sequence={[item[1] for item in records]}\n'
            f'conclusion={conclusion}', flush=True)

    def scan_endpoints(self, scan: LaserScan, target_frame: str
                       ) -> List[Endpoint]:
        source_frame = scan.header.frame_id.lstrip('/')
        transform = self.tf_buffer.lookup_transform(
            target_frame, source_frame, Time(),
            timeout=Duration(seconds=0.5))
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        yaw = yaw_from_quaternion(transform.transform.rotation)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        endpoints = []
        for index, value in enumerate(scan.ranges):
            distance = float(value)
            if (not math.isfinite(distance)
                    or distance < scan.range_min
                    or distance > scan.range_max):
                continue
            angle = scan.angle_min + index * scan.angle_increment
            lx = distance * math.cos(angle)
            ly = distance * math.sin(angle)
            endpoints.append(Endpoint(
                tx + cosine * lx - sine * ly,
                ty + sine * lx + cosine * ly,
                angle,
                distance,
            ))
        return endpoints

    @staticmethod
    def nearest_endpoint(x: float, y: float,
                         endpoints: Sequence[Endpoint]
                         ) -> Tuple[float, Optional[Endpoint]]:
        if not endpoints:
            return float('inf'), None
        endpoint = min(
            endpoints, key=lambda item: (item.x - x) ** 2 + (item.y - y) ** 2)
        return math.hypot(endpoint.x - x, endpoint.y - y), endpoint

    @staticmethod
    def rectangle_gap(x: float, y: float) -> Tuple[bool, float]:
        min_x = min(point[0] for point in PADDED_FOOTPRINT)
        max_x = max(point[0] for point in PADDED_FOOTPRINT)
        min_y = min(point[1] for point in PADDED_FOOTPRINT)
        max_y = max(point[1] for point in PADDED_FOOTPRINT)
        inside = min_x <= x <= max_x and min_y <= y <= max_y
        if inside:
            boundary_gap = min(
                x - min_x, max_x - x, y - min_y, max_y - y)
            return True, -boundary_gap
        dx = max(min_x - x, 0.0, x - max_x)
        dy = max(min_y - y, 0.0, y - max_y)
        return False, math.hypot(dx, dy)

    def rear_self_return_report(self) -> None:
        if self.rear_scan is None:
            raise RuntimeError('rear scan unavailable')
        points = self.scan_endpoints(self.rear_scan, 'base_link')
        bins = {'inside': 0, '<0.10': 0, '0.10~0.30': 0, '0.30~0.50': 0}
        close = []
        for point in points:
            inside, gap = self.rectangle_gap(point.x, point.y)
            if inside:
                bins['inside'] += 1
                close.append((gap, point, 'inside'))
            elif gap < 0.10:
                bins['<0.10'] += 1
                close.append((gap, point, '<0.10'))
            elif gap < 0.30:
                bins['0.10~0.30'] += 1
                close.append((gap, point, '0.10~0.30'))
            elif gap < 0.50:
                bins['0.30~0.50'] += 1
                close.append((gap, point, '0.30~0.50'))
        close.sort(key=lambda item: (item[0], item[1].angle))
        print(
            '[REAR SELF-RETURN]\n'
            f'finite_endpoints={len(points)}\n'
            f'padded_footprint={PADDED_FOOTPRINT}\n'
            f'inside_padded_footprint={bins["inside"]}\n'
            f'distance_lt_0.10={bins["<0.10"]}\n'
            f'distance_0.10_0.30={bins["0.10~0.30"]}\n'
            f'distance_0.30_0.50={bins["0.30~0.50"]}', flush=True)
        for gap, point, band in close[:30]:
            print(
                'self_return_candidate '
                f'band={band} base=({point.x:.6f},{point.y:.6f}) '
                f'angle={point.angle:.6f}rad '
                f'range={point.distance:.6f}m '
                f'footprint_signed_gap={gap:.6f}m', flush=True)

    def collision_and_source_report(
            self, start: Tuple[float, float, float]) -> None:
        if (self.global_costmap is None or self.front_scan is None
                or self.rear_scan is None):
            raise RuntimeError('costmap or scans unavailable')
        grid = self.global_costmap
        poses = self.interpolated_poses(start)
        first = None
        for index, (x, y, yaw) in enumerate(poses):
            result = self.inspect_footprint(grid, x, y, yaw)
            if result.collision:
                first = (index, x, y, yaw, result)
                break
        if first is None:
            index = len(poses) - 1
            x, y, yaw = poses[-1]
            result = self.inspect_footprint(grid, x, y, yaw)
        else:
            index, x, y, yaw, result = first
        print(
            '[START-STAGING COLLISION]\n'
            f'index={index}\n'
            f'interpolation_poses={len(poses)}\n'
            f'max_step={START_TO_STAGING_MAX_STEP_M:.3f}m\n'
            f'x={x:.6f}\n'
            f'y={y:.6f}\n'
            f'yaw={yaw:.6f}\n'
            f'center_cost={result.center_cost}\n'
            f'footprint_max_cost={result.max_cost}\n'
            f'cost253={result.counts[253]}\n'
            f'cost254={result.counts[254]}\n'
            f'cost255={result.counts[255]}\n'
            f'static_occupied={result.static_occupied}\n'
            f'collision={str(result.collision).lower()}\n'
            'collision cell coordinates=' + (
                ', '.join(
                    f'({item[2]:.6f},{item[3]:.6f})'
                    for item in result.collision_cells)
                if result.collision_cells else 'none'), flush=True)

        front = self.scan_endpoints(self.front_scan, 'map')
        rear = self.scan_endpoints(self.rear_scan, 'map')
        resolution = grid.metadata.resolution
        for _, _, cell_x, cell_y, cost, static in result.collision_cells:
            front_distance, front_endpoint = self.nearest_endpoint(
                cell_x, cell_y, front)
            rear_distance, rear_endpoint = self.nearest_endpoint(
                cell_x, cell_y, rear)
            match_distance = 2.0 * resolution
            if static is not None and static >= 50:
                source = 'UNKNOWN'
            elif min(front_distance, rear_distance) <= match_distance:
                source = 'FRONT' if front_distance <= rear_distance else 'REAR'
            elif min(front_distance, rear_distance) > 3.0 * resolution:
                source = 'STALE'
            else:
                source = 'UNKNOWN'
            front_text = (
                f'({front_endpoint.x:.6f},{front_endpoint.y:.6f})'
                if front_endpoint is not None else 'none')
            rear_text = (
                f'({rear_endpoint.x:.6f},{rear_endpoint.y:.6f})'
                if rear_endpoint is not None else 'none')
            print(
                '[LETHAL SOURCE DEBUG]\n'
                f'cell=({cell_x:.6f},{cell_y:.6f})\n'
                f'static_map={static}\n'
                f'cost={cost}\n'
                f'nearest_front_scan_endpoint_distance={front_distance:.6f}\n'
                f'nearest_front_endpoint={front_text}\n'
                f'nearest_rear_scan_endpoint_distance={rear_distance:.6f}\n'
                f'nearest_rear_endpoint={rear_text}\n'
                f'probable_source={source}', flush=True)

    def road_lethal_field_report(self) -> None:
        """Report every road lethal cell near the common parking approach."""
        if (self.global_costmap is None or self.front_scan is None
                or self.rear_scan is None):
            raise RuntimeError('costmap or scans unavailable')
        grid = self.global_costmap
        front = self.scan_endpoints(self.front_scan, 'map')
        rear = self.scan_endpoints(self.rear_scan, 'map')
        cells = []
        for row in range(grid.metadata.size_y):
            for col in range(grid.metadata.size_x):
                if self.cost_value(grid, col, row) != 254:
                    continue
                x, y = self.grid_world(
                    grid.metadata.origin, grid.metadata.resolution, col, row)
                if not (5.5 <= x <= 7.5 and -2.16 <= y <= 2.16):
                    continue
                front_distance, front_endpoint = self.nearest_endpoint(x, y, front)
                rear_distance, rear_endpoint = self.nearest_endpoint(x, y, rear)
                source = (
                    'FRONT' if front_distance <= rear_distance
                    else 'REAR')
                cells.append((
                    x, y, self.occupancy_at(x, y), source,
                    front_distance, rear_distance, front_endpoint,
                    rear_endpoint))
        print(
            '[ROAD LETHAL FIELD]\n'
            f'cell_count={len(cells)}', flush=True)
        for (x, y, static, source, front_distance, rear_distance,
             front_endpoint, rear_endpoint) in cells:
            nearest = front_endpoint if source == 'FRONT' else rear_endpoint
            local_text = (
                f'angle={nearest.angle:.6f} range={nearest.distance:.6f}'
                if nearest is not None else 'angle=none range=none')
            print(
                f'cell=({x:.6f},{y:.6f}) static={static} '
                f'front_distance={front_distance:.6f} '
                f'rear_distance={rear_distance:.6f} source={source} '
                f'{local_text}', flush=True)

    def planner_distance_steps(self) -> None:
        if not self.plan_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('ComputePathToPose action unavailable')
        error_names = (
            'NONE', 'UNKNOWN', 'INVALID_PLANNER', 'TF_ERROR',
            'START_OUTSIDE_MAP', 'GOAL_OUTSIDE_MAP', 'START_OCCUPIED',
            'GOAL_OCCUPIED', 'TIMEOUT', 'NO_VALID_PATH')
        for goal_x in (9.5, 9.0, 8.5, 8.0, 7.80, 7.70, 7.65, 7.60,
                       7.55, STAGING_X):
            goal = ComputePathToPose.Goal()
            goal.goal.header.frame_id = 'map'
            goal.goal.header.stamp = self.get_clock().now().to_msg()
            goal.goal.pose.position.x = goal_x
            goal.goal.pose.position.y = STAGING_Y
            set_pose_yaw(goal.goal.pose, ROAD_YAW)
            goal.planner_id = 'GridBased'
            goal.use_start = False
            started = time.monotonic()
            send_future = self.plan_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
            goal_handle = send_future.result() if send_future.done() else None
            if goal_handle is None or not goal_handle.accepted:
                print(
                    f'[PLANNER STEP] goal_x={goal_x:.2f} result=FAIL '
                    'error_code=UNAVAILABLE error_msg=goal_not_accepted '
                    f'planning_time={time.monotonic() - started:.6f}s '
                    'poses=0 path_length=0.000000m', flush=True)
                continue
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=50.0)
            wrapped = result_future.result() if result_future.done() else None
            if wrapped is None:
                print(
                    f'[PLANNER STEP] goal_x={goal_x:.2f} result=FAIL '
                    'error_code=TIMEOUT error_msg=result_timeout '
                    f'planning_time={time.monotonic() - started:.6f}s '
                    'poses=0 path_length=0.000000m', flush=True)
                continue
            result = wrapped.result
            error_name = next(
                (name for name in error_names
                 if result.error_code == getattr(
                     ComputePathToPose.Result, name, -1)),
                'UNRECOGNIZED')
            succeeded = (
                wrapped.status == GoalStatus.STATUS_SUCCEEDED
                and result.error_code == ComputePathToPose.Result.NONE
                and bool(result.path.poses))
            planning_seconds = (
                result.planning_time.sec
                + result.planning_time.nanosec * 1.0e-9)
            print(
                f'[PLANNER STEP] goal_x={goal_x:.2f} '
                f'result={"SUCCESS" if succeeded else "FAIL"} '
                f'error_code={error_name}({result.error_code}) '
                f'error_msg={result.error_msg or "<empty>"} '
                f'planning_time={planning_seconds:.6f}s '
                f'wall_time={time.monotonic() - started:.6f}s '
                f'poses={len(result.path.poses)} '
                f'path_length={path_length(result.path.poses):.6f}m',
                flush=True)

    def through_pose_candidate_steps(self) -> None:
        """Run the exact 12 parking requests with a collision-free staging."""
        if not self.through_plan_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('ComputePathThroughPoses action unavailable')
        error_names = (
            'NONE', 'UNKNOWN', 'INVALID_PLANNER', 'TF_ERROR',
            'START_OUTSIDE_MAP', 'GOAL_OUTSIDE_MAP', 'START_OCCUPIED',
            'GOAL_OCCUPIED', 'TIMEOUT', 'NO_VALID_PATH')
        for offset in (0.45, 0.30, 0.15, 0.00):
            for depth in (0.10, 0.20, 0.30):
                goal = ComputePathThroughPoses.Goal()
                goal.planner_id = 'GridBased'
                goal.use_start = False
                coordinates = (
                    (7.65, 0.0, ROAD_YAW),
                    (3.25 + offset, 0.0, ROAD_YAW),
                    (5.10, -1.25 - depth, math.pi / 2.0),
                    (5.10, -7.455, math.pi / 2.0),
                )
                for x, y, yaw in coordinates:
                    pose = PoseStamped()
                    pose.header.frame_id = 'map'
                    pose.header.stamp = self.get_clock().now().to_msg()
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    set_pose_yaw(pose.pose, yaw)
                    goal.goals.append(pose)
                started = time.monotonic()
                send_future = self.through_plan_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(
                    self, send_future, timeout_sec=5.0)
                goal_handle = send_future.result() if send_future.done() else None
                if goal_handle is None or not goal_handle.accepted:
                    print(
                        f'[THROUGH CANDIDATE] offset={offset:.2f} '
                        f'depth={depth:.2f} result=FAIL '
                        'error_code=UNAVAILABLE error_msg=goal_not_accepted '
                        'poses=0 path_length=0.000000m', flush=True)
                    continue
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(
                    self, result_future, timeout_sec=50.0)
                wrapped = result_future.result() if result_future.done() else None
                if wrapped is None:
                    print(
                        f'[THROUGH CANDIDATE] offset={offset:.2f} '
                        f'depth={depth:.2f} result=FAIL error_code=TIMEOUT '
                        'error_msg=result_timeout poses=0 '
                        'path_length=0.000000m', flush=True)
                    continue
                result = wrapped.result
                error_name = next(
                    (name for name in error_names
                     if result.error_code == getattr(
                         ComputePathThroughPoses.Result, name, -1)),
                    'UNRECOGNIZED')
                succeeded = (
                    wrapped.status == GoalStatus.STATUS_SUCCEEDED
                    and result.error_code == ComputePathThroughPoses.Result.NONE
                    and bool(result.path.poses))
                print(
                    f'[THROUGH CANDIDATE] offset={offset:.2f} '
                    f'depth={depth:.2f} '
                    f'result={"SUCCESS" if succeeded else "FAIL"} '
                    f'error_code={error_name}({result.error_code}) '
                    f'error_msg={result.error_msg or "<empty>"} '
                    f'wall_time={time.monotonic() - started:.6f}s '
                    f'poses={len(result.path.poses)} '
                    f'path_length={path_length(result.path.poses):.6f}m',
                    flush=True)

    def run(self) -> None:
        self.wait_for_inputs()
        start = self.wait_for_localization_stability()
        self.clear_after_stability_and_wait_fresh()
        # Capture the post-clear TF pose.  It is the same actual live start
        # that the planner uses with use_start=false.
        start = self.current_pose()
        self.snapshot_persistence(start)
        self.rear_self_return_report()
        self.collision_and_source_report(start)
        self.road_lethal_field_report()
        self.planner_distance_steps()
        self.through_pose_candidate_steps()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StartStagingDiagnostics()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
