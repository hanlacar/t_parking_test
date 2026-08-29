#!/usr/bin/env python3
"""Pure helpers shared by the isolated wheels-off-ground BENCH tools."""

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple


BENCH_NAV_NODES = (
    '/planner_server',
    '/controller_server',
    '/lifecycle_manager_navigation',
)
MCU_NODES = ('/mcu_bridge', '/mcu_manager')
BENCH_ODOM_TOPIC = '/bench/odom'
BENCH_ODOM_FRAME = 'bench_odom'
BENCH_BASE_FRAME = 'bench_base_link'


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class BicycleStep:
    pose: Pose2D
    linear_velocity: float
    yaw_rate: float
    raw_delta_deg: float
    clamped_delta_deg: float


@dataclass(frozen=True)
class StartupSnapshot:
    drive_publishers: int
    wheel_publishers: int
    drive_zero_samples: int
    wheel_zero_samples: int
    nonzero_command_seen: bool
    mcu_connected: bool
    mcu_ready: bool
    vehicle_mode: str
    safety_state: str


def startup_gate_state(
        snapshot: StartupSnapshot,
        zero_samples_required: int = 3) -> str:
    """Return the fail-closed BENCH startup state for one observation."""
    if snapshot.drive_publishers > 1 or snapshot.wheel_publishers > 1:
        return 'FAIL_DUPLICATE_LIDAR_COMMAND_SOURCE'
    if snapshot.nonzero_command_seen:
        return 'FAIL_NONZERO_COMMAND_BEFORE_START'
    if snapshot.drive_publishers < 1 or snapshot.wheel_publishers < 1:
        return 'WAITING_FOR_LIDAR_COMMAND_SOURCE'
    if (snapshot.drive_zero_samples < zero_samples_required
            or snapshot.wheel_zero_samples < zero_samples_required):
        return 'WAITING_FOR_ZERO_COMMAND'
    if not (
            snapshot.mcu_connected
            and snapshot.vehicle_mode == 'T_PARK'
            and snapshot.safety_state == 'OK'):
        return 'WAITING_FOR_MCU_STATUS'
    return 'READY'


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def compose(left: Pose2D, right: Pose2D) -> Pose2D:
    cosine = math.cos(left.yaw)
    sine = math.sin(left.yaw)
    return Pose2D(
        left.x + cosine * right.x - sine * right.y,
        left.y + sine * right.x + cosine * right.y,
        normalize_angle(left.yaw + right.yaw),
    )


def inverse(pose: Pose2D) -> Pose2D:
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    return Pose2D(
        -cosine * pose.x - sine * pose.y,
        sine * pose.x - cosine * pose.y,
        normalize_angle(-pose.yaw),
    )


def compute_map_to_odom(
        odom_to_base_start: Pose2D,
        desired_map_to_base_start: Pose2D) -> Pose2D:
    """Return T_map_odom = T_map_start * inverse(T_odom_base_start)."""
    return compose(desired_map_to_base_start, inverse(odom_to_base_start))


def bench_motion_allowed(
        bench_mode: bool, wheels_off_ground: bool, execute: bool) -> bool:
    """Actual motion is possible only under the explicit three-way gate."""
    return bool(bench_mode and wheels_off_ground and execute)


def integrate_bicycle(
        pose: Pose2D,
        linear_x: float,
        angular_z: float,
        dt: float,
        wheel_base: float = 0.77,
        steering_limit_deg: float = 27.0,
        stopped_speed_epsilon: float = 0.01,
        motion_allowed: bool = True) -> BicycleStep:
    """Integrate one converter-compatible Ackermann bicycle-model step."""
    if (not motion_allowed or dt <= 0.0
            or not math.isfinite(linear_x)
            or not math.isfinite(angular_z)
            or abs(linear_x) < stopped_speed_epsilon):
        return BicycleStep(pose, 0.0, 0.0, 0.0, 0.0)
    if wheel_base <= 0.0 or steering_limit_deg <= 0.0:
        raise ValueError('wheel_base and steering_limit_deg must be positive')

    raw_delta = math.atan(wheel_base * angular_z / linear_x)
    limit = math.radians(steering_limit_deg)
    clamped_delta = max(-limit, min(limit, raw_delta))
    yaw_rate = linear_x / wheel_base * math.tan(clamped_delta)
    result = Pose2D(
        pose.x + linear_x * math.cos(pose.yaw) * dt,
        pose.y + linear_x * math.sin(pose.yaw) * dt,
        normalize_angle(pose.yaw + yaw_rate * dt),
    )
    return BicycleStep(
        result,
        linear_x,
        yaw_rate,
        math.degrees(raw_delta),
        math.degrees(clamped_delta),
    )


def fully_qualified_node_name(name: str, namespace: str) -> str:
    namespace = namespace.rstrip('/')
    if not namespace:
        namespace = '/'
    if namespace == '/':
        return '/' + name.lstrip('/')
    return namespace + '/' + name.lstrip('/')


def count_nodes(
        nodes: Iterable[Tuple[str, str]], targets: Sequence[str]) -> dict:
    counts = {target: 0 for target in targets}
    for name, namespace in nodes:
        full_name = fully_qualified_node_name(name, namespace)
        if full_name in counts:
            counts[full_name] += 1
    return counts


def duplicate_nav_nodes(
        nodes: Iterable[Tuple[str, str]],
        expected_count: int = 1) -> List[str]:
    counts = count_nodes(nodes, BENCH_NAV_NODES)
    return [name for name, count in counts.items() if count != expected_count]


def duplicate_node_names(
        nodes: Iterable[Tuple[str, str]],
        targets: Sequence[str]) -> List[str]:
    """Return only node names with more than one graph owner."""
    counts = count_nodes(nodes, targets)
    return [name for name, count in counts.items() if count > 1]


def preflight_graph_conflicts(
        nodes: Iterable[Tuple[str, str]]) -> List[str]:
    """Find graph owners which must not pre-exist before BENCH starts."""
    full_names = [fully_qualified_node_name(*node) for node in nodes]
    conflicts = [name for name in BENCH_NAV_NODES if name in full_names]
    conflicts.extend(name for name in full_names if name == '/amcl')
    return sorted(set(conflicts))
