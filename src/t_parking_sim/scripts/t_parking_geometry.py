#!/usr/bin/env python3
"""Pure geometry helpers shared by T-parking planning and tests."""

from dataclasses import dataclass
import math
from typing import List, Tuple


@dataclass(frozen=True)
class ParkingPoseAssessment:
    """Clearances of one vehicle footprint relative to an axis-aligned bay."""

    footprint_inside: bool
    end_clearance: float
    front_clearance: float
    side_clearance: float
    slot_depth: float
    front_extent: float
    rear_extent: float


def vehicle_longitudinal_extents(
        vehicle_length: float,
        vehicle_center_x_offset: float) -> Tuple[float, float]:
    """Return physical front/rear bumper distances from the base origin."""
    if vehicle_length <= 0.0:
        raise ValueError('vehicle_length must be positive')
    half_length = 0.5 * vehicle_length
    front_extent = half_length + vehicle_center_x_offset
    rear_extent = half_length - vehicle_center_x_offset
    if front_extent <= 0.0 or rear_extent <= 0.0:
        raise ValueError('vehicle_center_x_offset lies outside the vehicle')
    return front_extent, rear_extent


def _axes(yaw: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    forward = (math.cos(yaw), math.sin(yaw))
    left = (-forward[1], forward[0])
    return forward, left


def _slot_corners(
        min_x: float, max_x: float,
        min_y: float, max_y: float) -> List[Tuple[float, float]]:
    if min_x >= max_x or min_y >= max_y:
        raise ValueError('slot bounds must have positive area')
    return [
        (min_x, min_y), (min_x, max_y),
        (max_x, min_y), (max_x, max_y),
    ]


def _projection_bounds(
        points: List[Tuple[float, float]],
        axis: Tuple[float, float]) -> Tuple[float, float]:
    values = [x * axis[0] + y * axis[1] for x, y in points]
    return min(values), max(values)


def parking_target_from_end_clearance(
        *,
        slot_center_x: float,
        slot_center_y: float,
        slot_yaw: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        vehicle_length: float,
        vehicle_center_x_offset: float,
        parking_end_clearance_m: float) -> Tuple[float, float]:
    """
    Place the base origin at an exact rear-bumper-to-end clearance.

    ``slot_yaw`` points from the closed end of the bay toward its mouth.  The
    vehicle faces that direction and reverses into the bay, so its rear bumper
    is the footprint edge nearest the minimum longitudinal slot projection.
    """
    if parking_end_clearance_m < 0.0:
        raise ValueError('parking_end_clearance_m must be non-negative')
    front_extent, rear_extent = vehicle_longitudinal_extents(
        vehicle_length, vehicle_center_x_offset)
    forward, left = _axes(slot_yaw)
    corners = _slot_corners(min_x, max_x, min_y, max_y)
    end_projection, mouth_projection = _projection_bounds(corners, forward)
    slot_depth = mouth_projection - end_projection
    maximum_clearance = slot_depth - front_extent - rear_extent
    if parking_end_clearance_m > maximum_clearance + 1.0e-9:
        raise ValueError(
            'parking_end_clearance_m leaves the front bumper outside the slot: '
            f'clearance={parking_end_clearance_m:.3f}m '
            f'maximum={maximum_clearance:.3f}m')

    base_projection = (
        end_projection + rear_extent + parking_end_clearance_m)
    lateral_projection = (
        slot_center_x * left[0] + slot_center_y * left[1])
    return (
        base_projection * forward[0] + lateral_projection * left[0],
        base_projection * forward[1] + lateral_projection * left[1],
    )


def footprint_corners(
        *,
        base_x: float,
        base_y: float,
        vehicle_yaw: float,
        vehicle_length: float,
        vehicle_width: float,
        vehicle_center_x_offset: float) -> List[Tuple[float, float]]:
    """Return the four physical footprint corners in the world frame."""
    if vehicle_width <= 0.0:
        raise ValueError('vehicle_width must be positive')
    front_extent, rear_extent = vehicle_longitudinal_extents(
        vehicle_length, vehicle_center_x_offset)
    half_width = 0.5 * vehicle_width
    cosine = math.cos(vehicle_yaw)
    sine = math.sin(vehicle_yaw)
    local = [
        (front_extent, half_width),
        (front_extent, -half_width),
        (-rear_extent, -half_width),
        (-rear_extent, half_width),
    ]
    return [
        (base_x + cosine * x - sine * y,
         base_y + sine * x + cosine * y)
        for x, y in local
    ]


def assess_parking_pose(
        *,
        base_x: float,
        base_y: float,
        vehicle_yaw: float,
        slot_yaw: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        vehicle_length: float,
        vehicle_width: float,
        vehicle_center_x_offset: float,
        tolerance: float = 1.0e-9) -> ParkingPoseAssessment:
    """Measure end, mouth, and side clearance for the whole footprint."""
    front_extent, rear_extent = vehicle_longitudinal_extents(
        vehicle_length, vehicle_center_x_offset)
    forward, left = _axes(slot_yaw)
    slot_corners = _slot_corners(min_x, max_x, min_y, max_y)
    end_projection, mouth_projection = _projection_bounds(
        slot_corners, forward)
    slot_left_min, slot_left_max = _projection_bounds(slot_corners, left)
    vehicle_corners = footprint_corners(
        base_x=base_x,
        base_y=base_y,
        vehicle_yaw=vehicle_yaw,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        vehicle_center_x_offset=vehicle_center_x_offset)
    vehicle_forward_min, vehicle_forward_max = _projection_bounds(
        vehicle_corners, forward)
    vehicle_left_min, vehicle_left_max = _projection_bounds(
        vehicle_corners, left)
    inside = all(
        min_x - tolerance <= x <= max_x + tolerance
        and min_y - tolerance <= y <= max_y + tolerance
        for x, y in vehicle_corners)
    return ParkingPoseAssessment(
        footprint_inside=inside,
        end_clearance=vehicle_forward_min - end_projection,
        front_clearance=mouth_projection - vehicle_forward_max,
        side_clearance=min(
            vehicle_left_min - slot_left_min,
            slot_left_max - vehicle_left_max),
        slot_depth=mouth_projection - end_projection,
        front_extent=front_extent,
        rear_extent=rear_extent,
    )
