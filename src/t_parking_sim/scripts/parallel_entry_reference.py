"""Reference conversions for the parallel-in-T entry trajectory."""

import math
from typing import Tuple


Pose2D = Tuple[float, float, float]


def rear_axle_to_vehicle_center(
        pose: Pose2D, wheel_base: float) -> Pose2D:
    """Convert a rear-axle bicycle pose to the base-footprint centre."""
    if not math.isfinite(wheel_base) or wheel_base <= 0.0:
        raise ValueError('wheel_base must be finite and positive')
    x, y, yaw = pose
    rear_to_center = 0.5 * wheel_base
    return (
        x + rear_to_center * math.cos(yaw),
        y + rear_to_center * math.sin(yaw),
        yaw,
    )


def vehicle_center_to_rear_axle(
        pose: Pose2D, wheel_base: float) -> Pose2D:
    """Convert a base-footprint centre pose to the rear bicycle state."""
    if not math.isfinite(wheel_base) or wheel_base <= 0.0:
        raise ValueError('wheel_base must be finite and positive')
    x, y, yaw = pose
    rear_to_center = 0.5 * wheel_base
    return (
        x - rear_to_center * math.cos(yaw),
        y - rear_to_center * math.sin(yaw),
        yaw,
    )
