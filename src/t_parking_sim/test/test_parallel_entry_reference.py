"""Deterministic tests for rear-axle and vehicle-centre conversion."""

import importlib.util
import math
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / 'scripts' / 'parallel_entry_reference.py')
SPEC = importlib.util.spec_from_file_location(
    'parallel_entry_reference', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rear_axle_to_center_at_zero_yaw():
    """Half of the configured wheelbase lies forward at zero yaw."""
    assert MODULE.rear_axle_to_vehicle_center(
        (0.0, 0.0, 0.0), 0.73) == pytest.approx((0.365, 0.0, 0.0))


def test_rear_axle_to_center_at_positive_quarter_turn():
    """Half of the configured wheelbase lies left at positive 90 degrees."""
    assert MODULE.rear_axle_to_vehicle_center(
        (0.0, 0.0, 0.5 * math.pi), 0.73
    ) == pytest.approx((0.0, 0.365, 0.5 * math.pi), abs=1.0e-12)


@pytest.mark.parametrize('wheel_base', [0.50, 0.73, 1.20])
def test_center_rear_center_round_trip_uses_supplied_wheelbase(wheel_base):
    """Round trips remain exact for geometry other than the current car."""
    center = (5.595, 4.775, -0.63)
    rear = MODULE.vehicle_center_to_rear_axle(center, wheel_base)
    restored = MODULE.rear_axle_to_vehicle_center(rear, wheel_base)
    assert restored == pytest.approx(center, abs=1.0e-12)
