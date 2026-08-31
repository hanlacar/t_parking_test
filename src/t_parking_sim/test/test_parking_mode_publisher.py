#!/usr/bin/env python3
"""Unit tests for the Gazebo parking-mode lifecycle policy."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'parking_mode_publisher.py'
SPEC = importlib.util.spec_from_file_location('parking_mode_publisher', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_waiting_and_terminal_statuses_restore_non_parking_mode():
    """Waiting and every terminal outcome must restore normal mode."""
    for status in (
            'WAITING', 'FINISHED', 'PARKING_SUCCESS: slot=slot_1',
            'SUCCESS', 'FAILED: entry FollowPath failed', 'CANCELLED'):
        assert MODULE.mode_for_status(
            status, 'PARALLEL_PARK', 'NORMAL') == 'NORMAL'


def test_active_state_machine_statuses_grant_parallel_parking_mode():
    """Every active state-machine phase must grant parallel mode."""
    for status in (
            'WAIT_SYSTEM', 'SELECT_SLOT', 'DRIVING_TO_APPROACH',
            'EXECUTE_ENTRY', 'CONFIRM_PARKED', 'STOP'):
        assert MODULE.mode_for_status(
            status, 'PARALLEL_PARK', 'NORMAL') == 'PARALLEL_PARK'
