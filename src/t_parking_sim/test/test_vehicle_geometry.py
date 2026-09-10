"""Regression checks for the measured real-vehicle geometry contract."""

import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest
import yaml


PACKAGE = Path(__file__).parents[1]
XACRO = PACKAGE / 'urdf' / 'turtle_car.urdf.xacro'


def _expanded_urdf(*arguments):
    executable = shutil.which('xacro')
    if executable is None:
        pytest.skip('xacro is not installed')
    completed = subprocess.run(
        [executable, str(XACRO), *arguments],
        check=True, capture_output=True, text=True)
    return ET.fromstring(completed.stdout)


def _joint(root, name):
    return root.find(f"./joint[@name='{name}']")


def _xyz(joint):
    return tuple(float(value) for value in joint.find('origin').attrib['xyz'].split())


def test_xacro_matches_measured_axles_tracks_wheels_and_steering():
    root = _expanded_urdf('include_gazebo:=false')
    assert _xyz(_joint(root, 'front_left_steering_joint')) == pytest.approx(
        (0.365, 0.3875, 0.0))
    assert _xyz(_joint(root, 'front_right_steering_joint')) == pytest.approx(
        (0.365, -0.3875, 0.0))
    assert _xyz(_joint(root, 'rear_left_wheel_joint')) == pytest.approx(
        (-0.365, 0.3925, 0.0))
    assert _xyz(_joint(root, 'rear_right_wheel_joint')) == pytest.approx(
        (-0.365, -0.3925, 0.0))

    limit = _joint(root, 'front_left_steering_joint').find('limit').attrib
    assert float(limit['lower']) == pytest.approx(-math.radians(22.0))
    assert float(limit['upper']) == pytest.approx(math.radians(22.0))
    wheel = root.find("./link[@name='front_left_wheel_link']/visual/geometry/cylinder")
    assert float(wheel.attrib['radius']) == pytest.approx(0.135)


def test_lidar_transforms_are_relative_to_base_link_without_double_z_offset():
    root = _expanded_urdf('include_gazebo:=false', 'use_base_footprint:=false')
    assert _joint(root, 'laser_joint').find('parent').attrib['link'] == 'base_link'
    assert _xyz(_joint(root, 'laser_joint')) == pytest.approx(
        (0.730, 0.0, -0.030))
    rear = _joint(root, 'rear_laser_joint')
    assert rear.find('parent').attrib['link'] == 'base_link'
    assert _xyz(rear) == pytest.approx((-0.680, 0.0, 0.020))
    rear_rpy = tuple(
        float(value) for value in rear.find('origin').attrib['rpy'].split())
    assert rear_rpy == pytest.approx((0.0, 0.0, math.pi))
    assert root.find("./link[@name='base_footprint']") is None


def test_sim_base_link_height_and_planner_geometry_are_consistent():
    root = _expanded_urdf('include_gazebo:=false')
    assert _xyz(_joint(root, 'base_footprint_joint')) == pytest.approx(
        (0.0, 0.0, 0.135))

    with (PACKAGE / 'config' / 't_parking_auto.yaml').open() as stream:
        parking = yaml.safe_load(stream)['t_parking_auto']['ros__parameters']
    with (PACKAGE / 'config' / 'nav2_params.yaml').open() as stream:
        nav2 = yaml.safe_load(stream)
    planner = nav2['planner_server']['ros__parameters']
    reverse = nav2['controller_server']['ros__parameters']['ParkingReverse']

    assert parking['wheel_base'] == 0.73
    assert parking['front_wheel_track'] == 0.775
    assert parking['rear_wheel_track'] == 0.785
    assert parking['wheel_radius'] == 0.135
    assert parking['maximum_command_steering_deg'] == 22.0
    assert parking['minimum_turning_radius'] == 1.82
    assert parking['exit_planning_turn_radius'] == 2.30
    assert planner['GridBased']['minimum_turning_radius'] == 1.82
    assert planner['ForwardExit']['minimum_turning_radius'] == 2.30
    assert reverse['reverse_wheel_base'] == 0.73
    assert reverse['reverse_hard_steering_limit_deg'] == 22.0
    assert 0.73 / math.tan(math.radians(22.0)) < 1.82
