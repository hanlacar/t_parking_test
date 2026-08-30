"""Protect the Gazebo-to-Nav2 front and rear LaserScan contract."""

import importlib.util
import os
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
from launch_ros.actions import SetRemap
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('ROS_LOG_DIR', '/tmp/t_parking_launch_tests')


def _load_launch(filename):
    path = PACKAGE_ROOT / 'launch' / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def _walk(actions):
    for action in actions:
        yield action
        if hasattr(action, 'get_sub_entities'):
            children = action.get_sub_entities()
        else:
            children = getattr(action, 'entities', ())
        if not children and hasattr(action, 'actions'):
            children = action.actions
        yield from _walk(children)


def _launch_defaults(description):
    context = LaunchContext()
    defaults = {}
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            defaults[action.name] = perform_substitutions(
                context, action.default_value)
    return defaults


def _scan_remaps(description):
    context = LaunchContext()
    context.launch_configurations.update(_launch_defaults(description))
    return {
        perform_substitutions(context, action.src):
        perform_substitutions(context, action.dst)
        for action in _walk(description.entities)
        if isinstance(action, SetRemap)
    }


def test_gazebo_bridge_publishes_raw_front_and_rear_scans():
    bridge = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'bridge.yaml').read_text())
    laser_bridges = {
        item['ros_topic_name']: item
        for item in bridge
        if item['ros_type_name'] == 'sensor_msgs/msg/LaserScan'
    }

    assert laser_bridges['/scan']['gz_topic_name'] == '/scan'
    assert laser_bridges['/scan']['frame_id'] == 'laser_link'
    assert laser_bridges['/scan_rear']['gz_topic_name'] == '/scan_rear'
    assert laser_bridges['/scan_rear']['frame_id'] == 'rear_laser_link'


def test_nav2_keeps_hardware_canonical_scan_names():
    nav2 = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'nav2_params.yaml').read_text())
    assert nav2['amcl']['ros__parameters']['scan_topic'] == '/scan_front'

    for costmap_name in ('local_costmap', 'global_costmap'):
        obstacle = nav2[costmap_name][costmap_name][
            'ros__parameters']['obstacle_layer']
        assert obstacle['observation_sources'] == 'scan_front scan_rear'
        assert obstacle['scan_front']['topic'] == '/scan_front'
        assert obstacle['scan_rear']['topic'] == '/scan_rear'


def test_gazebo_nav_launches_alias_canonical_topics_to_publishers():
    for filename in ('saved_map_nav.launch.py', 'nav2_mapping.launch.py'):
        description = _load_launch(filename)
        defaults = _launch_defaults(description)
        remaps = _scan_remaps(description)

        assert defaults['front_scan_topic'] == '/scan'
        assert defaults['rear_scan_topic'] == '/scan_rear'
        assert remaps['/scan_front'] == '/scan'
        assert remaps['/scan_rear'] == '/scan_rear'


def test_parking_launches_expose_the_same_gazebo_scan_defaults():
    for filename in (
            'auto_t_parking.launch.py',
            'auto_parallel_in_t_slot.launch.py'):
        defaults = _launch_defaults(_load_launch(filename))
        assert defaults['front_scan_topic'] == '/scan'
        assert defaults['rear_scan_topic'] == '/scan_rear'
        if filename == 'auto_parallel_in_t_slot.launch.py':
            assert defaults['spawn_parking_obstacles'] == 'true'
            assert defaults['parking_obstacle_seed'] == '-1'
