"""Launch saved-map parallel parking against real MCU and LiDAR only."""

import math
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _as_bool(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in (
        '1', 'true', 'yes', 'on')


def _validate_hardware_launch(context):
    if LaunchConfiguration('map_mode').perform(context) != 'saved':
        raise RuntimeError(
            'real_parallel_parking supports map_mode:=saved only')
    if _as_bool(context, 'use_sim_time'):
        raise RuntimeError(
            'real_parallel_parking is hardware-only and requires '
            'use_sim_time:=false')

    requested = LaunchConfiguration('map').perform(context)
    map_yaml = os.path.abspath(os.path.expandvars(os.path.expanduser(requested)))
    if not os.path.isfile(map_yaml):
        raise RuntimeError(f'Saved map YAML does not exist: {map_yaml}')

    image_name = None
    with open(map_yaml, encoding='utf-8') as stream:
        for line in stream:
            key, separator, value = line.partition(':')
            if separator and key.strip() == 'image':
                image_name = value.strip().strip('"\'')
                break
    if not image_name:
        raise RuntimeError(f'Saved map YAML has no image entry: {map_yaml}')
    image_path = os.path.expandvars(os.path.expanduser(image_name))
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml), image_path)
    if not os.path.isfile(image_path):
        raise RuntimeError(
            f'Saved map image does not exist: {os.path.abspath(image_path)}')

    for name in ('front_scan_topic', 'rear_scan_topic'):
        if not LaunchConfiguration(name).perform(context).strip():
            raise RuntimeError(f'{name} must not be empty')
    return [SetLaunchConfiguration('map', map_yaml)]


def _read_vehicle_geometry():
    xacro_path = os.path.join(
        get_package_share_directory('t_parking_sim'),
        'urdf', 'turtle_car.urdf.xacro')
    root = ET.parse(xacro_path).getroot()
    xacro_property = '{http://www.ros.org/wiki/xacro}property'
    properties = {
        element.attrib['name']: element.attrib['value']
        for element in root.iter(xacro_property)
        if 'name' in element.attrib and 'value' in element.attrib
    }
    try:
        return (
            float(properties['wheel_base']),
            math.degrees(float(properties['steering_limit'])))
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f'Cannot read vehicle geometry from {xacro_path}') from exc


def generate_launch_description():
    package_share = FindPackageShare('t_parking_sim')
    wheel_base, steering_limit_deg = _read_vehicle_geometry()

    map_yaml = LaunchConfiguration('map')
    use_sim_time = ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)
    front_scan = LaunchConfiguration('front_scan_topic')
    rear_scan = LaunchConfiguration('rear_scan_topic')
    initial_x = LaunchConfiguration('initial_pose_x')
    initial_y = LaunchConfiguration('initial_pose_y')
    initial_yaw = LaunchConfiguration('initial_pose_yaw')
    auto_start = LaunchConfiguration('auto_start')
    execute = LaunchConfiguration('execute')

    nav2_params = PathJoinSubstitution(
        [package_share, 'config', 'nav2_params.yaml'])
    parking_params = PathJoinSubstitution(
        [package_share, 'config', 'parallel_parking_auto.yaml'])
    default_map = PathJoinSubstitution([
        package_share, 'maps', 'combined_parking_map_real_vehicle.yaml'])
    rviz_config = PathJoinSubstitution(
        [package_share, 'rviz', 'nav2_mapping.rviz'])

    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        output='screen', parameters=[{
            'use_sim_time': use_sim_time,
            'yaml_filename': map_yaml,
        }])

    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl',
        output='screen',
        parameters=[nav2_params, {
            'use_sim_time': use_sim_time,
            'initial_pose.x': ParameterValue(initial_x, value_type=float),
            'initial_pose.y': ParameterValue(initial_y, value_type=float),
            'initial_pose.yaw': ParameterValue(initial_yaw, value_type=float),
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
            ('/scan_front', front_scan),
        ])

    localization_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'bond_timeout': 4.0,
            'node_names': ['map_server', 'amcl'],
        }])

    navigation = GroupAction([
        SetRemap(src='/scan_front', dst=front_scan),
        SetRemap(src='/scan_rear', dst=rear_scan),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [package_share, 'launch', 'nav2_minimal.launch.py'])),
            launch_arguments={
                'use_sim_time': 'false',
                'params_file': nav2_params,
                'cmd_vel_output_topic': '/t_parking/cmd_vel_control',
            }.items()),
    ])

    parking = Node(
        package='t_parking_sim', executable='auto_parallel_parking.py',
        name='parallel_parking_auto', output='screen',
        parameters=[parking_params, {
            'use_sim_time': use_sim_time,
            'auto_start': ParameterValue(auto_start, value_type=bool),
            'execute': ParameterValue(execute, value_type=bool),
            'target_slot': LaunchConfiguration('target_slot'),
            'freeze_slam_during_execution': False,
        }],
        remappings=[
            ('/scan_front', front_scan),
            ('/scan_rear', rear_scan),
        ])

    converter = Node(
        package='t_parking_sim', executable='cmd_vel_to_lidar_cmd.py',
        name='cmd_vel_to_lidar_cmd', output='screen', parameters=[{
            'use_sim_time': use_sim_time,
            'input_topic': '/t_parking/cmd_vel_control',
            'wheel_base': wheel_base,
            'steering_limit_deg': steering_limit_deg,
            'mcu_wheel_limit_deg': 27,
            'stopped_speed_epsilon': 0.01,
            'forward_drive_stage': 1.0,
            'reverse_drive_stage': -1.0,
            'output_frequency': 10.0,
            'input_timeout_sec': 0.50,
        }])

    rviz = Node(
        package='rviz2', executable='rviz2',
        name='real_parallel_parking_rviz', output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('start_rviz')))

    return LaunchDescription([
        DeclareLaunchArgument('map_mode', default_value='saved'),
        DeclareLaunchArgument('map', default_value=default_map),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('front_scan_topic', default_value='/front/scan'),
        DeclareLaunchArgument('rear_scan_topic', default_value='/rear/scan'),
        DeclareLaunchArgument('initial_pose_x', default_value='14.705'),
        DeclareLaunchArgument('initial_pose_y', default_value='-0.50'),
        DeclareLaunchArgument(
            'initial_pose_yaw', default_value='1.57079632679'),
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument('execute', default_value='false'),
        DeclareLaunchArgument('target_slot', default_value='auto'),
        DeclareLaunchArgument('start_rviz', default_value='false'),
        OpaqueFunction(function=_validate_hardware_launch),
        map_server,
        amcl,
        localization_lifecycle,
        TimerAction(period=3.0, actions=[navigation]),
        parking,
        converter,
        TimerAction(period=5.0, actions=[rviz]),
    ])
