"""Launch saved-map or online Nav2 with the parallel-parking node."""

import math
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _validate_map_mode(context):
    map_mode = LaunchConfiguration('map_mode').perform(context)
    if map_mode not in ('saved', 'online'):
        raise RuntimeError(
            f'Invalid map_mode={map_mode!r}; expected "saved" or "online"')
    return []


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
            f'Cannot read numeric vehicle geometry from {xacro_path}') from exc


def generate_launch_description():
    package_share = FindPackageShare('t_parking_sim')
    wheel_base, steering_limit_deg = _read_vehicle_geometry()
    map_mode = LaunchConfiguration('map_mode')
    map_yaml = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    start_rviz = LaunchConfiguration('start_rviz')
    practice_mode = LaunchConfiguration('practice_mode')
    auto_start = LaunchConfiguration('auto_start')
    execute = LaunchConfiguration('execute')
    target_slot = LaunchConfiguration('target_slot')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')
    enable_gazebo_bridge = LaunchConfiguration('enable_gazebo_bridge')

    default_map = PathJoinSubstitution([
        package_share, 'maps',
        'combined_parking_map_real_vehicle.yaml'])

    online_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [package_share, 'launch', 'nav2_mapping.launch.py'])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'gui': gui,
            'start_rviz': start_rviz,
            'practice_mode': practice_mode,
            'spawn_parking_obstacles': 'true',
            'cmd_vel_output_topic': '/t_parking/cmd_vel_control',
        }.items(),
        condition=IfCondition(PythonExpression([
            "'", map_mode, "' == 'online'"])))

    saved_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [package_share, 'launch', 'saved_map_nav.launch.py'])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'gui': gui,
            'start_rviz': start_rviz,
            'practice_mode': practice_mode,
            'map': map_yaml,
            'spawn_parking_obstacles': 'true',
            'initial_pose_x': initial_pose_x,
            'initial_pose_y': initial_pose_y,
            'initial_pose_yaw': initial_pose_yaw,
            'cmd_vel_output_topic': '/t_parking/cmd_vel_control',
        }.items(),
        condition=IfCondition(PythonExpression([
            "'", map_mode, "' == 'saved'"])))

    parking = Node(
        package='t_parking_sim',
        executable='auto_parallel_parking.py',
        name='parallel_parking_auto',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                package_share, 'config', 'parallel_parking_auto.yaml']),
            {
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'auto_start': ParameterValue(auto_start, value_type=bool),
                'execute': ParameterValue(execute, value_type=bool),
                'target_slot': target_slot,
                'freeze_slam_during_execution': ParameterValue(
                    PythonExpression(["'", map_mode, "' == 'online'"]),
                    value_type=bool),
            },
        ])

    cmd_vel_to_lidar_cmd = Node(
        package='t_parking_sim',
        executable='cmd_vel_to_lidar_cmd.py',
        name='cmd_vel_to_lidar_cmd',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
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

    lidar_to_gazebo_bridge = Node(
        package='t_parking_sim',
        executable='lidar_to_gazebo_bridge.py',
        name='lidar_to_gazebo_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'drive_scale_mps': 0.30,
            'wheel_base': wheel_base,
            'wheel_limit_deg': 22,
            'command_timeout_sec': 0.50,
            'publish_frequency': 20.0,
        }],
        condition=IfCondition(enable_gazebo_bridge),
    )

    return LaunchDescription([
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument('execute', default_value='true'),
        DeclareLaunchArgument('target_slot', default_value='auto'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument(
            'enable_gazebo_bridge', default_value='true',
            description=(
                'Route the shared /lidar_* command contract into Gazebo. '
                'Disable only for a real-vehicle-only session.')),
        DeclareLaunchArgument(
            'map_mode', default_value='saved', description='saved | online'),
        DeclareLaunchArgument(
            'map', default_value=default_map,
            description='Saved map YAML used when map_mode:=saved.'),
        DeclareLaunchArgument(
            'practice_mode', default_value='parallel_parking',
            description=(
                'Keep parallel_parking unless the parallel geometry is '
                'explicitly re-based to another odom origin.')),
        # The combined map starts at world (-5, 0, 0), while this vehicle
        # starts at world (9.75, -0.50, pi/2).
        DeclareLaunchArgument('initial_pose_x', default_value='14.705'),
        DeclareLaunchArgument('initial_pose_y', default_value='-0.50'),
        DeclareLaunchArgument(
            'initial_pose_yaw', default_value='1.57079632679'),
        OpaqueFunction(function=_validate_map_mode),
        online_navigation,
        saved_navigation,
        parking,
        cmd_vel_to_lidar_cmd,
        lidar_to_gazebo_bridge,
    ])
