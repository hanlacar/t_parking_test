"""Launch Gazebo, online SLAM, Nav2, and the Nav2-only T-parking node."""

import math
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
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
    """Read the canonical wheelbase and steering limit from the xacro."""
    xacro_path = os.path.join(
        get_package_share_directory('t_parking_sim'),
        'urdf',
        'turtle_car.urdf.xacro',
    )
    root = ET.parse(xacro_path).getroot()
    xacro_property = '{http://www.ros.org/wiki/xacro}property'
    properties = {
        element.attrib['name']: element.attrib['value']
        for element in root.iter(xacro_property)
        if 'name' in element.attrib and 'value' in element.attrib
    }
    try:
        wheel_base = float(properties['wheel_base'])
        steering_limit_deg = math.degrees(float(properties['steering_limit']))
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f'Cannot read numeric vehicle geometry from {xacro_path}') from exc
    return wheel_base, steering_limit_deg


def generate_launch_description():
    package_share = FindPackageShare('t_parking_sim')
    wheel_base, steering_limit_deg = _read_vehicle_geometry()
    auto_start = LaunchConfiguration('auto_start')
    execute = LaunchConfiguration('execute')
    target_slot = LaunchConfiguration('target_slot')
    return_to_entrance = LaunchConfiguration('return_to_entrance')
    stop_when_all_wheels_inside = LaunchConfiguration(
        'stop_when_all_wheels_inside')
    exit_mode = LaunchConfiguration('exit_mode')
    wheel_inside_margin = LaunchConfiguration('wheel_inside_margin')
    wheel_inside_confirm_count = LaunchConfiguration(
        'wheel_inside_confirm_count')
    entrance_pose_sample_count = LaunchConfiguration(
        'entrance_pose_sample_count')
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    start_rviz = LaunchConfiguration('start_rviz')
    practice_mode = LaunchConfiguration('practice_mode')
    map_mode = LaunchConfiguration('map_mode')
    map_yaml = LaunchConfiguration('map')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')
    enable_gazebo_bridge = LaunchConfiguration('enable_gazebo_bridge')
    front_scan_topic = LaunchConfiguration('front_scan_topic')
    rear_scan_topic = LaunchConfiguration('rear_scan_topic')

    default_map = PathJoinSubstitution([
        package_share, 'maps',
        'combined_parking_map_real_vehicle.yaml'])

    online_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, 'launch', 'nav2_mapping.launch.py']
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'gui': gui,
            'start_rviz': start_rviz,
            'practice_mode': practice_mode,
            'cmd_vel_output_topic': '/t_parking/cmd_vel_control',
            'front_scan_topic': front_scan_topic,
            'rear_scan_topic': rear_scan_topic,
        }.items(),
        condition=IfCondition(PythonExpression([
            "'", map_mode, "' == 'online'"])),
    )

    saved_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, 'launch', 'saved_map_nav.launch.py']
            )
        ),
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
            'front_scan_topic': front_scan_topic,
            'rear_scan_topic': rear_scan_topic,
        }.items(),
        condition=IfCondition(PythonExpression([
            "'", map_mode, "' == 'saved'"])),
    )

    auto_parking = Node(
        package='t_parking_sim',
        executable='auto_t_parking.py',
        name='t_parking_auto',
        output='screen',
        parameters=[
            PathJoinSubstitution(
                [package_share, 'config', 't_parking_auto.yaml']
            ),
            {
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'auto_start': ParameterValue(auto_start, value_type=bool),
                'execute': ParameterValue(execute, value_type=bool),
                'target_slot': target_slot,
                'return_to_entrance': ParameterValue(
                    return_to_entrance, value_type=bool),
                'stop_when_all_wheels_inside': ParameterValue(
                    stop_when_all_wheels_inside, value_type=bool),
                'exit_mode': exit_mode,
                'wheel_inside_margin': ParameterValue(
                    wheel_inside_margin, value_type=float),
                'wheel_inside_confirm_count': ParameterValue(
                    wheel_inside_confirm_count, value_type=int),
                'entrance_pose_sample_count': ParameterValue(
                    entrance_pose_sample_count, value_type=int),
                # Saved-map localization has no slam_toolbox services to
                # pause; online mode retains the existing freeze behavior.
                'freeze_slam_during_execution': ParameterValue(
                    PythonExpression([
                        "'", map_mode, "' == 'online'"]),
                    value_type=bool),
            },
        ],
        remappings=[
            ('/scan_front', front_scan_topic),
            ('/scan_rear', rear_scan_topic),
        ],
    )

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
            # Existing mcu_manager/mcu_bridge contract, not the Gazebo limit.
            'mcu_wheel_limit_deg': 27,
            'stopped_speed_epsilon': 0.01,
            'forward_drive_stage': 1.0,
            'reverse_drive_stage': -1.0,
            'output_frequency': 10.0,
            'input_timeout_sec': 0.50,
        }],
    )

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
            # Nav2's velocity smoother emits at 20 Hz.  Ten missed samples
            # stop Gazebo while avoiding false trips from normal jitter.
            'command_timeout_sec': 0.50,
            'publish_frequency': 20.0,
        }],
        condition=IfCondition(enable_gazebo_bridge),
    )

    return LaunchDescription([
        DeclareLaunchArgument('auto_start', default_value='true'),
        DeclareLaunchArgument('execute', default_value='true'),
        DeclareLaunchArgument('target_slot', default_value='auto'),
        DeclareLaunchArgument('return_to_entrance', default_value='true'),
        DeclareLaunchArgument(
            'stop_when_all_wheels_inside', default_value='true'),
        DeclareLaunchArgument('exit_mode', default_value='forward_right'),
        DeclareLaunchArgument('wheel_inside_margin', default_value='0.01'),
        DeclareLaunchArgument(
            'wheel_inside_confirm_count', default_value='5'),
        DeclareLaunchArgument(
            'entrance_pose_sample_count', default_value='5'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument(
            'enable_gazebo_bridge',
            default_value='true',
            description=(
                'Convert shared /lidar_drive and /lidar_wheel to Gazebo '
                '/cmd_vel. Disable for a real-vehicle-only session.')),
        DeclareLaunchArgument(
            'front_scan_topic', default_value='/scan',
            description='Gazebo front LaserScan topic.'),
        DeclareLaunchArgument(
            'rear_scan_topic', default_value='/scan_rear',
            description='Gazebo rear LaserScan topic.'),
        DeclareLaunchArgument(
            'map_mode',
            default_value='saved',
            description='saved | online'),
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Saved map YAML used when map_mode:=saved.'),
        DeclareLaunchArgument('initial_pose_x', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_yaw', default_value='0.0'),
        # t_parking_auto.yaml derives its slot coordinates from the initial
        # world pose (-5, 0, yaw=0), which is the full_course start.  Changing
        # this shifts odom's origin and invalidates those slot bounds.
        DeclareLaunchArgument('practice_mode', default_value='full_course'),
        OpaqueFunction(function=_validate_map_mode),
        online_navigation,
        saved_navigation,
        auto_parking,
        cmd_vel_to_lidar_cmd,
        lidar_to_gazebo_bridge,
    ])
