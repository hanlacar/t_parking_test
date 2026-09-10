"""Run parallel-in-T-slot planning on the existing saved map and world."""

import math
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _read_vehicle_geometry():
    xacro_path = os.path.join(
        get_package_share_directory('t_parking_sim'),
        'urdf', 'turtle_car.urdf.xacro')
    root = ET.parse(xacro_path).getroot()
    tag = '{http://www.ros.org/wiki/xacro}property'
    properties = {
        element.attrib['name']: element.attrib['value']
        for element in root.iter(tag)
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
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    start_rviz = LaunchConfiguration('start_rviz')
    map_yaml = LaunchConfiguration('map')
    auto_start = LaunchConfiguration('auto_start')
    execute = LaunchConfiguration('execute')
    target_slot = LaunchConfiguration('target_slot')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')
    enable_gazebo_bridge = LaunchConfiguration('enable_gazebo_bridge')
    front_scan_topic = LaunchConfiguration('front_scan_topic')
    rear_scan_topic = LaunchConfiguration('rear_scan_topic')
    spawn_parking_obstacles = LaunchConfiguration(
        'spawn_parking_obstacles')
    parking_obstacle_seed = LaunchConfiguration('parking_obstacle_seed')

    default_map = PathJoinSubstitution([
        package_share, 'maps', 'combined_parking_map_real_vehicle.yaml'])

    saved_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [package_share, 'launch', 'saved_map_nav.launch.py'])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'gui': gui,
            # Use the dedicated configuration below so all diagnostics are
            # visible without changing the existing Nav2 RViz profile.
            'start_rviz': 'false',
            'practice_mode': 'full_course',
            'map': map_yaml,
            'spawn_parking_obstacles': spawn_parking_obstacles,
            'initial_pose_x': initial_pose_x,
            'initial_pose_y': initial_pose_y,
            'initial_pose_yaw': initial_pose_yaw,
            'cmd_vel_output_topic': '/parallel_in_t_slot/cmd_vel_control',
            'front_scan_topic': front_scan_topic,
            'rear_scan_topic': rear_scan_topic,
            'parking_obstacle_seed': parking_obstacle_seed,
        }.items())

    parking = Node(
        package='t_parking_sim',
        executable='parallel_in_t_slot.py',
        name='parallel_in_t_slot',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                package_share, 'config', 'parallel_in_t_slot.yaml']),
            {
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'auto_start': ParameterValue(auto_start, value_type=bool),
                'execute': ParameterValue(execute, value_type=bool),
                'target_slot': target_slot,
            },
        ],
        remappings=[
            ('/parallel_parking/start', '/parallel_in_t_slot/start'),
            ('/parallel_parking/cancel', '/parallel_in_t_slot/cancel'),
            ('/parallel_parking/status', '/parallel_in_t_slot/status'),
            ('/parallel_parking/planned_path',
             '/parallel_in_t_slot/planned_path'),
            ('/parallel_parking/exit_path', '/parallel_in_t_slot/exit_path'),
            ('/parallel_parking/forward_path',
             '/parallel_in_t_slot/forward_path'),
            ('/parallel_parking/reverse_path',
             '/parallel_in_t_slot/reverse_path'),
            ('/parallel_parking/markers', '/parallel_in_t_slot/markers'),
            ('/t_parking/active_segment',
             '/parallel_in_t_slot/active_segment'),
            ('/cmd_vel', '/parallel_in_t_slot/cmd_vel_control'),
            ('/scan', front_scan_topic),
            ('/scan_rear', rear_scan_topic),
        ])

    cmd_bridge = Node(
        package='t_parking_sim',
        executable='cmd_vel_to_lidar_cmd.py',
        name='parallel_in_t_slot_cmd_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'input_topic': '/parallel_in_t_slot/cmd_vel_control',
            'wheel_base': wheel_base,
            'steering_limit_deg': steering_limit_deg,
            'mcu_wheel_limit_deg': 27,
            'stopped_speed_epsilon': 0.01,
            'forward_drive_stage': 1.0,
            'reverse_drive_stage': -1.0,
            'output_frequency': 10.0,
            'input_timeout_sec': 0.50,
        }])

    # The real vehicle gets its applied mode from the mission/MCU manager.
    # Gazebo has no mode manager, so this launch owns the equivalent lifecycle:
    # grant LiDAR commands only while this parking state machine is active and
    # restore NORMAL on finish, failure, or cancellation.
    mode_publisher = Node(
        package='t_parking_sim',
        executable='parking_mode_publisher.py',
        name='parallel_in_t_slot_mode_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'status_topic': '/parallel_in_t_slot/status',
            'mode_topic': '/vehicle_mode',
            'active_mode': 'PARALLEL_PARK',
            'restore_mode': 'NORMAL',
            'publish_hz': 2.0,
        }])

    gazebo_bridge = Node(
        package='t_parking_sim',
        executable='lidar_to_gazebo_bridge.py',
        name='parallel_in_t_slot_gazebo_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'drive_scale_mps': 0.30,
            'wheel_base': wheel_base,
            'wheel_limit_deg': 22,
            'command_timeout_sec': 0.50,
            'publish_frequency': 20.0,
        }],
        condition=IfCondition(enable_gazebo_bridge))

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='parallel_in_t_slot_rviz',
        output='screen',
        arguments=['-d', PathJoinSubstitution([
            package_share, 'rviz', 'parallel_in_t_slot.rviz'])],
        parameters=[{'use_sim_time': ParameterValue(
            use_sim_time, value_type=bool)}],
        condition=IfCondition(start_rviz))

    return LaunchDescription([
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument(
            'execute', default_value='false',
            description='false validates and visualizes without motion.'),
        DeclareLaunchArgument(
            'target_slot', default_value='auto',
            description='auto | slot_1 (eastbound) | slot_2 (westbound)'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument(
            'enable_gazebo_bridge', default_value='true'),
        DeclareLaunchArgument(
            'front_scan_topic', default_value='/scan',
            description='Gazebo front LaserScan topic.'),
        DeclareLaunchArgument(
            'rear_scan_topic', default_value='/scan_rear',
            description='Gazebo rear LaserScan topic.'),
        DeclareLaunchArgument(
            'spawn_parking_obstacles', default_value='true',
            description='Keep the normal random-obstacle behavior by default.'),
        DeclareLaunchArgument(
            'parking_obstacle_seed', default_value='-1',
            description='Nonnegative values reproduce obstacle layouts.'),
        DeclareLaunchArgument('map', default_value=default_map),
        # The existing saved map was made from this original full-course pose.
        DeclareLaunchArgument('initial_pose_x', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_yaw', default_value='0.0'),
        saved_navigation,
        parking,
        mode_publisher,
        cmd_bridge,
        gazebo_bridge,
        rviz,
    ])
