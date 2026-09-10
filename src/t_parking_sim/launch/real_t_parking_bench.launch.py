"""Run the real-map, wheels-off-ground T-parking BENCH sequence."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    GroupAction,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


TRUE_VALUES = ('1', 'true', 'yes', 'on')


def bench_acknowledged(value):
    """Return true only for an explicit affirmative launch value."""
    return str(value).strip().lower() in TRUE_VALUES


def serial_bridge_enabled(value):
    """Return true only when FIELD serial output was explicitly requested."""
    return str(value).strip().lower() in TRUE_VALUES


def _validate_saved_map(map_yaml):
    path = Path(os.path.abspath(os.path.expandvars(os.path.expanduser(
        map_yaml))))
    if not path.is_file():
        raise RuntimeError(f'BENCH saved map does not exist: {path}')
    image_name = None
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            key, separator, value = line.partition(':')
            if separator and key.strip() == 'image':
                image_name = value.strip().strip('"\'')
                break
    if not image_name:
        raise RuntimeError(f'BENCH saved map has no image entry: {path}')
    image = Path(os.path.expandvars(os.path.expanduser(image_name)))
    if not image.is_absolute():
        image = path.parent / image
    if not image.is_file():
        raise RuntimeError(f'BENCH saved map image does not exist: {image}')
    return str(path)


def _validate_launch_arguments(context):
    if not bench_acknowledged(
            LaunchConfiguration('bench_ack').perform(context)):
        raise RuntimeError(
            'BENCH launch refused: pass bench_ack:=true only after the '
            'vehicle is securely supported with every wheel off the ground')
    serial_requested = serial_bridge_enabled(
        LaunchConfiguration('serial_bridge').perform(context))
    hardware_ack = bench_acknowledged(
        LaunchConfiguration('hardware_ack').perform(context))
    if serial_requested and not hardware_ack:
        raise RuntimeError(
            'FIELD serial bridge refused: serial_bridge:=true also requires '
            'hardware_ack:=true after checking the physical E-stop')
    target_slot = LaunchConfiguration('target_slot').perform(context)
    if target_slot not in ('auto', 'slot_1', 'slot_2'):
        raise RuntimeError(
            'target_slot must be one of: auto, slot_1, slot_2')
    map_yaml = _validate_saved_map(
        LaunchConfiguration('map').perform(context))
    parking_config = os.path.abspath(os.path.expandvars(os.path.expanduser(
        LaunchConfiguration('parking_config').perform(context))))
    if not Path(parking_config).is_file():
        raise RuntimeError(
            f'BENCH parking geometry config does not exist: {parking_config}')
    return [
        SetLaunchConfiguration('map', map_yaml),
        SetLaunchConfiguration('parking_config', parking_config),
    ]


def _bench_nodes(context):
    share = get_package_share_directory('t_parking_sim')
    map_yaml = LaunchConfiguration('map').perform(context)
    initial_x = LaunchConfiguration('initial_pose_x')
    initial_y = LaunchConfiguration('initial_pose_y')
    initial_yaw = LaunchConfiguration('initial_pose_yaw')
    target_slot = LaunchConfiguration('target_slot')
    auto_start = LaunchConfiguration('auto_start')
    production_nav = os.path.join(share, 'config', 'nav2_params.yaml')
    bench_nav = os.path.join(share, 'config', 'nav2_params_bench.yaml')
    production_auto = os.path.join(share, 'config', 't_parking_auto.yaml')
    bench_auto = os.path.join(share, 'config', 't_parking_bench.yaml')
    virtual_config = os.path.join(
        share, 'config', 'bench_virtual_vehicle.yaml')
    rviz_config = os.path.join(share, 'rviz', 'bench_t_parking.rviz')
    xacro_file = os.path.join(share, 'urdf', 'turtle_car.urdf.xacro')
    robot_description = ParameterValue(Command([
        FindExecutable(name='xacro'), ' ', xacro_file,
        ' include_gazebo:=false',
    ]), value_type=str)
    common = {'output': 'screen'}

    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        parameters=[{'use_sim_time': False, 'yaml_filename': map_yaml}],
        **common)
    map_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_map', parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'bond_timeout': 4.0,
            'node_names': ['map_server'],
        }], **common)
    anchor = Node(
        package='t_parking_sim', executable='bench_map_odom_anchor.py',
        name='bench_map_odom_anchor', parameters=[{
            'use_sim_time': False,
            'desired_x': ParameterValue(initial_x, value_type=float),
            'desired_y': ParameterValue(initial_y, value_type=float),
            'desired_yaw': ParameterValue(initial_yaw, value_type=float),
        }], **common)
    robot_state = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='bench_robot_state_publisher', parameters=[{
            'use_sim_time': False,
            'robot_description': robot_description,
            'frame_prefix': 'bench_',
        }], **common)
    joint_state = Node(
        package='joint_state_publisher', executable='joint_state_publisher',
        name='bench_joint_state_publisher', parameters=[{
            'use_sim_time': False,
            'robot_description': robot_description,
        }], **common)
    controller = Node(
        package='nav2_controller', executable='controller_server',
        name='controller_server', parameters=[production_nav, bench_nav],
        remappings=[('cmd_vel', '/cmd_vel_nav')], **common)
    planner = Node(
        package='nav2_planner', executable='planner_server',
        name='planner_server', parameters=[production_nav, bench_nav],
        **common)
    velocity_smoother = Node(
        package='nav2_velocity_smoother', executable='velocity_smoother',
        name='velocity_smoother', parameters=[production_nav, bench_nav],
        remappings=[
            ('cmd_vel', '/cmd_vel_nav'),
            ('cmd_vel_smoothed', '/t_parking/cmd_vel_control'),
        ], **common)
    navigation_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'bond_timeout': 4.0,
            'node_names': [
                'controller_server', 'planner_server', 'velocity_smoother'],
        }], **common)
    virtual_vehicle = Node(
        package='t_parking_sim', executable='bench_virtual_vehicle.py',
        name='bench_virtual_vehicle', parameters=[virtual_config, {
            'use_sim_time': False,
            'bench_mode': True,
            'wheels_off_ground': True,
            'execute': True,
            'input_topic': '/t_parking/cmd_vel_control',
            'drive_topic': '/lidar_drive',
            'wheel_topic': '/lidar_wheel',
            'stop_topic': '/lidar_stop',
            'base_frame': 'bench_base_footprint',
            'wheel_base': 0.73,
            'steering_limit_deg': 22.0,
            'stopped_speed_epsilon': 0.01,
            'input_timeout_sec': 0.50,
        }], **common)
    auto = Node(
        package='t_parking_sim', executable='auto_t_parking.py',
        name='t_parking_auto', parameters=[
            production_auto,
            bench_auto,
            LaunchConfiguration('parking_config'),
            {
                'use_sim_time': False,
                'bench_mode': True,
                'wheels_off_ground': True,
                'execute': True,
                'auto_start': ParameterValue(auto_start, value_type=bool),
                'target_slot': target_slot,
                'return_to_entrance': False,
                'robot_base_frame': 'bench_base_link',
                'odom_topic': '/bench/odom',
                'wheel_frames': [
                    'bench_front_left_wheel_link',
                    'bench_front_right_wheel_link',
                    'bench_rear_left_wheel_link',
                    'bench_rear_right_wheel_link',
                ],
                'require_mcu_status': False,
            },
        ], **common)
    rviz = Node(
        package='rviz2', executable='rviz2', name='real_t_parking_bench_rviz',
        arguments=['-d', rviz_config], parameters=[{'use_sim_time': False}],
        condition=IfCondition(LaunchConfiguration('rviz')), **common)

    return [GroupAction([
        LogInfo(msg='[BENCH STARTUP] all interlocks passed; starting Nav2'),
        map_server,
        map_lifecycle,
        anchor,
        robot_state,
        joint_state,
        virtual_vehicle,
        controller,
        planner,
        velocity_smoother,
        navigation_lifecycle,
        rviz,
        TimerAction(period=4.0, actions=[auto]),
    ])]


def _startup_preflight_exited(event, context):
    if event.returncode != 0:
        return [
            LogInfo(msg=(
                '[BENCH STARTUP] FAIL: MCU/zero-command preflight failed; '
                'Nav2 and T-parking were not started.')),
            EmitEvent(event=Shutdown(
                reason='BENCH MCU/zero-command preflight failed')),
        ]
    return _bench_nodes(context)


def generate_launch_description():
    graph_preflight = ExecuteProcess(
        cmd=[
            'ros2', 'run', 't_parking_sim', 'bench_preflight.py',
            '--ros-args',
            '-p', 'phase:=graph',
            '-p', 'bench_mode:=true',
            '-p', 'wheels_off_ground:=true',
            '-p', 'execute:=true',
            '-p', 'observation_sec:=1.5',
        ], output='screen')
    converter = Node(
        package='t_parking_sim', executable='cmd_vel_to_lidar_cmd.py',
        name='cmd_vel_to_lidar_cmd', output='screen', parameters=[{
            'use_sim_time': False,
            'input_topic': '/t_parking/cmd_vel_control',
            'wheel_base': 0.73,
            'steering_limit_deg': 22.0,
            'mcu_wheel_limit_deg': 22,
            'fault_on_steering_limit': True,
            'stopped_speed_epsilon': 0.01,
            'forward_drive_stage': 1.0,
            'reverse_drive_stage': -1.0,
            'output_frequency': 10.0,
            'input_timeout_sec': 0.50,
            'require_parking_mode': False,
            'bench_interlock_enabled': True,
            'bench_mode': True,
            'wheels_off_ground': True,
            'execute': True,
        }])
    startup_preflight = ExecuteProcess(
        cmd=[
            'ros2', 'run', 't_parking_sim', 'bench_preflight.py',
            '--ros-args',
            '-p', 'phase:=startup',
            '-p', 'bench_mode:=true',
            '-p', 'wheels_off_ground:=true',
            '-p', 'execute:=true',
            '-p', ['startup_timeout_sec:=',
                   LaunchConfiguration('startup_timeout_sec')],
            '-p', 'zero_samples_required:=3',
            '-p', 'require_mcu_status:=false',
        ], output='screen')
    field_serial_bridge = Node(
        package='t_parking_sim', executable='t870_field_serial_bridge.py',
        name='t870_field_serial_bridge', output='screen',
        condition=IfCondition(LaunchConfiguration('serial_bridge')),
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'serial_baud': ParameterValue(
                LaunchConfiguration('serial_baud'), value_type=int),
            'watchdog_timeout_sec': 0.50,
            'output_frequency': 10.0,
            'steering_limit_deg': 22,
        }])

    def graph_preflight_exited(event, _context):
        if event.returncode != 0:
            return [
                LogInfo(msg=(
                    '[BENCH STARTUP] FAIL: graph preflight failed; no '
                    'command publisher or Nav2 node was started.')),
                EmitEvent(event=Shutdown(
                    reason='BENCH graph preflight failed')),
            ]
        return [
            LogInfo(msg=(
                '[BENCH STARTUP] graph safe; starting zero-only command '
                'heartbeat under the FIELD direct-serial contract.')),
            converter,
            field_serial_bridge,
            startup_preflight,
        ]

    def field_serial_bridge_exited(event, _context):
        if event.returncode == 0:
            reason = 'FIELD serial bridge exited'
        else:
            reason = f'FIELD serial bridge failed (exit={event.returncode})'
        return [
            LogInfo(msg=f'[BENCH STARTUP] FAIL: {reason}'),
            EmitEvent(event=Shutdown(reason=reason)),
        ]

    banner = (
        '\n==================================================\n'
        ' T PARKING BENCH MODE\n'
        ' WHEELS OFF GROUND ONLY\n'
        ' DO NOT RUN THIS MODE ON THE GROUND\n'
        '==================================================')
    share = get_package_share_directory('t_parking_sim')
    return LaunchDescription([
        DeclareLaunchArgument(
            'map', default_value=os.path.expanduser(
                '~/t_parking_maps/real_cart_slam_T_park_handA.yaml')),
        DeclareLaunchArgument(
            'parking_config', default_value=os.path.join(
                share, 'config', 't_parking_rviz_real.yaml')),
        DeclareLaunchArgument('initial_pose_x', default_value='0.24247'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.05097'),
        DeclareLaunchArgument(
            'initial_pose_yaw', default_value='3.14159265'),
        DeclareLaunchArgument('target_slot', default_value='auto'),
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('startup_timeout_sec', default_value='15.0'),
        DeclareLaunchArgument(
            'serial_bridge', default_value='false',
            description=(
                'Enable the real T870 FIELD serial bridge. Default false.')),
        DeclareLaunchArgument(
            'serial_port', default_value='/dev/t870_mcu'),
        DeclareLaunchArgument('serial_baud', default_value='115200'),
        DeclareLaunchArgument(
            'hardware_ack', default_value='false',
            description=(
                'Required with serial_bridge=true after E-stop inspection.')),
        DeclareLaunchArgument(
            'bench_ack', default_value='false',
            description=(
                'Must be true after physically confirming all wheels are '
                'securely off the ground.')),
        LogInfo(msg=banner),
        OpaqueFunction(function=_validate_launch_arguments),
        RegisterEventHandler(OnProcessExit(
            target_action=graph_preflight,
            on_exit=graph_preflight_exited,
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=startup_preflight,
            on_exit=_startup_preflight_exited,
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=field_serial_bridge,
            on_exit=field_serial_bridge_exited,
        )),
        graph_preflight,
    ])
