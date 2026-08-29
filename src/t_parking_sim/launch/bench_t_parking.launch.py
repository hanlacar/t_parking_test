"""Saved-map, virtual-feedback Nav2 BENCH with real guarded MCU output."""

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
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _as_bool(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in (
        '1', 'true', 'yes', 'on')


def _validate_motion_arguments(context):
    bench_mode = _as_bool(context, 'bench_mode')
    wheels_off_ground = _as_bool(context, 'wheels_off_ground')
    execute = _as_bool(context, 'execute')
    if not bench_mode:
        raise RuntimeError(
            'BENCH launch requires the explicit argument bench_mode:=true')
    if execute and not wheels_off_ground:
        raise RuntimeError(
            'BENCH MOTION INHIBITED: bench_mode and wheels_off_ground '
            'must both be true')
    return bench_mode, wheels_off_ground, execute


def _validate_launch_arguments(context):
    _validate_motion_arguments(context)
    return []


def _bench_nodes(context):
    bench_mode, wheels_off_ground, execute = _validate_motion_arguments(context)
    auto_start = _as_bool(context, 'auto_start')
    map_yaml = os.path.abspath(os.path.expanduser(
        LaunchConfiguration('map').perform(context)))
    if not Path(map_yaml).is_file():
        raise RuntimeError(f'BENCH saved map does not exist: {map_yaml}')
    share = get_package_share_directory('t_parking_sim')
    production_nav = os.path.join(share, 'config', 'nav2_params.yaml')
    bench_nav = os.path.join(share, 'config', 'nav2_params_bench.yaml')
    production_auto = os.path.join(share, 'config', 't_parking_auto.yaml')
    bench_auto = os.path.join(share, 'config', 't_parking_bench.yaml')
    virtual_config = os.path.join(
        share, 'config', 'bench_virtual_vehicle.yaml')
    rviz_config = os.path.join(share, 'rviz', 'bench_t_parking.rviz')
    common = {'output': 'screen'}

    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        parameters=[{'use_sim_time': False, 'yaml_filename': map_yaml}],
        **common)
    map_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_map',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'bond_timeout': 4.0,
            'node_names': ['map_server'],
        }],
        **common)
    anchor = Node(
        package='t_parking_sim', executable='bench_map_odom_anchor.py',
        name='bench_map_odom_anchor', parameters=[{
            'use_sim_time': False,
            'desired_x': 9.70,
            'desired_y': 0.0,
            'desired_yaw': 3.141592653589793,
        }],
        **common)
    controller = Node(
        package='nav2_controller', executable='controller_server',
        name='controller_server',
        parameters=[production_nav, bench_nav],
        remappings=[('cmd_vel', '/cmd_vel_nav')],
        **common)
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
        ],
        **common)
    navigation_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'bond_timeout': 4.0,
            'node_names': [
                'controller_server', 'planner_server', 'velocity_smoother'],
        }],
        **common)
    virtual_vehicle = Node(
        package='t_parking_sim', executable='bench_virtual_vehicle.py',
        name='bench_virtual_vehicle', parameters=[virtual_config, {
            'use_sim_time': False,
            'bench_mode': bench_mode,
            'wheels_off_ground': wheels_off_ground,
            'execute': execute,
            'input_topic': '/t_parking/cmd_vel_control',
            'wheel_base': 0.77,
            'steering_limit_deg': 27.0,
            'stopped_speed_epsilon': 0.01,
            'input_timeout_sec': 0.50,
        }],
        **common)
    auto = Node(
        package='t_parking_sim', executable='auto_t_parking.py',
        name='t_parking_auto', parameters=[
            production_auto,
            bench_auto,
            {
                'use_sim_time': False,
                'bench_mode': bench_mode,
                'wheels_off_ground': wheels_off_ground,
                'execute': execute,
                'auto_start': auto_start,
                'target_slot': 'slot_1',
                'return_to_entrance': True,
            },
        ],
        **common)
    rviz = Node(
        package='rviz2', executable='rviz2', name='bench_t_parking_rviz',
        arguments=['-d', rviz_config], parameters=[{'use_sim_time': False}],
        condition=IfCondition(LaunchConfiguration('start_rviz')),
        **common)

    return [GroupAction([
        LogInfo(msg='[BENCH ONLY - VEHICLE MUST BE OFF THE GROUND]'),
        LogInfo(msg='[BENCH ONLY - VEHICLE MUST BE OFF THE GROUND]'),
        LogInfo(msg='[BENCH ONLY - VEHICLE MUST BE OFF THE GROUND]'),
        map_server,
        map_lifecycle,
        anchor,
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
                '[BENCH STARTUP] FAIL: Nav2 and auto parking were not '
                'started; converter remains at zero until launch shutdown.')),
        ]
    return _bench_nodes(context)


def generate_launch_description():
    default_map = os.path.expanduser('~/t_parking_maps/real_parking_map.yaml')
    graph_preflight = ExecuteProcess(
        cmd=[
            'ros2', 'run', 't_parking_sim', 'bench_preflight.py',
            '--ros-args',
            '-p', 'phase:=graph',
            '-p', ['bench_mode:=', LaunchConfiguration('bench_mode')],
            '-p', ['wheels_off_ground:=',
                   LaunchConfiguration('wheels_off_ground')],
            '-p', ['execute:=', LaunchConfiguration('execute')],
            '-p', 'observation_sec:=1.5',
        ],
        output='screen',
    )
    converter = Node(
        package='t_parking_sim', executable='cmd_vel_to_lidar_cmd.py',
        name='cmd_vel_to_lidar_cmd', output='screen', parameters=[{
            'use_sim_time': False,
            'input_topic': '/t_parking/cmd_vel_control',
            'wheel_base': 0.77,
            'steering_limit_deg': 27.0,
            'mcu_wheel_limit_deg': 27,
            'stopped_speed_epsilon': 0.01,
            'forward_drive_stage': 1.0,
            'reverse_drive_stage': -1.0,
            'output_frequency': 10.0,
            'input_timeout_sec': 0.50,
            'mode_topic': '/mcu/current_mode',
            'bench_interlock_enabled': True,
            'bench_mode': ParameterValue(
                LaunchConfiguration('bench_mode'), value_type=bool),
            'wheels_off_ground': ParameterValue(
                LaunchConfiguration('wheels_off_ground'), value_type=bool),
            'execute': ParameterValue(
                LaunchConfiguration('execute'), value_type=bool),
        }])
    startup_preflight = ExecuteProcess(
        cmd=[
            'ros2', 'run', 't_parking_sim', 'bench_preflight.py',
            '--ros-args',
            '-p', 'phase:=startup',
            '-p', ['bench_mode:=', LaunchConfiguration('bench_mode')],
            '-p', ['wheels_off_ground:=',
                   LaunchConfiguration('wheels_off_ground')],
            '-p', ['execute:=', LaunchConfiguration('execute')],
            '-p', ['startup_timeout_sec:=',
                   LaunchConfiguration('startup_timeout_sec')],
            '-p', 'zero_samples_required:=3',
        ],
        output='screen',
    )

    def graph_preflight_exited(event, _context):
        if event.returncode != 0:
            return [
                LogInfo(msg=(
                    '[BENCH STARTUP] FAIL: graph preflight failed; command '
                    'converter and Nav2 were not started.')),
                EmitEvent(event=Shutdown(
                    reason='BENCH graph preflight failed')),
            ]
        return [
            LogInfo(msg=(
                '[BENCH STARTUP] graph safe; starting zero-heartbeat '
                'converter before MCU status gate.')),
            converter,
            startup_preflight,
        ]

    return LaunchDescription([
        DeclareLaunchArgument('bench_mode', default_value='false'),
        DeclareLaunchArgument('wheels_off_ground', default_value='false'),
        DeclareLaunchArgument('execute', default_value='false'),
        DeclareLaunchArgument('auto_start', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('startup_timeout_sec', default_value='15.0'),
        DeclareLaunchArgument('map', default_value=default_map),
        OpaqueFunction(function=_validate_launch_arguments),
        RegisterEventHandler(OnProcessExit(
            target_action=graph_preflight,
            on_exit=graph_preflight_exited,
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=startup_preflight,
            on_exit=_startup_preflight_exited,
        )),
        graph_preflight,
    ])
