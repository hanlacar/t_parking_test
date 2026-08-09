"""Single entry point that picks the practice course from one `mode` argument.

`mode` is the only knob a user should have to change between courses.  It
selects three things at once:

  * the `practice_mode` parameter that vehicle_spawn_manager uses to look up
    the start pose (and therefore the odom origin),
  * which automatic parking node is started, and
  * that node's parameter file.

SLAM and the minimal Nav2 stack are shared by both courses and come from
nav2_mapping.launch.py.
"""

import os

import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


# mode -> practice_mode.  vehicle_spawn_manager.POSES holds the actual start
# poses: t_parking is (-3.50, 0.0, yaw 0.0) and parallel_parking is
# (9.75, -0.50, yaw 1.5708).
MODE_TO_PRACTICE_MODE = {
    't_parking': 't_parking',
    'parallel': 'parallel_parking',
}


# vehicle_spawn_manager.POSES: full_course starts at world x=-5.0 and
# t_parking at world x=-3.50, both with y=0 and yaw=0.  Every odom-frame
# constant in t_parking_auto.yaml was derived against the full_course origin
# (see the comment above its slot block), so running the T node from the
# t_parking spawn shifts the whole odom frame 1.5 m and puts the bay 1.5 m
# further along +x than the node believes.  The difference is a pure
# translation along x because both poses share y and yaw.
T_PARKING_ODOM_X_SHIFT = -5.0 - (-3.50)
# Odom-frame x constants that have to move with the origin.  Everything else
# in that file is a length, a tolerance, or a y coordinate, none of which the
# shift affects.
T_PARKING_SHIFTED_KEYS = (
    'slot_1.odom_x', 'slot_1.min_x', 'slot_1.max_x',
    'slot_2.odom_x', 'slot_2.min_x', 'slot_2.max_x',
    'observation_odom_x',
)


def _t_parking_origin_overrides(config_path: str) -> dict:
    """Re-base t_parking_auto.yaml's odom x constants on the t_parking spawn.

    Read from the file rather than restated here, so the two never drift.
    """
    with open(config_path, 'r', encoding='utf-8') as handle:
        parameters = yaml.safe_load(handle)['t_parking_auto']['ros__parameters']
    overrides = {}
    for key in T_PARKING_SHIFTED_KEYS:
        section, _, field = key.partition('.')
        value = parameters[section][field] if field else parameters[section]
        overrides[key] = float(value) + T_PARKING_ODOM_X_SHIFT
    return overrides


def _parking_node(mode: str, package_share: str, use_sim_time, auto_start,
                  execute):
    """Return the automatic parking node that belongs to ``mode``."""
    if mode == 't_parking':
        config_path = os.path.join(
            package_share, 'config', 't_parking_auto.yaml')
        return Node(
            package='t_parking_sim',
            executable='auto_t_parking.py',
            name='t_parking_auto',
            output='screen',
            parameters=[
                config_path,
                _t_parking_origin_overrides(config_path),
                {
                    'use_sim_time': ParameterValue(
                        use_sim_time, value_type=bool),
                    'auto_start': ParameterValue(auto_start, value_type=bool),
                    'execute': ParameterValue(execute, value_type=bool),
                },
            ],
        )
    return Node(
        package='t_parking_sim',
        executable='auto_parallel_parking.py',
        name='parallel_parking_auto',
        output='screen',
        parameters=[
            PathJoinSubstitution(
                [package_share, 'config', 'parallel_parking_auto.yaml']),
            {
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'auto_start': ParameterValue(auto_start, value_type=bool),
                'execute': ParameterValue(execute, value_type=bool),
            },
        ],
    )


def _launch_setup(context, *args, **kwargs):
    del args, kwargs
    mode = LaunchConfiguration('mode').perform(context)
    if mode not in MODE_TO_PRACTICE_MODE:
        valid = ' | '.join(sorted(MODE_TO_PRACTICE_MODE))
        raise RuntimeError(
            f"Invalid mode: '{mode}'.  Valid values are: {valid}.  "
            f"Example: ros2 launch t_parking_sim parking_bringup.launch.py "
            f"mode:=parallel")
    practice_mode = MODE_TO_PRACTICE_MODE[mode]
    package_share = FindPackageShare('t_parking_sim')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Gazebo + vehicle_spawn_manager (sole owner of vehicle and obstacle
    # spawning) + online SLAM + the minimal Nav2 stack.
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, 'launch', 'nav2_mapping.launch.py']
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'gui': LaunchConfiguration('gui'),
            'start_rviz': LaunchConfiguration('rviz'),
            'practice_mode': practice_mode,
        }.items(),
    )

    parking = _parking_node(
        mode, package_share.perform(context), use_sim_time,
        LaunchConfiguration('auto_start'), LaunchConfiguration('execute'))

    return [
        simulation,
        # Let Gazebo, SLAM and the Nav2 lifecycle nodes come up first; the
        # node's own readiness gate then waits for the rest.
        TimerAction(period=6.0, actions=[parking]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='t_parking',
            description=(
                't_parking (spawn at the T bay entrance) | '
                'parallel (spawn at the parallel course entrance).'),
        ),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Start the Gazebo graphical client.',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Start the Nav2 RViz instance.',
        ),
        DeclareLaunchArgument(
            'auto_start',
            default_value='false',
            description=(
                'Run the parking sequence without waiting for the start '
                'service.'),
        ),
        DeclareLaunchArgument(
            'execute',
            default_value='true',
            description=(
                'false plans and publishes without moving the vehicle.'),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
