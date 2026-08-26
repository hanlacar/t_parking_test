"""Launch Gazebo, saved-map localization, minimal Nav2, and RViz."""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _validate_saved_map(context):
    requested = LaunchConfiguration('map').perform(context)
    map_yaml = os.path.abspath(os.path.expandvars(os.path.expanduser(requested)))
    if not os.path.isfile(map_yaml):
        raise RuntimeError(
            f'Saved map YAML does not exist: {map_yaml}. '
            'Create it with nav2_map_server/map_saver_cli or pass map:=...')

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
            f'Saved map image referenced by {map_yaml} does not exist: '
            f'{os.path.abspath(image_path)}')

    # Also normalize ~ and environment variables before map_server receives it.
    return [SetLaunchConfiguration('map', map_yaml)]


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    start_rviz = LaunchConfiguration('start_rviz')
    world = LaunchConfiguration('world')
    practice_mode = LaunchConfiguration('practice_mode')
    spawn_parking_obstacles = LaunchConfiguration(
        'spawn_parking_obstacles')
    map_yaml = LaunchConfiguration('map')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')
    cmd_vel_output_topic = LaunchConfiguration('cmd_vel_output_topic')

    package_share = FindPackageShare('t_parking_sim')
    default_world = PathJoinSubstitution(
        [package_share, 'worlds', 't_parking_exam_real_vehicle.sdf'])
    default_map = PathJoinSubstitution([
        package_share, 'maps',
        'combined_parking_map_real_vehicle.yaml'])
    nav2_params = PathJoinSubstitution(
        [package_share, 'config', 'nav2_params.yaml'])
    rviz_config = PathJoinSubstitution(
        [package_share, 'rviz', 'nav2_mapping.rviz'])

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, 'launch', 'sim.launch.py'])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'gui': gui,
            'rviz': 'false',
            'world': world,
            'practice_mode': practice_mode,
            'spawn_parking_obstacles': spawn_parking_obstacles,
        }.items(),
    )

    # /scan_static remains useful to the global costmap even though AMCL uses
    # the full raw /scan directly.
    scan_filter = Node(
        package='t_parking_sim',
        executable='scan_filter.py',
        name='scan_filter',
        output='screen',
        parameters=[{'use_sim_time': ParameterValue(
            use_sim_time, value_type=bool), 'odom_frame': 'odom'}],
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'yaml_filename': map_yaml,
        }],
    )

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[
            nav2_params,
            {
                'use_sim_time': ParameterValue(
                    use_sim_time, value_type=bool),
                'initial_pose.x': ParameterValue(
                    initial_pose_x, value_type=float),
                'initial_pose.y': ParameterValue(
                    initial_pose_y, value_type=float),
                'initial_pose.yaw': ParameterValue(
                    initial_pose_yaw, value_type=float),
            },
        ],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    localization_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'autostart': True,
            'node_names': ['map_server', 'amcl'],
        }],
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, 'launch', 'nav2_minimal.launch.py'])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': nav2_params,
            'log_level': 'info',
            'cmd_vel_output_topic': cmd_vel_output_topic,
        }.items(),
    )

    nav2_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='t_parking_saved_map_rviz',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': ParameterValue(
            use_sim_time, value_type=bool)}],
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument('map', default_value=default_map),
        DeclareLaunchArgument(
            'practice_mode', default_value='full_course'),
        DeclareLaunchArgument(
            'spawn_parking_obstacles', default_value='true'),
        # The reference map was created with full_course world (-5, 0, 0) as
        # odom/map (0, 0, 0), so these are derived map coordinates.
        DeclareLaunchArgument('initial_pose_x', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_yaw', default_value='0.0'),
        DeclareLaunchArgument(
            'cmd_vel_output_topic', default_value='/cmd_vel'),
        OpaqueFunction(function=_validate_saved_map),
        simulation,
        scan_filter,
        map_server,
        amcl,
        localization_lifecycle,
        TimerAction(period=3.0, actions=[navigation]),
        TimerAction(period=5.0, actions=[nav2_rviz]),
    ])
