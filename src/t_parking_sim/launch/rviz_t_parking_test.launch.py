"""Launch a hardware-free saved-map T-parking animation in RViz."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _validate_map(context):
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
            f'Saved map image referenced by {map_yaml} does not exist: '
            f'{os.path.abspath(image_path)}')
    return [SetLaunchConfiguration('map', map_yaml)]


def generate_launch_description():
    package_share = FindPackageShare('t_parking_sim')
    map_yaml = LaunchConfiguration('map')
    target_slot = LaunchConfiguration('target_slot')
    initial_x = LaunchConfiguration('initial_pose_x')
    initial_y = LaunchConfiguration('initial_pose_y')
    initial_yaw = LaunchConfiguration('initial_pose_yaw')
    start_rviz = LaunchConfiguration('start_rviz')

    base_params = PathJoinSubstitution(
        [package_share, 'config', 'nav2_params.yaml'])
    dry_run_params = PathJoinSubstitution(
        [package_share, 'config', 'rviz_t_parking_test.yaml'])
    xacro_file = PathJoinSubstitution(
        [package_share, 'urdf', 'turtle_car.urdf.xacro'])
    rviz_file = PathJoinSubstitution(
        [package_share, 'rviz', 'rviz_t_parking_test.rviz'])
    robot_description = Command([
        FindExecutable(name='xacro'), ' ', xacro_file])

    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        output='screen', parameters=[{
            'use_sim_time': False,
            'yaml_filename': map_yaml,
        }])

    planner_server = Node(
        package='nav2_planner', executable='planner_server',
        name='planner_server', output='screen',
        parameters=[base_params, dry_run_params, {'use_sim_time': False}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')])

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_rviz_t_parking', output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'bond_timeout': 4.0,
            'node_names': ['map_server', 'planner_server'],
        }])

    map_to_odom = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='map_to_odom_static', output='screen',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'map', '--child-frame-id', 'odom',
        ])

    state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen', parameters=[{
            'use_sim_time': False,
            'robot_description': ParameterValue(
                robot_description, value_type=str),
            'publish_frequency': 50.0,
        }])

    fake_vehicle = Node(
        package='t_parking_sim', executable='rviz_fake_vehicle.py',
        name='rviz_fake_vehicle', output='screen', parameters=[{
            'use_sim_time': False,
            'initial_pose_x': ParameterValue(initial_x, value_type=float),
            'initial_pose_y': ParameterValue(initial_y, value_type=float),
            'initial_pose_yaw': ParameterValue(initial_yaw, value_type=float),
            'forward_speed_mps': ParameterValue(
                LaunchConfiguration('forward_speed_mps'), value_type=float),
            'reverse_speed_mps': ParameterValue(
                LaunchConfiguration('reverse_speed_mps'), value_type=float),
            'update_rate_hz': ParameterValue(
                LaunchConfiguration('update_rate_hz'), value_type=float),
            'cusp_pause_sec': ParameterValue(
                LaunchConfiguration('cusp_pause_sec'), value_type=float),
        }])

    planner = Node(
        package='t_parking_sim', executable='auto_t_parking.py',
        name='t_parking_auto', output='screen',
        parameters=[
            PathJoinSubstitution(
                [package_share, 'config', 't_parking_auto.yaml']),
            dry_run_params,
            {
                'use_sim_time': False,
                'rviz_only': True,
                'auto_start': False,
                'execute': False,
                'target_slot': target_slot,
            },
        ])

    rviz = Node(
        package='rviz2', executable='rviz2',
        name='rviz_t_parking_test', output='screen',
        arguments=['-d', rviz_file], parameters=[{'use_sim_time': False}],
        condition=IfCondition(start_rviz))

    return LaunchDescription([
        DeclareLaunchArgument(
            'map', default_value='/home/wan/t_parking_maps/real_parking_map.yaml'),
        DeclareLaunchArgument(
            'target_slot', default_value='auto',
            description='Existing planner convention: auto | slot_1 | slot_2'),
        # Canonical opposite-side start from rviz_t_parking_test.yaml.  Launch
        # arguments remain overrideable for the requested robustness checks.
        DeclareLaunchArgument('initial_pose_x', default_value='9.70'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        DeclareLaunchArgument(
            'initial_pose_yaw', default_value='3.14159265359'),
        DeclareLaunchArgument('forward_speed_mps', default_value='0.30'),
        DeclareLaunchArgument('reverse_speed_mps', default_value='0.20'),
        DeclareLaunchArgument('update_rate_hz', default_value='20.0'),
        DeclareLaunchArgument('cusp_pause_sec', default_value='0.5'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        OpaqueFunction(function=_validate_map),
        map_to_odom,
        state_publisher,
        fake_vehicle,
        map_server,
        planner_server,
        lifecycle_manager,
        planner,
        rviz,
    ])
