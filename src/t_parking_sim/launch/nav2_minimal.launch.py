"""Launch only the Nav2 servers this package actually uses.

nav2_bringup/navigation_launch.py brings up eleven lifecycle nodes.
auto_t_parking.py talks to exactly two of them -- planner_server via
/compute_path_through_poses and controller_server via /follow_path -- and
never calls /navigate_to_pose, so the behaviour tree stack and everything
hanging off it is dead weight in both the process list and rqt_graph.

Kept:
  planner_server      /compute_path_through_poses  (owns global_costmap)
  controller_server   /follow_path                 (owns local_costmap)
  velocity_smoother   auto_t_parking.py publishes its stop command on
                      /cmd_vel_nav.  Its output topic is configurable so the
                      T-parking launch can route the final Twist through the
                      shared lidar command contract before Gazebo sees it.
  lifecycle_manager_navigation

Dropped: bt_navigator, behavior_server, smoother_server, waypoint_follower,
docking_server, route_server, collision_monitor.

collision_monitor used to be the publisher of /cmd_vel (velocity_smoother
emitted /cmd_vel_smoothed into it).  With it gone, velocity_smoother's output
is remapped onto cmd_vel_output_topic.  The default remains /cmd_vel for
standalone and parallel-parking compatibility; T parking overrides it with a
private topic consumed by cmd_vel_to_lidar_cmd.
The costmaps are not separate nodes -- they live inside the planner and
controller servers -- so they must not appear in node_names.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # The launch argument arrives as a string; declaring the type keeps it
    # from being pushed to the nodes as a string parameter.
    use_sim_time = ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)
    params_file = LaunchConfiguration('params_file')
    log_level = LaunchConfiguration('log_level')
    cmd_vel_output_topic = LaunchConfiguration('cmd_vel_output_topic')

    default_params = PathJoinSubstitution(
        [FindPackageShare('t_parking_sim'), 'config', 'nav2_params.yaml']
    )

    # /tf and /tf_static are remapped relative so a namespace could still be
    # pushed onto this group later, matching nav2_bringup's convention.
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    common = {
        'output': 'screen',
        'arguments': ['--ros-args', '--log-level', log_level],
    }

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=remappings + [('cmd_vel', 'cmd_vel_nav')],
        **common,
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=remappings,
        **common,
    )

    velocity_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=remappings + [
            ('cmd_vel', 'cmd_vel_nav'),
            ('cmd_vel_smoothed', cmd_vel_output_topic),
        ],
        **common,
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'bond_timeout': 4.0,
            'node_names': [
                'controller_server',
                'planner_server',
                'velocity_smoother',
            ],
        }],
        **common,
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument(
            'cmd_vel_output_topic',
            default_value='/cmd_vel',
            description='Final velocity_smoother Twist output topic.'),
        controller_server,
        planner_server,
        velocity_smoother,
        lifecycle_manager,
    ])
