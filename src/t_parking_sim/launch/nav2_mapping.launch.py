"""Launch Gazebo, online SLAM, Nav2 navigation, and one Nav2 RViz."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    start_rviz = LaunchConfiguration('start_rviz')
    world = LaunchConfiguration('world')
    spawn_x = LaunchConfiguration('x')
    spawn_y = LaunchConfiguration('y')
    spawn_z = LaunchConfiguration('z')
    spawn_yaw = LaunchConfiguration('yaw')
    practice_mode = LaunchConfiguration('practice_mode')
    spawn_parking_obstacles = LaunchConfiguration(
        'spawn_parking_obstacles')
    cmd_vel_output_topic = LaunchConfiguration('cmd_vel_output_topic')

    package_share = FindPackageShare('t_parking_sim')
    default_world = PathJoinSubstitution(
        [package_share, 'worlds', 't_parking_exam_real_vehicle.sdf']
    )
    nav2_params = PathJoinSubstitution(
        [package_share, 'config', 'nav2_params.yaml']
    )
    nav2_rviz_config = PathJoinSubstitution(
        [package_share, 'rviz', 'nav2_mapping.rviz']
    )

    mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, 'launch', 'mapping.launch.py']
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'gui': gui,
            'rviz': 'false',
            'world': world,
            'x': spawn_x,
            'y': spawn_y,
            'z': spawn_z,
            'yaw': spawn_yaw,
            'practice_mode': practice_mode,
            'spawn_parking_obstacles': spawn_parking_obstacles,
        }.items(),
    )

    # nav2_bringup/navigation_launch.py would start eleven lifecycle nodes;
    # this package uses planner_server, controller_server and
    # velocity_smoother only.  See nav2_minimal.launch.py for the rationale.
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, 'launch', 'nav2_minimal.launch.py']
            )
        ),
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
        name='t_parking_nav2_rviz',
        output='screen',
        arguments=['-d', nav2_rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('use_sim_time', default_value='true'),
            DeclareLaunchArgument('gui', default_value='true'),
            DeclareLaunchArgument(
                'start_rviz',
                default_value='true',
                description='Start the single Nav2 mapping RViz instance.',
            ),
            DeclareLaunchArgument('world', default_value=default_world),
            DeclareLaunchArgument('x', default_value='-5.0'),
            DeclareLaunchArgument('y', default_value='0.0'),
            DeclareLaunchArgument('z', default_value='0.0'),
            DeclareLaunchArgument('yaw', default_value='0.0'),
            DeclareLaunchArgument('practice_mode', default_value='t_parking'),
            DeclareLaunchArgument(
                'spawn_parking_obstacles',
                default_value='true',
                description='Keep obstacles in online navigation sessions.'),
            DeclareLaunchArgument(
                'cmd_vel_output_topic', default_value='/cmd_vel'),
            mapping,
            # Let Gazebo, the robot, and online SLAM begin publishing first.
            TimerAction(period=3.0, actions=[navigation]),
            TimerAction(period=5.0, actions=[nav2_rviz]),
        ]
    )
