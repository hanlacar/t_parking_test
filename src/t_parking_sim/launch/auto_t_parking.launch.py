"""Launch Gazebo, online SLAM, Nav2, and the Nav2-only T-parking node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare('t_parking_sim')
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

    navigation = IncludeLaunchDescription(
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
        }.items(),
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
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument('execute', default_value='false'),
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
        # t_parking_auto.yaml derives its slot coordinates from the initial
        # world pose (-5, 0, yaw=0), which is the full_course start.  Changing
        # this shifts odom's origin and invalidates those slot bounds.
        DeclareLaunchArgument('practice_mode', default_value='full_course'),
        navigation,
        auto_parking,
    ])
