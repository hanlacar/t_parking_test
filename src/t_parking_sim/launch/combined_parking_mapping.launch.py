"""Map both parking areas and their connector in one SLAM session."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    start_rviz = LaunchConfiguration('rviz')
    auto_drive = LaunchConfiguration('auto_drive')
    package_share = FindPackageShare('t_parking_sim')
    world = LaunchConfiguration('world')
    default_world = PathJoinSubstitution(
        [package_share, 'worlds', 't_parking_exam_real_vehicle.sdf'])
    rviz_config = PathJoinSubstitution(
        [package_share, 'rviz', 'nav2_mapping.rviz'])

    mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [package_share, 'launch', 'mapping.launch.py'])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'gui': gui,
            'rviz': 'false',
            'world': world,
            'practice_mode': 'full_course',
            'spawn_parking_obstacles': 'false',
        }.items())

    survey_driver = Node(
        package='t_parking_sim',
        executable='combined_mapping_driver.py',
        name='combined_mapping_driver',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
        }],
        condition=IfCondition(auto_drive))

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='combined_parking_mapping_rviz',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
        }],
        condition=IfCondition(start_rviz))

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument(
            'auto_drive', default_value='false',
            description=(
                'Follow the continuous built-in survey route. Keep false '
                'when driving manually.')),
        mapping,
        TimerAction(period=5.0, actions=[survey_driver]),
        TimerAction(period=5.0, actions=[rviz]),
    ])
