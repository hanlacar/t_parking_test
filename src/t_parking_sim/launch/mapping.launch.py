"""Launch the complete T-parking simulation and asynchronous SLAM."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    gui = LaunchConfiguration("gui")
    rviz = LaunchConfiguration("rviz")
    world = LaunchConfiguration("world")
    spawn_x = LaunchConfiguration("x")
    spawn_y = LaunchConfiguration("y")
    spawn_z = LaunchConfiguration("z")
    spawn_yaw = LaunchConfiguration("yaw")
    practice_mode = LaunchConfiguration("practice_mode")
    spawn_parking_obstacles = LaunchConfiguration(
        "spawn_parking_obstacles")
    debug_scan_filter = LaunchConfiguration("debug_scan_filter")

    package_share = FindPackageShare("t_parking_sim")
    default_world = PathJoinSubstitution(
        [package_share, "worlds", "t_parking_exam_real_vehicle.sdf"]
    )
    slam_params = PathJoinSubstitution(
        [package_share, "config", "slam_toolbox.yaml"]
    )
    rviz_config = PathJoinSubstitution(
        [package_share, "rviz", "t_parking_mapping.rviz"]
    )

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [package_share, "launch", "sim.launch.py"]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "gui": gui,
            "rviz": "false",
            "world": world,
            "x": spawn_x,
            "y": spawn_y,
            "z": spawn_z,
            "yaw": spawn_yaw,
            "practice_mode": practice_mode,
            "spawn_parking_obstacles": spawn_parking_obstacles,
            "mapping_lidar_override": "true",
            "mapping_lidar_z": "0.15",
        }.items(),
    )

    # Publishes /scan_static, which slam_toolbox and the global costmap
    # consume in place of the raw /scan.  It has to be up before SLAM so the
    # very first scans SLAM integrates are already filtered.
    scan_filter = Node(
        package="t_parking_sim",
        executable="scan_filter.py",
        name="scan_filter",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "odom_frame": "odom",
            "debug_enable": debug_scan_filter,
        }],
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("slam_toolbox"),
                    "launch",
                    "online_async_launch.py",
                ]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "slam_params_file": slam_params,
            "autostart": "true",
            "use_lifecycle_manager": "false",
        }.items(),
    )

    mapping_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="t_parking_mapping_rviz",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start the mapping RViz configuration.",
            ),
            DeclareLaunchArgument("world", default_value=default_world),
            DeclareLaunchArgument("x", default_value="-5.0"),
            DeclareLaunchArgument("y", default_value="0.0"),
            DeclareLaunchArgument("z", default_value="0.0"),
            DeclareLaunchArgument("yaw", default_value="0.0"),
            DeclareLaunchArgument("practice_mode", default_value="t_parking"),
            DeclareLaunchArgument(
                "spawn_parking_obstacles",
                default_value="false",
                description="Keep the reference-map world free of slot obstacles.",
            ),
            DeclareLaunchArgument(
                "debug_scan_filter",
                default_value="false",
                description=(
                    "Publish scan_filter track markers.  Off by default so "
                    "the node adds no extra topics to rqt_graph."
                ),
            ),
            simulation,
            scan_filter,
            slam,
            mapping_rviz,
        ]
    )
