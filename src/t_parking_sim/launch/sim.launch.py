"""Launch the T-parking Gazebo Sim world and the turtle_car."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
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

    package_share = FindPackageShare("t_parking_sim")
    xacro_file = PathJoinSubstitution(
        [package_share, "urdf", "turtle_car.urdf.xacro"]
    )
    bridge_file = PathJoinSubstitution(
        [package_share, "config", "bridge.yaml"]
    )
    rviz_file = PathJoinSubstitution(
        [package_share, "rviz", "t_parking_mapping.rviz"]
    )
    default_world = PathJoinSubstitution(
        [package_share, "worlds", "t_parking_exam.sdf"]
    )

    robot_description = Command(
        [FindExecutable(name="xacro"), " ", xacro_file]
    )

    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={"gz_args": ["-r -v 3 ", world]}.items(),
        condition=IfCondition(gui),
    )

    gazebo_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={"gz_args": ["-s -r -v 3 ", world]}.items(),
        condition=UnlessCondition(gui),
    )

    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
                "publish_frequency": 50.0,
            }
        ],
    )

    # Sole owner of vehicle and obstacle spawning.  It waits for /clock to
    # advance before touching the world, so it needs no TimerAction here, and
    # it also serves /parking_practice/respawn for mid-session resets.
    spawn_manager = Node(
        package="t_parking_sim",
        executable="vehicle_spawn_manager.py",
        name="vehicle_spawn_manager",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "world_name": "t_parking_exam",
                "entity_name": "turtle_car",
                "practice_mode": practice_mode,
                "randomize_parking_obstacles": True,
                "parking_obstacle_seed": -1,
                # Only consulted when practice_mode is "custom"; every other
                # mode takes its start pose from the node's POSES table.
                "x": spawn_x,
                "y": spawn_y,
                "z": spawn_z,
                "yaw": spawn_yaw,
            }
        ],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="t_parking_bridge",
        output="screen",
        parameters=[
            {
                "config_file": bridge_file,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    optional_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="t_parking_rviz",
        output="screen",
        arguments=["-d", rviz_file],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use the Gazebo simulation clock.",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Start the Gazebo graphical client.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Start RViz (mapping.launch.py enables its own RViz).",
            ),
            DeclareLaunchArgument(
                "world",
                default_value=default_world,
                description="Absolute path to the Gazebo Sim world.",
            ),
            DeclareLaunchArgument("x", default_value="-5.0"),
            DeclareLaunchArgument("y", default_value="0.0"),
            DeclareLaunchArgument(
                "z",
                default_value="0.0",
                description="base_footprint height; wheel bottoms are at z=0.",
            ),
            DeclareLaunchArgument("yaw", default_value="0.0"),
            DeclareLaunchArgument(
                "practice_mode",
                default_value="t_parking",
                description=(
                    "t_parking | full_course | parallel_parking | "
                    "parallel_ready | custom.  The x/y/z/yaw arguments apply "
                    "to custom only."
                ),
            ),
            gazebo_gui,
            gazebo_headless,
            state_publisher,
            bridge,
            spawn_manager,
            optional_rviz,
        ]
    )
