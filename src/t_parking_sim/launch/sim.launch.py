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
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
    parking_obstacle_seed = LaunchConfiguration('parking_obstacle_seed')
    mapping_lidar_override = LaunchConfiguration("mapping_lidar_override")
    mapping_lidar_z = LaunchConfiguration("mapping_lidar_z")

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
        [package_share, "worlds", "t_parking_exam_real_vehicle.sdf"]
    )

    effective_lidar_z = PythonExpression([
        "", mapping_lidar_z, " if '", mapping_lidar_override,
        "'.lower() == 'true' else 0.52",
    ])
    robot_description = Command([
        FindExecutable(name="xacro"), " ", xacro_file,
        " lidar_ground_height:=", effective_lidar_z,
    ])

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
                "robot_description": ParameterValue(
                    robot_description, value_type=str),
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
                "spawn_parking_obstacles": ParameterValue(
                    spawn_parking_obstacles, value_type=bool),
                "randomize_parking_obstacles": True,
                'parking_obstacle_seed': ParameterValue(
                    parking_obstacle_seed, value_type=int),
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

    # Sole ROS authority for /odom and odom -> base_footprint.  Its input is
    # Gazebo's collision-aware model pose, not Ackermann wheel integration.
    pose_odom = Node(
        package="t_parking_sim",
        executable="gazebo_pose_odom.py",
        name="gazebo_pose_odom",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
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
            DeclareLaunchArgument(
                "spawn_parking_obstacles",
                default_value="true",
                description=(
                    "Spawn parking-slot obstacles. Set false only for a "
                    "clean reference-map mapping run."
                ),
            ),
            DeclareLaunchArgument(
                'parking_obstacle_seed',
                default_value='-1',
                description=(
                    'Negative selects system randomness; a nonnegative seed '
                    'makes obstacle layouts reproducible.'),
            ),
            DeclareLaunchArgument(
                "mapping_lidar_override",
                default_value="false",
                description=(
                    "Use mapping_lidar_z for this simulation. Normal and "
                    "saved-map launches must leave this false."),
            ),
            DeclareLaunchArgument(
                "mapping_lidar_z",
                default_value="0.15",
                description="Mapping-only LiDAR centre height above ground.",
            ),
            gazebo_gui,
            gazebo_headless,
            state_publisher,
            bridge,
            pose_odom,
            spawn_manager,
            optional_rviz,
        ]
    )
