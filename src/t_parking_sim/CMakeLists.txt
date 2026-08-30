cmake_minimum_required(VERSION 3.8)
project(t_parking_sim)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_regulated_pure_pursuit_controller REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)

add_library(direction_locked_rpp SHARED
  controller/direction_locked_rpp.cpp
)
target_compile_features(direction_locked_rpp PUBLIC cxx_std_17)
target_include_directories(direction_locked_rpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(direction_locked_rpp
  geometry_msgs
  nav2_core
  nav2_costmap_2d
  nav2_regulated_pure_pursuit_controller
  nav_msgs
  pluginlib
  rclcpp
  std_msgs
  tf2
)
pluginlib_export_plugin_description_file(nav2_core direction_locked_rpp_plugin.xml)

install(
  DIRECTORY
    launch
    urdf
    worlds
    config
    behavior_trees
    rviz
    maps
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS direction_locked_rpp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(PROGRAMS
  scripts/auto_parallel_parking.py
  scripts/auto_t_parking.py
  scripts/bench_map_odom_anchor.py
  scripts/bench_preflight.py
  scripts/bench_support.py
  scripts/bench_virtual_vehicle.py
  scripts/cmd_vel_to_lidar_cmd.py
  scripts/combined_mapping_driver.py
  scripts/compute_path_test.py
  scripts/gazebo_pose_odom.py
  scripts/lidar_to_gazebo_bridge.py
  scripts/lidar_visibility_check.py
  scripts/parallel_in_t_slot.py
  scripts/rviz_fake_vehicle.py
  scripts/scan_filter.py
  scripts/start_staging_diagnostics.py
  scripts/t_parking_geometry.py
  scripts/vehicle_spawn_manager.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  ament_add_gtest(test_terminal_capture test/test_terminal_capture.cpp)
  target_include_directories(test_terminal_capture PRIVATE include)
  ament_add_pytest_test(test_bench_mode test/test_bench_mode.py)
  ament_add_pytest_test(
    test_parallel_in_t_slot_geometry
    test/test_parallel_in_t_slot_geometry.py)
  ament_add_pytest_test(
    test_scan_topic_contract
    test/test_scan_topic_contract.py)
endif()

ament_package()
