cmake_minimum_required(VERSION 3.8)
project(t_parking_sim)

find_package(ament_cmake REQUIRED)

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

install(PROGRAMS
  scripts/auto_parallel_parking.py
  scripts/auto_t_parking.py
  scripts/compute_path_test.py
  scripts/lidar_visibility_check.py
  scripts/scan_filter.py
  scripts/vehicle_spawn_manager.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
