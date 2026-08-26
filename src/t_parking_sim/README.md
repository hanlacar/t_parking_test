# t_parking_sim

ROS 2 Jazzy + Gazebo Sim 기반 Ackermann 차량의 **T자 주차 / 평행주차 시뮬레이션 패키지**입니다.

## Quick Start

### 1. Dependencies

```bash
cd ~/t_parking_ws
source /opt/ros/jazzy/setup.bash

rosdep install --from-paths src --ignore-src -r -y
```

### 2. Build

```bash
cd ~/t_parking_ws
source /opt/ros/jazzy/setup.bash

colcon build --symlink-install
source install/setup.bash
```

### 3. T Parking

터미널 1:

```bash
cd ~/t_parking_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch t_parking_sim auto_t_parking.launch.py \
  map_mode:=saved \
  auto_start:=false \
  execute:=true \
  target_slot:=auto
```

터미널 2:

```bash
source /opt/ros/jazzy/setup.bash
source ~/t_parking_ws/install/setup.bash

ros2 service call /t_parking/start std_srvs/srv/Trigger "{}"
```

### 4. Parallel Parking

터미널 1:

```bash
cd ~/t_parking_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch t_parking_sim auto_parallel_parking.launch.py \
  map_mode:=saved \
  initial_pose_x:=14.705 \
  initial_pose_y:=-0.50 \
  initial_pose_yaw:=1.57079632679 \
  auto_start:=false \
  execute:=true \
  target_slot:=auto
```

터미널 2:

```bash
source /opt/ros/jazzy/setup.bash
source ~/t_parking_ws/install/setup.bash

ros2 service call /parallel_parking/start std_srvs/srv/Trigger "{}"
```

## Environment

* Ubuntu 24.04
* ROS 2 Jazzy
* Gazebo Sim
* Nav2
* AMCL
* SLAM Toolbox
* RViz2

## Main Features

* Automatic T parking
* Automatic parallel parking
* Saved map + AMCL localization
* SLAM Toolbox mapping
* Nav2 path planning
* Forward / reverse segment execution
* Direction-locked RPP controller
* Ackermann steering
* Front / rear LiDAR obstacle safety
* Gazebo vehicle simulation

## Package Structure

```text
t_parking_sim/
├── behavior_trees/
├── config/
├── controller/
├── include/
├── launch/
├── maps/
├── rviz/
├── scripts/
├── urdf/
└── worlds/
```

## Saved Map

기본 saved map은 패키지에 포함되어 있습니다.

```text
maps/
├── combined_parking_map_real_vehicle.yaml
└── combined_parking_map_real_vehicle.pgm
```

따라서 기본 실행에서는 별도의 `map:=...` 인자가 필요하지 않습니다.

다른 map을 사용하려면:

```bash
ros2 launch t_parking_sim auto_t_parking.launch.py \
  map_mode:=saved \
  map:=/path/to/map.yaml \
  auto_start:=false
```

평행주차도 동일하게 `map:=/path/to/map.yaml`로 변경할 수 있습니다.

## SLAM Mapping

기본 mapping:

```bash
cd ~/t_parking_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch t_parking_sim mapping.launch.py
```

통합 주차장 mapping:

```bash
ros2 launch t_parking_sim combined_parking_mapping.launch.py auto_drive:=true
```

map 저장 예시:

```bash
mkdir -p ~/t_parking_maps

ros2 run nav2_map_server map_saver_cli \
  -f ~/t_parking_maps/combined_parking_map \
  --ros-args -p use_sim_time:=true
```

## Vehicle Configuration

현재 차량의 주요 기본값:

```text
Vehicle length     : 1.33 m
Overall width      : 0.78 m
Body width         : 0.74 m
Wheelbase          : 0.77 m
Wheel track        : 0.67 m
Wheel diameter     : 0.28 m
Wheel width        : 0.11 m
Steering limit     : ±27 deg
```

다른 차량에 적용할 경우 다음 값을 반드시 확인해야 합니다.

* vehicle length / width
* wheelbase / wheel track
* wheel dimensions
* maximum steering angle
* LiDAR position / height / direction
* parking slot geometry
* obstacle safety distance
* AMCL initial pose
* Nav2 controller parameters
* topic / frame names

주요 설정 위치:

```text
urdf/turtle_car.urdf.xacro
config/t_parking_auto.yaml
config/parallel_parking_auto.yaml
config/nav2_params.yaml
launch/
```

## LiDAR

전방 / 후방 LiDAR를 사용합니다.

기본 frame:

```text
laser_link
rear_laser_link
```

LiDAR 높이는 xacro argument 및 launch argument를 통해 변경할 수 있습니다.

## Vehicle Command Topics

현재 차량 명령 계약:

```text
/lidar_drive    std_msgs/Float32
/lidar_wheel    std_msgs/Int32
/lidar_stop     std_msgs/Bool
```

기본 조향 규약:

```text
negative wheel = left
positive wheel = right
maximum        = ±27 deg
```

Gazebo bridge 사용 시 `/lidar_*` 명령을 `/cmd_vel`로 변환합니다.

실차 등 외부 MCU 시스템을 사용할 경우 Gazebo bridge를 끌 수 있습니다.

```bash
ros2 launch t_parking_sim auto_t_parking.launch.py \
  enable_gazebo_bridge:=false \
  auto_start:=false \
  execute:=true
```

## Online SLAM Mode

saved map 대신 online SLAM을 사용하려면:

```bash
ros2 launch t_parking_sim auto_t_parking.launch.py \
  map_mode:=online
```

## Plan Only

차량을 움직이지 않고 주차 경로만 생성하려면:

```bash
ros2 launch t_parking_sim auto_t_parking.launch.py \
  execute:=false
```

## Troubleshooting

새 터미널에서는 항상 먼저:

```bash
source /opt/ros/jazzy/setup.bash
source ~/t_parking_ws/install/setup.bash
```

clean build가 필요하면:

```bash
cd ~/t_parking_ws
rm -rf build install log

source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install
```

정상적으로 빌드되면 마지막에 다음과 같이 표시됩니다.

```text
Summary: 1 package finished
```

## License

See `LICENSE`.
