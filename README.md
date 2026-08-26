# t_parking_ws

ROS 2 Jazzy + Gazebo Sim 기반 Ackermann 차량의 **T자 주차 / 평행주차 프로젝트**입니다.

## Quick Start

### 1. Clone

```bash
cd ~
git clone https://github.com/hanlacar/t_parking_ws.git
cd ~/t_parking_ws
```

### 2. Dependencies

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Build

```bash
colcon build --symlink-install
source install/setup.bash
```

### 4. T Parking

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

### 5. Parallel Parking

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
* SLAM Toolbox
* RViz2

## Features

* T자 자동주차
* 평행 자동주차
* Saved map + AMCL localization
* SLAM Toolbox mapping
* Ackermann steering
* 전진 / 후진 segment 실행
* Direction-locked RPP controller
* 전 / 후방 LiDAR obstacle safety
* Gazebo 차량 시뮬레이션

## Maps

기본 saved map은 프로젝트에 포함되어 있습니다.

```text
src/t_parking_sim/maps/
├── combined_parking_map_real_vehicle.yaml
└── combined_parking_map_real_vehicle.pgm
```

다른 map을 사용하려면:

```bash
ros2 launch t_parking_sim auto_t_parking.launch.py \
  map_mode:=saved \
  map:=/path/to/map.yaml
```

## Vehicle Settings

다른 차량에 적용할 경우 다음 값을 차량에 맞게 확인하거나 수정해야 합니다.

* vehicle length / width
* wheelbase / wheel track
* wheel size
* maximum steering angle
* LiDAR position / height / direction
* obstacle safety distance
* parking slot geometry
* AMCL initial pose
* Nav2 controller parameters
* topic / frame names

주요 설정 위치:

```text
src/t_parking_sim/urdf/
src/t_parking_sim/config/
src/t_parking_sim/launch/
```

현재 차량 명령 토픽은 다음 계약을 사용합니다.

```text
/lidar_drive
/lidar_wheel
/lidar_stop
```

## Troubleshooting

새 터미널에서는 ROS 2와 workspace를 다시 source해야 합니다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/t_parking_ws/install/setup.bash
```

빌드 문제가 발생하면 clean build를 진행합니다.

```bash
cd ~/t_parking_ws
rm -rf build install log

source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

## License

라이선스는 다음 파일을 확인하세요.

```text
src/t_parking_sim/LICENSE
```
