# t_parking_sim

ROS 2 Jazzy와 Gazebo Sim용 T자 직각주차 코스이다. 커스텀 4륜
`turtle_car`, 실제 앞바퀴 조향/뒷바퀴 구동 Ackermann 시스템, 전·후방
2D LiDAR, odometry, TF, SLAM Toolbox 온라인 비동기 매핑, Nav2 경로 계획과
`FollowPath` 기반 자동 T자 주차를 포함한다.

## 1. 환경 요구사항

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Sim 8 (`gz sim`), Gazebo Classic 아님
- `ament_cmake`, `colcon`

필요한 ROS 패키지:

- `ros_gz_sim`, `ros_gz_bridge`
- `xacro`, `robot_state_publisher`
- `slam_toolbox`, `rviz2`
- `nav2_map_server`, `nav2_planner`, `nav2_controller`, `nav2_bt_navigator`
- `teleop_twist_keyboard`, `tf2_ros`

현재 설치 상태는 다음처럼 확인한다.

```bash
source /opt/ros/jazzy/setup.bash
gz sim --versions
ros2 pkg prefix ros_gz_sim
ros2 pkg prefix ros_gz_bridge
ros2 pkg prefix slam_toolbox
ros2 pkg prefix nav2_map_server
```

## 2. 의존성 설치와 빌드

워크스페이스 루트는 `/home/wan/t_parking_ws`이다.

```bash
cd /home/wan/t_parking_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select t_parking_sim
source install/setup.bash
```

새 터미널에서는 ROS 명령을 실행하기 전에 반드시 두 환경을 다시 source한다.
이 절차를 생략한 `Package 't_parking_sim' not found` 오류는 코드나 launch
오류가 아니다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/t_parking_ws/install/setup.bash
```

`rosdep`가 `nav2_map_server` 또는 키보드 주행 패키지의 누락을 보고하면:

```bash
sudo apt update
sudo apt install ros-jazzy-nav2-map-server ros-jazzy-teleop-twist-keyboard
```

위 `sudo` 명령은 사용자가 직접 검토한 뒤 실행해야 한다.

## 3. 시뮬레이션만 실행

```bash
cd /home/wan/t_parking_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch t_parking_sim sim.launch.py
```

GUI가 없는 환경:

```bash
ros2 launch t_parking_sim sim.launch.py gui:=false
```

`sim.launch.py`의 주요 인자는 `use_sim_time`, `gui`, `rviz`, `world`,
`practice_mode`, `x`, `y`, `z`, `yaw`이다.

차량과 장애물은 `vehicle_spawn_manager` 노드가 모두 스폰한다. 이 노드는
`/clock`이 흐르기 시작할 때까지 기다린 뒤 동작하므로 별도의 지연 설정이
필요 없다.

### 연습 모드

시작 자세는 `x`/`y`/`z`/`yaw`가 아니라 `practice_mode`가 정한다.

| `practice_mode` | 시작 자세 | 용도 |
| --- | --- | --- |
| `t_parking` (기본) | `(-3.50, 0.0)`, yaw `0` | T베이 바로 앞에서 시작 |
| `full_course` | `(-5.0, 0.0)`, yaw `0` | 메인 차로 서쪽 끝부터 전 구간 |
| `parallel_parking` | `(9.75, -0.50)`, yaw `90°` | 하단 출발존부터 평행주차 전 구간 |
| `parallel_ready` | `(8.75, 10.25)`, yaw `180°` | 평행 슬롯 바로 앞에서 시작 |
| `custom` | `x`/`y`/`z`/`yaw` 인자 | 임의 위치 |

```bash
ros2 launch t_parking_sim sim.launch.py practice_mode:=parallel_parking
```

`x`/`y`/`z`/`yaw`는 `practice_mode:=custom`일 때만 적용된다. 다른 모드에서
이 인자를 넘기면 조용히 무시된다.

### 연습용 장애물

실행할 때마다 T베이 두 칸 중 하나와 평행 슬롯 두 칸 중 하나가 무작위로
막힌다. 슬롯 이름은 T베이가 `A`(서), `B`(동), 평행 슬롯이 `C`(서),
`D`(동)이다.

```bash
# 현재 배치 확인 (예: "A-C")
ros2 topic echo /parking_practice/obstacle_layout --once

# 스폰 진행 상태
ros2 topic echo /parking_practice/spawn_status

# 장애물 재배치와 차량 시작 자세 복귀 (재실행 없이)
ros2 service call /parking_practice/respawn std_srvs/srv/Trigger
```

## 4. SLAM mapping 실행

다음 명령 하나가 시뮬레이션, bridge, robot state publisher, SLAM
Toolbox online asynchronous node와 mapping RViz를 시작한다.

```bash
cd /home/wan/t_parking_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch t_parking_sim mapping.launch.py
```

headless mapping:

```bash
ros2 launch t_parking_sim mapping.launch.py gui:=false rviz:=false
```

`mapping.launch.py`와 `nav2_mapping.launch.py`도 `practice_mode`를 그대로
`sim.launch.py`에 전달한다.

```bash
ros2 launch t_parking_sim mapping.launch.py practice_mode:=parallel_parking
```

## 5. 키보드 주행

새 터미널에서 환경을 다시 source한 뒤 실행한다.

```bash
source /opt/ros/jazzy/setup.bash
source /home/wan/t_parking_ws/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=/cmd_vel
```

`i` 전진, `,` 후진, `j` 좌회전, `l` 우회전, `k` 정지이다.
주차 코스에서는 시작 직후 `z`를 여러 번 눌러 선속도를 약
`0.3~0.5 m/s`까지 낮추고, `x`로 각속도도 낮춘 뒤 조작한다.
Ackermann 입력에서 `linear.x`는 선속도, `angular.z`는 목표 yaw rate이다.
후진 시 차체의 회전은 실제 자동차처럼 속도 부호에 따라 반대로 나타난다.

## 6. Nav2 자동 T자 주차

다음 명령 하나가 `auto_t_parking -> nav2_mapping -> mapping -> sim` 순서로
Gazebo, bridge, SLAM, Nav2와 자동주차 노드를 각각 한 번만 시작한다.

```bash
cd ~/t_parking_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch t_parking_sim auto_t_parking.launch.py \
  auto_start:=false execute:=true target_slot:=auto
```

`auto_start=false`이면 모든 readiness 항목이 true가 된 뒤에도 차량은
정지한 채 시작 서비스를 기다린다.

```bash
ros2 service call /t_parking/start std_srvs/srv/Trigger "{}"
```

`execute=false`는 검증된 계획과 시각화 토픽만 발행하는 계획 전용 모드이고,
`execute=true`는 검증 후 `/follow_path`를 호출하는 실제 실행 모드이다.
실행 상태는 `/t_parking/status`, 계획은 `/t_parking/planned_path`에서 확인한다.

이 launch의 `practice_mode` 기본값은 `full_course`이며 바꾸면 안 된다.
`config/t_parking_auto.yaml`의 슬롯 좌표가 초기 월드 자세 `(-5, 0, yaw=0)`
기준으로 `odom_x = world_x + 5`처럼 계산돼 있어서, 시작 자세를 바꾸면 odom
원점이 이동해 슬롯 경계가 모두 어긋난다.

평행주차용 자동 노드는 없다. 평행주차는 5절의 키보드 주행으로 연습한다.

## 7. 토픽 확인

```bash
ros2 topic list
ros2 topic hz /scan
ros2 topic echo /scan --once
ros2 topic hz /scan_rear
ros2 topic echo /scan_rear --once
ros2 topic hz /odom
ros2 topic echo /odom --once
ros2 topic echo /joint_states --once
ros2 topic echo /map --once
```

핵심 토픽은 `/clock`, `/cmd_vel`, `/scan`, `/scan_rear`, `/odom`, `/tf`,
`/tf_static`, `/joint_states`, `/robot_description`, `/map`,
`/map_metadata`이다. `/scan`의 `header.frame_id`는 `laser_link`,
`/odom`은 `header.frame_id=odom`, `child_frame_id=base_footprint`이어야
한다. `/scan_rear`의 `header.frame_id`는 `rear_laser_link`이다.

Gazebo 내부 이름은 다음으로 확인한다.

```bash
gz topic -l
```

## 8. TF 확인

```bash
ros2 run tf2_ros tf2_echo odom base_footprint
ros2 run tf2_ros tf2_echo base_link laser_link
ros2 run tf2_ros tf2_echo base_footprint rear_laser_link
ros2 run tf2_ros tf2_echo map odom
```

책임은 SLAM Toolbox `map -> odom`, Gazebo Ackermann odometry
`odom -> base_footprint`, robot_state_publisher가 그 아래 차량 링크들이다.
`map -> odom`과 `odom -> base_footprint`는 static TF가 아니다.

## 9. 지도 저장

충분히 주행해 지도가 누적된 뒤 새 터미널에서:

```bash
mkdir -p ~/t_parking_maps
source /opt/ros/jazzy/setup.bash
source /home/wan/t_parking_ws/install/setup.bash
ros2 run nav2_map_server map_saver_cli \
  -f ~/t_parking_maps/t_parking_exam_map \
  --ros-args -p use_sim_time:=true
```

결과 파일:

- `~/t_parking_maps/t_parking_exam_map.yaml`
- `~/t_parking_maps/t_parking_exam_map.pgm`

## 10. 자주 발생하는 오류

- `Package 't_parking_sim' not found`: 빌드 후
  `source /home/wan/t_parking_ws/install/setup.bash`를 다시 실행한다.
- `Package 'nav2_map_server' not found`: 위 의존성 설치 절의
  `ros-jazzy-nav2-map-server`를 설치한다.
- `teleop_twist_keyboard`를 찾지 못함:
  `ros-jazzy-teleop-twist-keyboard`를 설치한다.
- `/scan` 없음: Gazebo가 재생 중인지 확인하고 `gz topic -l`에서
  `/scan`이 있는지 본다. GPU/OGRE 오류가 나는 원격 환경에서는
  `gui:=false`로 먼저 검사한다.
- `/scan_rear` 없음: `gz topic -l`과 `gz topic -i -t /scan_rear`로 Gazebo
  센서 출력을 먼저 확인하고, 그 다음 `config/bridge.yaml`의 GZ→ROS bridge와
  새 터미널의 workspace source 여부를 확인한다.
- RViz의 `No transform from laser_link`: `/joint_states`, `/tf`,
  `/tf_static`을 확인하고 모든 프로세스가 `use_sim_time=true`인지 본다.
- `/map`이 갱신되지 않음: `/scan`과 `/odom`의 주기, `map -> odom`,
  `/clock`을 차례로 확인한다. 차량을 저속으로 0.05 m 이상 이동한다.
- 차량이 움직이지 않음: Gazebo pause를 해제하고 `/cmd_vel`을 echo해
  키보드 입력이 도착하는지 확인한다.
- 이전 Gazebo 프로세스 때문에 모델 생성이 실패함: 이전 launch를
  `Ctrl-C`로 정상 종료한 후 `gz topic -l`이 비었는지 확인하고 다시
  실행한다.
- 차량이나 장애물이 스폰되지 않음: `/parking_practice/spawn_status`를
  echo해 어느 단계에서 멈췄는지 본다. `WAITING_FOR_SIM`이 계속되면
  `/clock`이 흐르지 않는 것이고, `WAITING_FOR_DESCRIPTION`이 계속되면
  `robot_state_publisher`가 `/robot_description`을 발행하지 못한 것이다.
  `gz model --list`에는 `turtle_car` 한 개와 장애물 두 개가 보여야 한다.
