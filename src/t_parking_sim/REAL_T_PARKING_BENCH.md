# T-parking wheels-off-ground BENCH

> **Immediate stop:** `ros2 topic pub --once /lidar_stop std_msgs/msg/Bool
> "{data: true}"`
>
> This launch is only for a vehicle securely supported with every wheel off
> the ground. Never run it with a wheel touching the ground.

The BENCH path uses the production T-parking planner and the existing
`ParkingForward` / `ParkingReverse` controllers. The final controller Twist is
converted to `/lidar_drive`, `/lidar_wheel`, and `/lidar_stop`. An isolated
`/bench/odom` closes the Nav2 feedback loop, but advances only while those
actuator topics are fresh and agree with the requested direction. Production
`/odom`, `odom`, and `base_link` are not replaced.

## FIELD MCU contract

The competition vehicle uses the Arduino Mega 2560 with the directly flashed
`T870_FIELD_v9_0904` firmware at 115200 baud. This launch neither needs nor
starts `mcu_manager`, `mcu_bridge`, or a `/vehicle_mode` publisher. Its optional
`t870_field_serial_bridge.py` is the only adapter between the existing ROS
topics and the firmware's serial characters.

| ROS input | Serial output | Meaning |
|---|---|---|
| `/lidar_drive=-1` | `b\n` | reverse PWM stage 1 |
| `/lidar_drive=0` | `0` | immediate drive stop |
| `/lidar_drive=1` | `w\n` | forward PWM stage 1 |
| `/lidar_drive=2` | `e\n` | forward PWM stage 2 |
| `/lidar_drive=3` | `r\n` | forward PWM stage 3 |
| `/lidar_wheel=-1..-22` | `L1\n` .. `L22\n` | left steering |
| `/lidar_wheel=0` | `C\n` | centre steering |
| `/lidar_wheel=1..22` | `R1\n` .. `R22\n` | right steering |
| `/lidar_stop=true` | `0` | immediate drive stop |

An invalid drive value or steering outside `-22..22` latches a bridge fault,
writes repeated `0` stops, publishes `/t_parking/emergency_stop_request=true`,
and requests `/t_parking/cancel` when available. It does not clamp an invalid
steering command. If any of the three command inputs is older than 0.5 s, the
bridge sends `0`; shutdown, serial exceptions, and disconnects also send
repeated stops.

The final BENCH/real steering contract is +/-22 degrees. It uses wheelbase
0.73 m and a 1.82 m minimum turning radius. The bicycle-model theoretical
limit is `0.73 / tan(22 deg) = 1.8068 m`; the planner value is rounded up and
must not be reduced to force a path through. Gazebo, BENCH, and FIELD paths
all use the same measured geometry contract.

The FIELD firmware declares `STEER_STAGES = 22`, so the ROS and firmware
command ranges agree. Any future steering calibration must be performed and
verified as a separate hardware procedure; this repository must not invent
ADC endpoint measurements.

## Before connecting the vehicle

Do not infer the steering centre from a comment or replace it with 497. With
the car supported, use the firmware's `S` command and physically verify the
reported centre ADC and left/right direction first. Also confirm which sketch
is actually flashed: the checked-out `T870_FIELD_v9_0904.ino` currently differs
from its header comments in centre/PWM constants.

Confirm the serial symlink and that nothing else owns it:

```bash
readlink -f /dev/t870_mcu
fuser /dev/t870_mcu
```

## Run without serial output

```bash
cd ~/t_parking_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch t_parking_sim real_t_parking_bench.launch.py \
  map:=$HOME/t_parking_maps/real_cart_slam_T_park_handB.yaml \
  initial_pose_x:=0.24247 \
  initial_pose_y:=0.05097 \
  initial_pose_yaw:=3.14159265 \
  target_slot:=auto \
  auto_start:=false \
  rviz:=true \
  serial_bridge:=false \
  hardware_ack:=false \
  bench_ack:=true
```

`bench_ack:=true` is always mandatory. `serial_bridge:=false` is the default
and cannot open a serial port. When `serial_bridge:=true`, the launch additionally
requires the explicit `hardware_ack:=true`; otherwise it refuses to start before
opening the device.

Inspect the command path before starting:

```bash
ros2 topic info /lidar_drive -v
ros2 topic info /lidar_wheel -v
ros2 topic info /lidar_stop -v
ros2 topic hz /lidar_drive
ros2 topic hz /lidar_wheel
ros2 topic hz /lidar_stop
```

Start the planner/controller sequence from another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/t_parking_ws/install/setup.bash
ros2 service call /t_parking/start std_srvs/srv/Trigger "{}"
```

`[BENCH CMD]`, `[BENCH SEGMENT START]`, `[BENCH SEGMENT END]`, and
`[BENCH FINISHED]` logs show the motion contract and emitted commands. Normal
completion is `/lidar_drive=0`, `/lidar_wheel=0`, `/lidar_stop=false`; `true`
is reserved for a fault stop.

## Final wheels-off-ground launch

Only after checking `S`, the physical E-stop, wheel support, serial ownership,
and steering direction:

```bash
ros2 launch t_parking_sim real_t_parking_bench.launch.py map:=$HOME/t_parking_maps/real_cart_slam_T_park_handB.yaml target_slot:=auto auto_start:=false rviz:=true serial_bridge:=true serial_port:=/dev/t870_mcu serial_baud:=115200 hardware_ack:=true bench_ack:=true
```

This command only brings the guarded BENCH stack up; it does not auto-start the
parking sequence. Keep the immediate-stop command above ready in another
terminal.
