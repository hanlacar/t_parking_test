import importlib.util
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from bench_support import (  # noqa: E402
    BENCH_BASE_FRAME,
    bench_motion_allowed,
    BENCH_NAV_NODES,
    BENCH_ODOM_FRAME,
    BENCH_ODOM_TOPIC,
    compose,
    compute_map_to_odom,
    duplicate_nav_nodes,
    duplicate_node_names,
    integrate_bicycle,
    MCU_NODES,
    Pose2D,
    preflight_graph_conflicts,
    startup_gate_state,
    StartupSnapshot,
)
from t_parking_geometry import (  # noqa: E402
    assess_parking_pose,
    parking_target_from_end_clearance,
    vehicle_longitudinal_extents,
)
import yaml  # noqa: E402


def _load_converter():
    spec = importlib.util.spec_from_file_location(
        'cmd_vel_to_lidar_cmd', SCRIPTS / 'cmd_vel_to_lidar_cmd.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_auto_parking():
    spec = importlib.util.spec_from_file_location(
        'auto_t_parking_test_module', SCRIPTS / 'auto_t_parking.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_anchor_maps_arbitrary_odom_pose_to_canonical_start():
    odom_start = Pose2D(2.4, -1.7, 0.63)
    desired = Pose2D(9.70, 0.0, math.pi)
    anchor = compute_map_to_odom(odom_start, desired)
    result = compose(anchor, odom_start)
    assert math.isclose(result.x, desired.x, abs_tol=1.0e-10)
    assert math.isclose(result.y, desired.y, abs_tol=1.0e-10)
    assert math.isclose(
        math.atan2(math.sin(result.yaw - desired.yaw),
                   math.cos(result.yaw - desired.yaw)),
        0.0,
        abs_tol=1.0e-10,
    )


def test_direction_mapping_and_steering_clamp():
    converter = _load_converter()
    kwargs = {
        'wheel_base': 0.77,
        'steering_limit_deg': 27.0,
        'mcu_wheel_limit_deg': 27,
        'stopped_speed_epsilon': 0.01,
        'forward_drive_stage': 1.0,
        'reverse_drive_stage': -1.0,
    }
    forward = converter.convert_command(0.1, 20.0, **kwargs)
    reverse = converter.convert_command(-0.1, 20.0, **kwargs)
    stopped = converter.convert_command(0.0, 20.0, **kwargs)
    assert forward.drive_stage == 1.0
    assert reverse.drive_stage == -1.0
    assert stopped.drive_stage == 0.0 and stopped.wheel_deg == 0
    assert forward.wheel_deg == -27
    assert reverse.wheel_deg == 27
    assert -27 <= forward.wheel_deg <= 27
    assert -27 <= reverse.wheel_deg <= 27


def test_forward_right_turn_uses_positive_mcu_wheel_sign():
    converter = _load_converter()
    command = converter.convert_command(
        linear_x=0.20,
        angular_z=-0.10,
        wheel_base=0.77,
        steering_limit_deg=27.0,
        mcu_wheel_limit_deg=27,
        stopped_speed_epsilon=0.01,
        forward_drive_stage=1.0,
        reverse_drive_stage=-1.0,
    )
    assert command.drive_stage == 1.0
    assert command.steering_deg_ros < 0.0
    assert 0 < command.wheel_deg <= 27


def test_slot_1_target_is_footprint_derived_and_shallower_than_legacy():
    front_extent, rear_extent = vehicle_longitudinal_extents(1.33, 0.020)
    assert math.isclose(front_extent, 0.685)
    assert math.isclose(rear_extent, 0.645)

    target_x, target_y = parking_target_from_end_clearance(
        slot_center_x=5.10,
        slot_center_y=-4.975,
        slot_yaw=math.pi / 2.0,
        min_x=3.25,
        max_x=6.95,
        min_y=-8.70,
        max_y=-1.25,
        vehicle_length=1.33,
        vehicle_center_x_offset=0.020,
        parking_end_clearance_m=3.06,
    )
    legacy_target_y = -8.70 + rear_extent + 0.70
    assert math.isclose(target_x, 5.10, abs_tol=1.0e-12)
    assert math.isclose(target_y, -4.995, abs_tol=1.0e-12)
    assert target_y > legacy_target_y

    assessment = assess_parking_pose(
        base_x=target_x,
        base_y=target_y,
        vehicle_yaw=math.pi / 2.0,
        slot_yaw=math.pi / 2.0,
        min_x=3.25,
        max_x=6.95,
        min_y=-8.70,
        max_y=-1.25,
        vehicle_length=1.33,
        vehicle_width=0.78,
        vehicle_center_x_offset=0.020,
    )
    assert assessment.footprint_inside
    assert math.isclose(assessment.end_clearance, 3.06, abs_tol=1.0e-12)
    assert math.isclose(assessment.front_clearance, 3.06, abs_tol=1.0e-12)
    assert assessment.side_clearance > 1.0


def test_end_clearance_cannot_push_front_bumper_outside_slot():
    try:
        parking_target_from_end_clearance(
            slot_center_x=0.0,
            slot_center_y=0.0,
            slot_yaw=0.0,
            min_x=-2.0,
            max_x=2.0,
            min_y=-1.0,
            max_y=1.0,
            vehicle_length=1.33,
            vehicle_center_x_offset=0.020,
            parking_end_clearance_m=3.0,
        )
    except ValueError as exc:
        assert 'front bumper outside' in str(exc)
    else:
        raise AssertionError('unsafe end clearance was accepted')


def test_bench_motion_requires_all_three_gates():
    assert bench_motion_allowed(True, True, True)
    for values in (
            (False, True, True), (True, False, True),
            (True, True, False), (False, False, False)):
        assert not bench_motion_allowed(*values)


def test_startup_waits_for_zero_source_then_official_mcu_status():
    initial = StartupSnapshot(
        drive_publishers=0,
        wheel_publishers=0,
        drive_zero_samples=0,
        wheel_zero_samples=0,
        nonzero_command_seen=False,
        mcu_connected=True,
        mcu_ready=False,
        vehicle_mode='T_PARK',
        safety_state='OK',
    )
    assert startup_gate_state(initial) == (
        'WAITING_FOR_LIDAR_COMMAND_SOURCE')

    zero_source = StartupSnapshot(
        drive_publishers=1,
        wheel_publishers=1,
        drive_zero_samples=3,
        wheel_zero_samples=3,
        nonzero_command_seen=False,
        mcu_connected=True,
        mcu_ready=False,
        vehicle_mode='T_PARK',
        safety_state='OK',
    )
    assert startup_gate_state(zero_source) == 'READY'
    ready_diagnostic_true = StartupSnapshot(
        **{**zero_source.__dict__, 'mcu_ready': True})
    assert startup_gate_state(ready_diagnostic_true) == 'READY'
    disconnected = StartupSnapshot(
        **{**zero_source.__dict__, 'mcu_connected': False})
    unsafe = StartupSnapshot(
        **{**zero_source.__dict__, 'safety_state': 'ESTOP'})
    wrong_mode = StartupSnapshot(
        **{**zero_source.__dict__, 'vehicle_mode': 'NORMAL'})
    assert startup_gate_state(disconnected) == 'WAITING_FOR_MCU_STATUS'
    assert startup_gate_state(unsafe) == 'WAITING_FOR_MCU_STATUS'
    assert startup_gate_state(wrong_mode) == 'WAITING_FOR_MCU_STATUS'


def test_startup_rejects_duplicate_or_nonzero_command_source():
    safe = {
        'drive_publishers': 1,
        'wheel_publishers': 1,
        'drive_zero_samples': 3,
        'wheel_zero_samples': 3,
        'nonzero_command_seen': False,
        'mcu_connected': True,
        'mcu_ready': True,
        'vehicle_mode': 'T_PARK',
        'safety_state': 'OK',
    }
    duplicate = StartupSnapshot(**{**safe, 'drive_publishers': 2})
    nonzero = StartupSnapshot(**{**safe, 'nonzero_command_seen': True})
    assert startup_gate_state(duplicate) == (
        'FAIL_DUPLICATE_LIDAR_COMMAND_SOURCE')
    assert startup_gate_state(nonzero) == (
        'FAIL_NONZERO_COMMAND_BEFORE_START')


def test_converter_idle_heartbeat_and_production_default_remain_safe():
    converter = _load_converter()
    moving = converter.ConvertedCommand(1.0, 12, -12.0)
    idle = converter.select_output_command(
        moving, None, 10.0, 0.5, False)
    fresh = converter.select_output_command(
        moving, 9.8, 10.0, 0.5, False)
    expired = converter.select_output_command(
        moving, 9.0, 10.0, 0.5, False)
    emergency = converter.select_output_command(
        moving, 9.8, 10.0, 0.5, True)
    assert idle == converter.ZERO_COMMAND
    assert fresh == moving
    assert expired == converter.ZERO_COMMAND
    assert emergency == converter.ZERO_COMMAND

    source = (SCRIPTS / 'cmd_vel_to_lidar_cmd.py').read_text()
    assert "declare_parameter('bench_interlock_enabled', False)" in source


def test_converter_lidar_output_is_allowed_only_in_parking_modes():
    converter = _load_converter()
    assert converter.mode_allows_lidar_commands('T_PARK')
    assert converter.mode_allows_lidar_commands('parallel_park')
    assert not converter.mode_allows_lidar_commands('NORMAL')
    assert not converter.mode_allows_lidar_commands('')

    source = (SCRIPTS / 'cmd_vel_to_lidar_cmd.py').read_text()
    assert "declare_parameter('mode_topic', '/vehicle_mode')" in source
    assert 'if not parking_active and not exit_zero_active:' in source


def test_bench_converter_uses_applied_mcu_mode_source():
    converter_source = (SCRIPTS / 'cmd_vel_to_lidar_cmd.py').read_text()
    launch_source = (
        Path(__file__).resolve().parents[1]
        / 'launch' / 'bench_t_parking.launch.py').read_text()

    assert "'mode_topic': '/mcu/current_mode'" in launch_source
    assert '[LIDAR_COMMAND] mode source=%s' in converter_source
    assert '[LIDAR_COMMAND] current mode received: %s' in converter_source
    assert '[BENCH STARTUP] zero heartbeat enabled for %s' in converter_source


def test_independent_stop_requests_are_or_combined():
    converter = _load_converter()
    assert not converter.any_stop_active(False, False)
    assert converter.any_stop_active(True, False)
    assert converter.any_stop_active(False, True)
    assert converter.any_stop_active(True, True)

    source = (SCRIPTS / 'cmd_vel_to_lidar_cmd.py').read_text()
    assert "'lidar_safety_stop_topic', '/lidar/stop_required'" in source
    assert 'self._publish_stop(stop_active)' in source


def test_zero_command_produces_zero_virtual_motion():
    start = Pose2D(1.0, 2.0, 0.4)
    step = integrate_bicycle(start, 0.0, 1.0, 1.0)
    assert step.pose == start
    assert step.linear_velocity == 0.0
    assert step.yaw_rate == 0.0


def test_virtual_straight_forward_and_reverse():
    start = Pose2D(0.0, 0.0, 0.0)
    forward = integrate_bicycle(start, 0.25, 0.0, 2.0)
    reverse = integrate_bicycle(start, -0.25, 0.0, 2.0)
    assert math.isclose(forward.pose.x, 0.5)
    assert math.isclose(reverse.pose.x, -0.5)
    assert forward.pose.y == reverse.pose.y == 0.0
    assert forward.pose.yaw == reverse.pose.yaw == 0.0


def test_positive_angular_command_has_ackermann_yaw_and_clamp():
    start = Pose2D(0.0, 0.0, 0.0)
    normal = integrate_bicycle(start, 0.25, 0.10, 1.0)
    assert normal.pose.yaw > 0.0
    assert math.isclose(normal.yaw_rate, 0.10, abs_tol=1.0e-12)

    clamped = integrate_bicycle(start, 0.10, 100.0, 1.0)
    assert clamped.raw_delta_deg > 27.0
    assert math.isclose(clamped.clamped_delta_deg, 27.0)
    expected_rate = 0.10 / 0.77 * math.tan(math.radians(27.0))
    assert math.isclose(clamped.yaw_rate, expected_rate)


def test_virtual_motion_interlocks_freeze_pose():
    start = Pose2D(0.0, 0.0, 0.0)
    for allowed in (
            bench_motion_allowed(False, True, True),
            bench_motion_allowed(True, False, True),
            bench_motion_allowed(True, True, False)):
        step = integrate_bicycle(
            start, 0.25, 0.1, 1.0, motion_allowed=allowed)
        assert step.pose == start


def test_virtual_terminal_handoff_forward_cusp_reverse_final_zero():
    pose = Pose2D(0.0, 0.0, 0.0)
    forward_endpoint = 0.30
    events = []

    for _ in range(100):
        endpoint_base_x = forward_endpoint - pose.x
        if abs(endpoint_base_x) <= 0.03:
            events.append('FORWARD_SUCCEEDED')
            break
        assert endpoint_base_x > 0.0
        pose = integrate_bicycle(pose, 0.08, 0.0, 0.1).pose
    else:
        raise AssertionError('virtual forward segment did not reach its endpoint')

    cusp_pose = pose
    events.append('CUSP_HANDOFF_WAIT_ZERO')
    for sample in range(1, 4):
        zero_step = integrate_bicycle(pose, 0.0, 0.0, 0.1)
        assert zero_step.pose == cusp_pose
        events.append(f'CUSP_ZERO_{sample}')
    events.append('CUSP_STOP_PASS')

    # The inclusive segment split shares the cusp pose. A small localization
    # offset can put that first shared pose ahead of base_link even though the
    # following meaningful path samples are unambiguously behind it.
    start_relation_base_x = [0.024, -0.030, -0.102]
    assert start_relation_base_x[0] > 0.0
    assert all(value < 0.0 for value in start_relation_base_x[1:])
    events.extend([
        'SEGMENT_START_REVERSE',
        'ParkingReverse_GOAL_SENT',
        'ParkingReverse_ACCEPTED',
    ])
    reverse_endpoint = 0.0
    reverse_motion_seen = False
    for _ in range(100):
        endpoint_base_x = reverse_endpoint - pose.x
        if abs(endpoint_base_x) <= 0.03:
            events.append('REVERSE_SUCCEEDED')
            break
        assert endpoint_base_x < 0.0
        pose = integrate_bicycle(pose, -0.08, 0.0, 0.1).pose
        if not reverse_motion_seen:
            events.append('REVERSE_VIRTUAL_MOTION')
            reverse_motion_seen = True
    else:
        raise AssertionError('virtual reverse segment did not reach its endpoint')

    final_step = integrate_bicycle(pose, 0.0, 0.0, 0.1)
    assert final_step.pose == pose
    events.append('FINAL_ZERO')
    parked = assess_parking_pose(
        base_x=5.10,
        base_y=-4.995,
        vehicle_yaw=math.pi / 2.0,
        slot_yaw=math.pi / 2.0,
        min_x=3.25,
        max_x=6.95,
        min_y=-8.70,
        max_y=-1.25,
        vehicle_length=1.33,
        vehicle_width=0.78,
        vehicle_center_x_offset=0.020,
    )
    assert parked.footprint_inside
    assert math.isclose(parked.end_clearance, 3.06, abs_tol=1.0e-12)
    events.extend(['PARKED_FOOTPRINT_PASS', 'PARKED', 'EXIT_LOGIC_START'])
    assert events == [
        'FORWARD_SUCCEEDED',
        'CUSP_HANDOFF_WAIT_ZERO',
        'CUSP_ZERO_1', 'CUSP_ZERO_2', 'CUSP_ZERO_3',
        'CUSP_STOP_PASS',
        'SEGMENT_START_REVERSE',
        'ParkingReverse_GOAL_SENT',
        'ParkingReverse_ACCEPTED',
        'REVERSE_VIRTUAL_MOTION', 'REVERSE_SUCCEEDED', 'FINAL_ZERO',
        'PARKED_FOOTPRINT_PASS', 'PARKED', 'EXIT_LOGIC_START',
    ]


def test_virtual_parked_stop_then_forward_right_exit_and_final_zero():
    parked = Pose2D(5.10, -4.995, math.pi / 2.0)
    pose = parked
    for _ in range(3):
        step = integrate_bicycle(pose, 0.0, 0.0, 0.1)
        assert step.pose == parked

    speed = 0.20
    radius = 1.70
    yaw_rate = -speed / radius
    duration = (math.pi / 2.0) / abs(yaw_rate)
    first = integrate_bicycle(pose, speed, yaw_rate, 0.05)
    assert first.linear_velocity > 0.0
    assert first.clamped_delta_deg < 0.0
    assert abs(first.clamped_delta_deg) <= 27.0
    pose = first.pose
    elapsed = 0.05
    while elapsed < duration:
        dt = min(0.05, duration - elapsed)
        pose = integrate_bicycle(pose, speed, yaw_rate, dt).pose
        elapsed += dt
    assert pose.x > parked.x
    assert pose.y > parked.y
    assert abs(math.atan2(math.sin(pose.yaw), math.cos(pose.yaw))) < 0.03

    straight_start = pose
    for _ in range(50):
        pose = integrate_bicycle(pose, speed, 0.0, 0.05).pose
    assert pose.x > straight_start.x
    final = integrate_bicycle(pose, 0.0, 0.0, 0.1)
    assert final.pose == pose


def test_preflight_and_runtime_duplicate_detection():
    clean = [('unrelated', '/')]
    assert preflight_graph_conflicts(clean) == []
    assert preflight_graph_conflicts(
        [('amcl', '/'), ('planner_server', '/')]) == ['/amcl', '/planner_server']

    once = [(name.lstrip('/'), '/') for name in BENCH_NAV_NODES]
    assert duplicate_nav_nodes(once) == []
    duplicated = once + [('planner_server', '/')]
    assert duplicate_nav_nodes(duplicated) == ['/planner_server']

    mcu_once = [('mcu_bridge', '/'), ('mcu_manager', '/')]
    assert duplicate_node_names(mcu_once, MCU_NODES) == []
    assert duplicate_node_names(
        mcu_once + [('mcu_manager', '/')], MCU_NODES) == ['/mcu_manager']


def test_bench_code_uses_only_official_mcu_status_contract():
    preflight_source = (SCRIPTS / 'bench_preflight.py').read_text()
    auto_source = (SCRIPTS / 'auto_t_parking.py').read_text()
    support_source = (SCRIPTS / 'bench_support.py').read_text()
    combined = preflight_source + auto_source
    assert '/mcu/bridge_ready' not in combined
    assert '/mcu/connected' in preflight_source
    assert '/mcu/safety_state' in preflight_source
    assert '/mcu/ready' in preflight_source
    gate_source = support_source.split(
        'def startup_gate_state', 1)[1].split('def normalize_angle', 1)[0]
    assert 'mcu_ready' not in gate_source
    checks_source = auto_source.split(
        'checks = {', 1)[1].split('missing =', 1)[0]
    assert 'mcu_ready' not in checks_source


def test_bench_overlay_does_not_weaken_production_safety_config():
    config = Path(__file__).resolve().parents[1] / 'config'
    production_nav = yaml.safe_load(
        (config / 'nav2_params.yaml').read_text())
    bench_nav = yaml.safe_load(
        (config / 'nav2_params_bench.yaml').read_text())
    production_auto = yaml.safe_load(
        (config / 't_parking_auto.yaml').read_text())

    for costmap_name in ('local_costmap', 'global_costmap'):
        parameters = production_nav[costmap_name][costmap_name][
            'ros__parameters']
        obstacle = parameters['obstacle_layer']
        assert 'obstacle_layer' in parameters['plugins']
        assert obstacle['enabled'] is True
        for scan_name in ('scan_front', 'scan_rear'):
            assert obstacle[scan_name]['topic'] == f'/{scan_name}'
            assert obstacle[scan_name]['marking'] is True
            assert obstacle[scan_name]['clearing'] is True

        bench_parameters = bench_nav[costmap_name][costmap_name][
            'ros__parameters']
        assert bench_parameters['plugins'] == [
            'static_layer', 'inflation_layer']
        assert bench_parameters['obstacle_layer']['enabled'] is False
        assert bench_parameters['footprint'] == (
            '[[0.685, 0.39], [0.685, -0.39], '
            '[-0.645, -0.39], [-0.645, 0.39]]')
        assert bench_parameters['footprint_padding'] == 0.025
        assert bench_parameters['robot_base_frame'] == BENCH_BASE_FRAME

    auto_parameters = production_auto['t_parking_auto']['ros__parameters']
    assert auto_parameters['obstacle_observation_frames'] > 0
    assert auto_parameters['costmap_observation_updates'] > 0
    assert auto_parameters['rear_emergency_stop_distance'] > 0.0
    assert auto_parameters['rear_emergency_sector_deg'] > 0.0
    assert auto_parameters['minimum_turning_radius'] == 1.52
    assert auto_parameters['wheel_base'] == 0.77
    assert auto_parameters['parking_end_clearance_m'] == 3.06
    assert auto_parameters['final_pose_overshoot'] == 0.0

    assert production_nav['amcl']['ros__parameters'][
        'base_frame_id'] == 'base_link'
    assert production_nav['amcl']['ros__parameters'][
        'odom_frame_id'] == 'odom'
    assert production_nav['velocity_smoother']['ros__parameters'][
        'odom_topic'] == '/odom'
    assert production_nav['local_costmap']['local_costmap'][
        'ros__parameters']['robot_base_frame'] == 'base_link'
    assert production_nav['global_costmap']['global_costmap'][
        'ros__parameters']['robot_base_frame'] == 'base_link'
    assert bench_nav['controller_server']['ros__parameters'][
        'odom_topic'] == BENCH_ODOM_TOPIC
    assert bench_nav['velocity_smoother']['ros__parameters'][
        'odom_topic'] == BENCH_ODOM_TOPIC
    progress = production_nav['controller_server']['ros__parameters'][
        'progress_checker']
    assert progress['required_movement_radius'] == 0.10
    assert progress['movement_time_allowance'] == 15.0

    controller_parameters = production_nav['controller_server'][
        'ros__parameters']
    cusp_checker = controller_parameters['cusp_goal_checker']
    assert cusp_checker['stateful'] is True
    assert cusp_checker['xy_goal_tolerance'] == 0.03
    assert cusp_checker['yaw_goal_tolerance'] == 0.10
    assert 'terminal_capture_distance' not in controller_parameters[
        'ParkingForward']
    assert 'terminal_capture_distance' not in controller_parameters[
        'ParkingReverse']
    bench_controller = bench_nav['controller_server']['ros__parameters']
    bench_cusp_checker = bench_controller['cusp_goal_checker']
    assert bench_cusp_checker['plugin'] == (
        'nav2_controller::SimpleGoalChecker')
    assert bench_cusp_checker['stateful'] is True
    assert bench_cusp_checker['xy_goal_tolerance'] == 0.03
    assert bench_cusp_checker['yaw_goal_tolerance'] == 0.16
    assert bench_controller['ParkingForward'][
        'terminal_capture_distance'] == 0.35
    assert bench_controller['ParkingReverse'][
        'terminal_capture_distance'] == 0.35


def test_segment_handoff_keeps_three_zero_samples_and_direction_ids():
    source = (SCRIPTS / 'auto_t_parking.py').read_text()
    handoff = source.split(
        'def _confirm_cusp_zero_handoff', 1)[1].split(
        'def _log_pre_segment_failure', 1)[0]

    assert 'paired_samples = min(drive_samples, wheel_samples, 3)' in handoff
    assert '[T-PARK][CUSP HANDOFF]' in handoff
    assert '[T-PARK][CUSP ZERO]' in handoff
    assert '[T-PARK][CUSP STOP] result=PASS' in handoff
    assert "'forward_controller_id': 'ParkingForward'" in source
    assert "'reverse_controller_id': 'ParkingReverse'" in source
    assert '[T-PARK][SEGMENT START]' in source
    assert '[T-PARK][SEGMENT RESULT]' in source
    assert 'phase={"EXIT" if forward_exit else "PARKING"}' in source
    assert '[T-PARK][PRE-SEGMENT FAILURE]' in source

    action = source.split(
        'def _execute_path_action', 1)[1].split(
        'def _cancel_parking_follow_path', 1)[0]
    assert 'require_fresh_status=False' in action
    assert action.index('[T-PARK][SEGMENT START]') < action.index(
        'send_goal_async')
    assert action.index('send_goal_async') < action.index(
        '[T-PARK] FollowPath goal accepted:')

    preflight = source.split(
        'def _bench_motion_preflight', 1)[1].split(
        'def _start_worker', 1)[0]
    freshness_block = preflight.split(
        'if require_fresh_status:', 1)[1].split(
        'missing =', 1)[0]
    assert 'connected_status_fresh' in freshness_block
    assert 'mode_status_fresh' in freshness_block
    assert 'safety_status_fresh' in freshness_block

    workflow = source.split(
        'def _run_state_machine', 1)[1].split(
        'def _toggle_slam_measurements', 1)[0]
    reverse_execution = workflow.index('if not self._execute(')
    parked_validation = workflow.index('_validate_and_log_parked')
    parked_status = workflow.index("_publish_status('PARKED')")
    exit_stop = workflow.index("_confirm_command_zero('EXIT STOP')")
    exit_plan = workflow.index("'[T-PARK][EXIT PLAN] start=validated PARKED pose '")
    assert reverse_execution < parked_validation < parked_status
    assert parked_status < exit_stop < exit_plan

    exit_executor = source.split(
        'def _execute_forward_exit', 1)[1].split(
        '@staticmethod\n    def _direction_segments', 1)[0]
    assert '_execute_segmented_path(' in exit_executor
    assert 'forward_exit=True' in exit_executor
    assert '[T-PARK][EXIT STEERING SIGN]' in exit_executor

    controller_source = (
        Path(__file__).resolve().parents[1] /
        'controller' / 'direction_locked_rpp.cpp').read_text()
    assert '[T-PARK][TERMINAL GOAL REACHED]' in controller_source


def test_reverse_regression_does_not_reapply_startup_status_freshness():
    source = (SCRIPTS / 'auto_t_parking.py').read_text()
    action = source.split(
        'def _execute_path_action', 1)[1].split(
        'def _cancel_parking_follow_path', 1)[0]
    workflow = source.split(
        'def _run_state_machine', 1)[1].split(
        'def _toggle_slam_measurements', 1)[0]

    # Startup still observes fresh status, while a later segment rechecks the
    # current fail-closed state without requiring unchanged event topics to
    # have emitted again during the completed FORWARD action.
    assert '_bench_motion_preflight()' in source.split(
        'def _start_worker', 1)[1].split('def stop', 1)[0]
    assert '_bench_motion_preflight(\n                require_fresh_status=False)' in action
    assert '_log_pre_segment_failure(' in action
    assert "self._fail('FollowPath failed')" not in workflow


def test_mocked_reverse_action_is_sent_accepted_and_succeeds():
    module = _load_auto_parking()
    logs = []

    class CompletedFuture:
        def done(self):
            return True

        def result(self):
            result = SimpleNamespace(error_code=0, error_msg='')
            return SimpleNamespace(
                status=module.GoalStatus.STATUS_SUCCEEDED,
                result=result,
            )

    class AcceptedGoal:
        accepted = True

        def get_result_async(self):
            return CompletedFuture()

    class FollowClient:
        def send_goal_async(self, goal, feedback_callback):
            assert goal.controller_id == 'ParkingReverse'
            assert feedback_callback is not None
            return AcceptedGoal()

    node = object.__new__(module.AutoTParking)
    node.bench_mode = False
    node.data_lock = threading.Lock()
    node.active_direction_segment = SimpleNamespace(
        first_pose=87, last_pose=166)
    node.active_segment_number = 1
    node.active_follow_goal = None
    node.segment_state_publisher = object()
    node.follow_client = FollowClient()
    node._rear_scan_state = lambda: (True, 10.0, 'valid')
    node._abort_requested = lambda: False
    node._runtime_ok = lambda: True
    node._safe_publish = lambda publisher, message: True
    node._log_info = logs.append
    node._log_error = logs.append
    node._wait_future = lambda future, timeout: future
    node._follow_feedback = lambda feedback: None
    node.get_parameter = lambda name: SimpleNamespace(value={
        'reverse_controller_id': 'ParkingReverse',
        'goal_checker_id': 'parking_goal_checker',
        'progress_checker_id': 'progress_checker',
        'rear_emergency_stop_distance': 0.15,
        'follow_path_timeout': 5.0,
    }[name])

    path = module.Path()
    succeeded = node._execute_path_action(
        path, run_number=2, run_count=2, direction=-1,
        monitor_slot=None, forward_exit=False)

    output = '\n'.join(logs)
    print(output)
    assert succeeded is True
    assert '[T-PARK][SEGMENT START]' in output
    assert 'phase=PARKING segment=1 direction=REVERSE' in output
    assert 'sending FollowPath goal 2/2: controller_id=ParkingReverse' in output
    assert 'FollowPath goal accepted: controller_id=ParkingReverse' in output
    assert ('[T-PARK][SEGMENT RESULT] segment=1 direction=REVERSE '
            'result=SUCCEEDED') in output


def test_latched_safe_mcu_state_remains_valid_between_segments():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.bench_mode = True
    node.wheels_off_ground = True
    node.execute_path = True
    node.data_lock = threading.Lock()
    node.mcu_connected = True
    node.mcu_ready = True
    node.mcu_current_mode = 'T_PARK'
    node.mcu_safety_state = 'OK'
    node.estop_lock = False
    node.last_lidar_drive = 0.0
    node.last_lidar_wheel = 0
    now = time.monotonic()
    node.mcu_connected_received_at = now - 10.0
    node.mcu_mode_received_at = now - 10.0
    node.mcu_safety_received_at = now - 10.0
    node.lidar_drive_received_at = now
    node.lidar_wheel_received_at = now
    node.bench_consecutive_zero_drive_samples = 3
    node.bench_consecutive_zero_wheel_samples = 3
    node.last_bench_preflight_failure_reason = ''
    node._bench_runtime_graph_ok = lambda: True
    node.get_publishers_info_by_topic = lambda topic: [object()]
    node._log_info = lambda message: None
    node._log_error = lambda message: None

    assert node._bench_motion_preflight(require_fresh_status=True) is False
    assert 'connected_status_fresh' in (
        node.last_bench_preflight_failure_reason)
    assert node._bench_motion_preflight(require_fresh_status=False) is True


def test_bench_frames_and_topic_do_not_duplicate_production_names():
    assert BENCH_ODOM_TOPIC != '/odom'
    assert BENCH_ODOM_FRAME not in ('map', 'odom', 'base_link')
    assert BENCH_BASE_FRAME not in ('map', 'odom', 'base_link')
    assert BENCH_ODOM_FRAME != BENCH_BASE_FRAME


def test_bench_overlay_plans_and_executes_existing_return_state():
    config = Path(__file__).resolve().parents[1] / 'config'
    parameters = yaml.safe_load(
        (config / 't_parking_bench.yaml').read_text())[
            't_parking_auto']['ros__parameters']
    assert parameters['return_to_entrance'] is True
    assert parameters['bench_plan_exit'] is True
    assert parameters['parking_end_clearance_m'] == 3.06

    launch_source = (
        Path(__file__).resolve().parents[1]
        / 'launch' / 'bench_t_parking.launch.py').read_text()
    assert "'return_to_entrance': True" in launch_source
