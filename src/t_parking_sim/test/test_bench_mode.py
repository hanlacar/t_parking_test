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
    actuator_output_allows_progress,
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
    lidar_drive_matches_twist,
    MCU_NODES,
    Pose2D,
    preflight_graph_conflicts,
    startup_gate_state,
    StartupSnapshot,
)
import pytest  # noqa: E402
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


def _load_auto_parallel_parking():
    spec = importlib.util.spec_from_file_location(
        'auto_parallel_parking_test_module',
        SCRIPTS / 'auto_parallel_parking.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_mocked_reverse_action(
        bench_mode, wheels_off_ground, rear_state):
    module = _load_auto_parking()
    logs = []
    rear_calls = []
    sent_goals = []

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
            sent_goals.append(goal)
            assert goal.controller_id == 'ParkingReverse'
            assert feedback_callback is not None
            return AcceptedGoal()

    def rear_scan_state():
        rear_calls.append(rear_state)
        return rear_state

    node = object.__new__(module.AutoTParking)
    node.bench_mode = bench_mode
    node.wheels_off_ground = wheels_off_ground
    node.data_lock = threading.Lock()
    node.active_direction_segment = SimpleNamespace(
        first_pose=87, last_pose=166)
    node.active_segment_number = 1
    node.active_follow_goal = None
    node.last_execution_failure_reason = ''
    node.last_bench_preflight_failure_reason = ''
    node.segment_state_publisher = object()
    node.follow_client = FollowClient()
    node._bench_motion_preflight = lambda require_fresh_status: True
    node._rear_scan_state = rear_scan_state
    node._abort_requested = lambda: False
    node._runtime_ok = lambda: True
    node._safe_publish = lambda publisher, message: True
    node._log_info = logs.append
    node._log_error = logs.append
    node._wait_future = lambda future, timeout: future
    node._follow_feedback = lambda feedback: None
    node._emergency_stop = lambda **kwargs: None
    node.get_parameter = lambda name: SimpleNamespace(value={
        'reverse_controller_id': 'ParkingReverse',
        'goal_checker_id': 'parking_goal_checker',
        'progress_checker_id': 'progress_checker',
        'rear_emergency_stop_distance': 0.15,
        'follow_path_timeout': 5.0,
    }[name])

    succeeded = node._execute_path_action(
        module.Path(), run_number=2, run_count=2, direction=-1,
        monitor_slot=None, forward_exit=False)
    return succeeded, '\n'.join(logs), rear_calls, sent_goals


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


def test_direction_mapping_uses_physical_clamp_and_keeps_mcu_sign():
    converter = _load_converter()
    kwargs = {
        'wheel_base': 0.73,
        'steering_limit_deg': 22.0,
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
    assert forward.wheel_deg == -22
    assert reverse.wheel_deg == 22
    assert -22 <= forward.wheel_deg <= 22
    assert -22 <= reverse.wheel_deg <= 22


@pytest.mark.parametrize('value', [-1.0, 0.0, 1.0, 2.0, 3.0])
def test_lidar_drive_contract_accepts_only_exact_stages(value):
    converter = _load_converter()
    assert converter.validated_lidar_drive_value(value) == value
    assert value in converter.VALID_LIDAR_DRIVE_VALUES


@pytest.mark.parametrize(
    'value', [-3.0, -2.0, 0.5, 1.5, 2.5, float('nan'), float('inf')])
def test_lidar_drive_contract_rejects_invalid_values(value):
    converter = _load_converter()
    with pytest.raises(ValueError, match='invalid /lidar_drive value'):
        converter.validated_lidar_drive_value(value)


@pytest.mark.parametrize('linear_x, forward, reverse', [
    (0.1, 0.5, -1.0),
    (-0.1, 1.0, -2.0),
])
def test_converter_cannot_emit_an_invalid_lidar_drive_stage(
        linear_x, forward, reverse):
    converter = _load_converter()
    with pytest.raises(ValueError, match='invalid /lidar_drive value'):
        converter.convert_command(
            linear_x=linear_x,
            angular_z=0.0,
            wheel_base=0.73,
            steering_limit_deg=22.0,
            mcu_wheel_limit_deg=27,
            stopped_speed_epsilon=0.01,
            forward_drive_stage=forward,
            reverse_drive_stage=reverse,
        )


def test_forward_right_turn_uses_positive_mcu_wheel_sign():
    converter = _load_converter()
    command = converter.convert_command(
        linear_x=0.20,
        angular_z=-0.10,
        wheel_base=0.73,
        steering_limit_deg=22.0,
        mcu_wheel_limit_deg=22,
        stopped_speed_epsilon=0.01,
        forward_drive_stage=1.0,
        reverse_drive_stage=-1.0,
    )
    assert command.drive_stage == 1.0
    assert command.steering_deg_ros < 0.0
    assert 0 < command.wheel_deg <= 22


def test_field_converter_accepts_22_and_faults_above_limit_or_nan():
    converter = _load_converter()
    kwargs = {
        'wheel_base': 0.73,
        'steering_limit_deg': 22.0,
        'mcu_wheel_limit_deg': 22,
        'stopped_speed_epsilon': 0.01,
        'forward_drive_stage': 1.0,
        'reverse_drive_stage': -1.0,
        'fault_on_steering_limit': True,
    }
    linear_x = 0.20
    left = converter.convert_command(
        linear_x, linear_x * math.tan(math.radians(22.0)) / 0.73,
        **kwargs)
    right = converter.convert_command(
        linear_x, -linear_x * math.tan(math.radians(22.0)) / 0.73,
        **kwargs)
    assert left.wheel_deg == -22
    assert right.wheel_deg == 22

    with pytest.raises(ValueError, match='steering limit exceeded'):
        converter.convert_command(
            linear_x, linear_x * math.tan(math.radians(23.0)) / 0.73,
            **kwargs)
    with pytest.raises(ValueError, match='steering limit exceeded'):
        converter.convert_command(
            linear_x, -linear_x * math.tan(math.radians(23.0)) / 0.73,
            **kwargs)
    with pytest.raises(ValueError, match='non-finite Twist'):
        converter.convert_command(float('nan'), 0.0, **kwargs)


def test_planner_steering_contract_accepts_22_and_rejects_above_22():
    auto = _load_auto_parking()
    curvature_22 = math.tan(math.radians(22.0)) / 0.73
    curvature_23 = math.tan(math.radians(23.0)) / 0.73
    assert auto.steering_within_command_limit(0.73, curvature_22, 22.0)
    assert not auto.steering_within_command_limit(
        0.73, curvature_23, 22.0)


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
        stop_publishers=0,
        drive_zero_samples=0,
        wheel_zero_samples=0,
        stop_false_samples=0,
        nonzero_command_seen=False,
        stop_active_seen=False,
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
        stop_publishers=1,
        drive_zero_samples=3,
        wheel_zero_samples=3,
        stop_false_samples=3,
        nonzero_command_seen=False,
        stop_active_seen=False,
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
    assert startup_gate_state(
        disconnected, require_mcu_status=False) == 'READY'
    assert startup_gate_state(
        unsafe, require_mcu_status=False) == 'READY'
    assert startup_gate_state(
        wrong_mode, require_mcu_status=False) == 'READY'


def test_startup_rejects_duplicate_or_nonzero_command_source():
    safe = {
        'drive_publishers': 1,
        'wheel_publishers': 1,
        'stop_publishers': 1,
        'drive_zero_samples': 3,
        'wheel_zero_samples': 3,
        'stop_false_samples': 3,
        'nonzero_command_seen': False,
        'stop_active_seen': False,
        'mcu_connected': True,
        'mcu_ready': True,
        'vehicle_mode': 'T_PARK',
        'safety_state': 'OK',
    }
    duplicate = StartupSnapshot(**{**safe, 'drive_publishers': 2})
    nonzero = StartupSnapshot(**{**safe, 'nonzero_command_seen': True})
    stopped = StartupSnapshot(**{**safe, 'stop_active_seen': True})
    assert startup_gate_state(duplicate) == (
        'FAIL_DUPLICATE_LIDAR_COMMAND_SOURCE')
    assert startup_gate_state(nonzero) == (
        'FAIL_NONZERO_COMMAND_BEFORE_START')
    assert startup_gate_state(stopped) == 'FAIL_STOP_ACTIVE_BEFORE_START'


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


def test_bench_terminal_paths_publish_explicit_safe_commands():
    converter = _load_converter()
    auto = _load_auto_parking()
    outputs = {'drive': [], 'wheel': [], 'stop': []}

    class Publisher:
        def __init__(self, key):
            self.key = key

        def publish(self, message):
            outputs[self.key].append(message.data)

    class Logger:
        def warning(self, _message):
            pass

    command_node = object.__new__(converter.CmdVelToLidarCmd)
    command_node.drive_publisher = Publisher('drive')
    command_node.wheel_publisher = Publisher('wheel')
    command_node.stop_publisher = Publisher('stop')
    command_node.get_logger = lambda: Logger()
    command_node.publish_shutdown_stop(repeats=1)
    assert outputs == {'drive': [0.0], 'wheel': [0], 'stop': [True]}

    terminal_node = object.__new__(auto.AutoTParking)
    terminal_node.bench_mode = True
    terminal_node.execute_path = True
    terminal_node._cancel_active_goals = lambda: None
    emergency_calls = []
    statuses = []
    terminal_node._emergency_stop = (
        lambda **kwargs: emergency_calls.append(kwargs))
    terminal_node._publish_status = (
        lambda status, detail='': statuses.append((status, detail)))
    terminal_node._fail('test failure')
    terminal_node._cancelled()
    assert emergency_calls == [
        {'emergency': True}, {'emergency': True}]
    assert statuses == [
        ('FAILED', 'test failure'), ('CANCELLED', '')]

    auto_source = (SCRIPTS / 'auto_t_parking.py').read_text()
    assert "'FINISHED', 'return_to_entrance=false'" in auto_source


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
    assert clamped.raw_delta_deg > 22.0
    assert math.isclose(clamped.clamped_delta_deg, 22.0)
    expected_rate = 0.10 / 0.73 * math.tan(math.radians(22.0))
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


def test_virtual_progress_requires_matching_emitted_drive_direction():
    assert lidar_drive_matches_twist(0.20, 1.0)
    assert lidar_drive_matches_twist(-0.20, -1.0)
    assert lidar_drive_matches_twist(0.0, 0.0)
    assert not lidar_drive_matches_twist(0.20, -1.0)
    assert not lidar_drive_matches_twist(-0.20, 1.0)
    assert not lidar_drive_matches_twist(0.20, 0.0)

    assert actuator_output_allows_progress(True, False, True)
    assert not actuator_output_allows_progress(False, False, True)
    assert not actuator_output_allows_progress(True, True, True)
    assert not actuator_output_allows_progress(True, False, False)


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
    events.append('PARKING_STOP')
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
    events.extend([
        'PARKED_FOOTPRINT_PASS', 'PARKED', 'EXIT_STOP', 'EXIT_PLAN',
        'FORWARD_EXIT', 'EXIT_COMPLETE', 'FINAL_ZERO',
    ])
    assert events == [
        'FORWARD_SUCCEEDED',
        'CUSP_HANDOFF_WAIT_ZERO',
        'CUSP_ZERO_1', 'CUSP_ZERO_2', 'CUSP_ZERO_3',
        'CUSP_STOP_PASS',
        'SEGMENT_START_REVERSE',
        'ParkingReverse_GOAL_SENT',
        'ParkingReverse_ACCEPTED',
        'REVERSE_VIRTUAL_MOTION', 'REVERSE_SUCCEEDED', 'PARKING_STOP',
        'PARKED_FOOTPRINT_PASS', 'PARKED', 'EXIT_STOP', 'EXIT_PLAN',
        'FORWARD_EXIT', 'EXIT_COMPLETE', 'FINAL_ZERO',
    ]


def test_virtual_parked_stop_then_forward_right_exit_and_final_zero():
    parked = Pose2D(5.10, -4.995, math.pi / 2.0)
    pose = parked
    for _ in range(3):
        step = integrate_bicycle(pose, 0.0, 0.0, 0.1)
        assert step.pose == parked

    speed = 0.20
    radius = 2.30
    yaw_rate = -speed / radius
    duration = (math.pi / 2.0) / abs(yaw_rate)
    first = integrate_bicycle(pose, speed, yaw_rate, 0.05)
    assert first.linear_velocity > 0.0
    assert first.clamped_delta_deg < 0.0
    assert abs(first.clamped_delta_deg) <= 22.0
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


def test_real_bench_uses_field_direct_contract_without_mcu_manager_status():
    preflight_source = (SCRIPTS / 'bench_preflight.py').read_text()
    auto_source = (SCRIPTS / 'auto_t_parking.py').read_text()
    support_source = (SCRIPTS / 'bench_support.py').read_text()
    launch_source = (
        Path(__file__).resolve().parents[1]
        / 'launch' / 'real_t_parking_bench.launch.py').read_text()
    assert "'require_mcu_status': False" in launch_source
    assert "'require_parking_mode': False" in launch_source
    assert "'require_mcu_status:=false'" in launch_source
    assert "'mode_topic': '/mcu/current_mode'" not in launch_source
    assert 'mcu_connected_topic' not in launch_source
    gate_source = support_source.split(
        'def startup_gate_state', 1)[1].split('def normalize_angle', 1)[0]
    assert 'require_mcu_status' in gate_source
    checks_source = auto_source.split(
        'checks = {', 1)[1].split('missing =', 1)[0]
    assert 'if self.require_mcu_status:' in checks_source
    assert 'vehicle_mode_T_PARK' in checks_source
    assert 'field_firmware_direct_serial=true' in auto_source
    assert 'require_mcu_status' in preflight_source


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
    assert auto_parameters['minimum_turning_radius'] == 1.82
    assert auto_parameters['wheel_base'] == 0.73
    assert auto_parameters['front_wheel_track'] == 0.775
    assert auto_parameters['rear_wheel_track'] == 0.785
    assert auto_parameters['wheel_radius'] == 0.135
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
    bench_controller = bench_nav['controller_server']['ros__parameters']
    assert bench_controller['ParkingReverse'][
        'reverse_wheel_base'] == 0.73
    assert bench_controller['ParkingReverse'][
        'reverse_hard_steering_limit_deg'] == 22.0
    assert bench_controller['ParkingReverse'][
        'reverse_fault_on_steering_limit'] is True
    assert bench_nav['planner_server']['ros__parameters'][
        'GridBased']['minimum_turning_radius'] == 1.82
    assert bench_nav['planner_server']['ros__parameters'][
        'ForwardExit']['minimum_turning_radius'] == 2.30
    assert bench_nav['velocity_smoother']['ros__parameters'][
        'scale_velocities'] is True
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
        'if require_fresh_status and self.require_mcu_status:', 1)[1].split(
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
    exit_plan = workflow.index('forward_exit = self._consume_exit_preplan')
    exit_complete = workflow.index('[T-PARK][EXIT COMPLETE]')
    final_zero = workflow.index(
        "_confirm_command_zero('FINAL ZERO')", exit_complete)
    assert reverse_execution < parked_validation < parked_status
    assert parked_status < exit_stop < exit_plan
    assert exit_plan < exit_complete < final_zero

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
    succeeded, output, rear_calls, sent_goals = _run_mocked_reverse_action(
        False, False, (True, 10.0, 'valid'))
    print(output)
    assert succeeded is True
    assert len(rear_calls) == 1
    assert len(sent_goals) == 1
    assert '[T-PARK][SEGMENT START]' in output
    assert 'phase=PARKING segment=1 direction=REVERSE' in output
    assert 'sending FollowPath goal 2/2: controller_id=ParkingReverse' in output
    assert 'FollowPath goal accepted: controller_id=ParkingReverse' in output
    assert ('[T-PARK][SEGMENT RESULT] segment=1 direction=REVERSE '
            'result=SUCCEEDED') in output


def test_bench_reverse_skips_missing_or_stale_physical_rear_scan():
    for rear_state in (
            (False, float('inf'), 'has no message'),
            (False, float('inf'), 'is stale (age=10.00s)')):
        succeeded, output, rear_calls, sent_goals = (
            _run_mocked_reverse_action(True, True, rear_state))
        print(output)
        assert succeeded is True
        assert rear_calls == []
        assert len(sent_goals) == 1
        assert '[BENCH SAFETY]' in output
        assert ('physical rear LiDAR availability check skipped:'
                in output)
        assert 'bench_mode=true wheels_off_ground=true' in output
        assert '[T-PARK][SEGMENT START]' in output
        assert 'controller_id=ParkingReverse' in output
        assert '[T-PARK] FollowPath goal accepted:' in output


def test_production_reverse_blocks_missing_or_stale_physical_rear_scan():
    for rear_state in (
            (False, float('inf'), 'has no message'),
            (False, float('inf'), 'is stale (age=10.00s)')):
        succeeded, output, rear_calls, sent_goals = (
            _run_mocked_reverse_action(False, False, rear_state))
        assert succeeded is False
        assert len(rear_calls) == 1
        assert sent_goals == []
        assert '[BENCH SAFETY]' not in output
        assert '[T-PARK][PRE-SEGMENT FAILURE]' in output
        assert 'rear_lidar_sector_' in output


def test_latched_safe_mcu_state_remains_valid_between_segments():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.bench_mode = True
    node.wheels_off_ground = True
    node.execute_path = True
    node.require_mcu_status = True
    node.data_lock = threading.Lock()
    node.mcu_connected = True
    node.mcu_ready = True
    node.mcu_current_mode = 'T_PARK'
    node.mcu_safety_state = 'OK'
    node.estop_lock = False
    node.last_lidar_drive = 0.0
    node.last_lidar_wheel = 0
    node.last_lidar_stop = False
    now = time.monotonic()
    node.mcu_connected_received_at = now - 10.0
    node.mcu_mode_received_at = now - 10.0
    node.mcu_safety_received_at = now - 10.0
    node.lidar_drive_received_at = now
    node.lidar_wheel_received_at = now
    node.lidar_stop_received_at = now
    node.bench_consecutive_zero_drive_samples = 3
    node.bench_consecutive_zero_wheel_samples = 3
    node.bench_consecutive_stop_false_samples = 3
    node.last_bench_preflight_failure_reason = ''
    node._bench_runtime_graph_ok = lambda: True
    node.get_publishers_info_by_topic = lambda topic: [object()]
    node._log_info = lambda message: None
    node._log_error = lambda message: None

    assert node._bench_motion_preflight(require_fresh_status=True) is False
    assert 'connected_status_fresh' in (
        node.last_bench_preflight_failure_reason)
    assert node._bench_motion_preflight(require_fresh_status=False) is True

    node.require_mcu_status = False
    node.mcu_connected = False
    node.mcu_current_mode = ''
    node.mcu_safety_state = 'UNKNOWN'
    assert node._bench_motion_preflight(require_fresh_status=True) is True
    node.require_mcu_status = True
    node.mcu_connected = True
    node.mcu_current_mode = 'T_PARK'
    node.mcu_safety_state = 'OK'

    node.estop_lock = True
    assert node._bench_motion_preflight(require_fresh_status=False) is False
    assert 'estop_unlocked' in node.last_bench_preflight_failure_reason
    node.estop_lock = False

    node.mcu_safety_state = 'ESTOP'
    assert node._bench_motion_preflight(require_fresh_status=False) is False
    assert 'mcu_safety_state_OK' in node.last_bench_preflight_failure_reason
    node.mcu_safety_state = 'OK'

    node.bench_consecutive_zero_wheel_samples = 2
    assert node._bench_motion_preflight(require_fresh_status=False) is False
    assert ('lidar_wheel_zero_continuous' in
            node.last_bench_preflight_failure_reason)
    node.bench_consecutive_zero_wheel_samples = 3

    node.get_publishers_info_by_topic = lambda topic: [object(), object()]
    assert node._bench_motion_preflight(require_fresh_status=False) is False
    assert ('lidar_drive_single_publisher' in
            node.last_bench_preflight_failure_reason)


def test_bench_frames_and_topic_do_not_duplicate_production_names():
    assert BENCH_ODOM_TOPIC != '/odom'
    assert BENCH_ODOM_FRAME not in ('map', 'odom', 'base_link')
    assert BENCH_BASE_FRAME not in ('map', 'odom', 'base_link')
    assert BENCH_ODOM_FRAME != BENCH_BASE_FRAME


def test_real_bench_overlay_stops_after_parking_and_keeps_safety_geometry():
    config = Path(__file__).resolve().parents[1] / 'config'
    parameters = yaml.safe_load(
        (config / 't_parking_bench.yaml').read_text())[
            't_parking_auto']['ros__parameters']
    assert parameters['return_to_entrance'] is False
    assert parameters['bench_plan_exit'] is True
    assert parameters['parking_end_clearance_m'] == 3.06
    assert parameters['minimum_turning_radius'] == 1.82
    assert parameters['exit_planning_turn_radius'] == 2.30
    assert parameters['wheel_base'] == 0.73
    assert parameters['mechanical_steering_max_deg'] == 22.0
    assert parameters['maximum_command_steering_deg'] == 22.0
    assert parameters['maximum_cusps'] == 2

    launch_source = (
        Path(__file__).resolve().parents[1]
        / 'launch' / 'real_t_parking_bench.launch.py').read_text()
    assert "'return_to_entrance': False" in launch_source


def test_real_bench_launch_has_explicit_ack_and_no_simulator_process():
    launch_path = (
        Path(__file__).resolve().parents[1]
        / 'launch' / 'real_t_parking_bench.launch.py')
    spec = importlib.util.spec_from_file_location(
        'real_t_parking_bench_launch', launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.bench_acknowledged('true')
    assert not module.bench_acknowledged('false')
    assert not module.bench_acknowledged('')
    assert module.serial_bridge_enabled('true')
    assert not module.serial_bridge_enabled('false')
    source = launch_path.read_text()
    assert "'bench_ack', default_value='false'" in source
    assert "'serial_bridge', default_value='false'" in source
    assert "'hardware_ack', default_value='false'" in source
    assert "executable='t870_field_serial_bridge.py'" in source
    assert 'WHEELS OFF GROUND ONLY' in source
    assert "package='ros_gz_sim'" not in source
    assert "package='ros_gz_bridge'" not in source
    assert "executable='gzserver'" not in source
    assert 'sim.launch.py' not in source
    assert "'bench_mode': True" in source
    assert "'wheels_off_ground': True" in source
    assert "'execute': True" in source
    assert 't_parking_rviz_real.yaml' in source
    assert "'output_frequency': 10.0" in source
    assert "'steering_limit_deg': 22.0" in source
    assert "'mcu_wheel_limit_deg': 22" in source
    assert "'fault_on_steering_limit': True" in source


def test_rviz_plan_only_exposes_optional_exit_without_motion_nodes():
    package = Path(__file__).resolve().parents[1]
    launch_source = (
        package / 'launch' / 'rviz_t_parking_test.launch.py').read_text()
    fake_vehicle_source = (
        package / 'scripts' / 'rviz_fake_vehicle.py').read_text()
    dry_run = yaml.safe_load(
        (package / 'config' / 'rviz_t_parking_test.yaml').read_text())
    real = yaml.safe_load(
        (package / 'config' / 't_parking_rviz_real.yaml').read_text())
    auto_source = (package / 'scripts' / 'auto_t_parking.py').read_text()

    planner = dry_run['planner_server']['ros__parameters']
    assert planner['GridBased']['minimum_turning_radius'] == 1.82
    assert planner['ForwardExit']['minimum_turning_radius'] == 2.30
    real_parameters = real['t_parking_auto']['ros__parameters']
    assert real_parameters['wheel_base'] == 0.73
    assert real_parameters['mechanical_steering_max_deg'] == 22.0
    assert real_parameters['maximum_command_steering_deg'] == 22.0
    assert real_parameters['minimum_turning_radius'] == 1.82
    assert real_parameters['exit_planning_turn_radius'] == 2.30
    assert real_parameters['exit_goal'] == {
        'x': 0.24247, 'y': 0.05097, 'yaw': 0.0,
        'position_tolerance': 0.05, 'yaw_tolerance': 0.05}
    assert real_parameters['fast_planning'] is True
    assert real_parameters['preferred_candidates'] == {
        'slot_1': {'setup_offset': 0.30, 'entry_depth': 0.30},
        'slot_2': {'setup_offset': 0.15, 'entry_depth': 0.30},
    }
    assert real_parameters['exit']['slot_2'][
        'clearance_margin_candidates'] == [0.20, 0.40, 0.60]
    assert real_parameters['recenter'] == {
        'enabled': True,
        'lateral_step': 0.20,
        'max_lateral_shift': 1.50,
        'turn_radius': 2.30,
        'straight_lead': 0.50,
        'longitudinal_margin': 0.30,
        'position_tolerance': 0.05,
        'yaw_tolerance': 0.05,
        'join_position_tolerance': 0.002,
        'join_yaw_tolerance': 0.01,
        'max_heading_change_deg': 120.0,
        'recoverable_start_max_index': 2,
        'slot_2': {
            'correction_candidates': [
                0.0, 0.10, 0.20, 0.30, 0.40, 0.50,
                0.60, 0.80, 1.00, 1.20, 1.40, 1.50],
            'straight_lead_candidates': [0.0, 0.25, 0.50],
            'heading_modes': ['preserve', 'road'],
            'turn_radius': 3.00,
            'longitudinal_margin': 0.30,
            'preferred_unknown': {
                'correction': 0.0,
                'straight_lead': 0.25,
                'heading_mode': 'preserve',
            },
            'preferred_geometry': {
                'correction': 0.50,
                'straight_lead': 0.25,
                'heading_mode': 'preserve',
            },
        },
    }
    assert real_parameters['preplan_exit_during_parking'] is True
    assert real_parameters['pre_race_readiness'] is True
    assert real_parameters['planner_warmup_before_start'] is False
    assert real_parameters['readiness_poll_period'] == 0.10

    assert "'plan_exit', default_value='false'" in launch_source
    assert "'plan_exit': ParameterValue(plan_exit" in launch_source
    assert "'fast_planning', default_value='true'" in launch_source
    assert "'execute', default_value='false'" in launch_source
    assert "candidate.named_poses['final'], candidate.slot" in auto_source
    assert "[named['exit_goal']]" in auto_source
    assert 'named.get(' + repr('exit_goal') + ')' in auto_source
    assert 'self._configured_exit_goal()' in auto_source
    assert 'def _verify_exit_goal_reached' in auto_source
    assert "else named['parking_stop']" in auto_source
    assert "self._publish_status('PLANNING_EXIT')" in auto_source
    assert "self._publish_status('VALIDATE_EXIT_PATH')" in auto_source
    assert "self._publish_status('EXIT_PLAN_VALID')" in auto_source
    assert "self._publish_status('FINISHED')" in auto_source
    assert "parameters['exit_goal']" in launch_source
    assert "'exit_goal.x': ParameterValue(" in launch_source
    assert 'self.exit_goal_x' in fake_vehicle_source
    assert 'self.return_x' not in fake_vehicle_source
    assert "get_parameter('return_yaw_tolerance')" not in fake_vehicle_source
    assert "'/t_parking/approach_path'" in auto_source
    rviz_source = (
        package / 'rviz' / 'rviz_t_parking_test.rviz').read_text()
    assert 'Value: /t_parking/approach_path' in rviz_source

    forbidden = (
        "package='nav2_controller'", 'cmd_vel_to_lidar_cmd.py',
        't870_field_serial_bridge.py', "package='ros_gz_sim'",
        "package='ros_gz_bridge'", "executable='gzserver'",
        "executable='gzclient'",
    )
    for token in forbidden:
        assert token not in launch_source


def _mock_fast_planning_node(
        module, fast_result, recenter_enabled=False,
        rejection_code='STEERING_LIMIT', fast_enabled=True):
    node = object.__new__(module.AutoTParking)
    values = {
        'opposite_setup_inset_candidates': [0.30, 0.45],
        'entry_depth_candidates': [0.30, 0.20],
        'preferred_candidate.slot': 'slot_1',
        'preferred_candidate.setup_offset': 0.30,
        'preferred_candidate.entry_depth': 0.30,
        'preferred_candidates.slot_1.setup_offset': -1.0,
        'preferred_candidates.slot_1.entry_depth': -1.0,
        'preferred_candidates.slot_2.setup_offset': -1.0,
        'preferred_candidates.slot_2.entry_depth': -1.0,
        'fast_planning': fast_enabled,
        'recenter.enabled': recenter_enabled,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.planning_request_count = 0
    node._log_live_start_costmap_snapshot = lambda: None
    node._abort_requested = lambda: False
    logs = []
    summaries = []
    calls = []
    node._log_info = logs.append
    node._log_candidate_summary = (
        lambda total, valid, counts, aborted=False:
        summaries.append((total, valid, dict(counts), aborted)))

    def plan_one(slot, offset, depth, counts):
        calls.append((slot.name, offset, depth))
        node.planning_request_count += 1
        if (slot.name, offset, depth) == ('slot_1', 0.30, 0.30):
            if fast_result:
                return SimpleNamespace(name='preferred'), 'VALID', 'valid'
            counts[rejection_code] = counts.get(rejection_code, 0) + 1
            return None, rejection_code, 'test rejection'
        counts['PLANNER_FAILED'] = counts.get('PLANNER_FAILED', 0) + 1
        return None, 'PLANNER_FAILED', 'test fallback rejection'

    node._plan_one_candidate = plan_one
    return node, logs, summaries, calls


def test_fast_candidate_pass_stops_after_one_fully_validated_nav2_call():
    module = _load_auto_parking()
    node, logs, summaries, calls = _mock_fast_planning_node(module, True)
    slots = [SimpleNamespace(name='slot_1'), SimpleNamespace(name='slot_2')]

    results = module.AutoTParking._plan_all_candidates(node, slots)

    assert [result.name for result in results] == ['preferred']
    assert calls == [('slot_1', 0.30, 0.30)]
    assert summaries == [(1, 1, {}, False)]
    fast_log = next(log for log in logs if '[FAST PLAN]' in log)
    assert 'preferred candidate PASS' in fast_log
    assert 'Nav2 calls=1' in fast_log
    assert 'fallback=false' in fast_log


def test_fast_candidate_reject_falls_back_without_duplicate_or_other_slot():
    module = _load_auto_parking()
    node, logs, summaries, calls = _mock_fast_planning_node(module, False)
    # The state machine has already filtered target_slot=slot_1. Supplying
    # only that slot must never generate a slot_2 Nav2 request.
    results = module.AutoTParking._plan_all_candidates(
        node, [SimpleNamespace(name='slot_1')])

    assert results == []
    assert calls[0] == ('slot_1', 0.30, 0.30)
    assert len(calls) == 4
    assert len(set(calls)) == 4
    assert all(slot == 'slot_1' for slot, _, _ in calls)
    assert summaries[-1][0:2] == (4, 0)
    fast_log = next(log for log in logs if '[FAST PLAN]' in log)
    assert 'preferred candidate REJECT reason=STEERING_LIMIT' in fast_log
    assert 'fallback=true' in fast_log


def test_fast_planning_false_keeps_full_exhaustive_search():
    module = _load_auto_parking()
    node, logs, summaries, calls = _mock_fast_planning_node(
        module, True, fast_enabled=False)

    results = module.AutoTParking._plan_all_candidates(
        node, [SimpleNamespace(name='slot_1')])

    assert [result.name for result in results] == ['preferred']
    assert len(calls) == 4
    assert len(set(calls)) == 4
    assert summaries == [(4, 1, {'PLANNER_FAILED': 3}, False)]
    assert not any('[FAST PLAN]' in log for log in logs)


def test_slot_specific_fast_candidate_selects_slot_2_history():
    module = _load_auto_parking()
    node, logs, summaries, calls = _mock_fast_planning_node(module, True)
    values = {
        'opposite_setup_inset_candidates': [0.30, 0.15],
        'entry_depth_candidates': [0.30, 0.20],
        'preferred_candidate.slot': 'slot_1',
        'preferred_candidate.setup_offset': 0.30,
        'preferred_candidate.entry_depth': 0.30,
        'preferred_candidates.slot_1.setup_offset': 0.30,
        'preferred_candidates.slot_1.entry_depth': 0.30,
        'preferred_candidates.slot_2.setup_offset': 0.15,
        'preferred_candidates.slot_2.entry_depth': 0.30,
        'fast_planning': True,
        'recenter.enabled': False,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])

    def plan_one(slot, offset, depth, counts):
        calls.append((slot.name, offset, depth))
        node.planning_request_count += 1
        if (slot.name, offset, depth) == ('slot_2', 0.15, 0.30):
            return SimpleNamespace(name='slot_2_preferred'), 'VALID', 'valid'
        return None, 'PLANNER_FAILED', 'unexpected candidate'

    calls.clear()
    node._plan_one_candidate = plan_one
    results = module.AutoTParking._plan_all_candidates(
        node, [SimpleNamespace(name='slot_2')])

    assert [result.name for result in results] == ['slot_2_preferred']
    assert calls == [('slot_2', 0.15, 0.30)]
    assert summaries == [(1, 1, {}, False)]
    assert 'fallback=false' in next(
        log for log in logs if '[FAST PLAN]' in log)


def test_recenter_is_tried_only_for_recoverable_direct_failure():
    module = _load_auto_parking()
    node, _, _, _ = _mock_fast_planning_node(
        module, False, recenter_enabled=True)
    calls = []
    node._plan_recenter_candidates = (
        lambda slot, offset, depth, code:
        calls.append((slot.name, offset, depth, code))
        or SimpleNamespace(name='recentered'))

    results = module.AutoTParking._plan_all_candidates(
        node, [SimpleNamespace(name='slot_1')])

    assert [result.name for result in results] == ['recentered']
    assert calls == [('slot_1', 0.30, 0.30, 'STEERING_LIMIT')]

    node, _, _, _ = _mock_fast_planning_node(
        module, False, recenter_enabled=True, rejection_code='COLLISION')
    calls = []
    node._plan_recenter_candidates = (
        lambda *args: calls.append(args) or SimpleNamespace(name='unsafe'))
    module.AutoTParking._plan_all_candidates(
        node, [SimpleNamespace(name='slot_1')])
    assert calls == []


def test_recenter_recoverable_unknown_is_limited_to_initial_path_fringe():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.get_parameter = lambda name: SimpleNamespace(value={
        'recenter.recoverable_start_max_index': 2,
    }[name])

    assert node._recenter_rejection_is_recoverable(
        'UNKNOWN_SPACE', 'footprint index 1 reaches unknown')
    assert node._recenter_rejection_is_recoverable(
        'ROAD_BOUNDARY', 'path index 2 reaches road boundary')
    assert not node._recenter_rejection_is_recoverable(
        'UNKNOWN_SPACE', 'footprint index 3 reaches unknown')
    assert not node._recenter_rejection_is_recoverable(
        'UNKNOWN_SPACE', 'unknown location unavailable')
    assert not node._recenter_rejection_is_recoverable(
        'COLLISION', 'footprint index 1 overlaps obstacle')


def test_b_parking_reject_log_reports_initial_fringe_unknown_details():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.get_parameter = lambda name: SimpleNamespace(value={
        'wheel_base': 0.73,
    }[name])
    node.course_entrance_pose = module.PoseStamped()
    node.course_entrance_pose.pose.position.x = 0.24247
    node.course_entrance_pose.pose.position.y = 1.55097
    module.set_pose_yaw(node.course_entrance_pose.pose, math.pi)
    path = module.Path()
    path.poses = [
        node.course_entrance_pose,
        node._offset_pose(node.course_entrance_pose, 0.05, 0.0),
    ]
    logs = []
    node._log_info = logs.append
    metrics = SimpleNamespace(max_curvature=0.37315)

    node._log_b_parking_reject(
        SimpleNamespace(name='slot_2'), 0.15, 0.30,
        'UNKNOWN_SPACE', 'footprint index 1 reaches unknown',
        metrics, path, 'direct', 'SUCCEEDED')

    log = logs[-1]
    assert '[B PARKING REJECT]' in log
    assert 'initial_pose=(0.242470,1.550970,3.141593)' in log
    assert 'direct/recenter=direct' in log
    assert 'nav2_result=SUCCEEDED' in log
    assert 'unknown_collision=true' in log
    assert 'reject_index=1' in log
    assert 'required_steering=15.238' in log
    assert 'Rmin=2.680' in log


def test_recenter_straight_prefix_join_and_safety_contract():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    values = {
        'recenter.position_tolerance': 0.05,
        'recenter.yaw_tolerance': 0.05,
        'recenter.max_heading_change_deg': 120.0,
        'wheel_base': 0.73,
        'maximum_command_steering_deg': 22.0,
        'minimum_turning_radius': 1.82,
        'curvature_tolerance_factor': 1.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace()))

    start = module.PoseStamped()
    start.header.frame_id = 'map'
    start.pose.position.x = 0.24247
    start.pose.position.y = 1.55097
    module.set_pose_yaw(start.pose, math.pi)
    boundary = node._offset_pose(start, 0.50, 0.0)
    prefix = node._sample_straight_path(start, boundary)

    suffix = module.Path()
    suffix.header = prefix.header
    suffix.poses = [boundary, node._offset_pose(boundary, 0.10, 0.0)]
    position_error, yaw_error = node._path_join_errors(prefix, suffix)
    combined = node._prepend_path(prefix, suffix)

    assert position_error == pytest.approx(0.0)
    assert yaw_error == pytest.approx(0.0)
    assert len(combined.poses) == len(prefix.poses) + 1
    metrics = node._analyze_path(combined, suffix.poses[-1])
    valid, code, reason = node._validate_recenter_path(
        combined, metrics, footprint_already_validated=True)
    assert (valid, code, reason) == (True, 'VALID', 'valid')
    assert metrics.reverse_length == pytest.approx(0.0)
    assert metrics.cusp_count == 0
    assert metrics.max_curvature == pytest.approx(0.0)

    reverse_metrics = module.PathMetrics(
        metrics.total_length, 0.0, metrics.total_length, 0, 1,
        [-1] * len(metrics.directions), 0.0, 0.0, 0.0)
    valid, code, _ = node._validate_recenter_path(
        combined, reverse_metrics, footprint_already_validated=True)
    assert not valid
    assert code == 'DIRECTION'


def _scalar_parking_footprint_reference(
        node, path, map_msg, costmap, shadow_pose):
    """Preserve the pre-vectorization collision loop as a test oracle."""
    length = float(node.get_parameter('vehicle_length').value)
    width = float(node.get_parameter('vehicle_width').value)
    center_x = float(node.get_parameter('vehicle_center_x_offset').value)
    occupied_threshold = int(node.get_parameter('occupied_threshold').value)
    start_position = shadow_pose.pose.position
    start_yaw = node._yaw_from_pose(shadow_pose)
    start_cos = math.cos(start_yaw)
    start_sin = math.sin(start_yaw)
    start_shadow_margin = 0.05

    def inside_shadow(x, y):
        dx = x - start_position.x
        dy = y - start_position.y
        local_x = start_cos * dx + start_sin * dy
        local_y = -start_sin * dx + start_cos * dy
        return (
            center_x - 0.5 * length - start_shadow_margin
            <= local_x
            <= center_x + 0.5 * length + start_shadow_margin
            and abs(local_y) <= 0.5 * width + start_shadow_margin)

    for index, pose in enumerate(path.poses):
        center_cost = node._cost_value(
            costmap, pose.pose.position.x, pose.pose.position.y)
        if center_cost is None:
            return False, f'path center index {index} is outside costmap'
        if (center_cost == 255
                and not inside_shadow(
                    pose.pose.position.x, pose.pose.position.y)):
            return False, f'path center index {index} is unknown'
        if 253 <= center_cost <= 254:
            return False, f'path center index {index} is lethal/inscribed'
        for x, y in node._footprint_samples(pose, length, width):
            map_value = node._occupancy_value(map_msg, x, y)
            cost_value = node._cost_value(costmap, x, y)
            if map_value is None or cost_value is None:
                return False, f'footprint index {index} is outside map bounds'
            if cost_value == 255 and not inside_shadow(x, y):
                return False, (
                    f'footprint index {index} reaches unknown at '
                    f'({x:.3f},{y:.3f}); map={map_value} '
                    f'cost={cost_value}')
            if map_value >= occupied_threshold or cost_value == 254:
                return False, f'footprint index {index} overlaps obstacle'
    return True, 'valid'


def test_vectorized_parking_footprint_matches_scalar_reference():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    values = {
        'maximum_cusps': 2,
        'wheel_base': 0.73,
        'maximum_command_steering_deg': 22.0,
        'mechanical_steering_max_deg': 22.0,
        'minimum_turning_radius': 1.82,
        'curvature_tolerance_factor': 1.0,
        'vehicle_length': 0.20,
        'vehicle_width': 0.10,
        'vehicle_center_x_offset': 0.0,
        'occupied_threshold': 65,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.data_lock = threading.Lock()
    node._footprint_local_cache = {}
    node._setup_inside_road = lambda pose: True
    node._final_footprint_inside_slot = lambda slot, pose: True
    node._path_clears_slot_walls = lambda slot, path: (True, 'valid')
    node._timing_add = lambda name, elapsed_ms: None
    node._yaw_from_pose = lambda pose: module.yaw_from_quaternion(
        pose.pose.orientation)

    def make_pose(x, y, yaw=0.0):
        pose = module.PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        module.set_pose_yaw(pose.pose, yaw)
        return pose

    def make_grids():
        occupancy = module.OccupancyGrid()
        occupancy.info.resolution = 0.05
        occupancy.info.width = 40
        occupancy.info.height = 40
        occupancy.info.origin.position.x = -1.0
        occupancy.info.origin.position.y = -1.0
        module.set_pose_yaw(occupancy.info.origin, 0.0)
        occupancy.data = [0] * 1600
        costmap = module.Costmap()
        costmap.metadata.resolution = 0.05
        costmap.metadata.size_x = 40
        costmap.metadata.size_y = 40
        costmap.metadata.origin.position.x = -1.0
        costmap.metadata.origin.position.y = -1.0
        module.set_pose_yaw(costmap.metadata.origin, 0.0)
        costmap.data = [0] * 1600
        return occupancy, costmap

    def set_map_value(grid, x, y, value):
        col, row = node._grid_coordinates(
            grid.info.origin, grid.info.resolution, x, y)
        grid.data[row * grid.info.width + col] = value

    def set_cost_value(grid, x, y, value):
        col, row = node._grid_coordinates(
            grid.metadata.origin, grid.metadata.resolution, x, y)
        grid.data[row * grid.metadata.size_x + col] = value

    start = make_pose(0.0, 0.0)
    moved = make_pose(0.40, 0.0)
    rotated = make_pose(0.20, 0.20, math.pi / 4.0)
    rotated_corner_x = 0.20 + math.sqrt(0.5) * 0.05
    rotated_corner_y = 0.20 + math.sqrt(0.5) * 0.15
    cases = []

    occupancy, costmap = make_grids()
    cases.append(('free', [start, moved], start, occupancy, costmap))

    occupancy, costmap = make_grids()
    set_map_value(occupancy, 0.0, 0.0, 100)
    cases.append(('occupied', [start], start, occupancy, costmap))

    occupancy, costmap = make_grids()
    set_cost_value(costmap, moved.pose.position.x, moved.pose.position.y, 255)
    cases.append(('unknown', [start, moved], start, occupancy, costmap))

    occupancy, costmap = make_grids()
    set_cost_value(costmap, 0.0, 0.0, 255)
    cases.append(('initial_shadow', [start], start, occupancy, costmap))

    occupancy, costmap = make_grids()
    near_edge = make_pose(0.98, 0.0)
    cases.append(('out_of_map', [near_edge], near_edge, occupancy, costmap))

    occupancy, costmap = make_grids()
    cases.append(('rotated_free', [rotated], rotated, occupancy, costmap))

    occupancy, costmap = make_grids()
    set_map_value(
        occupancy, rotated_corner_x, rotated_corner_y, 100)
    cases.append((
        'rotated_map_collision', [rotated], rotated, occupancy, costmap))

    occupancy, costmap = make_grids()
    set_cost_value(costmap, 0.10, 0.05, 254)
    cases.append(('costmap_collision', [start], start, occupancy, costmap))

    metrics = module.PathMetrics(
        1.0, 0.5, 0.5, 1, 1, [1, -1], 0.0, 0.0, 0.0)
    named = {'setup': start, 'final': start}
    slot = SimpleNamespace(name='slot_1')
    for case_name, poses, shadow, occupancy, costmap in cases:
        path = module.Path()
        path.header.frame_id = 'map'
        path.poses = poses
        node.map_msg = occupancy
        node.global_costmap = costmap
        scalar = _scalar_parking_footprint_reference(
            node, path, occupancy, costmap, shadow)
        vector = node._validate_path(
            slot, named, path, metrics, unknown_shadow_pose=shadow)
        assert vector == scalar, case_name


def test_readiness_timing_categories_cover_rviz_and_bench_names():
    module = _load_auto_parking()
    rviz = module.AutoTParking._readiness_timing_states({
        'robot_spawned': True,
        'odom': True,
        'map': True,
        'map_to_base_tf': True,
        'planner_active': True,
        'compute_path_server': True,
        'global_costmap': True,
    })
    assert all(rviz.values())
    bench = module.AutoTParking._readiness_timing_states({
        'virtual_bench_odom': True,
        'saved_map': True,
        'map_to_base_tf': True,
        'planner_active': True,
        'compute_path_server': True,
        'global_costmap_static_only': True,
    })
    assert all(bench.values())


def test_target_slot_list_prevents_other_slot_occupancy_check():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    slot_1 = SimpleNamespace(name='slot_1')
    slot_2 = SimpleNamespace(name='slot_2')
    node.slots = [slot_1, slot_2]
    node.slot_timing = {}
    node.data_lock = threading.Lock()
    node.map_msg = object()
    node.global_costmap = object()
    node._abort_requested = lambda: False
    node.get_parameter = lambda name: SimpleNamespace(value={
        'map_wait_timeout': 1.0,
    }[name])
    calls = []
    node._slot_state = lambda slot: calls.append(slot.name) or module.SLOT_FREE
    node._log_info = lambda message: None

    available = node._wait_for_map_and_slots([slot_1])

    assert available == [slot_1]
    assert calls == ['slot_1']


def test_fresh_stable_entrance_pose_is_reused_without_settle_wait():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.pre_race_lock = threading.Lock()
    cached = module.PoseStamped()
    cached.header.frame_id = 'map'
    cached.pose.position.x = 1.0
    cached.pose.position.y = 2.0
    module.set_pose_yaw(cached.pose, 0.3)
    node.pre_race_entrance_pose = cached
    node.pre_race_entrance_pose_at = time.monotonic()
    node.course_entrance_pose = None
    node.get_parameter = lambda name: SimpleNamespace(value={
        'pre_race_readiness': True,
        'pre_race_pose_max_age': 0.5,
        'entrance_pose_position_stability': 0.03,
        'entrance_pose_yaw_stability': 0.05,
    }[name])
    node._current_map_pose = lambda: (1.01, 2.0, 0.31)
    node._vehicle_stopped = lambda: True
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace()))
    logs = []
    node._log_info = logs.append

    assert node._reuse_pre_race_entrance_pose()
    assert node.course_entrance_pose is not None
    assert 'reused=true' in logs[-1]


@pytest.mark.parametrize('initial_y', [0.05097, -1.44903, 1.55097])
def test_configured_exit_goal_is_fresh_and_independent_of_initial_pose(
        initial_y):
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.course_entrance_pose = module.PoseStamped()
    node.course_entrance_pose.pose.position.x = 0.24247
    node.course_entrance_pose.pose.position.y = initial_y
    module.set_pose_yaw(node.course_entrance_pose.pose, math.pi)
    node.get_parameter = lambda name: SimpleNamespace(value={
        'exit_goal.x': 0.24247,
        'exit_goal.y': 0.05097,
        'exit_goal.yaw': 0.0,
        'exit_planning_turn_radius': 2.30,
        'lane_merge_distance_candidates': [0.80],
        'wheel_base': 0.73,
        'wheel_radius': 0.135,
        'wheel_inside_margin': 0.01,
    }[name])
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace()))
    node._log_error = lambda message: None

    first = node._configured_exit_goal()
    second = node._configured_exit_goal()

    assert first is not second
    assert first is not node.course_entrance_pose
    assert first.pose.position.x == pytest.approx(0.24247)
    assert first.pose.position.y == pytest.approx(0.05097)
    assert module.yaw_from_quaternion(
        first.pose.orientation) == pytest.approx(0.0)

    parking_final = module.PoseStamped()
    parking_final.pose.position.x = -9.690
    parking_final.pose.position.y = -4.395
    module.set_pose_yaw(parking_final.pose, math.pi / 2.0)
    named = node._build_rviz_return_poses(parking_final)
    assert named is not None
    assert named['exit_goal'] is not node.course_entrance_pose
    assert named['exit_goal'].pose.position.y == pytest.approx(0.05097)

    slot_2 = SimpleNamespace(
        name='slot_2', yaw=math.pi / 2.0,
        min_x=-7.850, max_x=-5.350, min_y=-8.100, max_y=-2.300)
    slot_2_final = module.PoseStamped()
    slot_2_final.pose.position.x = -6.600
    slot_2_final.pose.position.y = -4.395
    module.set_pose_yaw(slot_2_final.pose, math.pi / 2.0)
    slot_2_named = node._build_slot_clear_return_poses(
        slot_2_final, slot_2, 0.20)
    assert slot_2_named is not None
    assert slot_2_named['exit_goal'] is not named['exit_goal']
    assert slot_2_named['exit_goal'].pose.position.x == pytest.approx(0.24247)
    assert slot_2_named['exit_goal'].pose.position.y == pytest.approx(0.05097)
    assert slot_2_named['slot_forward_clear'].pose.position.x == pytest.approx(
        -6.600)
    assert slot_2_named['slot_forward_clear'].pose.position.y > slot_2.max_y
    assert slot_2_named['right_turn_entry'].pose.position.x != pytest.approx(
        named['right_turn_entry'].pose.position.x)


def test_slot_2_real_exit_dispatches_to_slot_aware_direct_first_planner():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    parking_final = module.PoseStamped()
    slot_2 = SimpleNamespace(name='slot_2')
    expected = object()
    calls = []
    node._log_info = lambda message: None
    node._plan_rviz_return = lambda start, slot: (
        calls.append((start, slot)) or expected)

    actual = node._plan_forward_right_exit(
        parking_final, slot_2, use_current_start=True)

    assert actual is expected
    assert calls == [(parking_final, slot_2)]


def test_exit_goal_check_uses_strict_configured_tolerances():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.get_parameter = lambda name: SimpleNamespace(value={
        'exit_goal.x': 0.24247,
        'exit_goal.y': 0.05097,
        'exit_goal.yaw': 0.0,
        'exit_goal.position_tolerance': 0.05,
        'exit_goal.yaw_tolerance': 0.05,
    }[name])
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace()))
    logs = []
    node._log_info = logs.append
    node._log_error = logs.append
    path = module.Path()
    final = module.PoseStamped()
    final.pose.position.x = 0.27247
    final.pose.position.y = 0.05097
    module.set_pose_yaw(final.pose, 0.02)
    path.poses = [final]

    passed, position_error, yaw_error = node._check_and_log_exit_goal(path)

    assert passed
    assert position_error == pytest.approx(0.03)
    assert yaw_error == pytest.approx(0.02)
    assert '[EXIT GOAL CHECK]' in logs[-1]
    assert 'PASS=true' in logs[-1]

    path.poses[-1].pose.position.x = 0.30248
    passed, _, _ = node._check_and_log_exit_goal(path)
    assert not passed
    assert 'PASS=false' in logs[-1]


def test_exit_preplan_cache_requires_parked_pose_and_safety_revalidation():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.exit_preplan_thread = SimpleNamespace(is_alive=lambda: False)
    node.exit_preplan_lock = threading.Lock()
    node.exit_preplan_error = ''
    expected = module.PoseStamped()
    expected.pose.position.x = -9.69
    expected.pose.position.y = -4.395
    module.set_pose_yaw(expected.pose, math.pi / 2.0)
    candidate = SimpleNamespace(
        named_poses={'parking_stop': expected}, path=object(),
        metrics=object(), turn_direction='NONE')
    node.exit_preplan_result = candidate
    node._abort_requested = lambda: False
    node.get_parameter = lambda name: SimpleNamespace(value={
        'exit_cache_start_position_tolerance': 0.08,
        'exit_cache_start_yaw_tolerance': 0.12,
    }[name])
    logs = []
    warnings = []
    node._log_info = logs.append
    node._log_warn = warnings.append
    validations = []
    node._validate_forward_exit_path = (
        lambda path, metrics, named, slot:
        (validations.append((path, metrics, named, slot))
         or (True, 'RIGHT', 'valid')))

    actual = module.PoseStamped()
    actual.pose.position.x = -9.66
    actual.pose.position.y = -4.395
    module.set_pose_yaw(actual.pose, math.pi / 2.0 + 0.02)
    slot = SimpleNamespace(name='slot_1')
    assert node._consume_exit_preplan(actual, slot) is candidate
    assert candidate.turn_direction == 'RIGHT'
    assert len(validations) == 1
    assert 'full_exit_validation=PASS' in logs[-1]

    actual.pose.position.x = -9.50
    assert node._consume_exit_preplan(actual, slot) is None
    assert len(validations) == 1
    assert 'parked pose mismatch' in warnings[-1]


def test_rviz_path_topics_are_enabled_and_distinctly_colored():
    package = Path(__file__).resolve().parents[1]
    rviz_source = (
        package / 'rviz' / 'rviz_t_parking_test.rviz').read_text()
    topics = (
        '/t_parking/planned_path', '/t_parking/forward_path',
        '/t_parking/reverse_path', '/t_parking/forward_exit_path',
        '/t_parking/exit_path',
    )
    for topic in topics:
        assert topic in rviz_source
    assert rviz_source.count('Enabled: true') >= len(topics)
    colors = {
        line.strip() for line in rviz_source.splitlines()
        if line.strip().startswith('Color:')}
    assert len(colors) >= len(topics)


def test_bench_command_logs_have_human_direction_meaning():
    source = (SCRIPTS / 'cmd_vel_to_lidar_cmd.py').read_text()
    assert '[BENCH CMD]' in source
    assert "'LEFT' if command.wheel_deg < 0" in source
    assert "'RIGHT' if command.wheel_deg > 0" in source

    auto_source = (SCRIPTS / 'auto_t_parking.py').read_text()
    assert '[BENCH SEGMENT START]' in auto_source
    assert '[BENCH SEGMENT END]' in auto_source
    assert '[BENCH FINISHED]' in auto_source


def _synthetic_observation_rays(
        module, roi, sensor_names=('front', 'rear'), frames=10,
        obstacle_slot=False):
    min_x, max_x, min_y, max_y = roi
    rays = []
    rows = 24
    for sensor in sensor_names:
        for frame in range(frames):
            key = f'{sensor}:{frame}'
            for row in range(rows):
                y = min_y + (row + 0.5) * (max_y - min_y) / rows
                rays.append(module.ObservationRay(
                    min_x - 1.0, y, max_x + 1.0, y, False, key))
    if obstacle_slot:
        for frame in range(2):
            for offset in (-0.03, 0.03):
                rays.append(module.ObservationRay(
                    min_x - 1.0,
                    0.5 * (min_y + max_y) + offset,
                    0.5 * (min_x + max_x),
                    0.5 * (min_y + max_y) + offset,
                    True, f'front:{frame}'))
    return rays


def _classify_synthetic_slot(module, rays, roi, coverage=0.50):
    return module.AutoTParking._classify_slot_rays(
        rays, roi, required_frames=10,
        required_sensors=('front', 'rear'),
        minimum_coverage=coverage, minimum_valid_rays=40,
        obstacle_point_threshold=4, obstacle_frame_threshold=2,
        coverage_resolution=0.10)


def test_slot_observation_synthetic_free_and_occupied_evidence():
    module = _load_auto_parking()
    roi = (0.0, 1.0, 0.0, 2.0)
    free = _classify_synthetic_slot(
        module, _synthetic_observation_rays(module, roi), roi)
    occupied = _classify_synthetic_slot(
        module,
        _synthetic_observation_rays(
            module, roi, obstacle_slot=True),
        roi)

    assert free.state == module.SLOT_FREE
    assert free.valid_frames == 20
    assert free.observed_ratio >= 0.50
    assert occupied.state == module.SLOT_OCCUPIED
    assert occupied.obstacle_points == 4


@pytest.mark.parametrize(
    'slot_1,slot_2,selected,terminal,reason', [
        ('FREE', 'OCCUPIED', 'slot_1', True, 'slot_1=FREE'),
        ('OCCUPIED', 'FREE', 'slot_2', True, 'slot_2=FREE'),
        ('UNKNOWN', 'FREE', 'slot_2', True, 'slot_2=FREE'),
        ('FREE', 'UNKNOWN', 'slot_1', True, 'slot_1=FREE'),
        ('UNKNOWN', 'UNKNOWN', '', False, 'next_viewpoint'),
        ('OCCUPIED', 'OCCUPIED', '', True, 'NO_FREE_SLOT'),
    ])
def test_slot_observation_decision_contract(
        slot_1, slot_2, selected, terminal, reason):
    module = _load_auto_parking()
    actual, detail, actual_terminal = module.AutoTParking._slot_decision(
        slot_1, slot_2, 'slot_1')
    assert actual == selected
    assert actual_terminal is terminal
    assert reason in detail


def test_slot_observation_free_free_uses_configured_preference():
    module = _load_auto_parking()
    selected, reason, terminal = module.AutoTParking._slot_decision(
        module.SLOT_FREE, module.SLOT_FREE, 'slot_2')
    assert (selected, terminal) == ('slot_2', True)
    assert 'preferred_free_slot=slot_2' in reason


@pytest.mark.parametrize(
    'internal,external', [
        ('slot_1', 'T_B'),
        ('slot_2', 'T_A'),
        ('', 'NONE'),
        ('unexpected', 'NONE'),
    ])
def test_selected_slot_external_contract(internal, external):
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    published = []
    node.selected_slot_publisher = object()
    node._safe_publish = (
        lambda publisher, message: published.append(message.data) or True)

    node._publish_selected_slot(internal)

    assert published == [external]
    assert published[0] not in ('A', 'B', 'slot_1', 'slot_2')


def test_selected_slot_topic_type_qos_and_reset_contract():
    source = (SCRIPTS / 'auto_t_parking.py').read_text()

    assert "String, '/t_parking/selected_slot', transient_qos" in source
    assert 'history=HistoryPolicy.KEEP_LAST' in source
    assert 'depth=1' in source
    assert 'reliability=ReliabilityPolicy.RELIABLE' in source
    assert 'durability=DurabilityPolicy.TRANSIENT_LOCAL' in source
    assert source.count('self._publish_selected_slot()') >= 2


def test_slot_observation_candidate_does_not_publish_external_selection():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.slot_state_publishers = {
        'slot_1': object(),
        'slot_2': object(),
    }
    node.selected_slot_publisher = object()
    node.marker_publisher = object()
    node._make_slot_observation_markers = lambda results, selected: object()
    published_to = []
    node._safe_publish = (
        lambda publisher, message: published_to.append(publisher) or True)
    results = {
        'slot_1': module.SlotObservation(
            module.SLOT_FREE, 10, 50, 0.8, 0, 'pass'),
        'slot_2': module.SlotObservation(
            module.SLOT_OCCUPIED, 10, 50, 0.8, 5, 'obstacle'),
    }

    node._publish_slot_observations(results, 'slot_1')

    assert node.selected_slot_publisher not in published_to


@pytest.mark.parametrize(
    'selection_mode,internal,external', [
        ('manual', 'slot_1', 'T_B'),
        ('manual', 'slot_2', 'T_A'),
        ('auto', 'slot_1', 'T_B'),
        ('auto', 'slot_2', 'T_A'),
    ])
def test_final_candidate_publishes_gps_slot_contract(
        selection_mode, internal, external):
    module = _load_auto_parking()

    class StopAfterSelection(Exception):
        pass

    node = object.__new__(module.AutoTParking)
    node.target_slot = 'auto' if selection_mode == 'auto' else internal
    node.rviz_only = selection_mode == 'manual'
    node.bench_mode = False
    node.slam_paused_by_node = False
    node.slot_observation_results = {}
    selected_slot = SimpleNamespace(name=internal)
    other_slot = SimpleNamespace(
        name='slot_2' if internal == 'slot_1' else 'slot_1')
    node.slots = [selected_slot, other_slot]
    node.data_lock = threading.Lock()
    node.map_msg = object()
    node.global_costmap = object()
    node.local_costmap = object()
    node.active_plan_goal = None
    node.active_follow_goal = None
    node.motion_phase = 'idle'
    node.active_motion_path = None
    node.active_motion_metrics = None
    node.active_motion_final = None
    node._abort_requested = lambda: False
    node._runtime_ok = lambda: False
    node._publish_status = lambda *a, **k: None
    node._timing_set = lambda *a, **k: None
    node._slot_timing_add = lambda *a, **k: None
    node._wait_for_system = lambda: True
    node._capture_course_entrance_pose = lambda: True
    node._wheel_centers_map = lambda: object()
    node._cancelled = lambda: None
    node._fail = lambda reason: None
    node._log_slot_timing = lambda *a, **k: None
    node._log_parking_geometry = lambda *a, **k: None
    node._log_pose_contract = lambda candidate: True
    node._log_info = lambda message: (
        (_ for _ in ()).throw(StopAfterSelection())
        if message == f'selected: {internal}' else True)
    metrics = SimpleNamespace(
        cusp_count=1,
        max_curvature=0.25,
        total_length=4.0,
        reverse_length=2.0,
    )
    node._plan_all_candidates = lambda slots: [SimpleNamespace(
        slot=selected_slot, metrics=metrics)]
    if selection_mode == 'auto':
        node._run_auto_slot_observation = (
            lambda: (selected_slot, f'{internal}=FREE'))

    published = []
    node.selected_slot_publisher = object()
    node._safe_publish = (
        lambda publisher, message: published.append(message.data) or True)

    node._run_state_machine()

    assert published == [external]
    assert node.selected_slot_name == internal


@pytest.mark.parametrize(
    'internal,external', [
        ('slot_1', 'V_A'),
        ('slot_2', 'V_B'),
        ('', 'NONE'),
        ('unexpected', 'NONE'),
    ])
def test_parallel_selected_slot_external_contract(internal, external):
    module = _load_auto_parallel_parking()
    node = object.__new__(module.AutoParallelParking)
    published = []
    node.selected_slot_publisher = object()
    node._safe_publish = (
        lambda publisher, message: published.append(message.data) or True)

    node._publish_selected_slot(internal)

    assert published == [external]
    assert published[0] not in ('A', 'B', 'slot_1', 'slot_2')


def test_parallel_selected_slot_topic_qos_reset_and_physical_mapping():
    source = (SCRIPTS / 'auto_parallel_parking.py').read_text()
    parallel_in_t_source = (SCRIPTS / 'parallel_in_t_slot.py').read_text()

    assert "LAYOUT_BLOCKS = {'A': 'slot_1', 'B': 'slot_2'}" in (
        parallel_in_t_source)
    assert (
        "String, '/parallel_parking/selected_slot', transient_qos"
        in source)
    assert 'history=HistoryPolicy.KEEP_LAST' in source
    assert 'depth=1' in source
    assert 'reliability=ReliabilityPolicy.RELIABLE' in source
    assert 'durability=DurabilityPolicy.TRANSIENT_LOCAL' in source
    assert source.count('self._publish_selected_slot()') >= 2


@pytest.mark.parametrize(
    'internal,external', [
        ('slot_1', 'V_A'),
        ('slot_2', 'V_B'),
    ])
def test_parallel_validated_entry_publishes_gps_slot_contract(
        internal, external):
    module = _load_auto_parallel_parking()

    class StopAfterSelection(Exception):
        pass

    node = object.__new__(module.AutoParallelParking)
    slot = SimpleNamespace(name=internal)
    candidate = SimpleNamespace(slot=slot)
    node.execute_path = False
    node.slam_paused_by_node = False
    node.active_plan_goal = None
    node.active_follow_goal = None
    node.motion_phase = 'idle'
    node.active_motion_path = None
    node.active_motion_metrics = None
    node.active_motion_final = None
    node._abort_requested = lambda: False
    node._runtime_ok = lambda: False
    node._publish_status = lambda *a, **k: None
    node._wait_for_system = lambda: True
    node._wheel_centers_map = lambda: object()
    node._wait_for_map = lambda: True
    node._select_slot = lambda: slot
    node._plan_entry = lambda selected: candidate
    node._publish_entry_plan = (
        lambda selected: (_ for _ in ()).throw(StopAfterSelection()))
    node._fail = lambda reason: None
    node._cancelled = lambda: None
    published = []
    node.selected_slot_publisher = object()
    node._safe_publish = (
        lambda publisher, message: published.append(message.data) or True)

    node._run_state_machine()

    assert published == [external]
    assert node.selected_slot_name == internal


def test_parallel_rejected_entry_keeps_external_selection_unselected():
    module = _load_auto_parallel_parking()
    node = object.__new__(module.AutoParallelParking)
    slot = SimpleNamespace(name='slot_1')
    node.execute_path = False
    node.slam_paused_by_node = False
    node.active_plan_goal = None
    node.active_follow_goal = None
    node.motion_phase = 'idle'
    node.active_motion_path = None
    node.active_motion_metrics = None
    node.active_motion_final = None
    node._abort_requested = lambda: False
    node._runtime_ok = lambda: False
    node._publish_status = lambda *a, **k: None
    node._wait_for_system = lambda: True
    node._wheel_centers_map = lambda: object()
    node._wait_for_map = lambda: True
    node._select_slot = lambda: slot
    node._plan_entry = lambda selected: None
    node._fail = lambda reason: None
    node._cancelled = lambda: None
    published = []
    node._publish_selected_slot = published.append

    node._run_state_machine()

    assert published == []


def test_slot_observation_two_view_sequence_never_guesses_unknown():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.slots = [
        SimpleNamespace(name='slot_1'),
        SimpleNamespace(name='slot_2'),
    ]
    node.slot_observation_results = {}
    node.selected_slot_name = ''
    node.get_parameter = lambda name: SimpleNamespace(value={
        'slot_observation.enabled': True,
        'slot_observation.preferred_free_slot': 'slot_1',
        'stop_wait_timeout': 1.0,
    }[name])
    node._observation_viewpoints = lambda: [object(), object()]
    node._move_to_observation_viewpoint = lambda index, pose: True
    node._wait_until_stopped = lambda timeout: True
    node._publish_status = lambda status: None
    node._publish_slot_observations = lambda results, selected: None
    node._log_info = lambda message: None
    observations = iter([
        {
            'slot_1': module.SlotObservation(
                'UNKNOWN', 20, 0, 0.0, 0, 'occluded'),
            'slot_2': module.SlotObservation(
                'UNKNOWN', 20, 0, 0.0, 0, 'occluded'),
        },
        {
            'slot_1': module.SlotObservation(
                'FREE', 20, 100, 0.8, 0, 'pass'),
            'slot_2': module.SlotObservation(
                'OCCUPIED', 20, 100, 0.8, 8, 'obstacle'),
        },
    ])
    node._collect_slot_observation = lambda index: (next(observations), 0.0)

    selected, reason = node._run_auto_slot_observation()

    assert selected.name == 'slot_1'
    assert reason.startswith('slot_1=FREE')


def test_slot_observation_stale_scan_is_unknown():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.slots = [SimpleNamespace(name='slot_1'),
                  SimpleNamespace(name='slot_2')]
    node.data_lock = threading.Lock()
    scan = module.LaserScan()
    scan.header.frame_id = 'laser'
    scan.header.stamp.sec = 10
    node.slot_scan_buffers = {
        'front': [(time.monotonic() - 5.0, 1, scan)],
        'rear': [(time.monotonic() - 5.0, 1, scan)],
    }
    values = {
        'slot_observation.max_scan_age': 0.5,
        'slot_observation.required_sensors': ['front', 'rear'],
        'slot_observation.min_frames': 10,
        'slot_observation.min_coverage': 0.5,
        'slot_observation.min_valid_rays': 40,
        'slot_observation.obstacle_point_threshold': 4,
        'slot_observation.obstacle_frame_threshold': 2,
        'slot_observation.coverage_resolution': 0.1,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node._slot_observation_roi = lambda slot: (0.0, 1.0, 0.0, 1.0)

    results = node._evaluate_slot_scan_buffers()

    assert all(result.state == module.SLOT_UNKNOWN
               for result in results.values())
    assert all('stale' in result.reason for result in results.values())


def test_slot_observation_missing_timestamp_tf_is_unknown():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    scan = module.LaserScan()
    scan.header.frame_id = 'laser'
    scan.header.stamp.sec = 10
    scan.angle_increment = 0.1
    scan.range_min = 0.1
    scan.range_max = 10.0
    scan.ranges = [1.0]

    class MissingTf:
        def lookup_transform(self, *args, **kwargs):
            raise module.tf2_ros.TransformException('synthetic missing TF')

    node.tf_buffer = MissingTf()
    rays, reason = node._scan_to_map_rays('front', 1, scan)
    assert rays == []
    assert reason == 'tf_missing_at_scan_stamp'


def test_slot_observation_insufficient_coverage_remains_unknown():
    module = _load_auto_parking()
    roi = (0.0, 1.0, 0.0, 2.0)
    rays = []
    for sensor in ('front', 'rear'):
        for frame in range(10):
            rays.append(module.ObservationRay(
                -1.0, 0.05, 2.0, 0.05, False,
                f'{sensor}:{frame}'))
    result = _classify_synthetic_slot(module, rays, roi, coverage=0.50)
    assert result.state == module.SLOT_UNKNOWN
    assert result.reason in ('insufficient_valid_rays',
                             'insufficient_coverage')


def test_slot_observation_roi_and_viewpoints_derive_from_slot_geometry():
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.slots = [
        module.Slot('slot_1', -9.69, -5.2, math.pi / 2,
                    -11.05, -8.33, -8.10, -2.30),
        module.Slot('slot_2', -6.60, -5.2, math.pi / 2,
                    -7.85, -5.35, -8.10, -2.30),
    ]
    node.course_entrance_pose = None
    values = {
        'slot_observation.wall_margin': 0.15,
        'slot_observation.curb_margin': 0.20,
        'slot_observation.lateral_clearance': 0.10,
        'vehicle_width': 0.78,
        'opposite_start_yaw': math.pi,
        'road_center_y': 0.05097,
        'slot_observation.viewpoints': [
            'nearest_slot_mouth', 'midpoint', 'farthest_slot_mouth'],
        'slot_observation.max_viewpoints': 3,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node._current_map_pose = lambda: (0.24247, 0.05097, math.pi)
    node._map_pose = lambda x, y, yaw: (x, y, yaw)
    node._log_warn = lambda message: None

    roi = node._slot_observation_roi(node.slots[0])
    viewpoints = node._observation_viewpoints()

    assert roi == pytest.approx((-10.18, -9.20, -7.90, -2.50))
    assert viewpoints[0] == pytest.approx((-6.60, 0.05097, math.pi))
    assert viewpoints[1] == pytest.approx((-8.145, 0.05097, math.pi))
    assert viewpoints[2] == pytest.approx((-9.69, 0.05097, math.pi))


def test_real_launch_auto_gate_and_manual_target_modes_are_explicit():
    package = Path(__file__).resolve().parents[1]
    launch_source = (
        package / 'launch' / 'real_t_parking.launch.py').read_text()
    auto_source = (SCRIPTS / 'auto_t_parking.py').read_text()
    assert "DeclareLaunchArgument('target_slot', default_value='auto')" in (
        launch_source)
    assert "'slot_observation.enabled'" in launch_source
    assert "requested == 'auto'" in auto_source
    assert 'elif self.rviz_only' in auto_source


@pytest.mark.parametrize('requested', ['slot_1', 'slot_2'])
def test_manual_target_slot_never_runs_auto_slot_observation(requested):
    """Manual target_slot must take the live-observation branch.

    Never the auto slot-observation branch, even though both live under
    the same if/elif/elif/else keyed on self.target_slot.
    """
    module = _load_auto_parking()
    node = object.__new__(module.AutoTParking)
    node.target_slot = requested
    node.rviz_only = False
    node.bench_mode = False
    node.slots = [
        SimpleNamespace(name='slot_1'), SimpleNamespace(name='slot_2')]
    node.slam_paused_by_node = False
    node._abort_requested = lambda: False
    node._publish_status = lambda *a, **k: None
    node._timing_set = lambda *a, **k: None
    node._slot_timing_add = lambda *a, **k: None
    node._wait_for_system = lambda: True
    node._capture_course_entrance_pose = lambda: True
    node._wheel_centers_map = lambda: object()
    node._cancelled = lambda: None
    fail_reasons = []
    node._fail = lambda reason: fail_reasons.append(reason)
    observation_calls = []
    node._run_auto_slot_observation = (
        lambda: observation_calls.append(True) or (None, 'unused'))
    node._ensure_live_observation = lambda: False

    node._run_state_machine()

    assert observation_calls == []
    assert fail_reasons == [
        'front/rear lidar sectors and costmaps did not provide a '
        'stable obstacle observation window']


def test_select_slot_status_is_published_exactly_once_per_path():
    """SELECT_SLOT must be published exactly once per completed slot.

    The auto path already publishes it from inside
    _run_auto_slot_observation for the winning viewpoint, so the shared
    code after the if/elif/elif/else must skip its own publish for that
    case while still publishing it for the other three paths.
    """
    module = _load_auto_parking()

    class StopHere(Exception):
        """Short-circuits the run right after slot selection."""

    def _build_node(requested, rviz_only, bench_mode):
        node = object.__new__(module.AutoTParking)
        node.target_slot = requested
        node.rviz_only = rviz_only
        node.bench_mode = bench_mode
        node.slam_paused_by_node = False
        node.slot_observation_results = {}
        slot_1 = SimpleNamespace(name='slot_1')
        slot_2 = SimpleNamespace(name='slot_2')
        node.slots = [slot_1, slot_2]
        node.data_lock = threading.Lock()
        node.map_msg = object()
        node.global_costmap = object()
        node.local_costmap = object()
        publishes = []
        node._publish_status = (
            lambda status, *a, **k: publishes.append(status))
        node._abort_requested = lambda: False
        node._timing_set = lambda *a, **k: None
        node._slot_timing_add = lambda *a, **k: None
        node._wait_for_system = lambda: True
        node._capture_course_entrance_pose = lambda: True
        node._wheel_centers_map = lambda: object()
        node._cancelled = lambda: None
        node._log_info = lambda *a, **k: None
        node._log_slot_timing = lambda *a, **k: None
        node._log_parking_geometry = lambda *a, **k: None
        node._runtime_ok = lambda: False
        node._fail = lambda reason: None

        def _stop(*a, **k):
            raise StopHere()
        node._plan_all_candidates = _stop

        if requested == 'auto':
            def _observe():
                node._publish_status('SELECT_SLOT')
                return slot_1, 'slot_1=FREE'
            node._run_auto_slot_observation = _observe
        elif not (rviz_only or bench_mode):
            node._ensure_live_observation = lambda: True
            node._wait_for_map_and_slots = lambda slots: slots

        return node, publishes

    for requested, rviz_only, bench_mode in (
            ('auto', False, False),
            ('slot_1', True, False),
            ('slot_1', False, True),
            ('slot_1', False, False)):
        node, publishes = _build_node(requested, rviz_only, bench_mode)

        node._run_state_machine()

        assert publishes.count('SELECT_SLOT') == 1, (
            requested, rviz_only, bench_mode, publishes)
