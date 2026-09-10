import importlib.util
import os
from pathlib import Path
import pty
import select
import sys
import threading
import time

import pytest
import serial


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / 'scripts' / 't870_field_serial_bridge.py')


def _load_bridge():
    spec = importlib.util.spec_from_file_location(
        't870_field_serial_bridge_test_module', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_exact(fd: int, size: int, timeout: float = 1.0) -> bytes:
    deadline = time.monotonic() + timeout
    result = b''
    while len(result) < size and time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if ready:
            result += os.read(fd, size - len(result))
    return result


class _Publisher:
    def __init__(self):
        self.values = []

    def publish(self, message):
        self.values.append(message.data)


class _CancelClient:
    def __init__(self):
        self.requests = 0

    def service_is_ready(self):
        return True

    def call_async(self, _request):
        self.requests += 1


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


def _pty_bridge():
    bridge = _load_bridge()
    master_fd, slave_fd = pty.openpty()
    transport = serial.Serial(
        os.ttyname(slave_fd), 115200, timeout=0.0, write_timeout=0.2)
    node = object.__new__(bridge.T870FieldSerialBridge)
    node._lock = threading.Lock()
    node._serial = transport
    node._closing = False
    node._fault_latched = False
    node._drive = 0.0
    node._wheel = 0
    node._stop = False
    now = time.monotonic()
    node._drive_received_at = now
    node._wheel_received_at = now
    node._stop_received_at = now
    node._last_output_state = ''
    node.watchdog_timeout_sec = 0.50
    node.steering_limit_deg = 22
    node.shutdown_stop_repeats = 5
    node._ready_publisher = _Publisher()
    node._fault_publisher = _Publisher()
    node._emergency_publisher = _Publisher()
    node._cancel_client = _CancelClient()
    node.get_logger = lambda: _Logger()
    return bridge, node, master_fd, slave_fd


def _close_pty(node, master_fd, slave_fd):
    if node._serial is not None and node._serial.is_open:
        node._serial.close()
    os.close(slave_fd)
    os.close(master_fd)


@pytest.mark.parametrize('value, expected', [
    (1.0, b'w\n'),
    (2.0, b'e\n'),
    (3.0, b'r\n'),
    (-1.0, b'b\n'),
    (0.0, b'0'),
])
def test_drive_mapping(value, expected):
    assert _load_bridge().encode_drive_command(value) == expected


@pytest.mark.parametrize('value, expected', [
    (-22, b'L22\n'),
    (-1, b'L1\n'),
    (0, b'C\n'),
    (1, b'R1\n'),
    (22, b'R22\n'),
])
def test_steering_mapping(value, expected):
    assert _load_bridge().encode_wheel_command(value) == expected


@pytest.mark.parametrize('value', [23, -23])
def test_steering_over_limit_is_rejected_instead_of_clamped(value):
    with pytest.raises(ValueError, match='exceeds'):
        _load_bridge().encode_wheel_command(value)


@pytest.mark.parametrize(
    'value', [-3.0, -2.0, 0.5, 1.5, 2.5, 4.0, float('nan')])
def test_invalid_drive_is_rejected(value):
    with pytest.raises(ValueError, match='invalid FIELD drive stage'):
        _load_bridge().encode_drive_command(value)


def test_watchdog_requires_fresh_drive_wheel_and_stop():
    bridge = _load_bridge()
    now = 10.0
    assert bridge.command_inputs_fresh(now, 9.6, 9.6, 9.6, 0.5)
    assert not bridge.command_inputs_fresh(now, 9.4, 9.6, 9.6, 0.5)
    assert not bridge.command_inputs_fresh(now, 9.6, 9.4, 9.6, 0.5)
    assert not bridge.command_inputs_fresh(now, 9.6, 9.6, 9.4, 0.5)
    assert not bridge.command_inputs_fresh(now, None, 9.6, 9.6, 0.5)


def test_best_effort_stop_attempts_every_requested_write():
    bridge = _load_bridge()

    class AlwaysFailingTransport:
        shutdown_stop_repeats = 5

        def __init__(self):
            self.payloads = []

        def _write(self, payload):
            self.payloads.append(payload)
            return False

    transport = AlwaysFailingTransport()
    bridge.T870FieldSerialBridge._best_effort_stop(transport, repeats=3)
    assert transport.payloads == [b'0', b'0', b'0']


@pytest.mark.parametrize('payload', [
    b'w\n', b'e\n', b'r\n', b'b\n', b'0',
    b'L22\n', b'L1\n', b'C\n', b'R1\n', b'R22\n',
])
def test_encoded_command_is_written_exactly_through_pseudo_terminal(payload):
    master_fd, slave_fd = pty.openpty()
    slave_name = os.ttyname(slave_fd)
    transport = serial.Serial(
        slave_name, 115200, timeout=0.0, write_timeout=0.2)
    try:
        assert transport.write(payload) == len(payload)
        transport.flush()
        assert _read_exact(master_fd, len(payload)) == payload
    finally:
        transport.close()
        os.close(slave_fd)
        os.close(master_fd)


@pytest.mark.parametrize('wheel, expected', [
    (-22, b'L22\n0'),
    (-1, b'L1\n0'),
    (0, b'C\n0'),
    (1, b'R1\n0'),
    (22, b'R22\n0'),
])
def test_bridge_tick_writes_exact_wheel_mapping_through_pty(wheel, expected):
    bridge, node, master_fd, slave_fd = _pty_bridge()
    try:
        node._wheel_callback(bridge.Int32(data=wheel))
        node._output_tick()
        assert _read_exact(master_fd, len(expected)) == expected
    finally:
        _close_pty(node, master_fd, slave_fd)


@pytest.mark.parametrize('wheel', [23, -23])
def test_bridge_over_limit_latches_fault_cancel_and_serial_stop_through_pty(
        wheel):
    bridge, node, master_fd, slave_fd = _pty_bridge()
    try:
        node._wheel_callback(bridge.Int32(data=wheel))
        assert node._fault_latched is True
        assert _read_exact(master_fd, 3) == b'000'
        assert node._ready_publisher.values == [False]
        assert node._emergency_publisher.values == [True]
        assert node._cancel_client.requests == 1
    finally:
        _close_pty(node, master_fd, slave_fd)


def test_bridge_stop_stale_and_shutdown_are_serial_stops_through_pty():
    bridge, node, master_fd, slave_fd = _pty_bridge()
    try:
        node._stop_callback(bridge.Bool(data=True))
        assert _read_exact(master_fd, 1) == b'0'

        node._stop = False
        node._drive_received_at = time.monotonic() - 1.0
        node._output_tick()
        assert _read_exact(master_fd, 1) == b'0'

        node.shutdown_serial()
        assert _read_exact(master_fd, 5) == b'00000'
    finally:
        _close_pty(node, master_fd, slave_fd)
