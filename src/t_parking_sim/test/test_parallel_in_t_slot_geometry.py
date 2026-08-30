"""Static-map regression checks for the parallel-in-T-slot experiment."""

import math
from pathlib import Path

import pytest


MAP_ORIGIN = (-2.005, -2.130)
MAP_RESOLUTION = 0.05
VEHICLE_LENGTH = 1.33
VEHICLE_WIDTH = 0.78
BODY_CENTER_X = 0.020
WHEELBASE = 0.77
STEERING_LIMIT = math.radians(27.0)
TURN_RADIUS = 1.70
SLOTS = {
    'slot_1': (4.73, 6.46, 2.16, 7.59, 0.0),
    'slot_2': (6.54, 8.27, 2.16, 7.59, math.pi),
}


def _read_pgm(path: Path):
    with path.open('rb') as stream:
        assert stream.readline().strip() == b'P5'
        line = stream.readline()
        while line.startswith(b'#'):
            line = stream.readline()
        width, height = (int(value) for value in line.split())
        assert int(stream.readline()) == 255
        pixels = stream.read()
    assert len(pixels) == width * height
    return width, height, pixels


def _pixel(width, height, pixels, x, y):
    col = math.floor((x - MAP_ORIGIN[0]) / MAP_RESOLUTION)
    row = math.floor((y - MAP_ORIGIN[1]) / MAP_RESOLUTION)
    if not (0 <= col < width and 0 <= row < height):
        return None
    return pixels[(height - 1 - row) * width + col]


def _footprint(x, y, yaw):
    nx = math.ceil(VEHICLE_LENGTH / MAP_RESOLUTION)
    ny = math.ceil(VEHICLE_WIDTH / MAP_RESOLUTION)
    for ix in range(nx + 1):
        for iy in range(ny + 1):
            local_x = (
                BODY_CENTER_X - 0.5 * VEHICLE_LENGTH
                + VEHICLE_LENGTH * ix / nx)
            local_y = -0.5 * VEHICLE_WIDTH + VEHICLE_WIDTH * iy / ny
            yield (
                x + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
                y + math.sin(yaw) * local_x + math.cos(yaw) * local_y)


def _integrate(samples, distance, curvature):
    steps = max(1, math.ceil(abs(distance) / MAP_RESOLUTION))
    step = distance / steps
    x, y, yaw = samples[-1]
    for _ in range(steps):
        if abs(curvature) < 1.0e-9:
            x += step * math.cos(yaw)
            y += step * math.sin(yaw)
        else:
            next_yaw = yaw + curvature * step
            x += (math.sin(next_yaw) - math.sin(yaw)) / curvature
            y += (-math.cos(next_yaw) + math.cos(yaw)) / curvature
            yaw = next_yaw
        samples.append((x, y, math.atan2(math.sin(yaw), math.cos(yaw))))


def _entry_path(slot):
    min_x, max_x, min_y, max_y, yaw = slot
    park_x = 0.5 * (min_x + max_x)
    park_y = 0.5 * (min_y + max_y)
    lane_y = min_y - 0.80
    left_y = math.cos(yaw)
    lateral = (park_y - lane_y) * left_y
    angle = math.acos(1.0 - abs(lateral) / (2.0 * TURN_RADIUS))
    required = 2.0 * TURN_RADIUS * math.sin(angle)
    approach = required + 0.10
    samples = [(
        park_x + approach * math.cos(yaw),
        lane_y + approach * math.sin(yaw),
        yaw)]
    _integrate(samples, -(approach - required), 0.0)
    sign = math.copysign(1.0, lateral)
    _integrate(samples, -TURN_RADIUS * angle, sign / TURN_RADIUS)
    _integrate(samples, -TURN_RADIUS * angle, -sign / TURN_RADIUS)
    return samples


def test_vehicle_and_steering_fit_are_physical():
    minimum_radius = WHEELBASE / math.tan(STEERING_LIMIT)
    assert minimum_radius == pytest.approx(1.511214, abs=1.0e-5)
    assert TURN_RADIUS >= minimum_radius
    for min_x, max_x, min_y, max_y, _yaw in SLOTS.values():
        assert max_x - min_x >= VEHICLE_LENGTH + 2.0 * 0.03
        assert max_y - min_y >= VEHICLE_WIDTH + 2.0 * 0.03


@pytest.mark.parametrize('slot_name', ['slot_1', 'slot_2'])
def test_reverse_s_curve_swept_footprint_is_free_in_saved_map(slot_name):
    map_path = (
        Path(__file__).parents[1]
        / 'maps' / 'combined_parking_map_real_vehicle.pgm')
    width, height, pixels = _read_pgm(map_path)
    path = _entry_path(SLOTS[slot_name])
    final = path[-1]
    min_x, max_x, min_y, max_y, yaw = SLOTS[slot_name]
    assert final[0] == pytest.approx(0.5 * (min_x + max_x), abs=1.0e-6)
    assert final[1] == pytest.approx(0.5 * (min_y + max_y), abs=1.0e-6)
    assert abs(math.atan2(math.sin(final[2] - yaw), math.cos(final[2] - yaw))) < 1.0e-6

    for pose in path:
        for x, y in _footprint(*pose):
            value = _pixel(width, height, pixels, x, y)
            assert value is not None
            # Map YAML is negate=0, occupied_thresh=0.65. Thus pixels <=89
            # are occupied; 205 is the map_server unknown convention here.
            assert value != 205
            assert value > 89
