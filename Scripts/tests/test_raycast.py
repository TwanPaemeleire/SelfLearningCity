import math

import pytest

from sim.geometry import Segment, Vec2, raycast

MAX = 100.0
RIGHT = 0.0
DOWN = math.pi / 2  # y points down, like pygame
LEFT = math.pi
UP = -math.pi / 2


def seg(x1: float, y1: float, x2: float, y2: float) -> Segment:
    return Segment(Vec2(x1, y1), Vec2(x2, y2))


# --- Hitting walls ---


def test_wall_straight_ahead():
    wall = seg(50, -10, 50, 10)
    assert raycast(Vec2(0, 0), RIGHT, MAX, [wall]) == pytest.approx(50)


def test_ray_pointing_down_hits_wall_below():
    wall = seg(-10, 30, 10, 30)
    assert raycast(Vec2(0, 0), DOWN, MAX, [wall]) == pytest.approx(30)


def test_diagonal_ray():
    wall = seg(10, -100, 10, 100)
    assert raycast(Vec2(0, 0), math.pi / 4, MAX, [wall]) == pytest.approx(10 * math.sqrt(2))


def test_origin_away_from_zero():
    wall = seg(120, 90, 120, 110)
    assert raycast(Vec2(100, 100), RIGHT, MAX, [wall]) == pytest.approx(20)


def test_nearest_of_two_walls_wins():
    near = seg(30, -10, 30, 10)
    far = seg(60, -10, 60, 10)
    assert raycast(Vec2(0, 0), RIGHT, MAX, [far, near]) == pytest.approx(30)


def test_ray_touching_end_of_wall_counts_as_hit():
    wall = seg(40, 0, 40, 20)
    assert raycast(Vec2(0, 0), RIGHT, MAX, [wall]) == pytest.approx(40)


def test_origin_on_a_wall_gives_zero():
    wall = seg(0, -10, 0, 10)
    assert raycast(Vec2(0, 0), RIGHT, MAX, [wall]) == pytest.approx(0)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(RIGHT, 70), (DOWN, 60), (LEFT, 30), (UP, 40)],
)
def test_inside_a_box(angle, expected):
    # Like a car on a track: a 100x100 box, standing at (30, 40).
    box = [seg(0, 0, 100, 0), seg(100, 0, 100, 100), seg(100, 100, 0, 100), seg(0, 100, 0, 0)]
    assert raycast(Vec2(30, 40), angle, MAX, box) == pytest.approx(expected)


# --- Missing walls: returns max_length ---


def test_no_walls_returns_max_length():
    assert raycast(Vec2(0, 0), RIGHT, MAX, []) == MAX


def test_wall_beyond_max_length():
    wall = seg(150, -10, 150, 10)
    assert raycast(Vec2(0, 0), RIGHT, MAX, [wall]) == MAX


def test_wall_exactly_at_max_length():
    wall = seg(100, -10, 100, 10)
    assert raycast(Vec2(0, 0), RIGHT, MAX, [wall]) == pytest.approx(MAX)


def test_wall_behind_the_ray():
    wall = seg(-50, -10, -50, 10)
    assert raycast(Vec2(0, 0), RIGHT, MAX, [wall]) == MAX


def test_wall_off_to_the_side():
    wall = seg(50, 10, 50, 30)
    assert raycast(Vec2(0, 0), RIGHT, MAX, [wall]) == MAX


def test_wall_parallel_to_ray():
    wall = seg(0, 5, 100, 5)
    assert raycast(Vec2(0, 0), RIGHT, MAX, [wall]) == MAX
