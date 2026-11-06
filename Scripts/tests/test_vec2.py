"""Tests for sim/geometry.py."""

import math

import pytest

from sim.geometry import Vec2

# --- Vec2: basics ---


def test_vec2_stores_x_and_y():
    v = Vec2(3.0, 4.0)
    assert v.x == 3.0
    assert v.y == 4.0


def test_vec2_equal_when_components_match():
    assert Vec2(1.0, 2.0) == Vec2(1.0, 2.0)
    assert Vec2(1.0, 2.0) != Vec2(2.0, 1.0)


# --- Vec2: arithmetic ---


def test_add():
    assert Vec2(1.0, 2.0) + Vec2(3.0, 4.0) == Vec2(4.0, 6.0)


def test_subtract():
    assert Vec2(5.0, 7.0) - Vec2(2.0, 3.0) == Vec2(3.0, 4.0)


def test_multiply_by_number():
    assert Vec2(1.5, -2.0) * 2 == Vec2(3.0, -4.0)


def test_multiply_works_with_number_on_the_left():
    assert 2 * Vec2(1.5, -2.0) == Vec2(3.0, -4.0)


def test_operations_return_new_vectors_and_leave_inputs_unchanged():
    a = Vec2(1.0, 2.0)
    b = Vec2(3.0, 4.0)
    _ = a + b
    _ = a - b
    _ = a * 3
    assert a == Vec2(1.0, 2.0)
    assert b == Vec2(3.0, 4.0)


# --- Vec2: length, dot, cross ---


def test_length():
    assert Vec2(3.0, 4.0).length() == pytest.approx(5.0)


def test_length_of_zero_vector_is_zero():
    assert Vec2(0.0, 0.0).length() == 0.0


def test_dot_product():
    assert Vec2(1.0, 2.0).dot(Vec2(3.0, 4.0)) == pytest.approx(11.0)


def test_dot_of_perpendicular_vectors_is_zero():
    assert Vec2(1.0, 0.0).dot(Vec2(0.0, 1.0)) == pytest.approx(0.0)


def test_cross_product():
    # 2D cross product: a.x * b.y - a.y * b.x
    assert Vec2(1.0, 2.0).cross(Vec2(3.0, 4.0)) == pytest.approx(-2.0)


def test_cross_flips_sign_when_order_is_swapped():
    a = Vec2(1.0, 2.0)
    b = Vec2(3.0, 4.0)
    assert a.cross(b) == pytest.approx(-b.cross(a))


def test_cross_of_parallel_vectors_is_zero():
    # Segment intersection (2.1.2) relies on this to detect parallel lines.
    assert Vec2(2.0, 4.0).cross(Vec2(1.0, 2.0)) == pytest.approx(0.0)


# --- Vec2: from_angle (radians, 0 = right, y points down like pygame) ---


def test_from_angle_zero_points_right():
    v = Vec2.from_angle(0.0)
    assert v.x == pytest.approx(1.0)
    assert v.y == pytest.approx(0.0)


def test_from_angle_quarter_turn_points_down_the_screen():
    v = Vec2.from_angle(math.pi / 2)
    assert v.x == pytest.approx(0.0, abs=1e-9)
    assert v.y == pytest.approx(1.0)


def test_from_angle_half_turn_points_left():
    v = Vec2.from_angle(math.pi)
    assert v.x == pytest.approx(-1.0)
    assert v.y == pytest.approx(0.0, abs=1e-9)


def test_from_angle_always_has_length_one():
    for angle in (0.0, 0.3, 1.0, 2.5, -1.2, 10.0):
        assert Vec2.from_angle(angle).length() == pytest.approx(1.0)
