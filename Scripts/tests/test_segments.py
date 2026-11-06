import pytest

from sim.geometry import Segment, Vec2


def seg(x1: float, y1: float, x2: float, y2: float) -> Segment:
    return Segment(Vec2(x1, y1), Vec2(x2, y2))


def assert_point(result: Vec2 | None, x: float, y: float) -> None:
    assert result is not None, "expected an intersection, got None"
    assert result.x == pytest.approx(x)
    assert result.y == pytest.approx(y)


# --- Segments that cross ---


def test_basic_intersection():
    s1 = Segment(Vec2(0, 0), Vec2(10, 10))
    s2 = Segment(Vec2(0, 10), Vec2(10, 0))
    assert s1.segment_intersection(s2) == Vec2(5.0, 5.0)


def test_order_of_segments_does_not_matter():
    s1 = seg(0, 0, 10, 10)
    s2 = seg(0, 10, 10, 0)
    assert_point(s2.segment_intersection(s1), 5, 5)


def test_horizontal_and_vertical_segments():
    # Most walls you draw will be like this.
    assert_point(seg(0, 5, 10, 5).segment_intersection(seg(3, 0, 3, 10)), 3, 5)


def test_intersection_at_non_integer_point():
    assert_point(seg(0, 0, 4, 0).segment_intersection(seg(1, -1, 2, 3)), 1.25, 0)


def test_negative_coordinates():
    assert_point(seg(-5, -5, 5, 5).segment_intersection(seg(-5, 5, 5, -5)), 0, 0)


# --- Touching exactly (counts as an intersection) ---


def test_touching_at_shared_endpoint():
    # Two walls meeting in a corner.
    assert_point(seg(0, 0, 5, 5).segment_intersection(seg(5, 5, 10, 0)), 5, 5)


def test_end_of_one_segment_touching_middle_of_other():
    # T-junction: a car edge just touching a wall.
    assert_point(seg(0, 0, 10, 0).segment_intersection(seg(5, 0, 5, 5)), 5, 0)


# --- Segments that don't touch ---


def test_parallel_segments_return_none():
    assert seg(0, 0, 10, 0).segment_intersection(seg(0, 5, 10, 5)) is None


def test_lines_would_cross_if_extended_but_segments_stop_short():
    # The crossing point (6, 0) is past the end of the first segment.
    assert seg(0, 0, 4, 0).segment_intersection(seg(6, -1, 6, 1)) is None


def test_segment_ends_before_reaching_the_other():
    # The vertical segment would hit y=0, but it starts at y=1.
    assert seg(0, 0, 10, 0).segment_intersection(seg(5, 1, 5, 3)) is None


def test_near_miss_returns_none():
    assert seg(0, 0, 10, 0).segment_intersection(seg(5, 0.001, 5, 10)) is None


# --- Collinear and degenerate cases (chosen behaviour: return None) ---


def test_collinear_overlapping_segments_return_none():
    # They overlap along a whole stretch, not at a single point.
    assert seg(0, 0, 10, 0).segment_intersection(seg(5, 0, 15, 0)) is None


def test_collinear_separate_segments_return_none():
    assert seg(0, 0, 4, 0).segment_intersection(seg(6, 0, 10, 0)) is None


def test_zero_length_segment_returns_none():
    assert seg(3, 3, 3, 3).segment_intersection(seg(0, 0, 10, 10)) is None
