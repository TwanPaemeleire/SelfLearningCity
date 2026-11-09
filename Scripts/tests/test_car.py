import math

import pytest

import config
from sim.car import Car
from sim.geometry import Segment, Vec2
from sim.track import Track

DT = 1 / 60
MAX = config.CAR_MAX_SPEED
HALF_L = config.CAR_LENGTH / 2
HALF_W = config.CAR_WIDTH / 2


def make_car(speed: float = 0.0, angle: float = 0.0) -> Car:
    car = Car(Vec2(100, 100), angle)
    car.speed = speed
    return car


def drive(car: Car, steer: float, throttle: float, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        car.update(steer, throttle, DT)


# --- Starting state ---


def test_new_car_is_still_and_alive():
    car = Car(Vec2(10, 20), 1.5)
    assert car.position == Vec2(10, 20)
    assert car.angle == 1.5
    assert car.speed == 0
    assert car.alive


# --- Throttle and speed ---


def test_throttle_increases_speed():
    car = make_car()
    car.update(0, 1, DT)
    assert car.speed > 0


def test_half_throttle_accelerates_less_than_full():
    half, full = make_car(), make_car()
    half.update(0, 0.5, DT)
    full.update(0, 1, DT)
    assert 0 < half.speed < full.speed


def test_no_throttle_slows_the_car_down():
    car = make_car(speed=100)
    car.update(0, 0, DT)
    assert car.speed < 100


def test_coasting_eventually_stops_completely():
    car = make_car(speed=MAX)
    drive(car, 0, 0, seconds=30)
    assert car.speed == 0


def test_braking_slows_the_car_down():
    car = make_car(speed=100)
    car.update(0, -1, DT)
    assert car.speed < 100


def test_braking_slows_faster_than_coasting():
    braking, coasting = make_car(speed=100), make_car(speed=100)
    braking.update(0, -1, DT)
    coasting.update(0, 0, DT)
    assert braking.speed < coasting.speed


def test_speed_never_exceeds_maximum():
    car = make_car()
    drive(car, 0, 1, seconds=30)
    assert car.speed == pytest.approx(MAX)


def test_braking_never_makes_speed_negative():
    car = make_car(speed=1)
    drive(car, 0, -1, seconds=5)
    assert car.speed == 0


# --- Steering ---


def test_steering_at_zero_speed_does_not_rotate():
    car = make_car(speed=0, angle=0.5)
    car.update(1, 0, DT)
    assert car.angle == 0.5


def test_steering_right_increases_angle():
    # Positive angle = clockwise on screen.
    car = make_car(speed=MAX)
    car.update(1, 1, DT)
    assert car.angle > 0


def test_steering_left_decreases_angle():
    car = make_car(speed=MAX)
    car.update(-1, 1, DT)
    assert car.angle < 0


def test_no_steering_keeps_the_angle():
    car = make_car(speed=MAX, angle=0.5)
    car.update(0, 1, DT)
    assert car.angle == 0.5


def test_faster_car_turns_more():
    slow, fast = make_car(speed=MAX / 4), make_car(speed=MAX)
    slow.update(1, 1, DT)
    fast.update(1, 1, DT)
    assert 0 < slow.angle < fast.angle


# --- Moving ---


def test_parked_car_does_not_move():
    car = make_car()
    car.update(0, 0, DT)
    assert car.position == Vec2(100, 100)


def test_moves_right_when_facing_right():
    car = make_car(speed=MAX, angle=0)
    car.update(0, 1, DT)
    assert car.position.x == pytest.approx(100 + MAX * DT)
    assert car.position.y == pytest.approx(100)


def test_moves_down_when_facing_down():
    # y points down on screen, like pygame.
    car = make_car(speed=MAX, angle=math.pi / 2)
    car.update(0, 1, DT)
    assert car.position.x == pytest.approx(100)
    assert car.position.y == pytest.approx(100 + MAX * DT)


def test_distance_matches_speed_over_one_second():
    car = make_car(speed=MAX, angle=0)
    drive(car, 0, 1, seconds=1)
    assert car.position.x == pytest.approx(100 + MAX)


# --- Dead cars ---


def test_dead_car_does_not_change():
    car = make_car(speed=100, angle=0.5)
    car.alive = False
    car.update(1, 1, DT)
    assert car.position == Vec2(100, 100)
    assert car.speed == 100
    assert car.angle == 0.5


# --- Crashing into walls ---
# make_car() puts the car at (100, 100). Facing right (angle 0), its body spans
# x = 100 +- HALF_L and y = 100 +- HALF_W.


def wall(x1: float, y1: float, x2: float, y2: float) -> Segment:
    return Segment(Vec2(x1, y1), Vec2(x2, y2))


def test_wall_through_the_middle_kills_the_car():
    car = make_car()
    car.check_collision([wall(100, 0, 100, 200)])
    assert not car.alive


def test_wall_clipping_the_front_kills_the_car():
    car = make_car()
    x = 100 + HALF_L - 1  # 1 px inside the front of the car
    car.check_collision([wall(x, 0, x, 200)])
    assert not car.alive


def test_wall_clipping_the_side_kills_the_car():
    car = make_car()
    y = 100 + HALF_W - 1
    car.check_collision([wall(0, y, 200, y)])
    assert not car.alive


def test_wall_exactly_touching_the_front_counts_as_a_crash():
    car = make_car()
    x = 100 + HALF_L
    car.check_collision([wall(x, 0, x, 200)])
    assert not car.alive


def test_one_hit_among_many_walls_is_enough():
    car = make_car()
    far_away = [wall(500, 0, 500, 200), wall(0, 500, 200, 500)]
    car.check_collision([*far_away, wall(100, 0, 100, 200)])
    assert not car.alive


def test_car_in_open_space_stays_alive():
    car = make_car()
    car.check_collision([wall(500, 0, 500, 200), wall(0, 500, 200, 500)])
    assert car.alive


def test_no_walls_stays_alive():
    car = make_car()
    car.check_collision([])
    assert car.alive


def test_wall_just_in_front_does_not_kill_the_car():
    car = make_car()
    x = 100 + HALF_L + 1  # 1 px gap
    car.check_collision([wall(x, 0, x, 200)])
    assert car.alive


def test_wall_just_beside_does_not_kill_the_car():
    car = make_car()
    y = 100 + HALF_W + 1
    car.check_collision([wall(0, y, 200, y)])
    assert car.alive


def test_rotated_car_uses_its_rotated_body():
    # Facing down, the car is long in y and narrow in x.
    car = make_car(angle=math.pi / 2)
    x = 100 + HALF_W + 1  # would hit an unrotated car, misses a rotated one
    car.check_collision([wall(x, 0, x, 200)])
    assert car.alive

    y = 100 + HALF_L - 1  # inside the front of the rotated car
    car.check_collision([wall(0, y, 200, y)])
    assert not car.alive


def test_dead_car_stays_dead_in_open_space():
    car = make_car()
    car.alive = False
    car.check_collision([])
    assert not car.alive


def test_car_at_oval_start_is_alive():
    track = Track.load(config.DATA_DIR / "tracks" / "oval.json")
    car = Car(track.start_pos, track.start_angle)
    car.check_collision(track.walls)
    assert car.alive
