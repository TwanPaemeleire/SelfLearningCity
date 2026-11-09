import pytest

import config
from sim.car import Car
from sim.geometry import Segment, Vec2
from sim.track import Track
from sim.world import World

DT = 1 / 60

# Three vertical checkpoints across a straight road, in driving order (left to right).
CHECKPOINTS = [
    Segment(Vec2(200, 0), Vec2(200, 200)),
    Segment(Vec2(300, 0), Vec2(300, 200)),
    Segment(Vec2(400, 0), Vec2(400, 200)),
]


def make_car() -> Car:
    return Car(Vec2(100, 100), 0.0)


def move(car: Car, from_x: float, to_x: float) -> None:
    """Teleport the car along y = 100 and check checkpoints, like one simulation step."""
    car.previous_position = Vec2(from_x, 100)
    car.position = Vec2(to_x, 100)
    car.check_next_checkpoint_reached(CHECKPOINTS)


# --- Starting state ---


def test_new_car_has_no_progress():
    car = make_car()
    assert car.progress == 0
    assert car.next_checkpoint_index == 0


# --- Crossing in order ---


def test_crossing_the_next_checkpoint_counts():
    car = make_car()
    move(car, 190, 210)
    assert car.progress == 1
    assert car.next_checkpoint_index == 1


def test_not_reaching_a_checkpoint_does_not_count():
    car = make_car()
    move(car, 150, 190)
    assert car.progress == 0
    assert car.next_checkpoint_index == 0


def test_crossing_several_in_order_counts_each():
    car = make_car()
    move(car, 190, 210)
    move(car, 290, 310)
    assert car.progress == 2
    assert car.next_checkpoint_index == 2


def test_after_the_last_checkpoint_wraps_to_the_first():
    car = make_car()
    move(car, 190, 210)
    move(car, 290, 310)
    move(car, 390, 410)
    assert car.progress == 3
    assert car.next_checkpoint_index == 0


def test_second_lap_keeps_counting():
    car = make_car()
    for _ in range(2):
        move(car, 190, 210)
        move(car, 290, 310)
        move(car, 390, 410)
    assert car.progress == 6
    assert car.next_checkpoint_index == 0


# --- Cheating doesn't count ---


def test_crossing_out_of_order_does_not_count():
    car = make_car()
    move(car, 290, 310)  # skips checkpoint 0
    assert car.progress == 0
    assert car.next_checkpoint_index == 0


def test_reversing_back_across_a_checkpoint_does_not_count_twice():
    car = make_car()
    move(car, 190, 210)  # forward: counts
    move(car, 210, 190)  # backward across it again
    move(car, 190, 210)  # forward again
    assert car.progress == 1
    assert car.next_checkpoint_index == 1


# --- Real driving ---


def test_driving_across_a_checkpoint_with_update_counts():
    # update() must remember previous_position, so a normal drive is detected.
    car = make_car()
    car.speed = 200
    for _ in range(60):  # 1 second at 200 px/s: from x=100 to x=300
        car.update(0, 1, DT)
        car.check_next_checkpoint_reached(CHECKPOINTS)
    assert car.progress >= 1


def test_track_without_checkpoints_does_not_crash():
    car = make_car()
    car.previous_position = Vec2(190, 100)
    car.position = Vec2(210, 100)
    car.check_next_checkpoint_reached([])
    assert car.progress == 0


def test_respawn_resets_progress():
    car = make_car()
    move(car, 190, 210)
    car.respawn()
    assert car.progress == 0
    assert car.next_checkpoint_index == 0


def test_world_step_tracks_checkpoints():
    track = Track(walls=[], checkpoints=CHECKPOINTS, start_pos=Vec2(100, 100), start_angle=0.0)
    world = World(track)
    for _ in range(2 * 60):  # 2 seconds of full throttle
        world.step(DT, [(0.0, 1.0)])
    assert world.cars[0].progress >= 1


# --- Fitness: checkpoints passed + fraction of the way to the next one ---
# The car spawns at x=100; checkpoint midpoints are at x=200, 300, 400 (y=100).


def test_fitness_is_zero_at_spawn():
    car = make_car()
    move(car, 100, 100)
    assert car.fitness == pytest.approx(0.0)


def test_fitness_halfway_to_the_first_checkpoint():
    car = make_car()
    move(car, 100, 150)  # halfway from spawn (100) to checkpoint 0 (200)
    assert car.fitness == pytest.approx(0.5)


def test_fitness_just_after_a_checkpoint():
    car = make_car()
    move(car, 190, 210)  # passed checkpoint 0, 10% of the way to checkpoint 1
    assert car.fitness == pytest.approx(1.1)


def test_fitness_rises_while_driving_forward():
    car = make_car()
    previous = 0.0
    for x in range(110, 400, 10):
        move(car, x - 10, x)
        assert car.fitness > previous
        previous = car.fitness


def test_fitness_never_goes_below_progress():
    car = make_car()
    move(car, 100, 50)  # backwards, away from the first checkpoint
    assert car.fitness == pytest.approx(0.0)


def test_fitness_after_a_full_lap():
    car = make_car()
    move(car, 190, 210)
    move(car, 290, 310)
    move(car, 390, 410)  # wraps: next is checkpoint 0 again, behind the car
    assert 3.0 <= car.fitness < 4.0


def test_respawn_resets_fitness():
    car = make_car()
    move(car, 190, 210)
    car.respawn()
    assert car.fitness == 0.0


# --- Stuck timer: a car that makes no progress for too long dies ---

LIMIT = config.CAR_TICK_COUNTER_KILL_THRESHOLD


def tick(car: Car, times: int, steer: float = 0.0, throttle: float = 0.0) -> None:
    for _ in range(times):
        car.update(steer, throttle, DT)
        car.check_next_checkpoint_reached(CHECKPOINTS)


def test_parked_car_survives_until_just_before_the_limit():
    car = make_car()
    tick(car, LIMIT - 1)
    assert car.alive


def test_parked_car_dies_at_the_limit():
    car = make_car()
    tick(car, LIMIT)
    assert not car.alive


def test_passing_a_checkpoint_resets_the_timer():
    car = make_car()
    tick(car, LIMIT - 1)
    move(car, 190, 210)  # progress: timer starts over
    tick(car, LIMIT - 1)
    assert car.alive


def test_driving_in_circles_still_dies():
    # Moving isn't enough: only passing checkpoints counts as progress.
    car = make_car()
    car.speed = config.CAR_MAX_SPEED / 4
    tick(car, LIMIT, steer=1.0, throttle=0.5)
    assert car.progress == 0
    assert not car.alive


def test_respawn_resets_the_timer():
    car = make_car()
    tick(car, LIMIT - 1)
    car.respawn()
    tick(car, LIMIT - 1)
    assert car.alive
