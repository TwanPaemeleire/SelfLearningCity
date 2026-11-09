import pytest

import config
from sim.geometry import Segment, Vec2
from sim.track import Track
from sim.world import World

DT = 1 / 60
FULL_THROTTLE = (0.0, 1.0)
NO_INPUT = (0.0, 0.0)


def box_track() -> Track:
    """A 400 x 200 box. The car starts at (100, 100) facing right, towards the wall at x = 400."""
    walls = [
        Segment(Vec2(0, 0), Vec2(400, 0)),
        Segment(Vec2(400, 0), Vec2(400, 200)),
        Segment(Vec2(400, 200), Vec2(0, 200)),
        Segment(Vec2(0, 200), Vec2(0, 0)),
    ]
    return Track(walls=walls, checkpoints=[], start_pos=Vec2(100, 100), start_angle=0.0)


def run(world: World, controls: list[tuple[float, float]], steps: int) -> int | None:
    """Step the world. Return the step on which the first car died, or None if it survived."""
    for i in range(1, steps + 1):
        world.step(DT, controls)
        if not world.cars[0].alive:
            return i
    return None


def test_world_starts_with_one_live_car_at_the_track_start():
    world = World(box_track())
    assert len(world.cars) == 1
    car = world.cars[0]
    assert car.alive
    assert car.position == Vec2(100, 100)
    assert car.angle == 0.0


def test_full_throttle_into_a_wall_dies_within_5_seconds():
    world = World(box_track())
    died_on = run(world, [FULL_THROTTLE], steps=5 * 60)
    assert died_on is not None


def test_car_does_not_die_instantly():
    # It needs to drive about 280 px before the front hits the wall.
    world = World(box_track())
    died_on = run(world, [FULL_THROTTLE], steps=5 * 60)
    assert died_on is not None
    assert died_on > 60


def test_car_without_input_survives_until_the_stuck_timer():
    world = World(box_track())
    assert run(world, [NO_INPUT], steps=config.CAR_TICK_COUNTER_KILL_THRESHOLD - 1) is None


def test_step_moves_the_car():
    world = World(box_track())
    world.step(DT, [FULL_THROTTLE])
    assert world.cars[0].position.x > 100


def test_step_uses_the_given_dt():
    world = World(box_track())
    world.step(0.0, [FULL_THROTTLE])
    assert world.cars[0].position == Vec2(100, 100)
    assert world.cars[0].speed == 0


def test_dead_car_stops_moving():
    world = World(box_track())
    run(world, [FULL_THROTTLE], steps=5 * 60)
    position = world.cars[0].position
    world.step(DT, [FULL_THROTTLE])
    assert world.cars[0].position == position


def test_wrong_number_of_controls_raises():
    world = World(box_track())
    with pytest.raises(ValueError):
        world.step(DT, [FULL_THROTTLE, FULL_THROTTLE])
