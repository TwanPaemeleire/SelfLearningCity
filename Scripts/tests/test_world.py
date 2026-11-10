import numpy as np

import config
from ai.brain import Brain
from sim.geometry import Segment, Vec2
from sim.track import Track
from sim.world import World

DT = 1 / 60


class FixedController:
    """Test controller: always gives the same steer and throttle, and remembers the inputs it saw."""

    def __init__(self, steer: float = 0.0, throttle: float = 0.0) -> None:
        self.steer = steer
        self.throttle = throttle
        self.seen_inputs: list[list[float]] = []

    def decide(self, inputs: list[float]) -> tuple[float, float]:
        self.seen_inputs.append(inputs)
        return self.steer, self.throttle


def full_throttle() -> FixedController:
    return FixedController(throttle=1.0)


def no_input() -> FixedController:
    return FixedController()


def box_track() -> Track:
    """A 400 x 200 box. The car starts at (100, 100) facing right, towards the wall at x = 400."""
    walls = [
        Segment(Vec2(0, 0), Vec2(400, 0)),
        Segment(Vec2(400, 0), Vec2(400, 200)),
        Segment(Vec2(400, 200), Vec2(0, 200)),
        Segment(Vec2(0, 200), Vec2(0, 0)),
    ]
    return Track(walls=walls, checkpoints=[], start_pos=Vec2(100, 100), start_angle=0.0)


def run(world: World, steps: int) -> int | None:
    """Step the world. Return the step on which the first car died, or None if it survived."""
    for i in range(1, steps + 1):
        world.step(DT)
        if not world.cars[0].alive:
            return i
    return None


# --- Setup ---


def test_world_starts_with_one_live_car_at_the_track_start():
    world = World(box_track(), [no_input()])
    assert len(world.cars) == 1
    car = world.cars[0]
    assert car.alive
    assert car.position == Vec2(100, 100)
    assert car.angle == 0.0


def test_world_makes_one_car_per_controller():
    world = World(box_track(), [no_input(), no_input(), no_input()])
    assert len(world.cars) == 3
    assert all(car.position == Vec2(100, 100) for car in world.cars)


def test_cars_are_separate_objects():
    world = World(box_track(), [no_input(), no_input()])
    assert world.cars[0] is not world.cars[1]


# --- Driving and crashing ---


def test_full_throttle_into_a_wall_dies_within_5_seconds():
    world = World(box_track(), [full_throttle()])
    died_on = run(world, steps=5 * 60)
    assert died_on is not None


def test_car_does_not_die_instantly():
    # It needs to drive about 280 px before the front hits the wall.
    world = World(box_track(), [full_throttle()])
    died_on = run(world, steps=5 * 60)
    assert died_on is not None
    assert died_on > 60


def test_car_without_input_survives_until_the_stuck_timer():
    world = World(box_track(), [no_input()])
    assert run(world, steps=config.CAR_TICK_COUNTER_KILL_THRESHOLD - 1) is None


def test_step_moves_the_car():
    world = World(box_track(), [full_throttle()])
    world.step(DT)
    assert world.cars[0].position.x > 100


def test_step_uses_the_given_dt():
    world = World(box_track(), [full_throttle()])
    world.step(0.0)
    assert world.cars[0].position == Vec2(100, 100)
    assert world.cars[0].speed == 0


def test_dead_car_stops_moving():
    world = World(box_track(), [full_throttle()])
    run(world, steps=5 * 60)
    position = world.cars[0].position
    world.step(DT)
    assert world.cars[0].position == position


# --- Controllers ---


def test_each_controller_drives_its_own_car():
    world = World(box_track(), [full_throttle(), no_input()])
    for _ in range(30):
        world.step(DT)
    assert world.cars[0].position.x > 100
    assert world.cars[1].position == Vec2(100, 100)


def test_controller_is_asked_once_per_step():
    controller = no_input()
    world = World(box_track(), [controller])
    for _ in range(5):
        world.step(DT)
    assert len(controller.seen_inputs) == 5


def test_controller_gets_the_cars_inputs():
    controller = no_input()
    world = World(box_track(), [controller])
    world.step(DT)
    car = world.cars[0]
    assert controller.seen_inputs[0] == car.inputs(world.track.walls)
    assert len(controller.seen_inputs[0]) == config.BRAIN_INPUTS


def test_dead_cars_controller_is_not_asked():
    controller = full_throttle()
    world = World(box_track(), [controller])
    run(world, steps=5 * 60)
    asked = len(controller.seen_inputs)
    world.step(DT)
    assert len(controller.seen_inputs) == asked


def test_one_dead_car_does_not_stop_the_others():
    # Car 0 crashes into the wall; car 1 is parked and keeps living.
    world = World(box_track(), [full_throttle(), no_input()])
    run(world, steps=5 * 60)
    assert not world.cars[0].alive
    assert world.cars[1].alive


# --- Nearby walls: a cheap pre-check so cars only test walls they can reach ---


def test_walls_near_includes_a_wall_in_sensor_range():
    wall = Segment(Vec2(100 + config.CAR_SENSOR_LENGTH - 1, -1000), Vec2(100 + config.CAR_SENSOR_LENGTH - 1, 1000))
    track = Track(walls=[wall], checkpoints=[], start_pos=Vec2(100, 100), start_angle=0.0)
    world = World(track, [no_input()])
    assert world.walls_near(Vec2(100, 100)) == [wall]


def test_walls_near_skips_a_wall_far_away():
    far = Segment(Vec2(5000, 5000), Vec2(5100, 5000))
    track = Track(walls=[far], checkpoints=[], start_pos=Vec2(100, 100), start_angle=0.0)
    world = World(track, [no_input()])
    assert world.walls_near(Vec2(100, 100)) == []


def test_walls_near_uses_the_whole_wall_not_just_its_ends():
    # A long wall whose ends are far away, but whose middle passes right by the car.
    long_wall = Segment(Vec2(-5000, 150), Vec2(5000, 150))
    track = Track(walls=[long_wall], checkpoints=[], start_pos=Vec2(100, 100), start_angle=0.0)
    world = World(track, [no_input()])
    assert world.walls_near(Vec2(100, 100)) == [long_wall]


def test_nearby_walls_give_the_same_inputs_as_all_walls_on_the_circuit():
    # Drive random brains around the real circuit; at every step the shortcut must match checking every wall.
    track = Track.load(config.DATA_DIR / "tracks" / "circuit.json")
    rng = np.random.default_rng(3)
    world = World(track, [Brain.random(rng) for _ in range(10)])
    for _ in range(120):
        for car in world.cars:
            if car.alive:
                assert car.inputs(world.walls_near(car.position)) == car.inputs(track.walls)
        world.step(DT)
