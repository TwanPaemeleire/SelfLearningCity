import config
from sim.car import Car
from sim.controller import Controller
from sim.geometry import Segment, Vec2
from sim.track import Track


class World:
    def __init__(self, track: Track, controllers: list[Controller]) -> None:
        self.track: Track = track
        self.controllers = controllers
        self.cars: list[Car] = [Car(track.start_pos, track.start_angle) for _ in controllers]

        self._wall_boxes = [
            (
                min(wall.start.x, wall.end.x) - config.NEAR_DISTANCE,
                min(wall.start.y, wall.end.y) - config.NEAR_DISTANCE,
                max(wall.start.x, wall.end.x) + config.NEAR_DISTANCE,
                max(wall.start.y, wall.end.y) + config.NEAR_DISTANCE,
                wall,
            )
            for wall in track.walls
        ]

    def walls_near(self, position: Vec2) -> list[Segment]:
        """Return the walls a car at this position could see or touch: a cheap pre-check before raycasts and collisions.

        It may include a few walls that are slightly too far, but never leaves out one that's in reach.
        """
        return [wall for left, top, right, bottom, wall in self._wall_boxes if left <= position.x <= right and top <= position.y <= bottom]

    def step(self, dt: float) -> None:
        """Update the world state by a time step.

        Args:
            dt: The time step in seconds.
        """

        for car, controller in zip(self.cars, self.controllers, strict=True):
            if not car.alive:
                continue
            nearby = self.walls_near(car.position)
            output: tuple[float, float] = controller.decide(car.inputs(nearby))
            car.update(output[0], output[1], dt)
            car.check_next_checkpoint_reached(self.track.checkpoints)
            car.check_collision(nearby)
