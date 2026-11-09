from sim.car import Car
from sim.controller import Controller
from sim.track import Track


class World:
    def __init__(self, track: Track, controllers: list[Controller]) -> None:
        self.track: Track = track
        self.controllers = controllers
        self.cars: list[Car] = [Car(track.start_pos, track.start_angle) for _ in controllers]

    def step(self, dt: float) -> None:
        """Update the world state by a time step.

        Args:
            dt: The time step in seconds.
        """

        for car, controller in zip(self.cars, self.controllers, strict=True):
            if not car.alive:
                continue
            output: tuple[float, float] = controller.decide(car.inputs(self.track.walls))
            car.update(output[0], output[1], dt)
            car.check_next_checkpoint_reached(self.track.checkpoints)
            car.check_collision(self.track.walls)
