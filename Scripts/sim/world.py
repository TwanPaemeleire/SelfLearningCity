from sim.car import Car
from sim.track import Track


class World:
    def __init__(self, track: Track) -> None:
        self.track: Track = track
        self.cars: list[Car] = [Car(self.track.start_pos, self.track.start_angle)]

    def step(self, dt: float, controls: list[tuple[float, float]]) -> None:
        """Update the world state by a time step.

        Args:
            dt: The time step in seconds.
            controls: A list of tuples containing (steer, throttle) for each car.
        """
        for car, (steer, throttle) in zip(self.cars, controls, strict=True):
            car.update(steer, throttle, dt)
            car.check_collision(self.track.walls)
            car.check_next_checkpoint_reached(self.track.checkpoints)
