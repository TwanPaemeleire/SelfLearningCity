import math

import config
from sim.geometry import Segment, Vec2


class Car:
    def __init__(self, position: Vec2, angle: float) -> None:
        self.position = position
        self.spawn_position = position
        self.angle = angle
        self.spawn_angle = angle
        self.speed = 0.0
        self.alive = True

    def update(self, steer: float, throttle: float, dt: float) -> None:
        if not self.alive:
            return

        if throttle == 0.0:
            self.speed -= config.FRICTION_PER_SECOND * dt
        elif throttle > 0.0:
            self.speed += config.ACCELERATION_PER_SECOND * dt * throttle
        else:
            self.speed -= config.DECELERATION_PER_SECOND * dt * abs(throttle)

        # Limit speed to a maximum value
        self.speed = max(0.0, min(self.speed, config.CAR_MAX_SPEED))

        # Apply steering to the car's angle
        steering_multiplier = self.speed / config.CAR_MAX_SPEED
        self.angle += steer * config.CAR_STEERING_SPEED_RAD * dt * steering_multiplier

        # Update the car's position based on its speed and angle
        direction: Vec2 = Vec2.from_angle(self.angle)
        self.position = self.position + direction * self.speed * dt

    def check_collision(self, walls: list[Segment]) -> None:
        """Check if the car collides with any walls and sets the alive flag accordingly.

        Args:
            walls: A list of wall segments to check against.
        """
        car_edges: list[Segment] = self.edges()
        for wall in walls:
            for edge in car_edges:
                if wall.segment_intersection(edge) is not None:
                    self.alive = False
                    return

    def edges(self) -> list[Segment]:
        """Return the edges of the car as segments for collision detection."""
        corners: list[Vec2] = self.corners()
        edges: list[Segment] = [
            Segment(corners[0], corners[1]),  # Front edge
            Segment(corners[1], corners[2]),  # Right edge
            Segment(corners[2], corners[3]),  # Rear edge
            Segment(corners[3], corners[0]),  # Left edge
        ]
        return edges

    def corners(self) -> list[Vec2]:
        """Return the corners of the car as a list of Vec2 points."""
        half_width = config.CAR_WIDTH / 2
        half_length = config.CAR_LENGTH / 2

        # Calculate the corners of the car based on its position and angle
        front_left = self.position + Vec2.from_angle(self.angle) * half_length + Vec2.from_angle(self.angle + math.pi / 2) * half_width
        front_right = self.position + Vec2.from_angle(self.angle) * half_length + Vec2.from_angle(self.angle - math.pi / 2) * half_width
        rear_left = self.position - Vec2.from_angle(self.angle) * half_length + Vec2.from_angle(self.angle + math.pi / 2) * half_width
        rear_right = self.position - Vec2.from_angle(self.angle) * half_length + Vec2.from_angle(self.angle - math.pi / 2) * half_width

        return [front_left, front_right, rear_right, rear_left]

    def respawn(self) -> None:
        """Respawn the car at its initial position and reset its state."""
        self.position = self.spawn_position
        self.angle = self.spawn_angle
        self.speed = 0.0
        self.alive = True
