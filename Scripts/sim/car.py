import math

import config
from sim.geometry import Segment, Vec2, raycast


class Car:
    def __init__(self, position: Vec2, angle: float) -> None:
        self.position = position
        self.spawn_position = position
        self.previous_position = position
        self.angle = angle
        self.spawn_angle = angle
        self.speed = 0.0
        self.alive = True
        self.next_checkpoint_index = 0
        self.progress = 0
        self.fitness = 0.0
        self.tick_counter = 0

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
        self.previous_position = self.position
        self.position = self.position + direction * self.speed * dt

        # Update tick counter and kill if necesary
        self.tick_counter += 1
        if self.tick_counter >= config.CAR_TICK_COUNTER_KILL_THRESHOLD:
            self.alive = False

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

    def check_next_checkpoint_reached(self, checkpoints: list[Segment]) -> None:
        """Check if the car reached the next checkpoint and update its progress.

        Args:
            checkpoints: The list of checkpoints to check
        """
        amount_of_checkpoints: int = len(checkpoints)
        if amount_of_checkpoints == 0:
            return

        pos_change_segment: Segment = Segment(self.previous_position, self.position)
        if pos_change_segment.segment_intersection(checkpoints[self.next_checkpoint_index]) is not None:
            if self.next_checkpoint_index >= amount_of_checkpoints - 1:
                self.next_checkpoint_index = 0
            else:
                self.next_checkpoint_index += 1
            self.progress += 1
            self.tick_counter = 0

        self.fitness = self.progress + self._fraction_to_next_checkpoint(checkpoints)

    def _fraction_to_next_checkpoint(self, checkpoints: list[Segment]) -> float:
        """Return how far the car is from the last checkpoint it passed to the next one, from 0 to 1."""
        next_point = checkpoints[self.next_checkpoint_index].midpoint()
        if self.progress == 0:
            previous_point = self.spawn_position
        else:
            previous_point = checkpoints[self.next_checkpoint_index - 1].midpoint()

        gap = (next_point - previous_point).length()
        if gap == 0.0:
            return 0.0
        left = (next_point - self.position).length()
        return max(0.0, min(1.0, 1 - left / gap))

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
        self.previous_position = self.spawn_position
        self.angle = self.spawn_angle
        self.speed = 0.0
        self.alive = True
        self.progress = 0
        self.next_checkpoint_index = 0
        self.tick_counter = 0
        self.fitness = 0.0

    def sensor_readings(self, walls: list[Segment]) -> list[float]:
        """Return the normalized distances from the car to the nearest wall for each sensor.

        Args:
            walls: A list of wall segments to check against.

        Returns:
            A list of normalized distances for each sensor, 1.0 means no wall detected within sensor range.
        """
        readings: list[float] = []
        for angle in self.sensor_angles():
            distance: float = raycast(self.position, angle, config.CAR_SENSOR_LENGTH, walls)
            readings.append(distance / config.CAR_SENSOR_LENGTH)

        return readings

    def sensor_angles(self) -> list[float]:
        """Return the world angle (radians) of each sensor ray, from the car's left to its right."""
        angle_step: float = (config.CAR_SENSOR_ANGLE_END - config.CAR_SENSOR_ANGLE_START) / (config.CAR_SENSOR_COUNT - 1) if config.CAR_SENSOR_COUNT > 1 else 0
        return [self.angle + math.radians(config.CAR_SENSOR_ANGLE_START + i * angle_step) for i in range(config.CAR_SENSOR_COUNT)]
