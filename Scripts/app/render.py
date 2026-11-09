import pygame

import config
from sim.car import Car
from sim.geometry import Segment, Vec2
from sim.track import Track


def draw_track(surface: pygame.Surface, track: Track, checkpoint_to_highlight: int | None = None) -> None:
    """
    Draws the track on the given surface.

    Args:
        surface: The surface to draw on.
        track: The track object to draw.
        checkpoint_to_highlight: The index of the checkpoint to highlight (optional).
    """

    for wall in track.walls:
        pygame.draw.line(surface, config.TRACK_WALL_COLOR, (wall.start.x, wall.start.y), (wall.end.x, wall.end.y))

    for i in range(len(track.checkpoints)):
        checkpoint = track.checkpoints[i]
        if checkpoint_to_highlight is not None and i == checkpoint_to_highlight:
            pygame.draw.line(surface, config.TRACK_CHECKPOINT_HIGHLIGHTED_COLOR, (checkpoint.start.x, checkpoint.start.y), (checkpoint.end.x, checkpoint.end.y))
        else:
            pygame.draw.line(surface, config.TRACK_CHECKPOINT_COLOR, (checkpoint.start.x, checkpoint.start.y), (checkpoint.end.x, checkpoint.end.y))


def draw_car(surface: pygame.Surface, car: Car) -> None:
    """Draws the car on the given surface.

    Args:
        surface: The surface to draw on.
        car: The car object to draw.
    """
    color = config.CAR_DRAW_COLOR if car.alive else config.CAR_DEAD_DRAW_COLOR
    pygame.draw.polygon(surface, color, [(corner.x, corner.y) for corner in car.corners()])


def draw_sensors(surface: pygame.Surface, car: Car, walls: list[Segment]) -> None:
    """Draws the car's sensor rays, each stopping where it hits a wall.

    Args:
        surface: The surface to draw on.
        car: The car whose sensors to draw.
        walls: The walls the sensors can see.
    """
    readings = car.sensor_readings(walls)
    for angle, reading in zip(car.sensor_angles(), readings, strict=True):
        end = car.position + Vec2.from_angle(angle) * (reading * config.CAR_SENSOR_LENGTH)
        pygame.draw.line(surface, config.CAR_SENSOR_RAY_COLOR, (car.position.x, car.position.y), (end.x, end.y))
        if reading < 1.0:
            pygame.draw.circle(surface, config.CAR_SENSOR_RAY_COLOR, (end.x, end.y), 3)


def draw_fitness_and_kill_timer(surface: pygame.Surface, car: Car, font: pygame.Font) -> None:
    """Draws the fitness and kill timer of a car.

    Args:
        surface: The surface to draw on.
        car: The car whose data to draw.
        font: The font to draw with.
    """
    text = font.render(f"Fitness: {car.fitness:.2f}", True, (255, 255, 255))
    surface.blit(text, (10, 10))
    text = font.render(f"Death Countdown: {config.CAR_TICK_COUNTER_KILL_THRESHOLD - car.tick_counter}", True, (255, 255, 255))
    surface.blit(text, (10, 35))
