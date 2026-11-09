import pygame

import config
from sim.car import Car
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
