import pygame

import config
from sim.track import Track


def draw_track(surface: pygame.Surface, track: Track, checkpoint_to_highlight: int | None = None) -> None:
    """
    Draws the track on the given surface.

    Args:
        surface: The surface to draw on.
        track: The track object containing the track data.
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
