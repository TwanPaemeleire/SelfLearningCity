import json
from dataclasses import dataclass
from pathlib import Path

from sim.geometry import Segment, Vec2


class TrackLoadError(Exception):
    """Raised when a track cannot be loaded from a file."""


@dataclass(frozen=False)
class Track:
    """A track is a collection of segments that form a path for a vehicle to follow."""

    walls: list[Segment]
    checkpoints: list[Segment]
    start_pos: Vec2
    start_angle: float

    def to_dict(self) -> dict:
        """Convert a Track object to a dictionary representation.

        Returns:
            The dictionary representation of the track.
        """
        return {
            "walls": [
                {
                    "start": {"x": wall.start.x, "y": wall.start.y},
                    "end": {"x": wall.end.x, "y": wall.end.y},
                }
                for wall in self.walls
            ],
            "checkpoints": [
                {
                    "start": {"x": checkpoint.start.x, "y": checkpoint.start.y},
                    "end": {"x": checkpoint.end.x, "y": checkpoint.end.y},
                }
                for checkpoint in self.checkpoints
            ],
            "start_pos": {"x": self.start_pos.x, "y": self.start_pos.y},
            "start_angle": self.start_angle,
        }

    @staticmethod
    def from_dict(data: dict) -> "Track":
        """Convert a dictionary representation to a Track object.

        Args:
            data: The dictionary representation of the track.

        Returns:
            The Track object.
        """
        walls = [
            Segment(
                start=Vec2(wall["start"]["x"], wall["start"]["y"]),
                end=Vec2(wall["end"]["x"], wall["end"]["y"]),
            )
            for wall in data["walls"]
        ]
        checkpoints = [
            Segment(
                start=Vec2(checkpoint["start"]["x"], checkpoint["start"]["y"]),
                end=Vec2(checkpoint["end"]["x"], checkpoint["end"]["y"]),
            )
            for checkpoint in data["checkpoints"]
        ]
        start_pos = Vec2(data["start_pos"]["x"], data["start_pos"]["y"])
        start_angle = data["start_angle"]

        return Track(
            walls=walls, checkpoints=checkpoints, start_pos=start_pos, start_angle=start_angle
        )

    def save(self, path: Path) -> None:
        """Save a Track object to a JSON file.

        Args:
            path: The path to the JSON file.
        """
        json_string = json.dumps(self.to_dict())
        with open(path, "w") as f:
            f.write(json_string)

    @staticmethod
    def load(path: Path) -> "Track":
        """Load a Track object from a JSON file.

        Args:
            path: The path to the JSON file.

        Returns:
            The Track object.
        """
        with open(path) as f:
            try:
                data = json.loads(f.read())
            except json.JSONDecodeError as err:
                raise TrackLoadError(f"{path} is not valid JSON: {err}") from err

        try:
            track: Track = Track.from_dict(data)
            return track
        except (KeyError, TypeError) as err:
            raise TrackLoadError(f"{path} is missing track data: {err}") from err
