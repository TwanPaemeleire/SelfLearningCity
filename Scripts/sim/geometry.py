import math
from dataclasses import dataclass


@dataclass(frozen=False)
class Vec2:
    """A 2D point or direction.

    Angles are in radians: 0 points right, and positive angles turn clockwise
    on screen, because the y axis points down (like pygame).
    """

    x: float = 0.0
    y: float = 0.0

    @staticmethod
    def from_angle(angle: float) -> "Vec2":
        """Return a vector of length 1 pointing in the direction of angle (radians)."""
        return Vec2(math.cos(angle), math.sin(angle))

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vec2":
        return self.__mul__(scalar)

    def length(self) -> float:
        """Return the length of the vector."""
        return (self.x**2 + self.y**2) ** 0.5

    def dot(self, other: "Vec2") -> float:
        """Return the dot product. It is 0 when the two vectors are perpendicular."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        """Return the 2D cross product (a single number). It is 0 when the vectors are parallel."""
        return self.x * other.y - self.y * other.x


@dataclass(frozen=False)
class Segment:
    """A straight line between two points, such as a wall, a checkpoint or a ray."""

    start: Vec2
    end: Vec2

    def segment_intersection(self, other: "Segment") -> Vec2 | None:
        """Return the point where this segment crosses another, if any.

        Args:
            other: The segment to test against.

        Returns:
            The intersection point, or None if the segments don't touch. Parallel
            and collinear segments also return None.
        """
        p = self.start
        r = self.end - self.start
        q = other.start
        s = other.end - other.start

        r_cross_s = r.cross(s)
        q_minus_p = q - p

        if r_cross_s == 0:
            return None  # Segments are parallel or collinear

        t = q_minus_p.cross(s) / r_cross_s
        u = q_minus_p.cross(r) / r_cross_s

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_point = p + r * t
            return intersection_point

        return None


def raycast(origin: Vec2, angle: float, max_length: float, walls: list[Segment]) -> float:
    """Cast a ray and return the distance to the nearest wall it hits.

    Args:
        origin: Where the ray starts.
        angle: The ray's direction in radians.
        max_length: How far the ray reaches.
        walls: The walls the ray can hit.

    Returns:
        The distance to the closest hit, or max_length if nothing is hit
        within reach.
    """
    ray_segment = Segment(origin, origin + Vec2.from_angle(angle) * max_length)
    closest_distance = max_length
    for wall in walls:
        intersection_point = ray_segment.segment_intersection(wall)
        if intersection_point is not None:
            distance = (intersection_point - origin).length()
            if distance < closest_distance:
                closest_distance = distance
    return closest_distance
