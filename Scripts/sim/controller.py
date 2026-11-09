from typing import Protocol


class Controller(Protocol):
    """Anything that can drive a car: a keyboard, a brain, ..."""

    def decide(self, inputs: list[float]) -> tuple[float, float]:
        """Decide how to drive this step.

        Args:
            inputs: What the car senses: its sensor readings (0 to 1) followed by its speed divided by max speed.

        Returns:
            (steer, throttle), each between -1 and 1. Steer -1 is full left and 1 is full right.
            Throttle -1 is full brake and 1 is full gas.
        """
        ...
