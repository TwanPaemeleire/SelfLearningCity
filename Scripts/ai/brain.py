import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import config


def _shapes() -> list[tuple[int, ...]]:
    """Shapes of weight1, bias1, weight2 and bias2, from the layer sizes in config."""
    return [
        (config.BRAIN_INPUTS, config.BRAIN_HIDDEN),
        (config.BRAIN_HIDDEN,),
        (config.BRAIN_HIDDEN, config.BRAIN_OUTPUTS),
        (config.BRAIN_OUTPUTS,),
    ]


class Brain:
    def __init__(self, w1: NDArray, b1: NDArray, w2: NDArray, b2: NDArray) -> None:
        self.weight1 = w1
        self.bias1 = b1
        self.weight2 = w2
        self.bias2 = b2

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Brain":
        w1 = rng.normal(size=(config.BRAIN_INPUTS, config.BRAIN_HIDDEN))
        b1 = rng.normal(size=(config.BRAIN_HIDDEN,))
        w2 = rng.normal(size=(config.BRAIN_HIDDEN, config.BRAIN_OUTPUTS))
        b2 = rng.normal(size=(config.BRAIN_OUTPUTS,))
        return cls(w1, b1, w2, b2)

    def forward(self, inputs: NDArray) -> tuple[float, float]:
        """Turn the car's inputs into (steer, throttle), each between -1 and 1.

        Args:
            inputs: The car's input array.

        Returns:
            The steer and throttle."""
        hidden = np.tanh(inputs @ self.weight1 + self.bias1)
        output = np.tanh(hidden @ self.weight2 + self.bias2)
        return float(output[0]), float(output[1])

    @staticmethod
    def genome_length() -> int:
        """Return how many numbers a genome has (every weight and bias)."""
        return sum(math.prod(shape) for shape in _shapes())

    def to_genome(self) -> NDArray:
        """Return every weight and bias as one flat 1D array (a new copy)."""
        return np.concatenate([self.weight1.ravel(), self.bias1, self.weight2.ravel(), self.bias2])

    @classmethod
    def from_genome(cls, genome: NDArray) -> "Brain":
        """Build a brain from a flat genome made by `to_genome`.

        Raises:
            ValueError: If the genome isn't a 1D array of `genome_length()` numbers.
        """
        genome = np.asarray(genome, dtype=float)
        if genome.shape != (cls.genome_length(),):
            raise ValueError(f"A genome must be a 1D array of {cls.genome_length()} numbers, got shape {genome.shape}")

        arrays = []
        start = 0
        for shape in _shapes():
            end = start + math.prod(shape)
            arrays.append(genome[start:end].reshape(shape).copy())
            start = end
        return cls(*arrays)

    def save(self, path: Path) -> None:
        """Save the brain to an .npz file, with its layer sizes."""
        np.savez(
            path,
            layer_sizes=np.array([config.BRAIN_INPUTS, config.BRAIN_HIDDEN, config.BRAIN_OUTPUTS]),
            weight1=self.weight1,
            bias1=self.bias1,
            weight2=self.weight2,
            bias2=self.bias2,
        )

    @classmethod
    def load(cls, path: Path) -> "Brain":
        """Load a brain saved with `save`.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file isn't a saved brain, or its layer sizes don't match config.
        """
        with np.load(path) as data:
            try:
                layer_sizes = tuple(int(size) for size in data["layer_sizes"])
                arrays = [data[name] for name in config.BRAIN_ARRAY_NAMES]
            except KeyError as err:
                raise ValueError(f"{path} is not a saved brain: it has no {err}") from err

        expected = (config.BRAIN_INPUTS, config.BRAIN_HIDDEN, config.BRAIN_OUTPUTS)
        if layer_sizes != expected:
            raise ValueError(f"{path} has layer sizes {layer_sizes}, but config expects {expected}")
        for name, array, shape in zip(config.BRAIN_ARRAY_NAMES, arrays, _shapes(), strict=True):
            if array.shape != shape:
                raise ValueError(f"{path} has {name} with shape {array.shape}, expected {shape}")
        return cls(*arrays)
