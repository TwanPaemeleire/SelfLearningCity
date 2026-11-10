from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import config


@dataclass(frozen=True)
class EvolutionConfig:
    """Every evolution setting in one place. Defaults come from config; tests can pass their own."""

    population_size: int = config.POPULATION_SIZE
    elite_count: int = config.ELITE_COUNT
    parent_fraction: float = config.PARENT_FRACTION
    mutation_rate: float = config.MUTATION_RATE
    mutation_strength: float = config.MUTATION_STRENGTH
    generation_tick_limit: int = config.GENERATION_TICK_LIMIT
    seed: int = config.SEED


def mutate(genome: NDArray, rng: np.random.Generator, rate: float, strength: float) -> NDArray:
    """Return a slightly changed copy of a genome. The genome passed in is never changed.

    Each number has a `rate` chance of being picked. Picked numbers get a random
    nudge added: a normal random number scaled by `strength`.

    Args:
        genome: The flat array of weights and biases to start from.
        rng: The random generator to use (same seed gives the same result).
        rate: Chance for each number to change, from 0 (none) to 1 (all).
        strength: How big the nudges are, roughly.

    Returns:
        A new array with the same shape as `genome`.
    """
    picked = rng.random(genome.shape) < rate
    nudges = rng.normal(0.0, strength, size=genome.shape)
    return genome + nudges * picked


def next_generation(genomes: list[NDArray], fitnesses: list[float], rng: np.random.Generator, cfg: EvolutionConfig) -> list[NDArray]:
    """Build the next generation's genomes from this generation's genomes and scores.

    Args:
        genomes: This generation's genomes. `genomes[i]` scored `fitnesses[i]`.
        fitnesses: The fitness of each genome.
        rng: The random generator to use (same seed gives the same result).
        cfg: The evolution settings.

    Returns:
        A new list of `cfg.population_size` genomes, best elites first. The input genomes are never changed.
    """
    if len(genomes) == 0:
        raise ValueError("Need at least one genome to build a generation from")
    if len(genomes) != len(fitnesses):
        raise ValueError(f"Got {len(genomes)} genomes but {len(fitnesses)} fitnesses")
    order = np.argsort(fitnesses, kind="stable")[::-1]
    ranked = [genomes[i] for i in order]

    elite_count = min(cfg.elite_count, cfg.population_size)
    new_genomes = [genome.copy() for genome in ranked[:elite_count]]

    parent_count = max(1, round(cfg.parent_fraction * len(ranked)))
    parents = ranked[:parent_count]
    while len(new_genomes) < cfg.population_size:
        parent = parents[rng.integers(len(parents))]
        new_genomes.append(mutate(parent, rng, cfg.mutation_rate, cfg.mutation_strength))

    return new_genomes
