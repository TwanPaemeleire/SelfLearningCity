import numpy as np

import config
from ai.brain import Brain
from ai.evolution import EvolutionConfig, next_generation
from sim.track import Track
from sim.world import World


class Trainer:
    def __init__(self, track: Track, cfg: EvolutionConfig) -> None:
        self.track = track
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.generation = 0
        self.history: list[tuple[float, float]] = []  # (best, average)
        self.tick = 0

        self.brains = [Brain.random(self.rng) for _ in range(cfg.population_size)]
        self.world = World(track, self.brains)

    def step(self) -> None:
        """Advance the simulation one step. Ends the generation when every car is dead or time is up."""
        self.world.step(config.DT)
        self.tick += 1

        all_dead = not any(car.alive for car in self.world.cars)
        if all_dead or self.tick >= self.cfg.generation_tick_limit:
            self._end_generation()

    def _end_generation(self) -> None:
        """Score this generation, breed the next one from it, and start it in a fresh World."""
        fitnesses = [car.fitness for car in self.world.cars]
        self.history.append((max(fitnesses), sum(fitnesses) / len(fitnesses)))

        genomes = [brain.to_genome() for brain in self.brains]
        new_genomes = next_generation(genomes, fitnesses, self.rng, self.cfg)
        self.brains = [Brain.from_genome(genome) for genome in new_genomes]

        self.world = World(self.track, self.brains)
        self.tick = 0
        self.generation += 1
