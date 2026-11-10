import numpy as np
import pytest

import config
from ai.evolution import EvolutionConfig
from ai.trainer import Trainer
from sim.geometry import Segment, Vec2
from sim.track import Track


def oval() -> Track:
    return Track.load(config.DATA_DIR / "tracks" / "oval.json")


def deadly_track() -> Track:
    """A wall straight through the start, so every car dies on its first step."""
    return Track(walls=[Segment(Vec2(100, 0), Vec2(100, 200))], checkpoints=[], start_pos=Vec2(100, 100), start_angle=0.0)


def small(**overrides) -> EvolutionConfig:
    """Small, fast settings for tests."""
    values = {"population_size": 6, "elite_count": 2, "generation_tick_limit": 5, "seed": 1}
    values.update(overrides)
    return EvolutionConfig(**values)


def run_generations(trainer: Trainer, count: int) -> None:
    while trainer.generation < count:
        trainer.step()


# --- Starting state ---


def test_starts_at_generation_zero_with_no_history():
    trainer = Trainer(oval(), small())
    assert trainer.generation == 0
    assert trainer.history == []
    assert trainer.tick == 0


def test_starts_with_one_brain_and_car_per_population_slot():
    trainer = Trainer(oval(), small(population_size=7))
    assert len(trainer.brains) == 7
    assert len(trainer.world.cars) == 7


def test_the_world_is_driven_by_the_trainers_brains():
    trainer = Trainer(oval(), small())
    assert trainer.world.controllers == trainer.brains


# --- Stepping ---


def test_step_counts_ticks():
    trainer = Trainer(oval(), small(generation_tick_limit=100))
    for _ in range(3):
        trainer.step()
    assert trainer.tick == 3


def test_generation_does_not_end_before_the_tick_limit():
    trainer = Trainer(oval(), small(generation_tick_limit=5))
    for _ in range(4):
        trainer.step()
    assert trainer.generation == 0
    assert trainer.history == []


def test_generation_ends_at_the_tick_limit():
    trainer = Trainer(oval(), small(generation_tick_limit=5))
    for _ in range(5):
        trainer.step()
    assert trainer.generation == 1
    assert len(trainer.history) == 1
    assert trainer.tick == 0


def test_generation_ends_early_when_every_car_is_dead():
    trainer = Trainer(deadly_track(), small(generation_tick_limit=1000))
    trainer.step()
    assert trainer.generation == 1
    assert len(trainer.history) == 1


# --- A new generation ---


def test_new_generation_gets_a_fresh_world_with_live_cars():
    trainer = Trainer(deadly_track(), small(generation_tick_limit=1000))
    old_world = trainer.world
    trainer.step()  # everyone dies, a new generation starts
    assert trainer.world is not old_world
    assert all(car.alive for car in trainer.world.cars)
    assert all(car.fitness == 0.0 for car in trainer.world.cars)


def test_new_generation_keeps_the_population_size():
    trainer = Trainer(oval(), small(population_size=8))
    run_generations(trainer, 2)
    assert len(trainer.brains) == 8
    assert len(trainer.world.cars) == 8


def test_new_world_is_driven_by_the_new_brains():
    trainer = Trainer(oval(), small())
    old_brains = trainer.brains
    run_generations(trainer, 1)
    assert trainer.brains is not old_brains
    assert trainer.world.controllers == trainer.brains


def test_history_records_best_and_average_fitness():
    trainer = Trainer(oval(), small(population_size=4))
    for car, fitness in zip(trainer.world.cars, [1.0, 4.0, 2.0, 5.0], strict=True):
        car.fitness = fitness
    trainer._end_generation()
    best, average = trainer.history[0]
    assert best == pytest.approx(5.0)
    assert average == pytest.approx(3.0)


def test_best_brain_survives_unchanged_as_an_elite():
    trainer = Trainer(oval(), small(population_size=4, elite_count=1))
    for car, fitness in zip(trainer.world.cars, [1.0, 9.0, 2.0, 3.0], strict=True):
        car.fitness = fitness
    best_genome = trainer.brains[1].to_genome()
    trainer._end_generation()
    assert np.array_equal(trainer.brains[0].to_genome(), best_genome)


def test_history_grows_by_one_per_generation():
    trainer = Trainer(oval(), small())
    run_generations(trainer, 4)
    assert len(trainer.history) == 4


# --- Reproducibility ---


def test_same_seed_gives_the_same_training_run():
    a = Trainer(oval(), small(seed=3, generation_tick_limit=30))
    b = Trainer(oval(), small(seed=3, generation_tick_limit=30))
    run_generations(a, 3)
    run_generations(b, 3)
    assert a.history == b.history
    assert np.array_equal(a.brains[0].to_genome(), b.brains[0].to_genome())


def test_different_seeds_give_different_brains():
    a = Trainer(oval(), small(seed=1))
    b = Trainer(oval(), small(seed=2))
    assert not np.array_equal(a.brains[0].to_genome(), b.brains[0].to_genome())
