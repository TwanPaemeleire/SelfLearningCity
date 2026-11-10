import numpy as np
import pytest

import config
from ai.brain import Brain
from ai.evolution import EvolutionConfig, mutate, next_generation


def make_genome(length: int = 74, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=length)


def rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- EvolutionConfig ---


def test_config_defaults_come_from_config():
    cfg = EvolutionConfig()
    assert cfg.population_size == config.POPULATION_SIZE
    assert cfg.mutation_rate == config.MUTATION_RATE


def test_config_values_can_be_overridden():
    cfg = EvolutionConfig(population_size=10, elite_count=2)
    assert cfg.population_size == 10
    assert cfg.elite_count == 2
    assert cfg.mutation_rate == config.MUTATION_RATE  # the rest keep their defaults


def test_config_cannot_be_changed_after_creation():
    cfg = EvolutionConfig()
    with pytest.raises(AttributeError):
        cfg.population_size = 5  # type: ignore[misc]


# --- mutate ---


def test_rate_zero_changes_nothing():
    genome = make_genome()
    assert np.array_equal(mutate(genome, rng(), rate=0.0, strength=0.3), genome)


def test_rate_one_changes_every_number():
    genome = make_genome()
    child = mutate(genome, rng(), rate=1.0, strength=0.3)
    assert np.all(child != genome)


def test_strength_zero_changes_nothing():
    genome = make_genome()
    assert np.array_equal(mutate(genome, rng(), rate=1.0, strength=0.0), genome)


def test_the_input_genome_is_not_changed():
    genome = make_genome()
    before = genome.copy()
    mutate(genome, rng(), rate=1.0, strength=0.3)
    assert np.array_equal(genome, before)


def test_returns_a_new_array_with_the_same_shape():
    genome = make_genome()
    child = mutate(genome, rng(), rate=0.5, strength=0.3)
    assert child is not genome
    assert child.shape == genome.shape


def test_same_seed_gives_the_same_result():
    genome = make_genome()
    a = mutate(genome, rng(7), rate=0.5, strength=0.3)
    b = mutate(genome, rng(7), rate=0.5, strength=0.3)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_results():
    genome = make_genome()
    a = mutate(genome, rng(1), rate=0.5, strength=0.3)
    b = mutate(genome, rng(2), rate=0.5, strength=0.3)
    assert not np.array_equal(a, b)


def test_rate_controls_how_many_numbers_change():
    # With 10,000 numbers and rate 0.1, about 1,000 should change.
    genome = make_genome(length=10_000)
    child = mutate(genome, rng(), rate=0.1, strength=0.3)
    changed = np.count_nonzero(child != genome)
    assert 800 < changed < 1200


def test_strength_controls_how_big_the_changes_are():
    genome = make_genome(length=10_000)
    small = mutate(genome, rng(), rate=1.0, strength=0.1) - genome
    big = mutate(genome, rng(), rate=1.0, strength=1.0) - genome
    assert np.abs(small).mean() < np.abs(big).mean()
    assert np.abs(small).mean() == pytest.approx(0.08, abs=0.01)  # average size of a normal * 0.1


def test_changes_go_both_up_and_down():
    genome = make_genome(length=1000)
    diff = mutate(genome, rng(), rate=1.0, strength=0.3) - genome
    assert np.any(diff > 0)
    assert np.any(diff < 0)


def test_mutated_genome_still_makes_a_working_brain():
    genome = Brain.random(rng()).to_genome()
    child = mutate(genome, rng(1), rate=0.1, strength=0.3)
    brain = Brain.from_genome(child)
    steer, throttle = brain.forward(np.ones(config.BRAIN_INPUTS))
    assert -1.0 <= steer <= 1.0
    assert -1.0 <= throttle <= 1.0


# --- next_generation ---
# Genome i is filled with the number i, so you can tell which one it came from.
# Fitness i means genome 9 is the best and genome 0 the worst.


def numbered_genomes(count: int = 10) -> list[np.ndarray]:
    return [np.full(5, float(i)) for i in range(count)]


def settings(**overrides) -> EvolutionConfig:
    values = {"population_size": 10, "elite_count": 2, "parent_fraction": 0.3, "mutation_rate": 0.1, "mutation_strength": 0.3}
    values.update(overrides)
    return EvolutionConfig(**values)


def evolve(cfg: EvolutionConfig, count: int = 10, seed: int = 42) -> list[np.ndarray]:
    return next_generation(numbered_genomes(count), [float(i) for i in range(count)], rng(seed), cfg)


def test_population_size_stays_the_same():
    assert len(evolve(settings())) == 10


def test_population_size_follows_the_config():
    assert len(evolve(settings(population_size=25))) == 25
    assert len(evolve(settings(population_size=4))) == 4


def test_every_genome_keeps_its_length():
    for genome in evolve(settings()):
        assert genome.shape == (5,)


def test_top_elites_come_first_unchanged():
    new = evolve(settings(elite_count=3, mutation_rate=1.0))
    assert np.array_equal(new[0], np.full(5, 9.0))  # best
    assert np.array_equal(new[1], np.full(5, 8.0))
    assert np.array_equal(new[2], np.full(5, 7.0))


def test_elites_work_when_fitnesses_are_shuffled():
    genomes = numbered_genomes(5)
    fitnesses = [3.0, 10.0, -2.0, 7.0, 0.0]  # genome 1 is best, then genome 3
    new = next_generation(genomes, fitnesses, rng(), settings(population_size=5, elite_count=2))
    assert np.array_equal(new[0], genomes[1])
    assert np.array_equal(new[1], genomes[3])


def test_children_are_mutated():
    new = evolve(settings(elite_count=2, mutation_rate=1.0))
    for child in new[2:]:
        assert not any(np.array_equal(child, parent) for parent in numbered_genomes())


def test_children_only_come_from_the_top_parents():
    # No mutation, so each child is an exact copy of its parent.
    # parent_fraction 0.3 of 10 cars = the top 3: genomes 9, 8 and 7.
    new = evolve(settings(mutation_rate=0.0))
    for child in new:
        assert child[0] in (9.0, 8.0, 7.0)


def test_bad_genomes_never_survive():
    new = evolve(settings(mutation_rate=0.0))
    assert not any(child[0] == 0.0 for child in new)


def test_tiny_parent_fraction_still_has_one_parent():
    # One elite (the best) plus children of the only parent (also the best).
    new = evolve(settings(parent_fraction=0.0, elite_count=1, mutation_rate=0.0))
    assert all(child[0] == 9.0 for child in new)


def test_zero_elites_works():
    new = evolve(settings(elite_count=0, mutation_rate=1.0))
    assert len(new) == 10
    assert not any(np.array_equal(child, np.full(5, 9.0)) for child in new)


def test_more_elites_than_population_is_capped():
    assert len(evolve(settings(population_size=3, elite_count=5))) == 3


def test_input_genomes_are_not_changed():
    genomes = numbered_genomes()
    before = [g.copy() for g in genomes]
    new = next_generation(genomes, [float(i) for i in range(10)], rng(), settings())
    new[0][:] = -1.0  # editing an elite in the new list...
    for genome, original in zip(genomes, before, strict=True):
        assert np.array_equal(genome, original)  # ...doesn't touch the old list


def test_equal_fitness_everywhere_does_not_crash():
    new = next_generation(numbered_genomes(), [1.0] * 10, rng(), settings())
    assert len(new) == 10


def test_all_zero_and_negative_fitness_works():
    new = next_generation(numbered_genomes(), [-5.0] * 9 + [-1.0], rng(), settings())
    assert np.array_equal(new[0], np.full(5, 9.0))


def test_same_seed_gives_the_same_generation():
    a = evolve(settings(), seed=7)
    b = evolve(settings(), seed=7)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        next_generation(numbered_genomes(10), [1.0] * 9, rng(), settings())


def test_no_genomes_raises():
    with pytest.raises(ValueError):
        next_generation([], [], rng(), settings())


def test_works_with_real_brain_genomes():
    brains = [Brain.random(rng(seed)) for seed in range(6)]
    genomes = [brain.to_genome() for brain in brains]
    new = next_generation(genomes, [float(i) for i in range(6)], rng(), settings(population_size=6))
    for genome in new:
        Brain.from_genome(genome)  # every new genome builds a valid brain
