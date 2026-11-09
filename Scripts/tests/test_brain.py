import math

import numpy as np
import pytest

import config
from ai.brain import Brain

INPUTS = config.BRAIN_INPUTS
HIDDEN = config.BRAIN_HIDDEN
OUTPUTS = config.BRAIN_OUTPUTS


def random_brain(seed: int = 42) -> Brain:
    return Brain.random(np.random.default_rng(seed))


def zero_brain() -> Brain:
    """A brain with every weight and bias at 0."""
    return Brain(
        np.zeros((INPUTS, HIDDEN)),
        np.zeros(HIDDEN),
        np.zeros((HIDDEN, OUTPUTS)),
        np.zeros(OUTPUTS),
    )


# --- Brain.random ---


def test_random_brain_has_the_right_shapes():
    brain = random_brain()
    assert brain.weight1.shape == (INPUTS, HIDDEN)
    assert brain.bias1.shape == (HIDDEN,)
    assert brain.weight2.shape == (HIDDEN, OUTPUTS)
    assert brain.bias2.shape == (OUTPUTS,)


def test_random_brain_arrays_are_not_all_the_same():
    # Catches making a new generator per array with the same seed.
    brain = random_brain()
    assert not np.array_equal(brain.bias1[:OUTPUTS], brain.bias2)


def test_same_seed_gives_the_same_brain():
    a, b = random_brain(7), random_brain(7)
    assert np.array_equal(a.weight1, b.weight1)
    assert np.array_equal(a.weight2, b.weight2)


def test_different_seeds_give_different_brains():
    assert not np.array_equal(random_brain(1).weight1, random_brain(2).weight1)


# --- forward ---


def test_forward_returns_two_floats():
    result = random_brain().forward(np.ones(INPUTS))
    assert len(result) == 2
    steer, throttle = result
    assert isinstance(steer, float)
    assert isinstance(throttle, float)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, -1.0, 1000.0])
def test_forward_outputs_stay_between_minus_one_and_one(value):
    for seed in range(20):
        steer, throttle = random_brain(seed).forward(np.full(INPUTS, value))
        assert -1.0 <= steer <= 1.0
        assert -1.0 <= throttle <= 1.0


def test_forward_is_deterministic():
    brain = random_brain()
    inputs = np.array([0.1, 0.5, 1.0, 0.3, 0.8, 0.6])[:INPUTS]
    assert brain.forward(inputs) == brain.forward(inputs)


def test_different_inputs_give_different_outputs():
    brain = random_brain()
    assert brain.forward(np.zeros(INPUTS)) != brain.forward(np.ones(INPUTS))


def test_forward_does_not_change_the_brain():
    brain = random_brain()
    before = brain.weight1.copy()
    brain.forward(np.ones(INPUTS))
    assert np.array_equal(brain.weight1, before)


# --- forward, with numbers worked out by hand ---


def test_all_zero_brain_outputs_zero():
    assert zero_brain().forward(np.ones(INPUTS)) == (0.0, 0.0)


def test_output_biases_are_used():
    # Hidden layer is all 0, so the outputs are just tanh(bias2).
    brain = zero_brain()
    brain.bias2 = np.array([0.5, -0.5])
    steer, throttle = brain.forward(np.ones(INPUTS))
    assert steer == pytest.approx(math.tanh(0.5))
    assert throttle == pytest.approx(math.tanh(-0.5))


def test_weights_and_hidden_layer_are_used():
    # Every input is 1 and every first-layer weight is 0.1,
    # so each hidden neuron is tanh(INPUTS * 0.1).
    # Steer adds all hidden neurons up, throttle subtracts them.
    brain = zero_brain()
    brain.weight1 = np.full((INPUTS, HIDDEN), 0.1)
    brain.weight2 = np.zeros((HIDDEN, OUTPUTS))
    brain.weight2[:, 0] = 1.0
    brain.weight2[:, 1] = -1.0

    hidden = math.tanh(INPUTS * 0.1)
    steer, throttle = brain.forward(np.ones(INPUTS))
    assert steer == pytest.approx(math.tanh(HIDDEN * hidden))
    assert throttle == pytest.approx(math.tanh(-HIDDEN * hidden))


def assert_same_brain(a: Brain, b: Brain) -> None:
    assert np.array_equal(a.weight1, b.weight1)
    assert np.array_equal(a.bias1, b.bias1)
    assert np.array_equal(a.weight2, b.weight2)
    assert np.array_equal(a.bias2, b.bias2)


# --- Genome ---

GENOME_LENGTH = INPUTS * HIDDEN + HIDDEN + HIDDEN * OUTPUTS + OUTPUTS


def test_genome_length_counts_every_weight_and_bias():
    assert Brain.genome_length() == GENOME_LENGTH


def test_to_genome_is_flat_with_the_expected_length():
    genome = random_brain().to_genome()
    assert genome.shape == (GENOME_LENGTH,)


def test_genome_round_trip_gives_the_same_brain():
    brain = random_brain()
    rebuilt = Brain.from_genome(brain.to_genome())
    assert_same_brain(brain, rebuilt)


def test_genome_round_trip_gives_identical_outputs():
    brain = random_brain()
    rebuilt = Brain.from_genome(brain.to_genome())
    for seed in range(5):
        inputs = np.random.default_rng(seed).random(INPUTS)
        assert rebuilt.forward(inputs) == brain.forward(inputs)


def test_from_genome_fills_the_arrays_in_order():
    # Genome 0, 1, 2, ...: weight1 gets the first numbers, bias2 the last.
    brain = Brain.from_genome(np.arange(GENOME_LENGTH))
    assert brain.weight1[0, 0] == 0
    assert brain.weight1[0, 1] == 1  # row by row
    assert brain.bias1[0] == INPUTS * HIDDEN
    assert brain.weight2[0, 0] == INPUTS * HIDDEN + HIDDEN
    assert brain.bias2[-1] == GENOME_LENGTH - 1


@pytest.mark.parametrize("length", [0, GENOME_LENGTH - 1, GENOME_LENGTH + 1])
def test_from_genome_with_the_wrong_length_raises(length):
    with pytest.raises(ValueError):
        Brain.from_genome(np.zeros(length))


def test_from_genome_with_a_2d_array_raises():
    with pytest.raises(ValueError):
        Brain.from_genome(np.zeros((1, GENOME_LENGTH)))


def test_changing_the_genome_afterwards_does_not_change_the_brain():
    genome = random_brain().to_genome()
    brain = Brain.from_genome(genome)
    before = brain.to_genome()
    genome[:] = 0.0
    assert np.array_equal(brain.to_genome(), before)


def test_changing_a_genome_does_not_change_the_brain_it_came_from():
    brain = random_brain()
    before = brain.weight1.copy()
    brain.to_genome()[:] = 0.0
    assert np.array_equal(brain.weight1, before)


# --- Save and load ---


def test_save_creates_the_file(tmp_path):
    path = tmp_path / "brain.npz"
    random_brain().save(path)
    assert path.exists()


def test_save_then_load_gives_the_same_brain(tmp_path):
    path = tmp_path / "brain.npz"
    brain = random_brain()
    brain.save(path)
    assert_same_brain(Brain.load(path), brain)


def test_loaded_brain_drives_identically(tmp_path):
    path = tmp_path / "brain.npz"
    brain = random_brain()
    brain.save(path)
    loaded = Brain.load(path)
    inputs = np.linspace(0.0, 1.0, INPUTS)
    assert loaded.forward(inputs) == brain.forward(inputs)


def test_saved_file_stores_the_layer_sizes(tmp_path):
    path = tmp_path / "brain.npz"
    random_brain().save(path)
    with np.load(path) as data:
        assert list(data["layer_sizes"]) == [INPUTS, HIDDEN, OUTPUTS]


def test_loading_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Brain.load(tmp_path / "nope.npz")


def test_loading_a_brain_with_other_layer_sizes_raises(tmp_path):
    # As if it was saved before you changed the sensor count.
    path = tmp_path / "old.npz"
    inputs = INPUTS + 2
    np.savez(
        path,
        layer_sizes=np.array([inputs, HIDDEN, OUTPUTS]),
        weight1=np.zeros((inputs, HIDDEN)),
        bias1=np.zeros(HIDDEN),
        weight2=np.zeros((HIDDEN, OUTPUTS)),
        bias2=np.zeros(OUTPUTS),
    )
    with pytest.raises(ValueError, match="layer sizes"):
        Brain.load(path)


def test_loading_a_file_with_a_missing_array_raises(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez(path, layer_sizes=np.array([INPUTS, HIDDEN, OUTPUTS]), weight1=np.zeros((INPUTS, HIDDEN)))
    with pytest.raises(ValueError, match="not a saved brain"):
        Brain.load(path)


def test_loading_a_file_with_a_wrongly_shaped_array_raises(tmp_path):
    path = tmp_path / "bad_shape.npz"
    np.savez(
        path,
        layer_sizes=np.array([INPUTS, HIDDEN, OUTPUTS]),
        weight1=np.zeros((HIDDEN, INPUTS)),  # rows and columns swapped
        bias1=np.zeros(HIDDEN),
        weight2=np.zeros((HIDDEN, OUTPUTS)),
        bias2=np.zeros(OUTPUTS),
    )
    with pytest.raises(ValueError, match="weight1"):
        Brain.load(path)
