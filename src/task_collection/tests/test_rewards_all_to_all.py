"""Tests for deterministic all-to-all Fourier reward generation."""

import numpy as np
import pytest

from task_collection.env.stochastic import (
    StochasticEnv,
    generate_rewards_all_to_all,
    generate_rewards_circulant,
    generate_rewards_circulant_derived_task_variance,
    scaled_task_total_cycles,
)


@pytest.mark.parametrize(
    ("sigma_a", "sigma_b"),
    [(2.0, 6.0), (0.0, 6.0), (2.0, 0.0), (0.0, 0.0)],
)
def test_all_to_all_reward_invariants(sigma_a, sigma_b):
    N = 11
    rewards = generate_rewards_all_to_all(N, sigma_a, sigma_b)

    assert rewards.shape == (N, N)
    assert np.allclose(rewards.std(axis=0, ddof=0), sigma_a)
    assert np.isclose(rewards.sum(axis=0).std(ddof=0), sigma_b)
    assert np.allclose(rewards.mean(axis=1), 1.0 / N)
    assert np.allclose(
        rewards.std(axis=1, ddof=0),
        np.sqrt(sigma_a**2 + sigma_b**2 / N**2),
    )


def test_main_example_agent_standard_deviation():
    rewards = generate_rewards_all_to_all(11, 2.0, 6.0)
    assert np.allclose(rewards.std(axis=1, ddof=0), np.sqrt(2**2 + 6**2 / 11**2))


@pytest.mark.parametrize("sigma", [1.0, 2.0, 4.0, 8.0, 16.0])
def test_equal_sigma_sweep_for_plot_matrices(sigma):
    rewards = generate_rewards_all_to_all(11, sigma, sigma)

    assert np.allclose(rewards.std(axis=0, ddof=0), sigma)
    assert np.isclose(rewards.sum(axis=0).std(ddof=0), sigma)
    assert np.allclose(rewards.mean(axis=1), 1.0 / 11)
    assert np.allclose(
        rewards.std(axis=1, ddof=0),
        np.sqrt(sigma**2 + sigma**2 / 11**2),
    )


@pytest.mark.parametrize("sigma_b", [1.0, 2.0, 4.0, 8.0, 16.0])
def test_fixed_sigma_a_sweep_for_plot_matrices(sigma_b):
    rewards = generate_rewards_all_to_all(11, 4.0, sigma_b)

    assert np.allclose(
        rewards.std(axis=1, ddof=0),
        np.sqrt(4**2 + sigma_b**2 / 11**2),
    )


@pytest.mark.parametrize("num_agents", [0, 4])
def test_all_to_all_requires_at_least_five_agents(num_agents):
    with pytest.raises(ValueError, match="num_agents >= 5"):
        generate_rewards_all_to_all(num_agents, 1.0, 1.0)


@pytest.mark.parametrize(("sigma_a", "sigma_b"), [(-1.0, 0.0), (0.0, -1.0)])
def test_all_to_all_rejects_negative_sigmas(sigma_a, sigma_b):
    with pytest.raises(ValueError, match="non-negative"):
        generate_rewards_all_to_all(11, sigma_a, sigma_b)


def test_environment_uses_task_by_agent_orientation():
    rewards = StochasticEnv._generate_category_rewards(
        seed=123,
        n_task_types=11,
        N=11,
        sigma_a=2.0,
        sigma_b=6.0,
        relatedness_width=5,
        reward_generation="circulant_all_to_all",
    )

    assert rewards.shape == (11, 11)
    assert np.allclose(rewards.std(axis=1, ddof=0), 2.0)
    assert np.isclose(rewards.sum(axis=1).std(ddof=0), 6.0)

    other_seed_rewards = StochasticEnv._generate_category_rewards(
        seed=999,
        n_task_types=11,
        N=11,
        sigma_a=2.0,
        sigma_b=6.0,
        relatedness_width=5,
        reward_generation="circulant_all_to_all",
    )
    assert np.array_equal(rewards, other_seed_rewards)


@pytest.mark.parametrize("relatedness_width", [1, 2, 3, 4])
def test_partial_interest_uses_deterministic_fourier_generation(
    relatedness_width,
):
    rewards = StochasticEnv._generate_category_rewards(
        seed=123,
        n_task_types=11,
        N=11,
        sigma_a=2.0,
        sigma_b=6.0,
        relatedness_width=relatedness_width,
        reward_generation="circulant_all_to_all",
    )
    other_seed_rewards = StochasticEnv._generate_category_rewards(
        seed=999,
        n_task_types=11,
        N=11,
        sigma_a=2.0,
        sigma_b=6.0,
        relatedness_width=relatedness_width,
        reward_generation="circulant_all_to_all",
    )

    assert np.array_equal(rewards, other_seed_rewards)

    g = 2 * relatedness_width + 1
    for task in range(11):
        caring = np.array(
            [
                min(abs(task - agent), 11 - abs(task - agent))
                <= relatedness_width
                for agent in range(11)
            ]
        )
        assert caring.sum() == g
        assert np.all(rewards[task, ~caring] == 0.0)
        assert np.isclose(rewards[task, caring].std(ddof=0), 2.0)

    full_rewards = generate_rewards_all_to_all(11, 2.0, 6.0).T
    assert np.allclose(rewards.sum(axis=1), full_rewards.sum(axis=1))
    assert np.isclose(rewards.sum(axis=1).std(ddof=0), 6.0)


def test_partial_circulant_public_orientation_and_mask():
    rewards = generate_rewards_circulant(
        num_agents=11,
        relatedness_width=2,
        sigma_a=2.0,
        sigma_b=6.0,
    )

    assert rewards.shape == (11, 11)
    for task in range(11):
        caring = np.array(
            [
                min(abs(task - agent), 11 - abs(task - agent)) <= 2
                for agent in range(11)
            ]
        )
        assert np.all(rewards[~caring, task] == 0.0)


def test_full_interest_circulant_is_exactly_backward_compatible():
    rewards = generate_rewards_circulant(11, 5, 2.0, 6.0)

    assert np.array_equal(rewards, generate_rewards_all_to_all(11, 2.0, 6.0))


@pytest.mark.parametrize(
    ("num_agents", "expected_cycles"),
    [(11, 2), (17, 3), (23, 4)],
)
def test_scaled_task_total_cycles(num_agents, expected_cycles):
    assert scaled_task_total_cycles(num_agents) == expected_cycles


@pytest.mark.parametrize("num_agents", [11, 17, 23])
def test_scaled_frequency_reward_invariants(num_agents):
    rewards = StochasticEnv._generate_category_rewards(
        seed=123,
        n_task_types=num_agents,
        N=num_agents,
        sigma_a=1.0,
        sigma_b=8.0,
        relatedness_width=1,
        reward_generation="circulant_scaled_frequency",
    )

    task_totals = rewards.sum(axis=1)
    assert np.isclose(task_totals.mean(), 1.0)
    assert np.isclose(task_totals.std(ddof=0), 8.0)

    for task in range(num_agents):
        caring = np.array(
            [
                min(abs(task - agent), num_agents - abs(task - agent)) <= 1
                for agent in range(num_agents)
            ]
        )
        assert caring.sum() == 3
        assert np.all(rewards[task, ~caring] == 0.0)
        assert np.isclose(rewards[task, caring].std(ddof=0), 1.0)


@pytest.mark.parametrize("num_agents", [17, 23])
def test_scaled_frequency_reduces_adjacent_task_total_correlation(num_agents):
    legacy = generate_rewards_circulant(
        num_agents,
        relatedness_width=1,
        sigma_a=1.0,
        sigma_b=8.0,
    )
    scaled = generate_rewards_circulant(
        num_agents,
        relatedness_width=1,
        sigma_a=1.0,
        sigma_b=8.0,
        task_total_cycles=scaled_task_total_cycles(num_agents),
    )

    legacy_totals = legacy.sum(axis=0)
    scaled_totals = scaled.sum(axis=0)
    legacy_correlation = np.corrcoef(legacy_totals, np.roll(legacy_totals, 1))[0, 1]
    scaled_correlation = np.corrcoef(scaled_totals, np.roll(scaled_totals, 1))[0, 1]
    assert scaled_correlation < legacy_correlation


@pytest.mark.parametrize(
    ("num_agents", "relatedness_width"),
    [(11, 1), (11, 5), (33, 1), (33, 5), (33, 16)],
)
def test_derived_task_variance_reward_invariants(
    num_agents,
    relatedness_width,
):
    sigma_a = 1.25
    cycles = scaled_task_total_cycles(num_agents)
    rewards = generate_rewards_circulant_derived_task_variance(
        num_agents,
        relatedness_width,
        sigma_a,
        task_total_cycles=cycles,
    )

    assert rewards.shape == (num_agents, num_agents)
    g = min(2 * relatedness_width + 1, num_agents)
    for task in range(num_agents):
        caring = np.array(
            [
                min(abs(task - agent), num_agents - abs(task - agent))
                <= relatedness_width
                for agent in range(num_agents)
            ]
        )
        assert caring.sum() == g
        assert np.all(rewards[~caring, task] == 0.0)
        assert np.isclose(rewards[caring, task].std(ddof=0), sigma_a)

    task_totals = rewards.sum(axis=0)
    assert np.isclose(task_totals.mean(), 1.0)

    expected_task_variance = g * sigma_a**2
    assert np.isclose(task_totals.var(ddof=0), expected_task_variance)


def test_derived_task_variance_ignores_sigma_b_and_seed():
    kwargs = dict(
        n_task_types=11,
        N=11,
        sigma_a=1.0,
        relatedness_width=5,
        reward_generation="circulant_derived_task_variance",
    )
    rewards = StochasticEnv._generate_category_rewards(
        seed=123,
        sigma_b=0.0,
        **kwargs,
    )
    same_seed_other_sigma_b = StochasticEnv._generate_category_rewards(
        seed=123,
        sigma_b=999.0,
        **kwargs,
    )
    other_seed = StochasticEnv._generate_category_rewards(
        seed=124,
        sigma_b=0.0,
        **kwargs,
    )

    assert np.array_equal(rewards, same_seed_other_sigma_b)
    assert np.array_equal(rewards, other_seed)


def test_derived_full_interest_preserves_balanced_agent_variance():
    num_agents = 11
    sigma_a = 2.0
    rewards = generate_rewards_circulant_derived_task_variance(
        num_agents,
        relatedness_width=5,
        sigma_a=sigma_a,
    )

    realized_sigma_b = rewards.sum(axis=0).std(ddof=0)
    expected_agent_std = np.sqrt(
        sigma_a**2 + realized_sigma_b**2 / num_agents**2
    )
    assert np.allclose(rewards.std(axis=1, ddof=0), expected_agent_std)
