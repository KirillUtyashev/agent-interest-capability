"""Tests for regular and final-window evaluation scheduling."""

import pytest

from task_collection.datatypes import StoppingConfig
from task_collection.enums import StoppingCondition
from task_collection.train import EarlyStopper, should_run_evaluation


def _evaluation_steps(total_steps, regular_freq, final_freq=0, final_window=0):
    return [
        step
        for step in range(1, total_steps + 1)
        if should_run_evaluation(
            step,
            total_steps,
            regular_freq,
            final_freq,
            final_window,
        )
    ]


def test_disabled_final_schedule_preserves_regular_frequency():
    assert _evaluation_steps(100_000, 10_000) == list(range(10_000, 100_001, 10_000))


def test_final_million_contains_exactly_last_100_evaluations():
    steps = _evaluation_steps(8_000_000, 50_000, 10_000, 1_000_000)

    assert steps[:3] == [50_000, 100_000, 150_000]
    assert 7_000_000 in steps
    assert 7_010_000 in steps
    assert 7_005_000 not in steps
    assert steps[-100:] == list(range(7_010_000, 8_000_001, 10_000))
    assert len(steps) == 240


def test_final_window_covers_short_runs():
    assert _evaluation_steps(30_000, 50_000, 10_000, 1_000_000) == [
        10_000,
        20_000,
        30_000,
    ]


@pytest.mark.parametrize(
    "args",
    [
        (1, 100, 0, 10, 0),
        (1, 100, 10, -1, 100),
        (1, 100, 10, 10, -1),
        (1, 100, 10, 10, 0),
    ],
)
def test_invalid_schedule_raises(args):
    with pytest.raises(ValueError):
        should_run_evaluation(*args)


def test_early_stopping_counts_actual_variable_intervals():
    stopper = EarlyStopper(
        StoppingConfig(
            condition=StoppingCondition.RUNNING_MAX_RPS,
            patience_steps=70_000,
        )
    )

    assert not stopper.check(49_999, {"greedy_team_rps": 1.0})
    assert not stopper.check(99_999, {"greedy_team_rps": 1.0})
    assert not stopper.check(109_999, {"greedy_team_rps": 1.0})
    assert stopper.check(119_999, {"greedy_team_rps": 1.0})
