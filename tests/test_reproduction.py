"""Experiment example, recipe, baseline, and scoring regression tests."""

import csv
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import yaml

from task_collection.config import load_config
from task_collection.env import create_env
from task_collection.enums import AlgorithmName, EncoderType, LearningType
from task_collection.reproduce import (
    CONDITIONS, SCALING_POINTS, evaluate_heuristics, main, make_config, score_metrics,
)
from task_collection.seed import set_all_seeds
from task_collection.train import train


EXAMPLES = Path(str(files("task_collection").joinpath("configs/examples")))


def write_config(tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.stem)
def test_examples_are_self_contained_and_use_paper_core(path):
    cfg = load_config(path)
    assert cfg.env.n_agents == cfg.env.n_task_types
    assert cfg.env.stochastic.reward_generation == "circulant_derived_task_variance"
    assert cfg.env.stochastic.sigma_a == 16
    assert cfg.env.stochastic.spawn_at_round_end
    assert cfg.model.encoder == EncoderType.EVERYTHING_CNN_GRID
    assert cfg.train.algorithm.name == AlgorithmName.VALUE
    assert cfg.train.discount_method == "round_steps"
    assert cfg.model.conv_specs == tuple((cfg.model.mlp_dims[0], 3) for _ in range(2))
    assert not Path(cfg.logging.output_dir).is_absolute()


@pytest.mark.parametrize("agents", SCALING_POINTS)
@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("method", ("centralized", "decentralized"))
def test_main_recipes_set_all_coupled_fields(tmp_path, agents, condition, method):
    cfg = load_config(write_config(tmp_path, make_config(
        "main", agents=agents, condition=condition, method=method,
        learning_rate=1e-6, seed=1235,
    )))
    side, cap, budget = SCALING_POINTS[agents]
    assert (cfg.env.height, cfg.env.width) == (side, side)
    assert cfg.env.n_task_types == cfg.env.n_agents == agents
    assert cfg.env.n_tasks == cfg.env.max_tasks_per_type == cap
    assert cfg.train.total_steps == budget
    assert cfg.eval.eval_steps == 45 * agents
    assert cfg.train.seed == 1235
    assert cfg.train.lr.start == cfg.train.lr.end == 1e-6
    assert cfg.train.learning_type == LearningType[method.upper()]
    assert cfg.env.relatedness_width == (agents // 2 if condition == "capability-bounded" else 5)
    assert cfg.env.proficiency_width == (agents // 2 if condition == "interest-bounded" else 5)


def test_n11_environment_is_shared_across_all_main_conditions_and_width_sweep():
    configs = [make_config("main", agents=11, condition=c) for c in CONDITIONS]
    configs.append(make_config("width-sweep", agents=11, width=8))
    assert all(cfg["env"] == configs[0]["env"] for cfg in configs)
    assert all(cfg["model"] == configs[0]["model"] for cfg in configs)
    assert all(cfg["eval"] == configs[0]["eval"] for cfg in configs)
    assert all(cfg["train"] == configs[0]["train"] for cfg in configs)


@pytest.mark.parametrize("agents", SCALING_POINTS)
def test_derived_reward_properties_match_recipe(tmp_path, agents):
    cfg = load_config(write_config(tmp_path, make_config("main", agents=agents)))
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    # The environment stores rewards as (task type, receiving agent).
    for task in range(agents):
        offsets = np.abs(np.arange(agents) - task)
        interested = np.minimum(offsets, agents - offsets) <= 5
        assert np.all(env.category_rewards[task, ~interested] == 0)
        assert np.std(env.category_rewards[task, interested]) == pytest.approx(16, rel=1e-6)
    assert env.category_rewards.sum(axis=1).std() == pytest.approx(16 * np.sqrt(11), rel=1e-6)


def test_width64_recipe_and_independent_config_objects():
    cfg = make_config("centralized-width64", agents=41, method="centralized", gpu=True)
    assert cfg["model"]["conv_specs"] == [[64, 3], [64, 3]]
    assert cfg["model"]["mlp_dims"] == [64]
    assert cfg["train"]["use_gpu"]
    cfg["env"]["stochastic"]["sigma_a"] = 100
    assert make_config("main")["env"]["stochastic"]["sigma_a"] == 16


@pytest.mark.parametrize("family, kwargs", [
    ("unknown", {}),
    ("main", {"agents": 20}),
    ("main", {"learning_rate": float("nan")}),
    ("main", {"learning_rate": -1}),
    ("main", {"seed": -1}),
    ("width-sweep", {"agents": 31}),
    ("centralized-width64", {"method": "decentralized"}),
    ("centralized-width64", {"method": "centralized", "width": 8}),
])
def test_invalid_recipes_are_rejected(family, kwargs):
    with pytest.raises(ValueError):
        make_config(family, **kwargs)


def test_heuristic_reproducibility_and_same_units_as_training():
    path = EXAMPLES / "smoke_cpu.yaml"
    rows, summary = evaluate_heuristics(path, policies=("hungarian", "nearest"), rounds=2, seeds=2)
    repeated_rows, repeated_summary = evaluate_heuristics(path, policies=("hungarian", "nearest"), rounds=2, seeds=2)
    assert rows == repeated_rows
    assert summary == repeated_summary
    assert len(rows) == 4
    assert all(row["eval_steps"] == 22 for row in rows)
    assert {row["policy"] for row in summary} == {"hungarian", "nearest"}
    assert all(row["n_seeds"] == 2 for row in summary)


def test_parallel_heuristics_match_sequential_rollouts():
    path = EXAMPLES / "smoke_cpu.yaml"
    sequential = evaluate_heuristics(path, rounds=1, seeds=2, workers=1)
    parallel = evaluate_heuristics(path, rounds=1, seeds=2, workers=2)
    assert parallel == sequential


def write_metrics(path, steps, values):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "greedy_team_rps"])
        writer.writerows(zip(steps, values))


@pytest.mark.parametrize("cadence, expected_count", [(50_000, 20), (10_000, 100)])
def test_score_uses_final_million_not_last_k_rows(tmp_path, cadence, expected_count):
    path = tmp_path / "metrics.csv"
    steps = list(range(cadence, 10_500_001, cadence))
    values = [3. if 9_000_000 < step <= 10_000_000 else 999. for step in steps]
    write_metrics(path, steps, values)
    result = score_metrics([path], 10_000_000)
    assert result["performance"] == 3
    assert result["evaluations"] == expected_count
    assert result["window_start_exclusive"] == 9_000_000
    assert result["window_end"] == 10_000_000


def test_resumed_metrics_replace_overlapping_steps(tmp_path):
    initial, resumed = tmp_path / "initial.csv", tmp_path / "resumed.csv"
    write_metrics(initial, [1_000_000, 1_500_000, 2_000_000], [100, 1, 1])
    write_metrics(resumed, [2_000_000, 2_500_000], [5, 7])
    result = score_metrics([initial, resumed], 2_500_000)
    assert result["performance"] == 6
    assert result["evaluations"] == 2


def test_incomplete_score_requires_explicit_partial_flag(tmp_path):
    path = tmp_path / "metrics.csv"
    write_metrics(path, [500_000, 1_000_000, 1_500_000], [100, 2, 4])
    with pytest.raises(ValueError, match="only reached"):
        score_metrics([path], 2_000_000)
    result = score_metrics([path], 2_000_000, allow_partial=True)
    assert result["performance"] == 3
    assert result["window_end"] == 1_500_000


def test_config_cli_refuses_to_overwrite_existing_yaml(tmp_path, monkeypatch):
    output = tmp_path / "saved.yaml"
    output.write_text("existing user config\n")
    monkeypatch.setattr("sys.argv", ["task-collection-reproduce", "config", "--family", "main", "--output", str(output)])
    with pytest.raises(SystemExit):
        main()
    assert output.read_text() == "existing user config\n"


def test_config_cli_creates_loadable_yaml(tmp_path, monkeypatch, capsys):
    output = tmp_path / "generated.yaml"
    monkeypatch.setattr("sys.argv", [
        "task-collection-reproduce", "config", "--family", "main", "--agents", "31",
        "--condition", "capability-bounded", "--output", str(output),
    ])
    main()
    assert load_config(output).env.relatedness_width == 15
    assert str(output) in capsys.readouterr().out


def test_example_short_training_produces_metrics_and_checkpoint(tmp_path):
    cfg = load_config(EXAMPLES / "smoke_cpu.yaml", [
        "train.total_steps=5", "eval.eval_steps=5", "eval.n_test_states=2",
        "logging.main_csv_freq=5", "logging.detail_csv_freq=5",
        f"logging.output_dir={tmp_path / 'runs'}",
    ])
    train(cfg)
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "metadata.yaml").exists()
    assert (runs[0] / "checkpoints" / "final.pt").exists()
    result = score_metrics([runs[0] / "metrics.csv"], 5)
    assert result["evaluations"] == 1
