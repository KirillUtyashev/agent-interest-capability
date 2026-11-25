"""Portable configuration, baseline, and scoring tools for paper experiments.

These helpers call the existing environment/policy code; they do not modify
the learning algorithm and do not depend on cluster scripts or notebooks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import statistics
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

import yaml

from task_collection.config import load_config
from task_collection.env import create_env
from task_collection.enums import Heuristic
from task_collection.eval import evaluate_policy_metrics
from task_collection.policy import heuristic_action
from task_collection.seed import set_all_seeds


# N: (square-grid side, initial tasks and cap per task type, reporting budget).
SCALING_POINTS = {
    11: (9, 5, 2_500_000),
    21: (12, 9, 5_000_000),
    31: (15, 14, 7_500_000),
    41: (17, 18, 10_000_000),
}
MAIN_LR_GRID = (5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4)
WIDTHS = (1, 2, 4, 8, 16, 32, 64)
CONDITIONS = ("both-bounded", "interest-bounded", "capability-bounded")


def make_config(
    family: str,
    *,
    agents: int = 21,
    condition: str = "both-bounded",
    method: str = "decentralized",
    width: int | None = None,
    learning_rate: float = 5e-6,
    seed: int = 1234,
    gpu: bool = False,
) -> dict:
    """Construct a complete YAML-compatible config for a paper experiment."""
    if family not in ("main", "width-sweep", "centralized-width64"):
        raise ValueError(f"Unknown experiment family: {family}")
    if agents not in SCALING_POINTS:
        raise ValueError(f"agents must be one of {tuple(SCALING_POINTS)}")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if method not in ("centralized", "decentralized"):
        raise ValueError("method must be centralized or decentralized")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if family == "width-sweep" and (agents != 11 or condition != "both-bounded"):
        raise ValueError("width-sweep requires agents=11 and condition=both-bounded")
    if family == "centralized-width64":
        if method != "centralized" or condition != "both-bounded":
            raise ValueError("centralized-width64 requires centralized, both-bounded")
        if width not in (None, 64):
            raise ValueError("centralized-width64 requires width=64")
    width = width if width is not None else (64 if family == "centralized-width64" else 8)
    if width not in WIDTHS:
        raise ValueError(f"width must be one of {WIDTHS}")

    template = files("task_collection").joinpath("configs/examples/main_both_bounded.yaml")
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    side, cap, budget = SCALING_POINTS[agents]
    full_radius = agents // 2
    interest_radius = full_radius if condition == "capability-bounded" else 5
    capability_radius = full_radius if condition == "interest-bounded" else 5
    config["env"].update({
        "height": side, "width": side, "n_agents": agents,
        "n_task_types": agents, "n_tasks": cap, "max_tasks_per_type": cap,
        "relatedness_width": interest_radius,
        "proficiency_width": capability_radius,
    })
    config["model"].update({"conv_specs": [[width, 3], [width, 3]], "mlp_dims": [width]})
    config["train"].update({
        "learning_type": method, "use_gpu": gpu,
        "total_steps": budget, "seed": seed,
        "lr": {"start": learning_rate, "end": learning_rate, "schedule": "none"},
    })
    config["eval"]["eval_steps"] = 45 * agents
    config["logging"]["output_dir"] = (
        f"outputs/{family}/{condition}/{method}/agents_{agents}/width_{width}"
        f"/lr_{learning_rate:g}/seed_{seed}"
    )
    return config


def _heuristic_rollout(job: tuple) -> dict:
    env_cfg, construction_seed, policy_name, seed, rounds = job
    # Reward construction uses the training seed, not the rollout seed.
    set_all_seeds(construction_seed)
    env = create_env(env_cfg)
    env.set_eval_mode(True, seed=seed)
    try:
        heuristic = Heuristic[policy_name.upper()]
        metrics = evaluate_policy_metrics(
            env.init_state(), lambda state: heuristic_action(state, env, heuristic),
            env, rounds * env_cfg.n_agents,
        )
    finally:
        env.set_eval_mode(False)
    return {
        "policy": policy_name, "seed": seed, "rounds": rounds,
        "eval_steps": rounds * env_cfg.n_agents,
        "team_rps": metrics["team_rps"], "actor_rps": metrics["rps"],
    }


def evaluate_heuristics(
    config_path: str | Path,
    *,
    policies: tuple[str, ...] = ("hungarian",),
    rounds: int = 100,
    seeds: int = 10,
    seed_start: int = 0,
    workers: int = 1,
    overrides: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Evaluate planner/greedy baselines without constructing any networks."""
    if rounds <= 0 or seeds < 2 or seed_start < 0 or workers <= 0:
        raise ValueError("rounds/workers must be positive; seeds >= 2; seed_start >= 0")
    if not policies or any(p not in ("hungarian", "nearest") for p in policies):
        raise ValueError("policies must contain hungarian and/or nearest")
    cfg = load_config(config_path, overrides)
    jobs = [
        (cfg.env, cfg.train.seed, policy, seed, rounds)
        for policy in dict.fromkeys(policies)
        for seed in range(seed_start, seed_start + seeds)
    ]
    if workers == 1:
        rows = [_heuristic_rollout(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            rows = list(executor.map(_heuristic_rollout, jobs))
    summaries = []
    for policy in dict.fromkeys(policies):
        values = [row["team_rps"] for row in rows if row["policy"] == policy]
        summaries.append({
            "policy": policy, "n_seeds": len(values), "rounds": rounds,
            "team_rps_mean": statistics.mean(values),
            "team_rps_std": statistics.stdev(values),
        })
    return rows, summaries


def score_metrics(
    metrics_paths: list[str | Path], cutoff: int, *,
    window_steps: int = 1_000_000, metric: str = "greedy_team_rps",
    allow_partial: bool = False,
) -> dict:
    """Merge one run's chronological segments and score (cutoff-window, cutoff]."""
    if cutoff <= 0 or window_steps <= 0:
        raise ValueError("cutoff and window_steps must be positive")
    rows_by_step = {}
    for path in metrics_paths:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"step", metric}.issubset(reader.fieldnames or ()):
                raise ValueError(f"Missing step or {metric} column in {path}")
            for row in reader:
                try:
                    step_value, value = float(row["step"]), float(row[metric])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(step_value) and step_value.is_integer() and math.isfinite(value):
                    rows_by_step[int(step_value)] = value
    if not rows_by_step:
        raise ValueError("No valid evaluations found")
    latest = max(rows_by_step)
    if latest < cutoff and not allow_partial:
        raise ValueError(f"Run has only reached {latest}; requested cutoff is {cutoff}")
    available = [step for step in rows_by_step if step <= cutoff]
    if not available:
        raise ValueError("No evaluations at or before cutoff")
    end = max(available) if allow_partial else cutoff
    values = [value for step, value in rows_by_step.items() if end - window_steps < step <= end]
    if not values:
        raise ValueError("No evaluations in the requested final window")
    return {
        "metric": metric, "performance": statistics.mean(values),
        "evaluations": len(values), "window_start_exclusive": end - window_steps,
        "window_end": end, "latest_training_step": latest,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    config_cmd = commands.add_parser("config", help="Generate a complete experiment YAML")
    config_cmd.add_argument("--family", choices=("main", "width-sweep", "centralized-width64"), required=True)
    config_cmd.add_argument("--agents", type=int, choices=tuple(SCALING_POINTS), default=21)
    config_cmd.add_argument("--condition", choices=CONDITIONS, default="both-bounded")
    config_cmd.add_argument("--method", choices=("centralized", "decentralized"), default="decentralized")
    config_cmd.add_argument("--width", type=int, choices=WIDTHS)
    config_cmd.add_argument("--lr", type=float, default=5e-6)
    config_cmd.add_argument("--seed", type=int, default=1234)
    config_cmd.add_argument("--gpu", action="store_true")
    config_cmd.add_argument("--output", type=Path, required=True)
    config_cmd.add_argument("--force", action="store_true")

    heuristic_cmd = commands.add_parser("heuristic", help="Evaluate checkpoint-free planner/greedy baselines")
    heuristic_cmd.add_argument("--config", type=Path, required=True)
    heuristic_cmd.add_argument("--policies", nargs="+", choices=("hungarian", "nearest"), default=["hungarian"])
    heuristic_cmd.add_argument("--rounds", type=int, default=100)
    heuristic_cmd.add_argument("--seeds", type=int, default=10)
    heuristic_cmd.add_argument("--seed-start", type=int, default=0)
    heuristic_cmd.add_argument("--workers", type=int, default=1)
    heuristic_cmd.add_argument("--override", nargs="*", default=[])
    heuristic_cmd.add_argument("--output-dir", type=Path, required=True)
    heuristic_cmd.add_argument("--force", action="store_true")

    score_cmd = commands.add_parser("score", help="Score one run across initial/resumed CSV segments")
    score_cmd.add_argument("--metrics", nargs="+", type=Path, required=True)
    score_cmd.add_argument("--cutoff", type=int, required=True)
    score_cmd.add_argument("--window-steps", type=int, default=1_000_000)
    score_cmd.add_argument("--metric", default="greedy_team_rps")
    score_cmd.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.command == "config":
        if args.output.exists() and not args.force:
            parser.error(f"Refusing to overwrite {args.output}; use --force to replace it")
        config = make_config(
            args.family, agents=args.agents, condition=args.condition,
            method=args.method, width=args.width, learning_rate=args.lr,
            seed=args.seed, gpu=args.gpu,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        load_config(args.output)
        print(args.output)
    elif args.command == "heuristic":
        paths = [args.output_dir / name for name in ("per_seed.csv", "summary.csv")]
        if not args.force and any(path.exists() for path in paths):
            parser.error("Refusing to overwrite baseline results; choose a new output directory or --force")
        rows, summary = evaluate_heuristics(
            args.config, policies=tuple(args.policies), rounds=args.rounds,
            seeds=args.seeds, seed_start=args.seed_start, workers=args.workers,
            overrides=args.override,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(paths[0], rows)
        _write_csv(paths[1], summary)
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(score_metrics(
            args.metrics, args.cutoff, window_steps=args.window_steps,
            metric=args.metric, allow_partial=args.allow_partial,
        ), indent=2))


if __name__ == "__main__":
    main()
