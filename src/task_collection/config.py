"""Config loading: YAML → ExperimentConfig with validation and enum conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from task_collection.enums import (
    Activation,
    AlgorithmName,
    DespawnMode,
    EncoderType,
    Heuristic,
    LearningType,
    Schedule,
    StoppingCondition,
    WeightInit,
)
from task_collection.datatypes import (
    AlgorithmConfig,
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    ScheduleConfig,
    StochasticConfig,
    StoppingConfig,
    TrainConfig,
)


# ---------------------------------------------------------------------------
# String → Enum maps
# ---------------------------------------------------------------------------
_ENUM_MAPS: dict[str, dict[str, Any]] = {
    "encoder": {
        "everything_cnn_grid": EncoderType.EVERYTHING_CNN_GRID,
        "filtered_dec_cnn_grid": EncoderType.FILTERED_DEC_CNN_GRID,
    },
    "learning_type": {
        "decentralized": LearningType.DECENTRALIZED,
        "centralized": LearningType.CENTRALIZED,
    },
    "algorithm_name": {
        "value": AlgorithmName.VALUE,
    },
    "heuristic": {
        "nearest": Heuristic.NEAREST,
        "hungarian": Heuristic.HUNGARIAN,
        "stochastic_mpc": Heuristic.STOCHASTIC_MPC,
        "raw_stochastic_mpc": Heuristic.RAW_STOCHASTIC_MPC,
        "clairvoyant_rollout": Heuristic.CLAIRVOYANT_ROLLOUT,
    },
    "activation": {
        "relu": Activation.RELU,
        "leaky_relu": Activation.LEAKY_RELU,
        "none": Activation.NONE,
    },
    "weight_init": {
        "default": WeightInit.DEFAULT,
        "zero_bias": WeightInit.ZERO_BIAS,
    },
    "schedule": {
        "none": Schedule.NONE,
        "linear": Schedule.LINEAR,
        "step": Schedule.STEP,
    },
    "stopping_condition": {
        "none": StoppingCondition.NONE,
        "running_max_rps": StoppingCondition.RUNNING_MAX_RPS,
    },
    "despawn_mode": {
        "none": DespawnMode.NONE,
        "probability": DespawnMode.PROBABILITY,
    },
}


def _enum(value: str, field: str) -> Any:
    mapping = _ENUM_MAPS.get(field)
    if mapping is None:
        raise ValueError(f"No enum map for field: {field}")
    key = value.lower().strip()
    if key not in mapping:
        raise ValueError(f"Invalid {field}: '{value}'. Valid: {list(mapping.keys())}")
    return mapping[key]


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------
def _parse_schedule(d: dict[str, Any], name: str) -> ScheduleConfig:
    return ScheduleConfig(
        start=float(d["start"]),
        end=float(d.get("end", d["start"])),
        schedule=_enum(d.get("schedule", "none"), "schedule"),
        step_size=int(d.get("step_size", 0)),
        step_factor=float(d.get("step_factor", 1.0)),
        step_start=int(d.get("step_start", 0)),
    )


def _parse_env(d: dict[str, Any]) -> EnvConfig:
    n_task_types = int(d.get("n_task_types", 1))
    n_agents = int(d["n_agents"])
    if n_task_types != n_agents:
        raise ValueError(
            f"env.n_task_types ({n_task_types}) must equal env.n_agents ({n_agents}): "
            "the spec uses one shared id space (T=N), where agent i's home task is task i."
        )

    sd = d.get("stochastic")
    if sd is None:
        raise ValueError("env.stochastic block is required")
    stochastic_cfg = StochasticConfig(
        spawn_prob=float(sd["spawn_prob"]),
        despawn_mode=_enum(sd.get("despawn_mode", "probability"), "despawn_mode"),
        despawn_prob=float(sd.get("despawn_prob", 0.0)),
        sigma_a=float(sd.get("sigma_a", 0.0)),
        sigma_b=float(sd.get("sigma_b", 0.0)),
        reward_generation=str(
            sd.get("reward_generation", "independent")
        ).lower().strip(),
        spawn_on_agent_cells=bool(sd.get("spawn_on_agent_cells", False)),
        spawn_at_round_end=bool(sd.get("spawn_at_round_end", False)),
    )

    return EnvConfig(
        height=int(d["height"]),
        width=int(d["width"]),
        n_agents=n_agents,
        n_tasks=int(d.get("n_tasks", d.get("n_apples", 3))),
        gamma=float(d["gamma"]),
        n_task_types=n_task_types,
        relatedness_width=int(d.get("relatedness_width", 0)),
        proficiency_width=int(d.get("proficiency_width", 0)),
        max_tasks_per_type=int(d.get("max_tasks_per_type", 3)),
        stochastic=stochastic_cfg,
    )


def _parse_model(d: dict[str, Any]) -> ModelConfig:
    encoder_str = d.get("encoder", d.get("input_type"))
    if encoder_str is None:
        raise ValueError("model.encoder is required")
    encoder = _enum(encoder_str, "encoder")

    mlp_dims = tuple(int(x) for x in d.get("mlp_dims", [64, 64]))
    conv_specs = None
    if "conv_specs" in d and d["conv_specs"] is not None:
        conv_specs = tuple((int(s[0]), int(s[1])) for s in d["conv_specs"])

    return ModelConfig(
        encoder=encoder,
        mlp_dims=mlp_dims,
        conv_specs=conv_specs,
        activation=_enum(d.get("activation", "leaky_relu"), "activation"),
        weight_init=_enum(d.get("weight_init", "zero_bias"), "weight_init"),
    )


def _parse_train(d: dict[str, Any]) -> TrainConfig:
    unknown = set(d) - set(TrainConfig.__dataclass_fields__) - {"use_gpu_batched"}
    if unknown:
        raise ValueError(f"Unsupported training settings: {sorted(unknown)}")
    lr_cfg = _parse_schedule(d["lr"], "train.lr")

    eps_d = d.get("epsilon", {"start": 0.1, "end": 0.01, "schedule": "linear"})
    eps_cfg = _parse_schedule(eps_d, "train.epsilon")

    heuristic = _enum(d.get("heuristic", "nearest"), "heuristic")

    stop_d = d.get("stopping", {})
    stopping = StoppingConfig(
        condition=_enum(stop_d.get("condition", "none"), "stopping_condition"),
        patience_steps=int(stop_d.get("patience_steps", 10000)),
        improvement_threshold=float(stop_d.get("improvement_threshold", 0.01)),
        min_steps_before_stop=int(stop_d.get("min_steps_before_stop", 0)),
    )

    algorithm_d = d.get("algorithm", {})
    algorithm_name = _enum(algorithm_d.get("name", "value"), "algorithm_name")
    use_gpu = bool(d.get("use_gpu", d.get("use_gpu_batched", True)))

    discount_method = str(d.get("discount_method", "team_steps"))
    if discount_method not in ("team_steps", "world_steps", "round_steps"):
        raise ValueError(f"train.discount_method must be 'team_steps', 'world_steps', or 'round_steps', got {discount_method!r}")

    behavior_policy = str(d.get("behavior_policy", "value_greedy")).lower().strip()
    if behavior_policy not in ("value_greedy", "heuristic"):
        raise ValueError(
            f"train.behavior_policy must be 'value_greedy' or 'heuristic', got {behavior_policy!r}"
        )

    return TrainConfig(
        total_steps=int(d["total_steps"]),
        seed=int(d.get("seed", 42)),
        lr=lr_cfg,
        epsilon=eps_cfg,
        algorithm=AlgorithmConfig(name=algorithm_name),
        learning_type=_enum(d.get("learning_type", "decentralized"), "learning_type"),
        use_gpu=use_gpu,
        td_lambda=float(d.get("td_lambda", 0.0)),
        heuristic=heuristic,
        stopping=stopping,
        train_only_teammates=bool(d.get("train_only_teammates", False)),
        discount_method=discount_method,
        behavior_policy=behavior_policy,
    )


def _parse_eval(d: dict[str, Any]) -> EvalConfig:
    mc_validation_path = d.get("mc_validation_path")
    eval_seed = d.get("eval_seed")
    final_eval_freq = int(d.get("final_eval_freq", 0))
    final_eval_window = int(d.get("final_eval_window", 0))
    if final_eval_freq < 0 or final_eval_window < 0:
        raise ValueError("eval.final_eval_freq and eval.final_eval_window must be nonnegative.")
    if (final_eval_freq == 0) != (final_eval_window == 0):
        raise ValueError(
            "eval.final_eval_freq and eval.final_eval_window must either both be zero "
            "or both be positive."
        )
    return EvalConfig(
        eval_steps=int(d.get("eval_steps", 1000)),
        n_test_states=int(d.get("n_test_states", 50)),
        checkpoint_freq=int(d.get("checkpoint_freq", 0)),
        final_eval_freq=final_eval_freq,
        final_eval_window=final_eval_window,
        eval_seed=int(eval_seed) if eval_seed is not None else None,
        mc_validation_path=str(mc_validation_path) if mc_validation_path else None,
        rollout_metrics=bool(d.get("rollout_metrics", True)),
    )


def _parse_logging(d: dict[str, Any]) -> LoggingConfig:
    return LoggingConfig(
        output_dir=str(d.get("output_dir", "runs/")),
        main_csv_freq=int(d.get("main_csv_freq", 10000)),
        detail_csv_freq=int(d.get("detail_csv_freq", 50000)),
        timing_csv_freq=int(d.get("timing_csv_freq", 0)),
        env_trace=bool(d.get("env_trace", False)),
    )


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------
def _apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: '{override}'")
        key, value_str = override.split("=", 1)
        parts = key.strip().split(".")
        d = raw
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = _parse_override_value(value_str.strip())
    return raw


def _parse_override_value(s: str) -> Any:
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts, depth, current = [], 0, []
        for ch in inner:
            if ch == "[":
                depth += 1; current.append(ch)
            elif ch == "]":
                depth -= 1; current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip()); current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return [_parse_override_value(p) for p in parts]

    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = _apply_overrides(raw, overrides)

    for section in ("env", "model", "train"):
        if section not in raw:
            raise ValueError(f"Missing required config section: '{section}'")

    unknown = set(raw) - {"env", "model", "train", "eval", "logging"}
    if unknown:
        raise ValueError(f"Unsupported config sections: {sorted(unknown)}")

    env_cfg = _parse_env(raw["env"])
    model_cfg = _parse_model(raw["model"])
    train_cfg = _parse_train(raw["train"])
    return ExperimentConfig(
        env=env_cfg,
        model=model_cfg,
        train=train_cfg,
        eval=_parse_eval(raw.get("eval", {})),
        logging=_parse_logging(raw.get("logging", {})),
    )
