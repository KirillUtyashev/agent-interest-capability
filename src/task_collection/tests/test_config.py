"""Tests for config loading, parsing, enum conversions, and overrides."""

import os
import tempfile
import pytest

from task_collection.config import load_config, _apply_overrides, _parse_override_value
from task_collection.enums import AlgorithmName, EncoderType, Heuristic, LearningType

def _write_yaml(content: str) -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path

VALID_YAML = """
env:
  height: 5
  width: 5
  n_agents: 2
  n_task_types: 2
  gamma: 0.99
  stochastic:
    spawn_prob: 0.1
    despawn_mode: none
model:
  encoder: everything_cnn_grid
  mlp_dims: [64]
train:
  total_steps: 100
  lr:
    start: 0.001
"""

class TestConfigParsing:
    def test_valid_parse(self):
        path = _write_yaml(VALID_YAML)
        cfg = load_config(path)

        assert cfg.env.height == 5
        assert cfg.env.gamma == 0.99
        assert cfg.model.encoder == EncoderType.EVERYTHING_CNN_GRID
        assert cfg.train.total_steps == 100
        assert cfg.train.algorithm.name == AlgorithmName.VALUE

        os.unlink(path)

    def test_missing_section_raises(self):
        bad_yaml = VALID_YAML.replace("env:", "environment:")
        path = _write_yaml(bad_yaml)
        with pytest.raises(ValueError, match="Missing required config section: 'env'"):
            load_config(path)
        os.unlink(path)

    def test_invalid_enum_raises(self):
        bad_yaml = VALID_YAML.replace("everything_cnn_grid", "magic_encoder")
        path = _write_yaml(bad_yaml)
        with pytest.raises(ValueError, match="Invalid encoder: 'magic_encoder'"):
            load_config(path)
        os.unlink(path)

    def test_relatedness_width_proficiency_width_parse(self):
        yaml_str = VALID_YAML.replace(
            "n_agents: 2\n  n_task_types: 2",
            "n_agents: 4\n  n_task_types: 4\n  relatedness_width: 1\n  proficiency_width: 2",
        )
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.relatedness_width == 1
        assert cfg.env.proficiency_width == 2
        assert cfg.env.n_task_types == 4
        os.unlink(path)

    def test_t_equals_n_raises(self):
        # n_agents=2 but n_task_types=3 violates the shared id space T=N.
        yaml_str = VALID_YAML.replace("n_task_types: 2", "n_task_types: 3")
        path = _write_yaml(yaml_str)
        with pytest.raises(ValueError, match="shared id space"):
            load_config(path)
        os.unlink(path)

    def test_sigma_a_sigma_b_parse(self):
        yaml_str = VALID_YAML.replace(
            "spawn_prob: 0.1", "spawn_prob: 0.1\n    sigma_a: 0.3\n    sigma_b: 0.5"
        )
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.stochastic.sigma_a == pytest.approx(0.3)
        assert cfg.env.stochastic.sigma_b == pytest.approx(0.5)
        os.unlink(path)

    def test_reward_generation_parse(self):
        yaml_str = VALID_YAML.replace(
            "spawn_prob: 0.1",
            "spawn_prob: 0.1\n    reward_generation: circulant_all_to_all",
        )
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.stochastic.reward_generation == "circulant_all_to_all"
        os.unlink(path)

    def test_final_eval_schedule_parse(self):
        yaml_str = VALID_YAML + """
eval:
  final_eval_freq: 10000
  final_eval_window: 1000000
"""
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.eval.final_eval_freq == 10000
        assert cfg.eval.final_eval_window == 1000000
        os.unlink(path)

    @pytest.mark.parametrize(
        "eval_yaml",
        [
            "final_eval_freq: 10000\n  final_eval_window: 0",
            "final_eval_freq: 0\n  final_eval_window: 1000000",
            "final_eval_freq: -1\n  final_eval_window: 1000000",
        ],
    )
    def test_invalid_final_eval_schedule_raises(self, eval_yaml):
        path = _write_yaml(VALID_YAML + f"\neval:\n  {eval_yaml}\n")
        with pytest.raises(ValueError, match="final_eval_freq"):
            load_config(path)
        os.unlink(path)


class TestBackwardCompatibility:
    def test_n_apples_maps_to_n_tasks(self):
        yaml_str = VALID_YAML.replace("env:\n", "env:\n  n_apples: 7\n")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.env.n_tasks == 7
        os.unlink(path)

    def test_input_type_maps_to_encoder(self):
        yaml_str = VALID_YAML.replace("encoder:", "input_type:")
        path = _write_yaml(yaml_str)
        cfg = load_config(path)
        assert cfg.model.encoder == EncoderType.EVERYTHING_CNN_GRID
        os.unlink(path)

class TestOverrides:
    def test_override_dot_notation(self):
        raw = {"env": {"height": 5, "width": 5}}
        result = _apply_overrides(raw, ["env.height=10"])
        assert result["env"]["height"] == 10
        assert result["env"]["width"] == 5

    def test_override_type_coercion(self):
        assert _parse_override_value("42") == 42
        assert _parse_override_value("3.14") == 3.14
        assert _parse_override_value("true") is True
        assert _parse_override_value("false") is False
        assert _parse_override_value("hello") == "hello"

    def test_override_list_parsing(self):
        assert _parse_override_value("[64, 128]") == [64, 128]
        assert _parse_override_value("[]") == []
        assert _parse_override_value("[[4,3], [2,2]]") == [[4, 3], [2, 2]]

    def test_config_load_with_overrides(self):
        path = _write_yaml(VALID_YAML)
        cfg = load_config(
            path,
            overrides=[
                "train.total_steps=999",
                "train.seed=1235",
                "train.td_lambda=0.3",
            ],
        )
        assert cfg.train.total_steps == 999
        assert cfg.train.seed == 1235
        assert cfg.train.td_lambda == pytest.approx(0.3)
        os.unlink(path)






class TestSupportedSettings:
    def test_unknown_training_setting_rejected(self):
        path = _write_yaml(VALID_YAML.replace("total_steps: 100", "total_steps: 100\n  unsupported_setting: true"))
        try:
            with pytest.raises(ValueError, match="Unsupported training settings"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_unknown_section_rejected(self):
        path = _write_yaml(VALID_YAML + "\nunsupported_section: {}\n")
        try:
            with pytest.raises(ValueError, match="Unsupported config sections"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_unknown_algorithm_rejected(self):
        path = _write_yaml(VALID_YAML.replace("total_steps: 100", "total_steps: 100\n  algorithm:\n    name: unsupported_algorithm"))
        try:
            with pytest.raises(ValueError, match="Invalid algorithm.name"):
                load_config(path)
        finally:
            os.unlink(path)
