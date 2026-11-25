"""Tests that value-learning CLI resume flags reach train()."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import task_collection.train as train_module
from task_collection.train import main


def _run_main(argv: list[str]) -> MagicMock:
    """Patch train() and load_config(), run main() with the given argv, return the train mock."""
    fake_cfg = MagicMock()
    with (
        patch.object(train_module, "load_config", return_value=fake_cfg),
        patch.object(train_module, "train") as mock_train,
        patch.object(sys, "argv", ["task_collection.train"] + argv),
    ):
        main()
    return mock_train


class TestCliResumeForwarding:
    """Every resume flag that argparse accepts must reach train()."""

    def test_resume_checkpoint_forwarded(self, tmp_path):
        ckpt = str(tmp_path / "final.pt")
        mock_train = _run_main(["--config", "cfg.yaml", "--resume", ckpt])
        mock_train.assert_called_once()
        _, kwargs = mock_train.call_args
        assert kwargs["resume_checkpoint"] == ckpt

    def test_resume_training_checkpoint_forwarded(self, tmp_path):
        ckpt = str(tmp_path / "step_10.pt")
        mock_train = _run_main(
            ["--config", "cfg.yaml", "--resume-training", ckpt]
        )
        mock_train.assert_called_once()
        _, kwargs = mock_train.call_args
        assert kwargs["resume_training_checkpoint"] == ckpt




    def test_omitted_resume_flags_are_none(self):
        """When no resume flags are given, both must be None, not absent."""
        mock_train = _run_main(["--config", "cfg.yaml"])
        _, kwargs = mock_train.call_args
        assert kwargs.get("resume_checkpoint") is None
        assert kwargs.get("resume_training_checkpoint") is None
