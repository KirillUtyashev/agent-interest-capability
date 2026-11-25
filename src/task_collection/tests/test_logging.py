"""Tests for logging utilities."""

import csv
import tempfile
from pathlib import Path

from task_collection.logging_ import (
    CSVLogger,
    build_main_csv_fieldnames,
)

class TestCSVLogger:
    def test_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.csv"
            fields = ["step", "loss"]
            logger = CSVLogger(path, fields)
            logger.close()

            with open(path) as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames == ["step", "loss"]
                rows = list(reader)
                assert len(rows) == 0

    def test_log_row(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.csv"
            fields = ["step", "loss"]
            logger = CSVLogger(path, fields)
            logger.log({"step": 1, "loss": 0.5})
            logger.log({"step": 2, "loss": 0.3})
            logger.close()

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 2
                assert rows[0]["step"] == "1"
                assert rows[1]["loss"] == "0.3"

class TestFieldnames:
    def test_main_csv_fields(self):
        fields = build_main_csv_fieldnames("nearest")

        assert "step" in fields
        assert "greedy_rps" in fields
        assert "greedy_team_rps" in fields
        assert "nearest_rps" in fields
        assert "nearest_team_rps" in fields
        assert "td_loss_avg" in fields
        # No correct_pps/wrong_pps in new framework
        assert "greedy_correct_pps" not in fields
        assert "greedy_wrong_pps" not in fields

    def test_main_csv_without_rollouts(self):
        assert build_main_csv_fieldnames("nearest", rollout_metrics=False) == [
            "step", "wall_time", "td_loss_avg",
        ]
