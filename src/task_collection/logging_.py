"""Logging: CSVLogger, run directory setup, metadata writing."""

from __future__ import annotations

import csv
import multiprocessing
import os
import socket
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from task_collection.datatypes import ExperimentConfig


class CSVLogger:
    """Incremental CSV writer with immediate flush."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: dict[str, float | int | str]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _get_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    d = asdict(cfg)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if hasattr(obj, "name"):
            return obj.name.lower()
        return obj

    return _convert(d)


def setup_logging(cfg: ExperimentConfig) -> Path:
    """Create run directory and write initial metadata. Returns run_dir."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    run_dir = Path(cfg.logging.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    metadata: dict[str, Any] = {
        "config": _config_to_dict(cfg),
        "run": {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_wall_seconds": None,
            "hostname": socket.gethostname(),
            "git_hash": _get_git_hash(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", None),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", None),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", None),
        },
    }
    hw: dict[str, Any] = {
        "cpu_logical_cores": multiprocessing.cpu_count(),
        "cpu_affinity_cores": len(os.sched_getaffinity(0)),
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        hw["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
        hw["gpu_vram_mb"] = pynvml.nvmlDeviceGetMemoryInfo(handle).total // 1024**2
        try:
            import torch
            hw["gpu_sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
        except Exception:
            hw["gpu_sm_count"] = None
        pynvml.nvmlShutdown()
    except Exception as e:
        hw["gpu_name"] = None
        hw["gpu_sm_count"] = None
        hw["gpu_vram_mb"] = None
        hw["gpu_pynvml_error"] = str(e)
    metadata["run"].update(hw)

    with open(run_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    return run_dir


def finalize_logging(run_dir: Path, start_time: float) -> None:
    meta_path = run_dir / "metadata.yaml"
    with open(meta_path) as f:
        metadata = yaml.safe_load(f)
    metadata["run"]["end_time"] = datetime.now().isoformat()
    metadata["run"]["total_wall_seconds"] = round(time.time() - start_time, 2)
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def build_main_csv_fieldnames(
    heuristic_name: str,
    *,
    mc_validation: bool = False,
    rollout_metrics: bool = True,
) -> list[str]:
    fields = ["step", "wall_time"]
    if rollout_metrics:
        fields.extend([
            "greedy_rps", "greedy_team_rps",
            f"{heuristic_name}_rps", f"{heuristic_name}_team_rps",
        ])
    fields.append("td_loss_avg")
    if mc_validation:
        from task_collection.mc_value_validation import build_mc_validation_summary_fieldnames
        fields.extend(build_mc_validation_summary_fieldnames())
    return fields


def build_detail_csv_fieldnames(
    critic_networks: list[Any],
) -> list[str]:
    n_networks = len(critic_networks)
    fields = ["step", "wall_time", "ram_mb", "current_lr", "current_epsilon"]
    for agent_idx in range(n_networks):
        for name, _ in critic_networks[agent_idx].named_parameters():
            if "weight" in name:
                fields.append(f"critic_weight_norm_agent_{agent_idx}_{name}")
                fields.append(f"critic_grad_norm_agent_{agent_idx}_{name}")
    fields.extend(["td_loss_step", "value_pred_mean", "value_pred_std"])
    fields.extend(["vram_allocated_mb", "vram_peak_mb", "vram_total_mb"])
    return fields
