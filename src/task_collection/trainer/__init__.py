"""Trainer package: create_trainer factory."""

from __future__ import annotations

import torch

from task_collection.datatypes import ExperimentConfig
from task_collection.env.base import BaseEnv
from task_collection.model import create_networks
from task_collection.trainer.base import TrainerBase
from task_collection.trainer.timer import Timer


def create_trainer(
    cfg: ExperimentConfig,
    env: BaseEnv,
) -> TrainerBase:
    """Create the appropriate trainer based on config.

    use_gpu=True  → GpuTrainer (BatchedTrainer + vmap, works for any n_networks)
    use_gpu=False → CpuTrainer (sequential forwards, works for any n_networks)
    """
    gpu_sync = cfg.train.use_gpu and torch.cuda.is_available()
    timer = Timer(enabled=cfg.logging.timing_csv_freq > 0, gpu_sync=gpu_sync)
    from task_collection.batched_training import BatchedTrainer

    networks = create_networks(cfg.model, cfg.env, cfg.train)
    if cfg.train.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bt = BatchedTrainer(
            networks, td_lambda=cfg.train.td_lambda, device=device, timer=timer,
        )
        print(f"GpuTrainer: {len(networks)} networks on {device}")
        if device == "cuda":
            alloc = torch.cuda.memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"  VRAM: {alloc:.0f}MB / {total:.0f}MB ({alloc/total*100:.1f}%)")

        from task_collection.trainer.gpu import GpuTrainer
        return GpuTrainer(
            network_list=networks,
            bt=bt,
            env=env,
            gamma=cfg.env.gamma,
            epsilon_schedule=cfg.train.epsilon,
            lr_schedule=cfg.train.lr,
            total_steps=cfg.train.total_steps,
            heuristic=cfg.train.heuristic,
            timer=timer,
            train_only_teammates=cfg.train.train_only_teammates,
            discount_method=cfg.train.discount_method,
            behavior_policy=cfg.train.behavior_policy,
        )
    else:
        from task_collection.trainer.cpu import CpuTrainer
        print(f"CpuTrainer: {len(networks)} networks on CPU")
        return CpuTrainer(
            network_list=networks,
            env=env,
            gamma=cfg.env.gamma,
            epsilon_schedule=cfg.train.epsilon,
            lr_schedule=cfg.train.lr,
            total_steps=cfg.train.total_steps,
            heuristic=cfg.train.heuristic,
            timer=timer,
            train_only_teammates=cfg.train.train_only_teammates,
            discount_method=cfg.train.discount_method,
            behavior_policy=cfg.train.behavior_policy,
        )
