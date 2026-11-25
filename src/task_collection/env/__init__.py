"""Environment package: create_env factory."""

from __future__ import annotations

from task_collection.env.base import BaseEnv
from task_collection.env.stochastic import StochasticEnv
from task_collection.datatypes import EnvConfig


def create_env(cfg: EnvConfig) -> BaseEnv:
    return StochasticEnv(cfg)
