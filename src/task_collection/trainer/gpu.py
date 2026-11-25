"""GPU trainer: vmap-batched training and action selection via BatchedTrainer."""

from __future__ import annotations

from typing import Any

import torch

import task_collection.encoding as encoding
from task_collection.batched_training import BatchedTrainer
from task_collection.datatypes import ScheduleConfig, State
from task_collection.enums import Heuristic
from task_collection.env.base import BaseEnv
from task_collection.model import ValueNetwork
from task_collection.schedule import compute_schedule_value
from task_collection.trainer.timer import Timer
from task_collection.trainer.value_base import ValueTrainerBase


class GpuTrainer(ValueTrainerBase):
    """Batched GPU training for N networks (decentralized or centralized).

    Uses BatchedTrainer (vmap over N networks) for both training and
    action selection.
    """

    def __init__(
        self,
        network_list: list[ValueNetwork],
        bt: BatchedTrainer,
        env: BaseEnv,
        gamma: float,
        epsilon_schedule: ScheduleConfig,
        lr_schedule: ScheduleConfig,
        total_steps: int,
        heuristic: Heuristic,
        timer: Timer | None = None,
        train_only_teammates: bool = False,
        discount_method: str = "team_steps",
        behavior_policy: str = "value_greedy",
    ) -> None:
        super().__init__(
            network_list=network_list, env=env, gamma=gamma,
            epsilon_schedule=epsilon_schedule, lr_schedule=lr_schedule,
            total_steps=total_steps, heuristic=heuristic, timer=timer,
            train_only_teammates=train_only_teammates,
            discount_method=discount_method,
            behavior_policy=behavior_policy,
        )
        self._bt = bt
        self._zero_rewards_tensor = torch.zeros(
            self._n_networks, dtype=torch.float32, device=self._bt.device,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode_all(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        return encoding.encode_all_agents(state)

    # ------------------------------------------------------------------
    # TD step
    # ------------------------------------------------------------------
    def _td_step(
        self, prev: Any, rewards: tuple[float, ...],
        discount: float, current: Any, t: int,
        teammate_indices: list[int] | None = None,
    ) -> torch.Tensor:
        grids_t, scalars_t = prev
        grids_next, scalars_next = current
        # Move transitions have the same all-zero reward vector. Reuse it on the
        # accelerator instead of allocating on CPU and copying it every update.
        rewards_t = (
            self._zero_rewards_tensor
            if rewards is self._zero_rewards
            else torch.tensor(rewards, dtype=torch.float32)
        )

        return self._bt.td_lambda_step_batched(
            grids_t, scalars_t, rewards_t, discount,
            grids_next, scalars_next,
            alpha=compute_schedule_value(self._lr_schedule, t, self._total_steps),
            agent_indices=teammate_indices,
            defer_loss_sync=True,
        )

    # ------------------------------------------------------------------
    # Value computation for greedy action selection
    # ------------------------------------------------------------------
    def _compute_team_values(
        self, state: State, after_states: list[State],
        teammate_indices: list[int] | None = None,
    ) -> list[float]:
        from task_collection.trainer.timer import TimerSection
        self._timer.start(TimerSection.ACTION_ENCODE)
        grids, scalars = encoding.encode_all_agents_for_actions(state, after_states)
        self._timer.stop()
        self._timer.start(TimerSection.ACTION_FORWARD)
        values = self._bt.forward_batched(grids, scalars)  # (N, B)
        self._timer.stop()
        self._last_action_grids = grids      # (N, B, C, H, W) — kept for enc caching
        self._last_action_scalars = scalars  # (N, B, S)
        if teammate_indices is not None:
            return values[teammate_indices].sum(dim=0).tolist()
        return values.sum(dim=0).tolist()

    def _cache_selected_enc(self, best_idx: int) -> None:
        """Cache the encoding of the selected after-state for reuse in train_move/train_pick."""
        grids = self._last_action_grids[:, best_idx]
        scalars = self._last_action_scalars[:, best_idx]
        # Preserve EverythingEncoder's stride-zero network broadcast.  Filtered
        # inputs differ by network and retain the previous contiguous cache.
        self._cached_enc = (
            grids if grids.stride(0) == 0 else grids.contiguous(),
            scalars if scalars.stride(0) == 0 else scalars.contiguous(),
        )

    @property
    def evaluation_requires_cpu_sync(self) -> bool:
        """Evaluation reads BatchedTrainer's live parameters directly."""
        return False

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def sync_to_cpu(self) -> None:
        self._bt.sync_to_networks()

    def load_checkpoint(self, path: str | Any) -> int | None:
        step = super().load_checkpoint(path)
        self._bt.sync_from_networks()
        return step
