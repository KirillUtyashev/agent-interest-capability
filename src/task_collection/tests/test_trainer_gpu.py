"""Tests for GpuTrainer and vmap batching correctness."""

import pytest
import torch

from task_collection.enums import EncoderType, Heuristic, LearningType, DespawnMode
from task_collection.datatypes import (
    EnvConfig, EvalConfig, ModelConfig, ScheduleConfig, StochasticConfig,
    State, Grid, TrainConfig, StoppingConfig
)
from task_collection.env.stochastic import StochasticEnv
from task_collection.model import create_networks
import task_collection.encoding as encoding
from task_collection.trainer.cpu import CpuTrainer
from task_collection.trainer.gpu import GpuTrainer
from task_collection.batched_training import (
    BatchedTrainer,
    _to_device_preserving_leading_broadcast,
)
from task_collection.seed import set_all_seeds


def _clone_networks(networks):
    import copy
    return [copy.deepcopy(net) for net in networks]


def _setup_dual_trainers():
    """Sets up a CPU and GPU trainer with perfectly mirrored network weights."""
    torch.manual_seed(42)
    set_all_seeds(42)

    env_cfg = EnvConfig(
        height=3, width=3, n_agents=4, n_tasks=2,
        gamma=0.99, n_task_types=1, max_tasks_per_type=2,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0)
    )
    model_cfg = ModelConfig(encoder=EncoderType.EVERYTHING_CNN_GRID, mlp_dims=(16,))
    lr_cfg = ScheduleConfig(start=0.1, end=0.1)
    eps_cfg = ScheduleConfig(start=0.1, end=0.1)

    train_cfg = TrainConfig(
        total_steps=100, seed=42, lr=lr_cfg, epsilon=eps_cfg,
        learning_type=LearningType.DECENTRALIZED, use_gpu=True,
        td_lambda=0.0, heuristic=Heuristic.NEAREST,
        stopping=StoppingConfig()
    )

    env = StochasticEnv(env_cfg)
    encoding.init_encoder(model_cfg.encoder, env)

    cpu_nets = create_networks(model_cfg, env_cfg, train_cfg)
    gpu_nets = _clone_networks(cpu_nets)

    cpu_trainer = CpuTrainer(
        network_list=cpu_nets, env=env, gamma=0.99,
        epsilon_schedule=eps_cfg, lr_schedule=lr_cfg,
        total_steps=100, heuristic=Heuristic.NEAREST,
    )

    bt = BatchedTrainer(gpu_nets, td_lambda=0.0, device="cpu")  # Run vmap on CPU for float match
    gpu_trainer = GpuTrainer(
        network_list=gpu_nets, bt=bt, env=env, gamma=0.99,
        epsilon_schedule=eps_cfg, lr_schedule=lr_cfg,
        total_steps=100, heuristic=Heuristic.NEAREST,
    )

    return cpu_trainer, gpu_trainer


class TestVmapCorrectness:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    def test_broadcast_transfer_stays_stride_zero_on_cuda(self):
        base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        broadcast = base.unsqueeze(0).expand(41, -1, -1, -1)

        moved = _to_device_preserving_leading_broadcast(
            broadcast, torch.device("cuda")
        )

        assert moved.device.type == "cuda"
        assert moved.stride(0) == 0
        assert torch.equal(moved[0].cpu(), base)
        assert torch.equal(moved[-1].cpu(), base)
        assert moved.untyped_storage().nbytes() == base.numel() * base.element_size()

    def test_deferred_loss_matches_synchronous_loss_and_updates(self):
        """Deferring .item() must not alter loss, traces, or parameter updates."""
        _, trainer = _setup_dual_trainers()
        deferred_bt = trainer._bt
        synchronous_bt = BatchedTrainer(
            _clone_networks(trainer.critic_networks), td_lambda=0.0, device="cpu"
        )

        s_t = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=0, task_types=(0,),
        )
        s_next = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=1, task_types=(0,),
        )
        grids_t, scalars_t = encoding.encode_all_agents(s_t)
        grids_next, scalars_next = encoding.encode_all_agents(s_next)
        rewards = torch.tensor((1.0, -0.5, 0.25, 0.0), dtype=torch.float32)

        synchronous_loss = synchronous_bt.td_lambda_step_batched(
            grids_t, scalars_t, rewards, 0.99, grids_next, scalars_next, alpha=0.01
        )
        deferred_loss = deferred_bt.td_lambda_step_batched(
            grids_t, scalars_t, rewards, 0.99, grids_next, scalars_next,
            alpha=0.01, defer_loss_sync=True,
        )

        assert isinstance(synchronous_loss, float)
        assert isinstance(deferred_loss, torch.Tensor)
        assert not deferred_loss.requires_grad
        assert deferred_loss.item() == pytest.approx(synchronous_loss, rel=1e-7)
        for name in synchronous_bt._params:
            assert torch.equal(synchronous_bt._params[name], deferred_bt._params[name])
            assert torch.equal(synchronous_bt._traces[name], deferred_bt._traces[name])

    def test_gpu_trainer_accumulates_loss_without_sync_until_read(self):
        _, trainer = _setup_dual_trainers()
        s_t = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=0, task_types=(0,),
        )
        s_next = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=1, task_types=(0,),
        )
        trainer.train_move(s_t, on_task=False, t=0)
        trainer.train_move(s_next, on_task=False, t=1)

        assert isinstance(trainer._td_loss_accum, torch.Tensor)
        expected = trainer._td_loss_accum.detach().item() / trainer._td_loss_count
        assert trainer.get_td_loss() == pytest.approx(expected, rel=1e-7)
        assert trainer._td_loss_accum == 0.0
        assert trainer._td_loss_count == 0

    def test_evaluation_uses_live_batched_params_without_cpu_sync(self):
        cpu_trainer, gpu_trainer = _setup_dual_trainers()
        assert cpu_trainer.evaluation_requires_cpu_sync
        assert not gpu_trainer.evaluation_requires_cpu_sync

        s_t = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=0, task_types=(0,),
        )
        s_next = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=1, task_types=(0,),
        )
        gpu_trainer.train_move(s_t, on_task=False, t=0)
        gpu_trainer.train_move(s_next, on_task=False, t=1)
        eval_cfg = EvalConfig(eval_steps=20, eval_seed=123)

        before_sync = gpu_trainer.evaluate(gpu_trainer._env, eval_cfg)
        assert gpu_trainer._cached_enc is None
        gpu_trainer.sync_to_cpu()
        after_sync = gpu_trainer.evaluate(gpu_trainer._env, eval_cfg)
        assert before_sync == after_sync
        assert gpu_trainer._cached_enc is None

    def test_optimized_gpu_checkpoint_round_trip(self, tmp_path):
        """Broadcast views and deferred loss must not alter serialized weights."""
        _, trainer = _setup_dual_trainers()
        s_t = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=0, task_types=(0,),
        )
        s_next = State(
            agent_positions=(Grid(1, 1), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=1, task_types=(0,),
        )
        trainer.train_move(s_t, on_task=False, t=0)
        trainer.train_move(s_next, on_task=False, t=1)
        trainer.sync_to_cpu()

        checkpoint = tmp_path / "optimized_value.pt"
        trainer.save_checkpoint(checkpoint, step=2)
        _, restored = _setup_dual_trainers()
        assert restored.load_checkpoint(checkpoint) == 2

        for name in trainer._bt._params:
            assert torch.equal(trainer._bt._params[name], restored._bt._params[name])

    def test_single_step_params_match(self):
        """After 1 training step, vmap and sequential paths must have identical weights."""
        cpu_trainer, gpu_trainer = _setup_dual_trainers()

        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=0, task_types=(0,)
        )

        # Step 1: Establish _prev
        cpu_trainer.train_move(s, on_task=False, t=0)
        gpu_trainer.train_move(s, on_task=False, t=0)

        # Step 2: The actual TD update
        s_next = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=1, task_types=(0,)
        )

        cpu_trainer.train_move(s_next, on_task=False, t=1)
        gpu_trainer.train_move(s_next, on_task=False, t=1)

        # Pull parameters from the vmap wrapper back to the individual PyTorch modules
        gpu_trainer.sync_to_cpu()

        # Compare every weight in every agent
        for i in range(4):
            for (name1, p1), (name2, p2) in zip(
                cpu_trainer.networks[i].named_parameters(),
                gpu_trainer.networks[i].named_parameters()
            ):
                assert torch.allclose(p1, p2, atol=1e-5), \
                    f"Agent {i}, Param {name1} mismatch after TD update."

    def test_team_values_match(self):
        """Action selection `_compute_team_values` must match exactly."""
        cpu_trainer, gpu_trainer = _setup_dual_trainers()

        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)),
            task_positions=(Grid(2, 2),), actor=0, task_types=(0,)
        )

        # 3 arbitrary candidate after-states
        after_states = [s, s, s]

        cpu_vals = cpu_trainer._compute_team_values(s, after_states)
        gpu_vals = gpu_trainer._compute_team_values(s, after_states)

        assert len(cpu_vals) == 3
        for c_val, g_val in zip(cpu_vals, gpu_vals):
            assert pytest.approx(c_val, rel=1e-5) == g_val
