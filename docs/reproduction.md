# Reproducing the paper's experiments

Start with the repository [README](../README.md) for installation, metric
definitions, baseline evaluation, checkpoint continuation, and scoring commands.
This page documents how to run the three experiment families from the paper
using the code and example configurations in this repository.

## Main scaling figure

Use `task-collection-reproduce config --family main` with one of these conditions:

| Condition | Interest radius | Capability radius | Networks |
|---|---:|---:|---|
| `both-bounded` | 5 | 5 | DEC width 8; CEN width 8 |
| `interest-bounded` | 5 | `N // 2` | DEC width 8 |
| `capability-bounded` | `N // 2` | 5 | DEC width 8 |

A radius of 5 represents breadth 11, not breadth 5. At `N=11` every condition
has full interest and capability, so reuse the same width-8 runs, planner score,
and seed aggregate in all three panels.

| Agents/task types | Grid | Task cap per type | Reporting cutoff |
|---:|---:|---:|---:|
| 11 | 9×9 | 5 | 2.5M turns |
| 21 | 12×12 | 9 | 5M turns |
| 31 | 15×15 | 14 | 7.5M turns |
| 41 | 17×17 | 18 | 10M turns |

The generator also sets the initial task count to the cap and evaluation length
to `45*N` turns. Spawn/despawn probabilities remain 0.16/0.1. All recipes use
`circulant_derived_task_variance`, `sigma_a=16`, and `everything_cnn_grid`.
Score a run at its reporting cutoff using evaluations in the final 1M turns.
Normalize by the assignment-based planner's mean score from ten independent
100-round rollouts of the same environment.

### Selected learning rates

These rates provide direct starting points for reproducing the plotted
configurations; the common LR grid and selection protocol are in the README.
Train each reporting seed independently at the selected rate.

| N | Both bounded: DEC | Both bounded: CEN | Interest bounded: DEC | Capability bounded: DEC |
|---:|---:|---:|---:|---:|
| 11 | `5e-6` | `1e-6` | `5e-6` | `5e-6` |
| 21 | `1e-5` | `1e-6` | `5e-6` | `5e-6` |
| 31 | `5e-6` | `5e-7` | `5e-6` | `1e-6` |
| 41 | `5e-6` | `5e-7` | `5e-6` | `1e-6` |

For example, generate one additional-seed configuration for both bounded at
`N=21`:

```bash
uv run --locked --extra cuda task-collection-reproduce config \
  --family main --agents 21 --condition both-bounded \
  --method decentralized --width 8 --lr 1e-5 --seed 1235 --gpu \
  --output outputs/configs/main_both_n21_dec_seed1235.yaml

uv run --locked --extra cuda task-collection-train \
  --config outputs/configs/main_both_n21_dec_seed1235.yaml
```

For centralized runs, change `--method`, choose its selected LR, omit `--gpu`
for CPU training, and use the CPU extra on both commands.

## Network-width sweep at fixed N=11

Use `--family width-sweep --agents 11 --width W`. The environment has full
interest/capability, a 9×9 grid, five tasks per type, and the same reward generator
and encoder as the main figure. Train for 2.5M turns at each width. Width changes
both CNN layers and the hidden MLP layer; depth stays fixed.

| Width W | DEC selected LR | CEN selected LR |
|---:|---:|---:|
| 1 | `5e-5` | `1e-6` |
| 2 | `5e-6` | `1e-6` |
| 4 | `5e-6` | `5e-7` |
| 8 | `5e-6` | `1e-6` |
| 16 | `5e-6` | `1e-6` |
| 32 | `5e-6` | `1e-6` |
| 64 | `1e-5` | `1e-6` |

```bash
uv run --locked --extra cpu task-collection-reproduce config \
  --family width-sweep --agents 11 --method centralized \
  --width 4 --lr 5e-7 --seed 1235 \
  --output outputs/configs/width4_n11_cen_seed1235.yaml
```

Run the resulting YAML with `task-collection-train`. Reuse the width-8 main-figure point
instead of independently retraining it for this comparison. Normalize every
width by the same `N=11` planner score.

## Centralized width-64 scaling

Use `--family centralized-width64 --method centralized`. Both radii remain 5;
grid, task counts, evaluation length, and budget follow the scaling table.
This compares CEN width 64 against CEN width 8 and DEC width 8 under the
same environment and reporting rules.

| N | CEN width-64 selected LR |
|---:|---:|
| 11 | `1e-6` |
| 21 | `5e-7` |
| 31 | `5e-7` |
| 41 | `5e-7` |

```bash
uv run --locked --extra cuda task-collection-reproduce config \
  --family centralized-width64 --agents 41 --method centralized \
  --lr 5e-7 --seed 1235 --gpu \
  --output outputs/configs/cen64_n41_seed1235.yaml
```

The CPU path is also supported; use the CPU extra and omit `--gpu`. Reuse the
width-64 `N=11` point from the fixed-N width sweep.

## Reproduction checklist

1. Record the code commit, dependency lock, resolved YAML, LR, and training seed.
2. Keep independent seeds separate; combine only initial/resumed CSV segments
   belonging to the same trained run, in chronological order.
3. Score at the specified cutoff over the final 1M training turns, not an
   arbitrary number of CSV rows.
4. Use one scalar score per trained seed, then report the mean and sample SD
   across seeds. Do not treat repeated evaluations as independent training seeds.
5. Use the same positive planner denominator for all widths/methods in a setting.
   Recompute the planner when environment or reward parameters change.
6. Reuse the shared `N=11` reporting point consistently across figures.

This release includes code and recipes for running the paper's experiments.
Experiment outputs are generated locally and are not bundled with the code.
Fresh training can vary with hardware, software builds, and random seeds; the
settings above do not guarantee bitwise reproduction of historical weights.
