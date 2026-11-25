# Usage and reproduction guide

Detailed setup, environment definitions, training, baselines, scoring, and
release packaging for [Agent Interest and Capability](../README.md).

## Contents

- [Installation](#installation)
- [First run and tests](#first-run-and-tests)
- [Environment and learning setup](#environment-and-learning-setup)
- [Reproducing the experiment families](#reproducing-the-experiment-families)
- [Planner and greedy baselines](#planner-and-greedy-baselines)
- [Outputs, scoring, and resuming](#outputs-scoring-and-resuming)
- [Cluster use and troubleshooting](#cluster-use-and-troubleshooting)
- [Code layout and release packaging](#code-layout-and-release-packaging)

## Installation

### Requirements

The reference installation targets **Linux x86-64 and Python 3.12**. A CPU-only
installation is sufficient for the smoke test, unit tests, and heuristic
evaluation. Full decentralized experiments are substantially faster with an
NVIDIA GPU. CUDA training requires a driver compatible with the CUDA 12.6
PyTorch wheel. macOS and native Windows are not validated reference platforms;
use Linux or a Linux VM/WSL2 for the documented workflow.

Dependencies are specified in [pyproject.toml](../pyproject.toml) and their
resolved versions in [uv.lock](../uv.lock). CPU and CUDA installations use separate,
explicit PyTorch indexes; do not enable both extras together.

### Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not
already available. The locked setup was tested with uv 0.12.24; this project
requires that version or newer. From the repository root, run:

```bash
uv python install 3.12
uv sync --locked --extra cpu
uv run --locked --extra cpu task-collection-train --help
```

For an NVIDIA GPU machine, use the CUDA extra **instead**:

```bash
uv sync --locked --extra cuda
uv run --locked --extra cuda python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Use the same extra on subsequent `uv run` commands. Otherwise uv may change
the installed PyTorch variant. The CPU/CUDA index setup follows
[uv's PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
No activation or manual `PYTHONPATH` setting is required after installation.

**Existing checkout:** do not let `uv sync` replace a virtual environment
being used by running experiments. Use a separate environment for every uv
command, for example:

```bash
UV_PROJECT_ENVIRONMENT=.venv-test uv sync --locked --extra cpu
UV_PROJECT_ENVIRONMENT=.venv-test uv run --locked --extra cpu pytest -q
```

### Install with pip instead

From the repository root, create a new environment rather than modifying an
existing training environment:

```bash
python3.12 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[cpu]'
python -m pip install 'pytest==9.0.3'
task-collection-train --help
```

For CUDA, replace the PyTorch index with
`https://download.pytorch.org/whl/cu126` and install `'.[cuda]'`. Pip does not
use `uv.lock` or uv's index configuration; prefer uv for the locked environment.

## First run and tests

All commands below assume the repository root is the current directory.

Run a short, CPU-only example:

```bash
uv run --locked --extra cpu task-collection-train \
  --config src/task_collection/configs/examples/smoke_cpu.yaml
```

This trains for only 100 turns and evaluates twice. It verifies configuration
loading, reward construction, training, CSV logging, and checkpoint writing.
It is not a meaningful performance benchmark. Results appear beneath
`outputs/examples/smoke_cpu/` in a timestamped directory.

Run the test suite:

```bash
uv run --locked --extra cpu pytest -q
```

CUDA-specific tests skip when CUDA is unavailable. To test a GPU installation,
use `uv run --locked --extra cuda pytest -q` on a GPU node. Passing CPU tests is
not a substitute for validating CUDA on the target hardware.

The module form remains supported:

```bash
uv run --locked --extra cpu python -m task_collection.train \
  --config src/task_collection/configs/examples/smoke_cpu.yaml
```

## Environment and learning setup

Agents move on a grid and collect task instances. A turn consists of one acting
agent's movement decision and, if an eligible task is present at its location,
a collect-or-decline decision. Agents act cyclically; one round contains
`N` turns. Multiple agents can occupy a cell, and different task types can
coexist at a cell. At most one instance of a given type occupies a cell.

Completing a task produces a reward vector over agents. Capability restricts
which task types an agent can collect; interest restricts which agents receive
nonzero rewards from a task. These assignments and the reward matrix remain
fixed during a run. Tasks spawn and despawn after completed rounds in the
paper examples, with spawn probability 0.16 and despawn probability 0.1.

### Breadths versus YAML radii

The YAML names are historical: `relatedness_width` is the **interest radius**
and `proficiency_width` is the **capability radius**. They are not set sizes.
For either radius `r`, the circular set cardinality is `min(2*r + 1, N)`.

- A constant radius of 5 gives breadth 11 at every tested `N`.
- Full breadth uses radius `N // 2` for the odd agent counts below.
- At `N=11`, radius 5 is already full breadth. All three main-figure conditions
  therefore share the same environment and reporting point at `N=11`.

Always set `env.n_task_types = env.n_agents`. Initial task counts and caps
are **per task type**, not total counts across the grid.

### Rewards and networks

The examples explicitly select
`env.stochastic.reward_generation: circulant_derived_task_variance` and
`sigma_a: 16`. This is a deterministic, task-dependent Fourier construction,
not iid reward sampling. Entries outside each task's interest set are zero.
Within that set, deviations are centered and rescaled to population standard
deviation `sigma_a`. Task-total variability is derived from the interest-set
size; `sigma_b` is ignored in this mode. The Fourier frequency scales
automatically with agent count; it does not require a separate YAML override.
Rewards can be negative. See [env/stochastic.py](../src/task_collection/env/stochastic.py).

The paper configuration uses:

- `everything_cnn_grid`: full, unmasked task/agent position channels; not the
  filtered encoder used in some exploratory experiments.
- Two 3×3 convolutional layers, each with `w` output channels; stride 1,
  padding 1, dilation 1, groups 1, bias enabled; ReLU after each layer.
- Flattened convolution outputs concatenated with scalar observations, then
  one hidden MLP layer of width `w`, ReLU, and one scalar linear output.
- PyTorch's default weight/bias initialization; no pooling, normalization,
  dropout, gradient clipping, or Adam optimizer in value learning.
- After-state semi-gradient TD(λ), direct parameter updates, λ=0.3,
  γ=0.99 per round, and constant epsilon-greedy exploration ε=0.05.

Changing model width requires changing **both** convolutional channel counts
and the hidden MLP width; depth and kernel sizes stay fixed. Decentralized
learning uses `N` independently parameterized networks; centralized learning
uses one. The `round_steps` setting changes discount timing, not the per-update
application of λ. The examples disable early stopping and use value learning.

## Reproducing the experiment families

See [the paper recipes](reproduction.md) for the selected learning rates
and concrete settings for all three experiment families.

### Example YAMLs

Every example is self-contained and can be passed directly to `task-collection-train`.
They use relative output paths, require no inherited config, and default to
CPU so opening an example does not require a GPU allocation. For full
decentralized training, install the CUDA extra and override `train.use_gpu=true`.
The learning rates in the files are starting examples, not a claim that one
rate is optimal for every setting.

| Experiment | Example YAML |
|---|---|
| Installation smoke test | [smoke_cpu.yaml](../src/task_collection/configs/examples/smoke_cpu.yaml) |
| Both breadths constant, DEC | [main_both_bounded.yaml](../src/task_collection/configs/examples/main_both_bounded.yaml) |
| Interest constant, capability full, DEC | [main_interest_bounded.yaml](../src/task_collection/configs/examples/main_interest_bounded.yaml) |
| Capability constant, interest full, DEC | [main_capability_bounded.yaml](../src/task_collection/configs/examples/main_capability_bounded.yaml) |
| Centralized width-8 baseline | [main_centralized.yaml](../src/task_collection/configs/examples/main_centralized.yaml) |
| Fixed `N=11`, DEC width sweep | [width_sweep_n11_decentralized.yaml](../src/task_collection/configs/examples/width_sweep_n11_decentralized.yaml) |
| Fixed `N=11`, CEN width sweep | [width_sweep_n11_centralized.yaml](../src/task_collection/configs/examples/width_sweep_n11_centralized.yaml) |
| Centralized width-64 scaling | [centralized_width64_scaling.yaml](../src/task_collection/configs/examples/centralized_width64_scaling.yaml) |

### Environment scaling and budgets

These are the reporting budgets, not a requirement to discard later training
data. A run can have a larger training target; scoring must still use the
chosen reporting cutoff. The training step counter counts agent turns, not
rounds or individual movement/pick sub-transitions.

| Agents/task types `N` | Grid | Initial tasks and cap per type | Reporting cutoff | Training-eval turns |
|---:|---:|---:|---:|---:|
| 11 | 9×9 | 5 | 2,500,000 | 495 |
| 21 | 12×12 | 9 | 5,000,000 | 945 |
| 31 | 15×15 | 14 | 7,500,000 | 1,395 |
| 41 | 17×17 | 18 | 10,000,000 | 1,845 |

Spawn/despawn probabilities, reward construction, gamma, network depth,
lambda, and exploration remain fixed. Training evaluation uses 45 rounds
(`eval.eval_steps = 45*N`) in each setting.

### Generate a complete configuration

`task-collection-reproduce config` sets the coupled environment fields together,
including task counts, grid size, radii, evaluation length, network width,
and reporting budget. It refuses to overwrite an existing YAML unless
`--force` is explicitly supplied. It only creates a config; it does not
launch or submit training.

```bash
# Main figure: N=31, interest breadth 11, capability breadth 31.
uv run --locked --extra cuda task-collection-reproduce config \
  --family main --agents 31 --condition interest-bounded \
  --method decentralized --width 8 --lr 5e-6 --seed 1234 --gpu \
  --output outputs/configs/interest_bounded_n31.yaml

uv run --locked --extra cuda task-collection-train \
  --config outputs/configs/interest_bounded_n31.yaml
```

Condition names are `both-bounded`, `interest-bounded`, and
`capability-bounded`. For the centralized curve in the both-bounded panel,
use `--method centralized` and choose its learning rate independently.

For the fixed-agent-count network-width experiment:

```bash
uv run --locked --extra cpu task-collection-reproduce config \
  --family width-sweep --agents 11 --method centralized \
  --width 16 --lr 1e-6 --seed 1234 \
  --output outputs/configs/width16_n11_centralized.yaml

uv run --locked --extra cpu task-collection-train \
  --config outputs/configs/width16_n11_centralized.yaml
```

Use widths `[1, 2, 4, 8, 16, 32, 64]` for both methods. Both breadths are
full at `N=11`, and the budget stays 2.5M turns. The width-8 point is the
same configuration as the shared `N=11` main-figure point; reuse the same
trained seeds and scoring rather than treating it as a separate experiment.

For centralized width-64 scaling:

```bash
uv run --locked --extra cpu task-collection-reproduce config \
  --family centralized-width64 --agents 31 --method centralized \
  --lr 5e-7 --seed 1234 \
  --output outputs/configs/centralized_w64_n31.yaml

uv run --locked --extra cpu task-collection-train \
  --config outputs/configs/centralized_w64_n31.yaml
```

This keeps both breadths at 11 and changes only the centralized network
width relative to its width-8 baseline. It also supports GPU training:
generate with `--gpu` and run with the CUDA extra.

### Learning rates, seeds, and overrides

The common main-figure grid is
`[5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]`.
The `N=11` model-width sweep additionally explored `1e-7`; exploratory
centralized sweeps also tested lower rates. Do not substitute those extra
rates into a claimed common-grid comparison without changing its protocol.

For each method/configuration, run a preliminary LR sweep with training
seed 1234, choose the rate using the same reporting cutoff and trailing
training-step window, then train independent reporting seeds at that rate.
Seeds 1235–1238 are a convenient four-seed reproduction set. These are
training seeds, not evaluation-rollout seeds. Estimate seed uncertainty from
one performance value per trained seed; evaluation points within one run
are not independent training seeds. Keep seed-selection rules explicit and
consistent in any new reproduction study.

Override a YAML directly when only one setting changes:

```bash
uv run --locked --extra cpu task-collection-train \
  --config src/task_collection/configs/examples/main_centralized.yaml \
  --override train.seed=1235 train.lr.start=1e-6 train.lr.end=1e-6 \
    logging.output_dir=outputs/main/cen_n21/lr_1e-6/seed_1235
```

Quote list-valued overrides, e.g.
`'model.conv_specs=[[16,3],[16,3]]' 'model.mlp_dims=[16]'`.
Setting only `model.mlp_dims` does not change the CNN channels. Overrides
use existing field names: `--override key=value ...`. To change `N`, prefer
the config generator rather than updating only `env.n_agents`.

## Planner and greedy baselines

The assignment-based planner is named `hungarian` in the code. It assigns
agents to available task instances using team reward, capability constraints,
and a survival adjustment based on Manhattan distance and task despawn rate.
It does not train a model or use an oracle future rollout to select actions.
The value-aware greedy baseline is named `nearest`; despite that historical
name, it prioritizes task value and uses distance as a tie-breaker.

Compute planner/greedy reference scores for exactly the same YAML:

```bash
uv run --locked --extra cpu task-collection-reproduce heuristic \
  --config src/task_collection/configs/examples/main_both_bounded.yaml \
  --policies hungarian nearest --rounds 100 --seeds 10 --workers 4 \
  --output-dir outputs/baselines/both_bounded_n21
```

This runs ten independently seeded 100-round rollouts per policy, without
weights or training. It writes `per_seed.csv` and `summary.csv`; the denominator
for planner normalization is the positive `hungarian` `team_rps_mean` from
the summary. The environment/reward-construction seed remains the YAML's
training seed; the rollout seeds vary. Recompute baselines when changing
grid, task cap, reward generator, sigma_a, or either breadth.

For a quick functional check, use `--rounds 2 --seeds 2 --workers 1`.
Use enough rounds and seeds for reported performance; a quick check is not
a replacement for the paper's baseline estimate. The helper will not overwrite
existing baseline CSVs unless `--force` is supplied.

## Outputs, scoring, and resuming

### Run outputs

`logging.output_dir` is a base directory. Every invocation creates a new
timestamped child, so repeated or resumed training does not overwrite an
earlier run:

```text
outputs/.../YYYY-MM-DD_HHMMSS_microseconds/
├── metadata.yaml       # resolved configuration, run identity and hardware
├── metrics.csv         # evaluation scores, step counter and elapsed wall time
├── details.csv         # training diagnostics
├── timing.csv          # optional timing breakdown
└── checkpoints/
    ├── step_0.pt
    ├── step_1000000.pt
    └── final.pt
```

The examples evaluate every 50k turns during regular training and every
10k within the final 1M of the configured training target. They checkpoint
every 1M turns. These frequencies are separate controls:
`logging.main_csv_freq`, `eval.final_eval_freq`, `eval.final_eval_window`,
and `eval.checkpoint_freq`. The smoke test uses shorter intervals.

`greedy_team_rps` is the undiscounted sum of all agents' collected rewards
divided by the number of evaluation **turns**. It is not discounted return,
not a per-round sum, and not the acting agent's reward alone. Since each
round contains `N` turns, it also equals total team reward per round divided
by `N`. Training uses discounting; this reported rollout metric does not.

### Score at a reporting cutoff

Score the mean over `(cutoff - 1M, cutoff]`, independent of evaluation cadence:

```bash
uv run --locked --extra cpu task-collection-reproduce score \
  --metrics outputs/YOUR_RUN/TIMESTAMP/metrics.csv \
  --cutoff 5000000 --window-steps 1000000
```

Replace `YOUR_RUN/TIMESTAMP` with the generated run directory. The returned
JSON includes the mean, number of evaluations, and exact window boundaries.
A 50k cadence contributes 20 observations to a complete 1M window; a 10k
cadence contributes 100. Do not use an unconditional last-K-row slice when
comparing runs with different evaluation cadences.

For resumed runs, list the initial CSV first and later segments after it:

```bash
uv run --locked --extra cpu task-collection-reproduce score \
  --metrics outputs/INITIAL/metrics.csv outputs/RESUMED/metrics.csv \
  --cutoff 7500000
```

Later-listed segments take precedence at overlapping steps. Only combine
segments from the same method, environment, width, LR, and training seed.
Do not concatenate different seeds or LRs as if they were continuations.
Incomplete runs fail by default; `--allow-partial` explicitly scores at the
latest available evaluation at or below the requested cutoff.

For each configuration, divide each trained seed's scalar score by the
same planner reference mean. Plot the mean of those normalized seed scores
and their **sample standard deviation** (`ddof=1`). This is not the standard
deviation across evaluations within one run. At `N=11`, reuse one DEC width-8
seed aggregate for all three main panels and the corresponding width-8
comparison points.

### Resume or warm-start

Continue the global step counter from a checkpoint:

```bash
uv run --locked --extra cuda task-collection-train \
  --config outputs/configs/interest_bounded_n31.yaml \
  --resume-training outputs/YOUR_RUN/TIMESTAMP/checkpoints/step_1000000.pt \
  --override train.total_steps=7500000
```

`train.total_steps` is the absolute final target, not an additional number
of turns. The checkpoint must match the method, environment dimensions,
encoder, and network architecture. Keep the LR/seed protocol consistent
unless intentionally running a new continuation experiment.

Resume restores weights and the recorded step, but starts a fresh environment
trajectory and eligibility traces; it is not an exact replay of the interrupted
RNG/environment state. A new timestamped metrics segment is written. Preserve
all segments when analyzing the full trajectory.

`--resume CHECKPOINT` is different: it loads weights but starts the step
counter at zero. Use it for a warm start, not for continued-budget training.
Load only trusted checkpoints.

## Cluster use and troubleshooting

Neither Slurm nor CUDA MPS is required for the package. On a cluster, request
resources using that cluster's own partition, account, GPU type, and limits,
then invoke the same `task-collection-train` command from your job script.
No site-specific job scripts or cluster configuration are included.

For one training process, a minimal GPU job body is:

```bash
cd /path/to/task-collection
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
uv run --locked --extra cuda task-collection-train \
  --config outputs/configs/interest_bounded_n31.yaml
```

Install/sync before submitting a cluster job if compute nodes cannot access the internet.
Concurrency and memory requirements depend on `N`, grid size, and width;
start with one process per GPU before experimenting with MPS sharing.
Centralized training supports both CPU and CUDA. GPU allocation alone does
not select the GPU trainer: also set `train.use_gpu=true`.

- **No CUDA detected:** run the PyTorch check above on the allocated GPU node,
  not a login node. Verify the CUDA extra and driver. The current trainer
  falls back to CPU if CUDA is unavailable; check its startup message.
- **Slow CPU runs:** use the thread limits above. Large centralized networks
  and full evaluation rollouts can dominate runtime. Do not disable evaluation
  when intending to reproduce reported scores.
- **Wrong reward/encoder:** use the self-contained examples and confirm the
  resolved values in `metadata.yaml`.
- **Unequal task/agent counts:** update both fields; the implementation
  requires one shared ID space.
- **Wrong breadth:** `*_width=5` means breadth 11, not breadth 5.
- **Missing results:** inspect `metadata.yaml`, the newest timestamped segment,
  and stderr. A new resume creates a new directory rather than appending to
  the old CSV.

## Code layout and release packaging

```text
README.md                         # project overview and quick start
docs/usage.md                     # detailed installation and experiment guide
pyproject.toml / uv.lock           # installable package and dependency lock
MANIFEST.in                       # source-distribution rules
src/task_collection/
├── configs/examples/             # self-contained paper experiment YAMLs
├── reproduce.py                  # portable config/baseline/scoring CLI
├── env/                          # dynamics and reward construction
├── encoding/                     # full and filtered CNN observations
├── model.py                      # scalar value networks and TD updates
├── batched_training.py           # vectorized multi-network training
├── trainer/                      # CPU/GPU training and evaluation
├── train.py                      # training CLI and scheduling
├── policy.py / eval.py            # heuristics and rollout evaluation
└── tests/                        # environment, reward, training regression tests
```

The source archive contains source code, regression tests, documentation,
example configurations, and the SVG figures used in the README. It excludes
checkpoints, generated plots, logs, local environments, notebooks, experiment
outputs, and version-control history.

Build a wheel and source archive:

```bash
uv build
python scripts/check_distribution.py dist
```

Artifacts appear in `dist/`. Test changes with the CPU suite before rebuilding.
Record the code commit, resolved YAML, dependency lock, reporting cutoff,
selected LR, and reporting seed set alongside any new results. Matching seeds
does not promise bitwise equality across different PyTorch builds or GPU types.

The distribution check verifies that required package files are present and that
generated research outputs are excluded.
