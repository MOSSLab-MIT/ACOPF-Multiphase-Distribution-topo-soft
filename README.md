# ACOPF Distribution (topology-aware)

AC Optimal Power Flow (ACOPF) for **distribution networks with changing topology**. Training data and OPF labels are generated in Julia via [FeederFlow.jl](datagen_topo/FeederFlow.jl/); learning and inference run in JAX.

Supported feeder scales include IEEE **13**, **123**, **240**, **906**, and **8500** buses (see `datagen_topo/grid_data/`).

**Core library:** for notation, data containers, linearization math, and topology backends, see [`docs/architecture.md`](docs/architecture.md). For Python file layout and naming, see [`docs/coding_style.md`](docs/coding_style.md). Contributor conventions and parameter defaults: [`AGENTS.md`](AGENTS.md).

## Methods


| Method             | Entry script                      | Idea                                                                                                                 |
| ------------------ | --------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Penalty PF**     | `methods/penalty_pf/train.py`     | NN → DER projection → AC power flow; soft-OPF cost (V/I/θ/substation penalties). Test = same pipeline                |
| **Penalty Direct** | `methods/penalty_direct/train.py` | Projected DER controls and predicted voltages; soft-OPF objective plus an adaptive or fixed power-flow equality penalty |
| **Linear OPF**     | `methods/linear_opf/test.py`      | Fixed-reference linear OPF baseline (CVXPY + Clarabel); test-only, optional sharding                                 |


Corresponding test drivers: `methods/penalty_pf/test.py`, `methods/penalty_direct/test.py`. Linear OPF is test-only.

**NN architectures** (`--model_type`, Penalty PF): `mlp` (default; compact load features), `gcn`, `modified_gcn` (full non-compact inputs).

**Penalty PF pipeline** (identical at train, eval, and test time)

1. NN predicts DER controls `(pder, qder)`.
2. Closed-form projection onto the DER device limits (`core/grids/der_projection.py`): active/reactive power bounds, per-unit availability cap, and apparent-power disk.
3. AC power flow at the projected controls gives the voltages; the soft-OPF cost (economic + `ρ_V, ρ_I, ρ_θ, ρ_sub` penalties on voltage / current / angle / substation limits, as in the Julia labels) is evaluated there and is the loss.

Topology changes use compact `changing_switch_status` in HDF5 (expanded to full branch status via `network_config.json`). `core/topology/` provides Woodbury updates (default) and a stamp reference backend, with batch precompute for training/eval.

## Technology stack


| Layer            | Tools                                                                              |
| ---------------- | ---------------------------------------------------------------------------------- |
| Compute          | **JAX** (JIT, `vmap`, `lax.scan`; 64-bit enabled in trainers)                      |
| Models           | **Flax NNX** MLP / GCN (`core/nn/model.py`)                                        |
| Data I/O         | **TensorFlow** datasets (CPU only; GPU hidden from TF)                             |
| Numerics         | **NumPy**, **SciPy**, **Optax**                                                    |
| Linear OPF       | **CVXPY** + **Clarabel**                                                           |
| Datagen / labels | **Julia** + **FeederFlow.jl** (`datagen_topo/FeederFlow.jl/`, `gen_scripts/`)      |


## Environment setup

Create the Python environment from the repo root:

```bash
conda env create -f environment.yml
conda activate new_opt
```

For **GPU training**, ensure JAX sees your CUDA build (the default pip JAX may be CPU-only):

```bash
pip install -U "jax[cuda12]"   # adjust for your CUDA version; see JAX install docs
```

Julia datagen uses a separate project:

```bash
julia --project=datagen_topo/FeederFlow.jl -e 'using Pkg; Pkg.instantiate()'
```

## Repository layout

```
├── core/                      # Shared JAX library (import from submodules, not package roots)
│   ├── grids/                 # OPFData, linear model, power flow, metrics, DER projection, linear OPF solver
│   ├── nn/                    # MLP / GCN + compact input layout
│   ├── topology/              # Woodbury (default) and stamp backends, batch precompute
│   └── utils/                 # types, loss, datasets, workflow, checkpoint, CLI
├── methods/
│   ├── penalty_pf/            # Penalty + power flow (NN → projection → PF)
│   ├── penalty_direct/        # Penalty direct (joint voltage outputs)
│   └── linear_opf/            # Linear OPF baseline (test-only)
├── datagen_topo/
│   ├── FeederFlow.jl/         # Julia package (power flow / OPF)
│   ├── grid_data/             # OpenDSS feeders (13, 123, 240, 906, 8500, …)
│   ├── gen_scripts/           # Julia dataset generators + USAGE.md
│   └── datasets/              # Generated HDF5 + network_config.json
├── scripts/
│   ├── example_scripts/       # Ready-made train/test commands per case
│   └── slurm_runs/            # <method>.sbatch (self-submitting) + per-grid YAML configs
├── analysis/                  # Standalone benchmarks (Woodbury, differentiation, …)
├── processing_results/        # Paper tables/plots from results/
├── tests/                     # Pytest + Julia suite
├── docs/
│   ├── architecture.md
│   └── coding_style.md
├── environment.yml            # Conda env (new_opt)
└── AGENTS.md
```

Import style: `from core.grids.data import OPFData` (defining submodule), not package-root re-exports.

## Quick start (IEEE 13-bus, Penalty PF)

Run from the repository root with `conda activate new_opt`.

### 1. Training dataset

Writes `datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5` and `network_config.json`.

```bash
# Full training set (50k samples)
julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_training_dataset.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --smin 1 --smax 50000
```

### 2. Test datasets

Base feasible test set (required for training/eval):

```bash
julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss \
  --num_scenarios 10 --feasible_samples_per_scenario 100 \
  --max_attempts_multiplier 5
```

Produces `IEEE13Nodeckt_test_feasible.h5` and `IEEE13Nodeckt_test_infeasible.h5` (if any infeasible samples were collected).

Optional load/DER regime splits (used by the example training scripts). Explicit CLI overrides are supported; for reusable defaults, start from `datagen_topo/gen_scripts/configs/testing_defaults.json`.

```bash
julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss \
  --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 \
  --load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 \
  --output_suffix low_load_high_der --save_config false

julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss \
  --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 \
  --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 \
  --output_suffix high_load_low_der --save_config false
```

Scenario layouts, shifted test sets, and CLI details: [`datagen_topo/gen_scripts/USAGE.md`](datagen_topo/gen_scripts/USAGE.md).

### 3. Train Penalty PF

```bash
bash scripts/example_scripts/train_example_13_penalty_pf.sh
```

Or invoke Python directly (lower `--train_samples` / `--epochs` if you only ran the smoke datagen):

```bash
python methods/penalty_pf/train.py \
  --train_h5 datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5 \
  --val_h5 datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5 \
  --test_h5_variants \
    datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5 \
    feasible_high_load_low_der=datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_high_load_low_der.h5 \
    feasible_low_load_high_der=datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_low_load_high_der.h5 \
  --data_json datagen_topo/datasets/IEEE13Nodeckt/network_config.json \
  --case_name IEEE13Nodeckt_penalty_pf \
  --results_dir results_draft \
  --epochs 20 --batch_size 64 --train_samples 100000 --eval_samples 2000 --test_samples 2000 \
  --hidden_size 512 --num_layers 4 --learning_rate 1e-4 \
  --power_flow_solver fixed_point --seed 42
```

Checkpoints and logs: `results_draft/IEEE13Nodeckt_penalty_pf/`.

**CLI notes**

- `--val_h5`: optional in-epoch validation; preferred source for cost normalization when set (else first `--test_h5_variants` entry).
- `--test_h5_variants`: post-training tests; the first entry is the reference test set and must exist; other entries (bare paths or `name=path`) are skipped with a warning if missing.

### Other cases

Example shell scripts under `scripts/example_scripts/`:


| Script                                  | Case / method            |
| --------------------------------------- | ------------------------ |
| `train_example_13_penalty_pf.sh`        | 13-bus Penalty PF        |
| `train_example_13_penalty_direct.sh`    | 13-bus Penalty Direct    |
| `train_example_123_penalty_pf.sh`       | 123-bus Penalty PF       |
| `train_example_240_penalty_pf.sh`       | 240-bus Penalty PF       |
| `train_example_906_penalty_pf.sh`       | 906-bus Penalty PF       |
| `train_example_8500_penalty_pf.sh`      | 8500-bus Penalty PF      |
| `train_example_13_penalty_pf_test.sh`   | 13-bus Penalty PF retest |


Linear OPF across grids: `scripts/slurm_runs/run_linear_opf_all_grids.sh`.

Grid paths follow `datagen_topo/grid_data/<case>/`; swap the `.dss` path in the Julia commands above.

### Cluster (Slurm)

One sbatch script per method + one YAML config per grid: [`scripts/slurm_runs/README.md`](scripts/slurm_runs/README.md).

```bash
scripts/slurm_runs/penalty_pf.sbatch             # grids 123 240 8500, seeds from configs/<grid>.yaml
scripts/slurm_runs/penalty_pf.sbatch 13          # one grid
DRY_RUN=1 scripts/slurm_runs/penalty_pf.sbatch   # print sbatch + train commands
```

Suites cover Direct equality-weight sweeps, GCN variants, and regime ablations.

## Dataset format

Generated under `datagen_topo/datasets/<case_name>/`. Generator CLI and examples: [`datagen_topo/gen_scripts/USAGE.md`](datagen_topo/gen_scripts/USAGE.md).

### On-disk layout


| File / path                   | Role                                                                               |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `network_config.json`         | Bus/branch/DER indexing, limits, base `branch_status`; maps compact switch indices |
| `*_training.h5`               | Training **inputs only** (no OPF solve)                                            |
| `*_test_feasible.h5`          | Feasible test samples (Ipopt labels)                                               |
| `*_test_infeasible.h5`        | Infeasible test samples (if any were collected)                                    |
| `*_test_feasible_<suffix>.h5` | Regime-specific feasible flat file (e.g. `low_load_high_der`)                      |
| `*_test_feasible_shifted*.h5` | Optional shifted test sets (`--variant shifted`)                                   |


Training scripts pass HDF5 paths via `--val_h5` and `--test_h5_variants` (see `core/utils/datasets.py`).

### HDF5 groups (current schema)

**Training** (`*_training.h5`):


| Group       | Fields                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------- |
| `input/`    | `seed`, `regime_id`, `pd`, `qd`, `pder_avai`, `changing_switch_status`, `is_radial`, `obj_weights` |
| `metadata/` | Dimensions, regime schedule, topology sampler settings                                             |


**Test** (flat `*_test_feasible*.h5` or per-scenario files):


| Group     | Fields                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------- |
| `input/`  | `seed`, `scenario_index`, `pd`, `qd`, `pder_avai`, `changing_switch_status`, `is_radial`, `obj_weights` |
| `primal/` | `vr`, `vi`, `pder`, `qder`, `psub`, `qsub`, `psub_phase`, `qsub_phase` (OPF solution)                  |
| `meta/`   | Ipopt termination, timings, objectives (`OPFData` uses `solve_time` from here)                          |


`OPFData` in `core/grids/data.py` expands compact `changing_switch_status` to full branch status using `network_config.json`. Legacy combined `*_test.h5` files and full-width `branch_status` columns in HDF5 are no longer produced.

## Training parameters (typical)


| Parameter            | Typical     | Notes                                                              |
| -------------------- | ----------- | ------------------------------------------------------------------ |
| `rho_eq_init`        | 10–1000     | Penalty Direct weight on the power-flow equality residual          |
| `topology_backend`   | `woodbury`  | `woodbury` (default) or `stamp`                                    |
| `woodbury_ir_steps`  | `auto`      | SMW iterative refinement: `auto` / `0` / `1` / `2`                 |
| `model_type`         | `mlp`       | `mlp`, `gcn`, or `modified_gcn`                                    |


## Tests

```bash
conda activate new_opt
pytest tests/ -q
# Example: power flow + linear model
pytest tests/test_power_flow_solvers.py tests/test_linear_model_math.py -q
```

Tests cover power-flow solvers, the AC-OPF formulation and linearization, DER projection, penalty training helpers, and Woodbury topology updates.

## Analysis and results

| Path | Role |
|------|------|
| `analysis/woodbury_vs_direct/` | Woodbury SMW-IR vs dense solve |
| `analysis/differentiation/` | PF autodiff: full / truncated / implicit |
| `processing_results/` | Aggregate `results/` into paper tables/plots |

## Notes

The codebase and methods are under active development, so significant changes should be expected.

### When Woodbury is expected to be faster

The Woodbury `Y_LL_inv` update builds rank-`r` factors (`r = n_changing_switches × 3 × max_switch_terms`) and solves a dense `r×r` system. It is faster than a direct `O(n³)` dense solve only when `r ≪ n_net`. On small feeders (e.g. 13-bus, `n_net=38, r=18`) the overhead dominates, making direct inversion faster. Woodbury pays off on large feeders (240 / 906 / 8500 buses) where `n_net` grows while `r` stays small.
