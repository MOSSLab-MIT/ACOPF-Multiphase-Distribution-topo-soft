# ACOPF Distribution — topology-aware, **soft-constraint** variant

Learning-based AC Optimal Power Flow (ACOPF) for **unbalanced multiphase distribution networks whose
topology changes** (switches open/close). OPF labels are generated in Julia (FeederFlow.jl + JuMP + Ipopt);
learning and inference run in JAX.

> **This repository is the _soft-constraint_ case.** Voltage-magnitude, branch-current and
> phase-angle-difference limits are *not* hard constraints: they are relaxed with nonnegative slacks and
> priced in the objective with per-sample L1 penalty weights `ρ_V, ρ_I, ρ_θ`. Only power balance, DER limits and
> the per-phase substation limit stay hard. Everything below — datasets, labels, losses, metrics, the linear-OPF
> baseline — follows this convention. See [Soft-constraint formulation](#soft-constraint-formulation).

Supported feeders: IEEE **13**, **123**, **240**, **906** and **8500** buses (OpenDSS files in
[`datagen_topo/grid_data/`](datagen_topo/grid_data/)).

## Contents

1. [Soft-constraint formulation](#soft-constraint-formulation)
2. [Methods](#methods)
3. [Installation](#installation)
4. [Quick start (IEEE 13-bus, Penalty PF)](#quick-start-ieee-13-bus-penalty-pf)
5. [Other grids, methods and cluster runs](#other-grids-methods-and-cluster-runs)
6. [Dataset format](#dataset-format)
7. [Key hyperparameters](#key-hyperparameters)
8. [Repository layout](#repository-layout)
9. [Tests](#tests)
10. [Analysis scripts](#analysis-scripts)
11. [Notes](#notes)

## Soft-constraint formulation

For one sample (loads, DER availability, topology, weights) the label OPF is

```
min   w_cost · (C_sub + C_der) + w_curtail · Σ_units (avail − Σ_phases p_der)
      + ρ_V · Σ s_V + ρ_I · Σ s_I + ρ_θ · Σ s_θ
s.t.  nodal AC power balance                          (hard)
      DER: p_der ≥ 0, q box, p² + q² ≤ s_max², Σ_unit p ≤ availability   (hard)
      substation: |S_sub,k| ≤ s_max,k  per phase k    (hard)
      vmin² − s_V ≤ |u_i|² ≤ vmax² + s_V              (soft, non-slack nodes)
      |I_lr|² ≤ i_max,l² + s_I                        (soft, every branch phase)
      |Im(ū_f u_t)| ≤ dvamax · Re(ū_f u_t) + s_θ      (soft, angle difference; dvamax = tan 60°)
      s_V, s_I, s_θ ≥ 0
```

* `obj_weights = [w_cost, w_curtail, ρ_V, ρ_I, ρ_θ]` is stored **per sample** in the HDF5 files and is also an
  input of the neural network pipeline. `w_*` are sampled uniformly, `ρ_*` log-uniformly (see
  [`datagen_topo/gen_scripts/USAGE.md`](datagen_topo/gen_scripts/USAGE.md)).
* The slacks are in the raw (squared) units of each constraint, so for `ρ` above the hard-OPF multiplier the soft
  optimum coincides with the hard optimum (exact penalty). Because voltage/current/angle violations are always
  feasible, voltage-driven OPF infeasibility no longer occurs; `*_test_infeasible.h5` is therefore normally empty
  and not written.
* **What counts as a "violation" in the Python code** (`core/grids/metrics.py`): only the *hard* constraints
  (substation limit, DER limits, power-flow equality). Soft-limit exceedances are part of the **cost**
  (`calc_cost` = economic terms + `Σ_k ρ_k · pen_k`), are reported unweighted as `soft_penalty = [pen_V, pen_I, pen_θ]`,
  and are printed as `Soft penalty ρ·Σs (V / I / θ)` in test summaries.
* Two different "penalties" appear in this repo — do not confuse them:

  | Symbol | Where | Role |
  | --- | --- | --- |
  | `ρ_V, ρ_I, ρ_θ` | data (`obj_weights`) | part of the **OPF problem** being learned (soft limits) |
  | `--rho` (`rho_eq`, `rho_ineq`) | training | weight on **hard-constraint** violation in the **training loss** |

## Methods

| Method             | Entry script                      | Idea                                                                                                                       |
| ------------------ | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Penalty PF**     | `methods/penalty_pf/train.py`     | NN → DER projection → AC power flow. Loss = soft-OPF cost + `rho` · (hard-constraint violation)². Test = same pipeline.    |
| **Penalty Direct** | `methods/penalty_direct/train.py` | NN predicts DER controls **and** voltages (no PF solve). Loss = soft-OPF cost + `rho_eq` · (PF residual)² + `rho_ineq` · (hard violation)²; fixed or adaptive `rho`. |
| **Linear OPF**     | `methods/linear_opf/test.py`      | Fixed-reference linearized OPF baseline (CVXPY + Clarabel) with linearized soft rows; test-only, optional sharding.        |

Test-only drivers for trained checkpoints: `methods/penalty_pf/test.py`, `methods/penalty_direct/test.py`.

**Penalty PF pipeline** (identical at train, validation and test time — no post-processing):

1. The NN predicts raw per-phase DER controls `(pder, qder)`.
2. A closed-form projection (`core/grids/der_projection.py`) enforces the hard DER limits: `p ≥ pdermin`, per-unit
   availability cap, apparent-power disk.
3. An AC power flow at the projected controls gives the voltages (so the power-flow equations hold by construction).
   Cost, soft penalties and hard violations are evaluated there; only the hard violations enter the `rho` term.

NN architectures for Penalty PF (`--model_type`): `mlp` (default; compact load features), `gcn`, `modified_gcn`
(full, non-compact inputs). Penalty Direct is MLP only.

**Topology changes.** Samples store a compact `changing_switch_status` vector, expanded to the full branch status via
`network_config.json`. `core/topology/` updates `Y_LL⁻¹` for each sampled topology with a Woodbury low-rank update
(default) or a full re-stamp reference backend; matrices are precomputed per batch.

## Installation

Python (JAX/Flax/TensorFlow/CVXPY), from the repository root:

```bash
conda env create -f environment.yml
conda activate new_opt
```

For **GPU training**, make sure JAX sees your CUDA build (the pinned pip wheel may be CPU-only):

```bash
pip install -U "jax[cuda12]"   # adjust for your CUDA version; see the JAX install docs
```

TensorFlow is used only for CPU-side data loading (GPUs are hidden from it).

Julia (dataset generation), Julia ≥ 1.10:

* [FeederFlow.jl](https://github.com/hoang-tn/FeederFlow.jl) is a git submodule at `datagen_topo/FeederFlow.jl`
  (pinned to commit `12f129d`). Clone with submodules, or initialize them in an existing clone:

  ```bash
  git clone --recurse-submodules https://github.com/hoang-tn/ACOPF-Multiphase-Distribution-topo-soft.git
  # or, in an existing clone:
  git submodule update --init
  ```
* Then instantiate the project used by all commands in this README:

  ```bash
  julia --project=datagen_topo/FeederFlow.jl -e 'using Pkg; Pkg.instantiate()'
  ```
* The OPF solves use Ipopt with an HSL linear solver (`ma97`, configured in `datagen_topo/src/opf/model.jl`).
  The default `HSL_jll` artifact is only a placeholder; point `ACOPF_HSL_PATH` to a licensed `libhsl` (or delete the
  `hsllib`/`linear_solver` lines in `model.jl` to use Ipopt's default solver).

## Quick start (IEEE 13-bus, Penalty PF)

Run everything from the repository root. Generated data goes to `datagen_topo/datasets/<case>/` (git-ignored).

### 1. Test datasets (also write `network_config.json`)

Generate the **test** set first: it writes both the labelled HDF5 files and `network_config.json`, which the
training dataset generator does *not* produce but every Python script needs (`--data_json`).

```bash
julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss \
  --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5
```

Output in `datagen_topo/datasets/IEEE13Nodeckt/`:

* `IEEE13Nodeckt_test_feasible.h5` — inputs + Ipopt labels (required: used for validation and cost normalization),
* `IEEE13Nodeckt_test_infeasible.h5` — only if the solver reported infeasible samples (rare in the soft case),
* `network_config.json`.

Optional load/DER-regime test splits (evaluated as extra test variants). Pass `--save_config false` so the
shared `network_config.json` is not rewritten:

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

Shifted / out-of-range test sets, JSON configs and every flag: [`datagen_topo/gen_scripts/USAGE.md`](datagen_topo/gen_scripts/USAGE.md).

### 2. Training dataset (inputs only — no OPF solve)

```bash
julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_training_dataset.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --smin 1 --smax 50000
```

Writes `IEEE13Nodeckt_training.h5` (one sample per seed in `smin:smax`). Use `--smax 300` for a quick smoke test.

### 3. Train Penalty PF

```bash
bash scripts/example_scripts/train_example_13_penalty_pf.sh
```

(this script targets a Slurm/GPU cluster — it calls `module load cuda/…`; delete that line elsewhere), or call Python
directly. `--train_samples` is capped by the size of the training file:

```bash
python methods/penalty_pf/train.py \
  --train_h5 datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5 \
  --val_h5 datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5 \
  --test_h5_variants \
    datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5 \
    datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_infeasible.h5 \
    feasible_high_load_low_der=datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_high_load_low_der.h5 \
    feasible_low_load_high_der=datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_low_load_high_der.h5 \
  --data_json datagen_topo/datasets/IEEE13Nodeckt/network_config.json \
  --case_name IEEE13Nodeckt_penalty_pf \
  --results_dir results_draft \
  --epochs 20 --batch_size 64 --train_samples 50000 --eval_samples 2000 --test_samples 2000 \
  --hidden_size 512 --num_layers 4 --learning_rate 1e-4 --rho 1e5 \
  --power_flow_solver fixed_point --seed 42
```

**Dataset flags**

* `--val_h5`: in-epoch validation set. Its `meta/primal_objective_value` is also used to normalize the cost in the
  loss; if omitted, validation is skipped and the first `--test_h5_variants` entry is used for normalization.
* `--test_h5_variants`: post-training test sets. The first entry is the *reference* dataset (it must exist; it also
  initializes the data layout). Each entry is either a bare path — its name is inferred from the `*_test_<name>.h5`
  file stem (`…_test_feasible.h5` → `feasible`) — or `name=path`. **Missing optional entries are skipped with a
  warning**, so listing `…_test_infeasible.h5` is harmless when no infeasible samples were produced.

**Outputs** are written to `results_draft/<case_name>/seed_<seed>/`:

```
models/last/checkpoint_epoch_<E>/        # final model; `models/best/` with --save_best (needs --val_h5);
                                          #   `models/checkpoint_epoch_<E>/` every --save_every epochs
training/training_metrics.json            # per-epoch training metrics (+ training_summary.json,
                                          #   testing_metrics_last.json = last in-epoch validation)
testing/<variant>/testing_results_batch_<B>.json   # one folder per --test_h5_variants entry
```

### 4. Evaluate a saved checkpoint

```bash
python methods/penalty_pf/test.py \
  --train_h5 datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5 \
  --test_h5_variants datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5 \
  --data_json datagen_topo/datasets/IEEE13Nodeckt/network_config.json \
  --checkpoint_path results_draft/IEEE13Nodeckt_penalty_pf/seed_42/models/last \
  --case_name IEEE13Nodeckt_penalty_pf_eval --results_dir results_draft --seed 42
```

`--checkpoint_path` may point at a `checkpoint_epoch_*` directory or at its parent (the latest checkpoint is used).
`--train_h5` is needed to rebuild the input statistics.

## Other grids, methods and cluster runs

Ready-made training scripts in [`scripts/example_scripts/`](scripts/example_scripts/):

| Script                                | Case / method                                    |
| ------------------------------------- | ------------------------------------------------ |
| `train_example_13_penalty_pf.sh`      | 13-bus Penalty PF                                |
| `train_example_13_penalty_direct.sh`  | 13-bus Penalty Direct (fixed `rho` via `--rho_rate 0`) |
| `train_example_123_penalty_pf.sh`     | 123-bus Penalty PF                               |
| `train_example_240_penalty_pf.sh`     | 240-bus Penalty PF                               |
| `train_example_906_penalty_pf.sh`     | 906-bus Penalty PF                               |
| `train_example_8500_penalty_pf.sh`    | 8500-bus Penalty PF                              |
| `train_example_13_penalty_pf_test.sh` | Penalty PF on the `13_bus_test` feeder variant   |

To use another feeder, generate its data with the Julia commands above, swapping the `.dss` file
(`123_bus/IEEE123Master.dss`, `240_bus/240bus_Master.dss`, `906_bus/906bus_Master.dss`,
`IEEE8500/8500_Master.dss`). Dataset folders are named after the OpenDSS circuit, e.g. `datagen_topo/datasets/240bus_Master/`.

* **Linear OPF baseline**: `python methods/linear_opf/test.py --data_json … --test_h5 …` (see
  `scripts/run_linear_opf_all_grids.sh` for the sharded Slurm workflow over all grids).
* **Dataset generation on Slurm**: [`datagen_topo/gen_scripts/cluster/README.md`](datagen_topo/gen_scripts/cluster/README.md).
* **Training sweeps on Slurm** (rho sweeps, GCN, regime ablation): [`scripts/slurm_runs/README.md`](scripts/slurm_runs/README.md).

  ```bash
  python scripts/slurm_runs/train_jobs.py generate --config scripts/slurm_runs/train_jobs_penalty_pf.yaml
  sbatch scripts/slurm_runs/penalty_pf/submit_small.sh
  ```

## Dataset format

All files live in `datagen_topo/datasets/<case_name>/`.

| File                              | Role                                                                                         |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| `network_config.json`             | Bus/branch/DER indexing, limits, base `branch_status`, switch maps (written by the test generator) |
| `<case>_training.h5`              | Training **inputs only** (no OPF solve)                                                      |
| `<case>_test_feasible.h5`         | Test samples with Ipopt labels (`LOCALLY_SOLVED` / `ALMOST_LOCALLY_SOLVED`)                  |
| `<case>_test_infeasible.h5`       | Infeasible samples, if any (normally absent in the soft case)                                |
| `<case>_test_feasible_<suffix>.h5`| Regime splits (`--output_suffix`, e.g. `low_load_high_der`)                                   |
| `<case>_test_feasible_shifted*.h5`| Distribution-shifted test sets (`--variant shifted`)                                         |

**Training file** (`*_training.h5`):

| Group       | Fields                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------- |
| `input/`    | `seed`, `regime_id`, `pd`, `qd`, `pder_avai`, `changing_switch_status`, `is_radial`, `obj_weights` |
| `metadata/` | Dimensions, regime schedule, topology-sampler settings                                             |

**Test files** (`*_test_*.h5`):

| Group     | Fields                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------- |
| `input/`  | `seed`, `scenario_index`, `pd`, `qd`, `pder_avai`, `changing_switch_status`, `is_radial`, `obj_weights` |
| `primal/` | `vr`, `vi`, `pder`, `qder`, `psub`, `qsub`, `psub_phase`, `qsub_phase` (OPF solution)                   |
| `meta/`   | Ipopt termination status, timings, `primal_objective_value` (the **soft** objective, penalties included) |

`obj_weights` has five columns `[w_cost, w_curtail, ρ_V, ρ_I, ρ_θ]`. `OPFData` (`core/grids/data.py`) expands the
compact `changing_switch_status` to the full branch status using `network_config.json`.

## Key hyperparameters

| Parameter           | Typical       | Notes                                                                                   |
| ------------------- | ------------- | --------------------------------------------------------------------------------------- |
| `--rho`             | `1e4`–`1e6`   | Penalty PF: weight on the squared **hard**-constraint violation (script default `1e-3` is a placeholder). Examples use `1e5`. |
| `--rho_eq_init`, `--rho_ineq_init`, `--rho_rate`, `--rho_decay_rate`, `--violation_threshold`, `--rho_max` | `100`, `100`, `0`, `1`, `1e-3`, `1e3` | Penalty Direct: initial weights and adaptation. With `--rho_rate 0 --rho_decay_rate 1` the weights stay fixed (as in the examples and sweeps). |
| `--model_type`      | `mlp`         | Penalty PF: `mlp`, `gcn`, `modified_gcn`                                                |
| `--include_topo_nn_input` / `--no-include_topo_nn_input` | on | Feed `changing_switch_status` to the MLP (turn off for the no-topology ablation) |
| `--topology_backend`| `woodbury`    | `woodbury` (default) or `stamp` (re-stamp + dense inverse reference)                    |
| `--woodbury_ir_steps` | `auto`      | Woodbury iterative-refinement steps `auto` / `0` / `1` / `2` (tuned values: 906/8500 → 0, 13/123 → 1, 240 → 2) |
| `--woodbury_batch_mode` | `vmap`    | `vmap` (fast, more VRAM) or `map` (`lax.map`, less VRAM)                                |
| `--power_flow_solver` | `fixed_point` | `fixed_point`, `newton_dense`, `newton_gmres`, `newton_gmres_explicit`                |

## Repository layout

```
├── core/                      # Shared JAX library (import from the defining submodule)
│   ├── grids/                 # OPFData/NetworkParams, power flow, metrics (soft cost + hard violations),
│   │                          #   DER projection, linearized model + CVXPY linear OPF
│   ├── nn/                    # MLP / GCN models, compact input layout
│   ├── topology/              # Woodbury (default) and stamp backends, batch precompute
│   └── utils/                 # types, loss/cost normalization, datasets, workflow, checkpoints, CLI
├── methods/
│   ├── penalty_pf/            # NN → projection → PF (train.py, test.py, trainer.py, reporting.py)
│   ├── penalty_direct/        # NN predicts controls + voltages
│   └── linear_opf/            # Linear OPF baseline (test-only, sharding + merge helpers)
├── datagen_topo/
│   ├── FeederFlow.jl/         # Julia power-flow package (git submodule; see Installation)
│   ├── src/                   # Soft-OPF model, samplers, topology graph, HDF5 writers
│   ├── gen_scripts/           # create_{training,testing}_dataset.jl, JSON configs, USAGE.md, cluster/
│   ├── grid_data/             # OpenDSS feeders
│   └── datasets/              # Generated HDF5 + network_config.json (git-ignored)
├── scripts/
│   ├── example_scripts/       # Ready-made per-grid training commands
│   ├── slurm_runs/            # YAML job matrices + train_jobs.py launcher
│   └── run_linear_opf_all_grids.sh
├── analysis/                  # Standalone benchmarks (Woodbury vs dense, PF differentiation)
├── tests/                     # pytest (Python) and Julia test files
├── environment.yml            # Conda env `new_opt`
└── pyproject.toml             # Ruff configuration
```

Import style: `from core.grids.data import OPFData` (defining submodule), not package-root re-exports.

## Tests

```bash
conda activate new_opt
pytest tests/ -q
pytest tests/test_power_flow_solvers.py tests/test_linear_model_math.py -q   # a subset
```

Most pytest files read generated data from `datagen_topo/datasets/{IEEE13Nodeckt,IEEE123Master,240bus_Master,906bus_Master}/`
(`*_test_feasible.h5`, `*_training.h5`, `network_config.json`), so generate those first (the 13-bus quick start covers
`IEEE13Nodeckt`). Soft-constraint coverage: `tests/test_soft_cost_parity.py` (Python soft objective = Julia objective;
soft limits excluded from violations) and `tests/test_linear_opf_solver.py` (soft rows of the linear OPF).

Julia tests need FeederFlow.jl and Ipopt (+ HSL) and are run one file at a time:

```bash
julia --project=datagen_topo/FeederFlow.jl tests/test_soft_constraints.jl   # exact penalty, KKT, complementarity
```

## Analysis scripts

| Path                           | Role                                                          |
| ------------------------------ | ------------------------------------------------------------- |
| `analysis/woodbury_vs_direct/` | Woodbury (SMW + iterative refinement) vs dense solve          |
| `analysis/differentiation/`    | Power-flow autodiff: full unroll / truncated / implicit adjoint |

## Notes

The codebase is under active development; expect changes.

**When is Woodbury faster?** The update builds rank-`r` factors (`r = n_changing_switches × 3 × max_switch_terms`) and
solves a dense `r × r` system, so it beats a direct `O(n³)` solve only when `r ≪ n_net`. On the 13-bus feeder
(`n_net = 38`, `r = 18`) the overhead dominates and direct inversion is faster; Woodbury pays off on the large
feeders (240 / 906 / 8500) where `n_net` grows while `r` stays small.
