# Slurm training job configs

YAML-driven job matrices for batch training on Slurm. Each config expands into
`methods × grids × seeds` jobs, generates `jobs.jsonl` + `submit.sh`, and runs
training entrypoints under `methods/*/train.py`.

**One YAML per method family** (three configs total):

| Method | Config | Output dir | Results |
|--------|--------|------------|---------|
| SLFS/FSNet | [`train_jobs_slfs_fsnet.yaml`](train_jobs_slfs_fsnet.yaml) | `slfs_fsnet/` | `results/` |
| SLFS/FSNet (GCN) | [`train_jobs_slfs_fsnet_gcn.yaml`](train_jobs_slfs_fsnet_gcn.yaml) | `slfs_fsnet_gcn/` | `results/slfs_fsnet_gcn/` |
| Penalty FS | [`train_jobs_penalty_fs_rho_sweep.yaml`](train_jobs_penalty_fs_rho_sweep.yaml) | `penalty_fs_rho_sweep/` | `results_rho_sweep/` |
| Penalty FS (GCN) | [`train_jobs_penalty_fs_gcn.yaml`](train_jobs_penalty_fs_gcn.yaml) | `penalty_fs_gcn/` | `results/penalty_fs_gcn/` |
| Penalty direct | [`train_jobs_penalty_direct.yaml`](train_jobs_penalty_direct.yaml) | `penalty_direct/` | `results/` |
| Penalty direct (rho sweep) | [`train_jobs_penalty_direct_rho_sweep.yaml`](train_jobs_penalty_direct_rho_sweep.yaml) | `penalty_direct_rho_sweep/` | `results_penalty_direct_rho_sweep/` |

All commands assume the **repository root** and the `new_opt` conda environment.

## Quick start

Generate and submit each method separately:

```bash
conda activate new_opt
cd /path/to/ACOPF-Multiphase-Distribution-topo

# SLFS/FSNet (weight_dist sweep + no-topo ablation)
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_slfs_fsnet.yaml
sbatch scripts/slurm_runs/slfs_fsnet/submit.sh

# Penalty FS (rho sweep)
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_penalty_fs_rho_sweep.yaml
sbatch scripts/slurm_runs/penalty_fs_rho_sweep/submit.sh

# Penalty direct (fixed rho=10)
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_penalty_direct.yaml
sbatch scripts/slurm_runs/penalty_direct/submit.sh

# Penalty direct (rho_eq = rho_ineq sweep: 10, 100, 500, 1000)
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_penalty_direct_rho_sweep.yaml
sbatch scripts/slurm_runs/penalty_direct_rho_sweep/submit.sh

# SLFS/FSNet GCN (weight_dist=1, grids 13/123/240)
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_slfs_fsnet_gcn.yaml
sbatch scripts/slurm_runs/slfs_fsnet_gcn/submit.sh

# Penalty FS GCN (rho=1e5, grids 13/123/240)
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_penalty_fs_gcn.yaml
sbatch scripts/slurm_runs/penalty_fs_gcn/submit.sh
```

Preview jobs without generating files:

```bash
python scripts/slurm_runs/train_jobs.py list \
  --config scripts/slurm_runs/train_jobs_slfs_fsnet.yaml
```

Dry-run one job locally:

```bash
python scripts/slurm_runs/train_jobs.py run \
  --jobs scripts/slurm_runs/slfs_fsnet/jobs.jsonl \
  --method slfs_fsnet_wd0p01 --grid 13 --seed 1 --dry-run
```

CLI entrypoint: [`train_jobs.py`](train_jobs.py).

After `generate`, artifacts land in `scripts/slurm_runs/<config.name>/`:

- `jobs.jsonl` — one JSON record per array task (includes full `command`)
- `submit.sh` — Slurm array script
- `README.md` — auto-generated summary

Regenerate whenever you edit a YAML config.

## Results layout

```
{results_dir}/{case_name}/seed_{seed}/
```

Examples:

- `results/IEEE13Nodeckt_fsnet_wd0p01/seed_1/` — SLFS/FSNet, weight_dist=0.01
- `results/slfs_fsnet_gcn/IEEE13Nodeckt_fsnet_wd1_gcn/seed_1/` — SLFS/FSNet GCN, weight_dist=1
- `results/IEEE13Nodeckt_fsnet_no_topo/seed_1/` — SLFS/FSNet without topo NN input
- `results/penalty_fs_gcn/IEEE13Nodeckt_penalty_fs_rho1e5_gcn/seed_1/` — Penalty FS GCN, rho=1e5
- `results_rho_sweep/IEEE13Nodeckt_penalty_fs_rho1e5/seed_1/` — Penalty FS rho sweep
- `results/IEEE13Nodeckt_penalty_direct/seed_1/` — Penalty direct (rho=10, fixed)
- `results_penalty_direct_rho_sweep/IEEE13Nodeckt_penalty_direct_rho500/seed_1/` — Penalty direct rho sweep

Different hyperparameter values use different `case_suffix` values so checkpoints
never collide.

Slurm logs: `logs/training/<config.name>/`.

## Config summaries

### `train_jobs_slfs_fsnet.yaml` — 125 jobs (5 × 5 × 5)

| Method key | case_suffix | Notes |
|------------|-------------|-------|
| `slfs_fsnet_wd0p01` | `fsnet_wd0p01` | `weight_dist=0.01` |
| `slfs_fsnet_wd0p1` | `fsnet_wd0p1` | `weight_dist=0.1` |
| `slfs_fsnet_wd1` | `fsnet_wd1` | `weight_dist=1` |
| `slfs_fsnet_wd10` | `fsnet_wd10` | `weight_dist=10` |
| `slfs_fsnet_no_topo` | `fsnet_no_topo` | no topo NN input, `weight_dist=0.01` |

### `train_jobs_penalty_fs_rho_sweep.yaml` — 100 jobs (4 × 5 × 5)

| Method key | case_suffix | `rho` |
|------------|-------------|-------|
| `penalty_fs_rho1e3` | `penalty_fs_rho1e3` | 1e3 |
| `penalty_fs_rho1e4` | `penalty_fs_rho1e4` | 1e4 |
| `penalty_fs_rho1e5` | `penalty_fs_rho1e5` | 1e5 |
| `penalty_fs_rho1e6` | `penalty_fs_rho1e6` | 1e6 |

### `train_jobs_slfs_fsnet_gcn.yaml` — 9 jobs (1 × 3 × 3)

| Method key | case_suffix | Notes |
|------------|-------------|-------|
| `slfs_fsnet_wd1_gcn` | `fsnet_wd1_gcn` | `model_type=gcn`, `weight_dist=1` |

Grids: 13, 123, 240 only.

### `train_jobs_penalty_fs_gcn.yaml` — 9 jobs (1 × 3 × 3)

| Method key | case_suffix | Notes |
|------------|-------------|-------|
| `penalty_fs_rho1e5_gcn` | `penalty_fs_rho1e5_gcn` | `model_type=gcn`, `rho=1e5` |

Grids: 13, 123, 240 only.

### `train_jobs_penalty_direct.yaml` — 25 jobs (1 × 5 × 5)

Single method `penalty_direct` with `rho_eq_init=rho_ineq_init=10`, fixed penalty
(`rho_rate=0`, `rho_decay_rate=1`).

### `train_jobs_penalty_direct_rho_sweep.yaml` — 100 jobs (4 × 5 × 5)

| Method key | case_suffix | `rho_eq_init` = `rho_ineq_init` |
|------------|-------------|----------------------------------|
| `penalty_direct_rho10` | `penalty_direct_rho10` | 10 |
| `penalty_direct_rho100` | `penalty_direct_rho100` | 100 |
| `penalty_direct_rho500` | `penalty_direct_rho500` | 500 |
| `penalty_direct_rho1000` | `penalty_direct_rho1000` | 1000 |

All variants use `rho_rate=0`, `rho_decay_rate=1` (penalty fixed at init).

## YAML structure

```yaml
name: slfs_fsnet              # → scripts/slurm_runs/slfs_fsnet/

matrix:
  methods: [slfs_fsnet_wd0p01, slfs_fsnet_no_topo]
  grids: ["13", "123"]
  seeds: [1, 2, 3, 4, 5]

results_dir: results
max_parallel: 8

method_profiles:
  slfs_fsnet_wd0p01:
    entrypoint: methods/slfs_fsnet/train.py
    case_suffix: fsnet_wd0p01
    args:
      weight_dist: 0.01

grid_profiles:
  "13":
    case_dir: IEEE13Nodeckt
    resource_profile: small
    val_h5: datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5
    test_variants: [...]   # post-training evaluation only
    method_overrides:
      slfs_fsnet_wd0p01:
        batch_size: 32
```

**Arg merge order** (later wins): grid defaults → `method_profiles.args` → `method_overrides`.

`val_h5` is passed as `--val_h5` for in-epoch validation. Cost normalization uses
`val_h5` when set, otherwise the reference test HDF5 (first `test_variants` entry).
`test_variants` maps to `--test_h5_variants` for post-training evaluation only.

## Adding variants

Each distinct hyperparameter or ablation needs its own method profile with a
unique `case_suffix`. See the rho sweep and weight_dist sweep configs for the
pattern. Prefer a dedicated YAML rather than merging sweeps into one file.

## Regime-weight ablation (W0–W3)

Train SLFS/FSNet and Penalty FS on suffixed training HDF5 generated by
`datagen_topo/gen_scripts/cluster/run_training_regime_ablation.sh`. The default
unsuffixed `<case>_training.h5` (W0, all-base mix) is **not** re-trained in the
ablation matrix; compare against existing default training runs.

| Config | Output dir | Results |
|--------|------------|---------|
| [`train_jobs_regime_ablation_slfs_fsnet.yaml`](train_jobs_regime_ablation_slfs_fsnet.yaml) | `slfs_fsnet_regime_ablation/` | `results/slfs_fsnet_regime_ablation/` |
| [`train_jobs_regime_ablation_penalty_fs.yaml`](train_jobs_regime_ablation_penalty_fs.yaml) | `penalty_fs_regime_ablation/` | `results/penalty_fs_regime_ablation/` |

**Training HDF5 suffixes** (per grid, 12 files total):

| Label | Weights (base / high / low) | Load scale (base) | `train_h5` suffix |
|-------|-----------------------------|-------------------|-------------------|
| W0 (default, not in ablation) | 1.0 / 0.0 / 0.0 | [0.2, 1.5] | *(unsuffixed)* `<case>_training.h5` |
| W1 | 0.6 / 0.05 / 0.35 | [0.2, 1.5] | `060_005_035` |
| W2 | 0.6 / 0.35 / 0.05 | [0.2, 1.5] | `060_035_005` |
| W3 | 0.6 / 0.2 / 0.2 | [0.2, 1.5] | `060_020_020` |
| OOR load cap | 1.0 / 0.0 / 0.0 | [0.2, 0.8] | `l02_u08` |

Paths: `datagen_topo/datasets/<case>/<case>_training_<suffix>.h5`

**W0 baseline results** (for comparison):

- SLFS: `results/slfs_fsnet/IEEE13Nodeckt_fsnet_wd1/seed_1/`
- Penalty FS: `results/penalty_fs/IEEE13Nodeckt_penalty_fs_rho1e5/seed_1/`

Each ablation job uses `matrix.train_variants` and `ablation_case_token` in the
method profile. Example `case_name`:
`IEEE13Nodeckt_fsnet_060_035_005`.

```bash
# Verify suffixed HDF5 exist, then generate (12 jobs per YAML)
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_regime_ablation_slfs_fsnet.yaml \
  --check-train-h5

python scripts/slurm_runs/train_jobs.py list \
  --config scripts/slurm_runs/train_jobs_regime_ablation_slfs_fsnet.yaml

# Dry-run one job
python scripts/slurm_runs/train_jobs.py run \
  --jobs scripts/slurm_runs/slfs_fsnet_regime_ablation/jobs.jsonl \
  --index 0 --dry-run

# Submit both resource groups (13/123 small, 240 medium). Do not override `--array`.
sbatch scripts/slurm_runs/slfs_fsnet_regime_ablation/submit_small.sh
sbatch scripts/slurm_runs/slfs_fsnet_regime_ablation/submit_medium.sh

python scripts/slurm_runs/train_jobs.py verify \
  --jobs-dir scripts/slurm_runs/slfs_fsnet_regime_ablation

# Repeat for penalty FS
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_regime_ablation_penalty_fs.yaml \
  --check-train-h5
sbatch scripts/slurm_runs/penalty_fs_regime_ablation/submit_small.sh
sbatch scripts/slurm_runs/penalty_fs_regime_ablation/submit_medium.sh

python scripts/slurm_runs/train_jobs.py verify \
  --jobs-dir scripts/slurm_runs/penalty_fs_regime_ablation
```

Post-training evaluation runs automatically inside `train.py` via
`--test_h5_variants` (no separate `test.py` jobs). Filter a single variant
locally with `--train-suffix` on the `run` subcommand.

## Slurm resources

`resources` defines `small` / `medium` / `large` profiles. Each grid sets
`resource_profile`. Edit `partition`, `gpu_type`, etc. for your cluster.

Set `group_by_resource: true` to emit separate `submit_<profile>.sh` scripts.
Each generated script documents the `jobs.jsonl` indices it runs and rejects
stale `sbatch --array=...` overrides at job start. After `generate`, run
`train_jobs.py verify --jobs-dir <output_dir>` to confirm arrays match
`jobs.jsonl`.
