# Slurm training job configs

YAML-driven job matrices for batch training on Slurm. Each config expands into
`methods × grids × seeds` jobs, generates `jobs.jsonl` + `submit.sh`, and runs
training entrypoints under `methods/*/train.py`.

**One YAML per method family / ablation:**

| Method | Config | Output dir | Results |
|--------|--------|------------|---------|
| Penalty PF (rho sweep) | [`train_jobs_penalty_pf.yaml`](train_jobs_penalty_pf.yaml) | `penalty_pf/` | `results/penalty_pf/` |
| Penalty PF (GCN) | [`train_jobs_penalty_pf_gcn.yaml`](train_jobs_penalty_pf_gcn.yaml) | `penalty_pf_gcn/` | `results/penalty_pf_gcn/` |
| Penalty direct (rho sweep) | [`train_jobs_penalty_direct.yaml`](train_jobs_penalty_direct.yaml) | `penalty_direct/` | `results/penalty_direct/` |
| Penalty PF regime ablation | [`train_jobs_regime_ablation_penalty_pf.yaml`](train_jobs_regime_ablation_penalty_pf.yaml) | `penalty_pf_regime_ablation/` | `results/penalty_pf_regime_ablation/` |

All commands assume the **repository root** and the `new_opt` conda environment.

## Quick start

Generate and submit each method separately (`group_by_resource: true` emits one
`submit_<profile>.sh` per resource profile):

```bash
conda activate new_opt
cd /path/to/ACOPF-Multiphase-Distribution-topo

# Penalty PF (rho sweep + no-topo ablation)
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_penalty_pf.yaml
sbatch scripts/slurm_runs/penalty_pf/submit_small.sh
sbatch scripts/slurm_runs/penalty_pf/submit_medium.sh
sbatch scripts/slurm_runs/penalty_pf/submit_large.sh

# Penalty direct (rho_eq = rho_ineq sweep: 10, 100, 500, 1000)
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_penalty_direct.yaml
sbatch scripts/slurm_runs/penalty_direct/submit_small.sh

# Penalty PF GCN (rho=1e5)
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_penalty_pf_gcn.yaml
sbatch scripts/slurm_runs/penalty_pf_gcn/submit_small.sh
```

Preview jobs without generating files:

```bash
python scripts/slurm_runs/train_jobs.py list \
  --config scripts/slurm_runs/train_jobs_penalty_pf.yaml
```

Dry-run one job locally:

```bash
python scripts/slurm_runs/train_jobs.py run \
  --jobs scripts/slurm_runs/penalty_pf/jobs.jsonl \
  --method penalty_pf_rho1e5 --grid 13 --seed 1 --dry-run
```

CLI entrypoint: [`train_jobs.py`](train_jobs.py).

After `generate`, artifacts land in `scripts/slurm_runs/<config.name>/`:

- `jobs.jsonl` — one JSON record per array task (includes full `command`)
- `submit_<profile>.sh` — Slurm array script(s)
- `README.md` — auto-generated summary

Regenerate whenever you edit a YAML config.

## Results layout

```
{results_dir}/{case_name}/seed_{seed}/
```

Examples:

- `results/penalty_pf/IEEE13Nodeckt_penalty_pf_rho1e5/seed_1/` — Penalty PF, rho=1e5
- `results/penalty_pf/IEEE13Nodeckt_penalty_pf_rho1e5_no_topo/seed_1/` — Penalty PF without topo NN input
- `results/penalty_pf_gcn/IEEE13Nodeckt_penalty_pf_rho1e5_gcn/seed_1/` — Penalty PF GCN, rho=1e5
- `results/penalty_direct/IEEE13Nodeckt_penalty_direct_rho500/seed_1/` — Penalty direct, rho=500

Different hyperparameter values use different `case_suffix` values so checkpoints
never collide.

Slurm logs: `logs/training/<config.name>/`.

## Config summaries

### `train_jobs_penalty_pf.yaml` — 60 jobs (4 × 5 × 3)

| Method key | case_suffix | `rho` |
|------------|-------------|-------|
| `penalty_pf_rho1e4` | `penalty_pf_rho1e4` | 1e4 |
| `penalty_pf_rho1e5` | `penalty_pf_rho1e5` | 1e5 |
| `penalty_pf_rho1e5_no_topo` | `penalty_pf_rho1e5_no_topo` | 1e5, no topo NN input |
| `penalty_pf_rho1e6` | `penalty_pf_rho1e6` | 1e6 |

(`penalty_pf_rho1e3` is defined as a profile but not in the default matrix.)

### `train_jobs_penalty_pf_gcn.yaml` — 12 jobs (1 × 4 × 3)

| Method key | case_suffix | Notes |
|------------|-------------|-------|
| `penalty_pf_rho1e5_gcn` | `penalty_pf_rho1e5_gcn` | `model_type=gcn`, `rho=1e5` |

Grids: 13, 123, 240, 906.

### `train_jobs_penalty_direct.yaml` — 60 jobs (4 × 5 × 3)

| Method key | case_suffix | `rho_eq_init` = `rho_ineq_init` |
|------------|-------------|----------------------------------|
| `penalty_direct_rho10` | `penalty_direct_rho10` | 10 |
| `penalty_direct_rho100` | `penalty_direct_rho100` | 100 |
| `penalty_direct_rho500` | `penalty_direct_rho500` | 500 |
| `penalty_direct_rho1000` | `penalty_direct_rho1000` | 1000 |

All variants use `rho_rate=0`, `rho_decay_rate=1` (penalty fixed at init).

## YAML structure

```yaml
name: penalty_pf              # → scripts/slurm_runs/penalty_pf/

matrix:
  methods: [penalty_pf_rho1e5, penalty_pf_rho1e5_no_topo]
  grids: ["13", "123"]
  seeds: [1, 2, 3, 4, 5]

results_dir: results
max_parallel: 8

method_profiles:
  penalty_pf_rho1e5:
    entrypoint: methods/penalty_pf/train.py
    case_suffix: penalty_pf_rho1e5
    args:
      rho: 1.0e5

grid_profiles:
  "13":
    case_dir: IEEE13Nodeckt
    resource_profile: small
    val_h5: datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5
    test_variants: [...]   # post-training evaluation only
    method_overrides:
      penalty_pf_rho1e5:
        batch_size: 32
```

**Arg merge order** (later wins): grid defaults → `method_profiles.args` → `method_overrides`.

`val_h5` is passed as `--val_h5` for in-epoch validation. Cost normalization uses
`val_h5` when set, otherwise the reference test HDF5 (first `test_variants` entry).
`test_variants` maps to `--test_h5_variants` for post-training evaluation only.

## Adding variants

Each distinct hyperparameter or ablation needs its own method profile with a
unique `case_suffix`. See the rho sweep configs for the
pattern. Prefer a dedicated YAML rather than merging sweeps into one file.

## Regime-weight ablation (W0–W3)

Train Penalty PF on suffixed training HDF5 generated by
`datagen_topo/gen_scripts/cluster/run_datagen.sh` (see its README). The default
unsuffixed `<case>_training.h5` (W0, all-base mix) is **not** re-trained in the
ablation matrix; compare against existing default training runs.

| Config | Output dir | Results |
|--------|------------|---------|
| [`train_jobs_regime_ablation_penalty_pf.yaml`](train_jobs_regime_ablation_penalty_pf.yaml) | `penalty_pf_regime_ablation/` | `results/penalty_pf_regime_ablation/` |

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

- Penalty PF: `results/penalty_pf/IEEE13Nodeckt_penalty_pf_rho1e5/seed_1/`

Each ablation job uses `matrix.train_variants` and `ablation_case_token` in the
method profile. Example `case_name`:
`IEEE13Nodeckt_penalty_pf_060_035_005`.

```bash
# Verify suffixed HDF5 exist, then generate
python scripts/slurm_runs/train_jobs.py generate \
  --config scripts/slurm_runs/train_jobs_regime_ablation_penalty_pf.yaml \
  --check-train-h5

python scripts/slurm_runs/train_jobs.py list \
  --config scripts/slurm_runs/train_jobs_regime_ablation_penalty_pf.yaml

# Dry-run one job
python scripts/slurm_runs/train_jobs.py run \
  --jobs scripts/slurm_runs/penalty_pf_regime_ablation/jobs.jsonl \
  --index 0 --dry-run

# Submit both resource groups (13/123 small, 240 medium). Do not override `--array`.
sbatch scripts/slurm_runs/penalty_pf_regime_ablation/submit_small.sh
sbatch scripts/slurm_runs/penalty_pf_regime_ablation/submit_medium.sh

python scripts/slurm_runs/train_jobs.py verify \
  --jobs-dir scripts/slurm_runs/penalty_pf_regime_ablation
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
