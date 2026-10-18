# SLFS vs Voltage Least-Squares Benchmark

Benchmark-only comparison of **SLFS inference** versus a **JAXopt voltage least-squares** solver on perturbed OPF target DER controls. No neural-network training or checkpoints are involved.

Entry point: [`benchmark.py`](benchmark.py)

## What it measures

Both solvers receive the same inputs:

1. Start from OPF target DER controls `(pder_target, qder_target)` in the test HDF5.
2. Add deterministic relative perturbations (default: 1%, 5%, 10%).
3. Project perturbed controls to the DER feasible set (box, unit active-power cap, inverter disk).
4. Run power flow from the projected controls to get a shared initial voltage `v0`.

Then:

| Solver | Variable | Initial point |
|--------|----------|---------------|
| **SLFS** | DER controls `(pder, qder)` | projected perturbed controls, warm-started at `v0` |
| **JAXopt LM** | non-slack voltage `[vr; vi]` | `v0` from the same PF step |

**Headline timing** is solver/refinement only (ms per sample). Topology inverse, perturbation/PF prep, and post-solve quality metrics are reported separately in the JSON output.

## Prerequisites

- Conda environment: `new_opt`
- `jaxopt` installed (`from jaxopt import LevenbergMarquardt` must work)
- Test HDF5 + `network_config.json` under `datagen_topo/datasets/`

## Quick start

From the repo root:

```bash
conda run -n new_opt python analysis/slfs_vs_voltage_ls/benchmark.py --grid 13
```

### Grid presets

Use `--grid` with one of:

| `--grid` | Case directory |
|----------|----------------|
| `13` | `datagen_topo/datasets/IEEE13Nodeckt/` |
| `123` | `datagen_topo/datasets/IEEE123Master/` |
| `240` | `datagen_topo/datasets/240bus_Master/` |
| `906` | `datagen_topo/datasets/906bus_Master/` |
| `8500` | `datagen_topo/datasets/8500_Master/` |

### Explicit dataset paths

Instead of `--grid`, pass all three paths:

```bash
conda run -n new_opt python analysis/slfs_vs_voltage_ls/benchmark.py \
  --train_h5 datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5 \
  --test_h5 datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5 \
  --data_json datagen_topo/datasets/IEEE13Nodeckt/network_config.json
```

`--train_h5` is required for `OPFData` session setup; timing uses `--test_h5` batches.

## Common options

```bash
conda run -n new_opt python analysis/slfs_vs_voltage_ls/benchmark.py \
  --grid 123 \
  --batch_size 4 \
  --test_samples 32 \
  --perturb_scales 0.01 0.05 0.10 \
  --lm_maxiter 30 \
  --fs_maxiter 100 \
  --lin_steps 2 \
  --out analysis/slfs_vs_voltage_ls/results_123.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | `8` | TensorFlow batch size for test HDF5 streaming |
| `--test_samples` | `64` | Max test samples (`None` = full test set; not exposed as CLI default) |
| `--perturb_scales` | `0.01 0.05 0.10` | Relative perturbation levels on target DER controls |
| `--perturb_seed` | `1` | PRNG seed for perturbations |
| `--seed` | `1` | Global random seed |
| `--lin_steps` | `2` | SLFS linearization iterations |
| `--fs_maxiter` | `100` | PDHG max iterations per SLFS step |
| `--fs_tol` | `1e-6` | SLFS convergence tolerance |
| `--fs_rho` | `0.0` | SLFS proximity weight (0 for pure feasibility seeking) |
| `--fs_damp` | `1.0` | SLFS update damping |
| `--lm_maxiter` | `30` | JAXopt Levenberg–Marquardt max iterations |
| `--lm_tol` | `1e-6` | LM convergence tolerance |
| `--lm_damping` | `1e-6` | LM damping parameter |
| `--eq_weight` | `1.0` | Equality residual weight in LM objective |
| `--ineq_weight` | `1.0` | Inequality hinge residual weight in LM objective |
| `--power_flow_solver` | `fixed_point` (grid `240`: `newton_gmres`) | PF solver for input prep (`fixed_point`, `newton_dense`, `newton_gmres`, …) |
| `--out` | none | Optional JSON path for full results |

### Batch size notes

- `--batch_size` controls how many samples are processed per timed batch.
- Batches use `drop_remainder=True`, so trailing samples fewer than `batch_size` are dropped.
- Use smaller batches on large feeders (e.g. `--batch_size 2` for 240-bus) if GPU memory is tight.

### Power-flow init (grid 240)

The **240-bus** feeder (`906` network nodes) is stiff: `fixed_point` PF (7 iterations) does
not converge reliably for the shared initial voltage `v0`. When `--grid 240` (or
`240bus_Master` paths) is used **without** `--power_flow_solver`, the benchmark automatically
selects **`newton_gmres`**. Grids 13, 123, 906, and 8500 keep **`fixed_point`** unless
overridden. The chosen solver is written to JSON as `power_flow_solver`.

## Example: scalability sweep

Run each feeder separately and save JSON:

```bash
for g in 13 123 240; do
  bs=4
  if [ "$g" = 240 ]; then bs=2; fi
  conda run -n new_opt python analysis/slfs_vs_voltage_ls/benchmark.py \
    --grid "$g" \
    --batch_size "$bs" \
    --test_samples 32 \
    --out "analysis/slfs_vs_voltage_ls/results_${g}.json"
done
```

Grid `240` auto-selects `newton_gmres` for PF init (see **Power-flow init** above).

Large feeders (`906`, `8500`) may need smaller `--batch_size`, lower `--lm_maxiter`, or fewer `--test_samples` because voltage-dimension LM Jacobians are expensive.

### Slurm (grids 13, 123, 240, GPU)

From the repository root:

```bash
cd /path/to/ACOPF-Multiphase-Distribution-topo
sbatch scripts/run_slfs_vs_voltage_ls.sh 13
sbatch scripts/run_slfs_vs_voltage_ls.sh 123
sbatch scripts/run_slfs_vs_voltage_ls.sh 240
```

`#SBATCH` directives (GPU partition, 1 GPU, 24 CPUs, 64G, 4h) are in the script.
Logs: `logs/slfs_vs_voltage_ls/<jobid>.{out,err}`.
Output: `analysis/slfs_vs_voltage_ls/results_<grid>.json`.

## Reading the output

Console table (solver-only ms/sample):

```
   scale     slfs_ms       lm_ms   lm/slfs    slfs_res      lm_res     slfs_eq       lm_eq   slfs_ineq     lm_ineq
------------------------------------------------------------------------------------------
   0.010       1.425     127.552    89.505   1.429e-05   2.492e-03   2.494e-13   2.385e-05
```

| Column | Meaning |
|--------|---------|
| `slfs_ms` / `lm_ms` | Average solver time per sample (excludes topo inverse & input prep) |
| `lm/slfs` | Ratio `lm_ms / slfs_ms` (>1 means SLFS is faster) |
| `slfs_res` / `lm_res` | Mean LM residual norm (solver objective; unchanged) |
| `slfs_eq` / `lm_eq` | Mean max Z-bus equality violation (`compute_ac_metrics_from_voltage`) |
| `slfs_ineq` / `lm_ineq` | Mean max inequality violation (same API) |

JSON (`--out`) additionally includes per-batch breakdown and separate averages for:

- `grid` / `power_flow_solver` — preset grid id and PF solver used for `v0` init
- `avg_inverse_time_per_sample` — topology `Y_LL_inv` precompute
- `avg_prep_time_per_sample` — perturbation, projection, PF init
- `avg_slfs_solver_time_per_sample` / `avg_lm_solver_time_per_sample` — headline solver times
- `avg_metrics_time_per_sample` — post-solve quality evaluation

## Warmup protocol

Before timed batches, the runner calls `_warmup_benchmark_solvers()`:

- One warmup batch at the configured `--batch_size`
- All `--perturb_scales` compiled
- SLFS (`_run_slfs_jit`) and LM (`_LmBatchRunner`) kernels warmed with `block_until_ready()`

Timed batches should therefore exclude JIT compile from headline solver columns. First-batch compile may still appear in `prep_time` when a new perturbation scale is seen if warmup and timed loops diverge; the warmup loop covers all scales explicitly.

## Limitations

- Voltage-to-DER recovery requires unique `der_node` mapping per DER phase; duplicated nodes raise an error.
- JAXopt LM optimizes in voltage space; on large feeders the cost grows with `n_net` and LM iteration count.
- This is a timing/quality benchmark, not an OPF training pipeline.
