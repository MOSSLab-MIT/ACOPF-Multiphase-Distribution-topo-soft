# Power-Flow Differentiation Benchmark

Compare **autodiff through AC power flow** w.r.t. net injection `s` on isolated PF
(no neural network). All code lives under this folder; `core/` is imported
read-only.

## Methods

| Key | Forward | Backward | AD mode (dense Jacobian) |
|-----|---------|----------|--------------------------|
| `full` | `solve_power_flow_fixed_point_full` | Full unroll through all `maxiter` iterations (reference Jacobian) | `jacfwd` |
| `truncated_diffiter_{k}` | `solve_power_flow_fixed_point(..., diffiter=k)` | Truncated unroll: only the last `k` FP steps receive gradients | `jacfwd` |
| `implicit` | FP forward with `diffiter=0` | Local GMRES implicit adjoint at the converged Z-bus root (`implicit_solver.py`) | `jacrev` |

## Metrics

- **`error_max`**: max elementwise `|J_method - J_full|` (primary)
- **`error_mean`**, **`error_rel_fro`**: optional aggregate error norms
- **Timing**: warmup, then timed dense `J_method` per batch (reference `J_full` for error checks is untimed in the comparison loop)
- **Memory**: host RSS peak and JAX GPU peak (when available)
- **`benchmark_mode`**: `full_reference` for the timed `full` run; `dense_jacobian` for all other methods

Before benchmarking, the driver checks the forward fixed-point residual
`||v - G(v,s)||` and warns if it is large.

## Grid presets (`--grid`)

| Grid | Dataset |
|------|---------|
| `13` | `datagen_topo/datasets/IEEE13Nodeckt/` |
| `123` | `datagen_topo/datasets/IEEE123Master/` |
| `906` | `datagen_topo/datasets/906bus_Master/` |
| `8500` | `datagen_topo/datasets/8500_Master/` |

## Recommended settings

| Grid | `--samples` | `--batch-size` | Notes |
|------|-------------|----------------|-------|
| 13, 123 | 128–256 | 4 | Fast; Jacobian is small |
| 906 | 256 | 4 | Large network; keep samples on host, transfer per batch (~128G host RAM) |
| 8500 | 4 | 1 | Dense Jacobian may OOM; reduce `--batch-size` manually |

Reduce `--batch-size` if the benchmark runs out of GPU memory. `run_all.sh` currently
runs grid `906` with `--samples 256 --batch-size 4` (requires ~128G host memory via Slurm).

## Run tests

```bash
conda run -n new_opt pytest analysis/differentiation/test_compare_jacobians.py -v
```

## Run benchmark (single grid)

```bash
conda run -n new_opt python analysis/differentiation/compare_jacobians.py \
  --grid 13 --samples 256 --batch-size 4 \
  --maxiter 12 --diffiters 1 2 3 4 \
  --out analysis/differentiation/results/results_13.json
```

## Run all grids

```bash
bash analysis/differentiation/run_all.sh
```

Results are written to `analysis/differentiation/results/results_{grid}.json`.

## Slurm

```bash
sbatch analysis/differentiation/run_slurm.sh
```

## Expected outcomes

All methods build a **dense** Jacobian w.r.t. `s` for error comparison against `full`.

- **13 / 123**: `implicit` matches `full` (`error_max ~ 1e-10`). Truncated error
  shrinks as `k → maxiter`, but small `k` can still show large bias (e.g. `diffiter=1`
  often has `error_max ~ 0.1`).
- **906** (`results_906.json`, 256 samples, `batch_size=4`):
  - `implicit` is accurate when forward converges (`error_max ~ 1e-13`).
  - Small `diffiter` shows large Jacobian bias (`diffiter=1`: `error_max ~ 0.13`;
    `diffiter=3`: `error_max ~ 8e-4`).
  - **Speed**: truncated unroll is much faster than `full` (roughly 4–12× for
    `diffiter` 1–3). On this dense-Jacobian benchmark, `implicit` is **slower** than
    `full` (~2× at 906-bus), because `full`/`truncated` use forward-mode `jacfwd`
    while `implicit` uses reverse-mode `jacrev` to materialize the full matrix.
  - **Memory**: ~8.8 GB GPU peak at `batch_size=4` (no OOM observed).

### Accuracy vs speed (906-bus snapshot)

| Method | `error_max` | Total time (s) |
|--------|-------------|----------------|
| `full` | 0 (reference) | 136 |
| `truncated_diffiter_1` | 0.13 | 11 |
| `truncated_diffiter_2` | 0.011 | 23 |
| `truncated_diffiter_3` | 8e-4 | 34 |
| `implicit` | 3e-13 | 273 |

Use **truncated unroll** when a biased-but-fast Jacobian is acceptable in production;
use **implicit** when you need reference-level accuracy without unrolling all FP steps
(e.g. vector-Jacobian products or when only a few columns are needed). For a full dense
Jacobian on large networks, **full unroll with `jacfwd` is currently faster** than the
implicit adjoint path in this driver.
