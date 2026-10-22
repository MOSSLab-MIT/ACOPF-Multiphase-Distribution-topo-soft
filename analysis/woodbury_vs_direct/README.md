# Woodbury vs Direct Inverse Benchmark

Compare **production JAX Woodbury** (slim stamping + `lax.map` batch JIT) against
**dense `jnp.linalg.solve(Y_LL, I)`** on identical stamped admittances. Each grid
is timed and accuracy-checked at **IR steps = 0, 1, 2** (iterative refinement)
to expose the effect of SMW-IR corrections on residual and runtime. Output is
structured JSON per grid for downstream result tables.

These benchmarks concern only the topology update (`Y_LL⁻¹`), so they are independent of the soft-constraint
formulation. They need the generated `*_training.h5` / `*_test_feasible.h5` and `network_config.json` of each grid
(see the [main README](../../README.md)).

## Comparison

| Path | Implementation |
|------|----------------|
| Woodbury | `woodbury_topology_matrices_slim_branchless(ir_steps=N)` batched with `lax.map`, for `N` in `{0, 1, 2}` |
| Direct | `woodbury_topology_matrices_direct` batched with `lax.map` (IR-independent reference) |

Both paths stamp the same `Y_LL` before inversion. This is **not** the legacy
`woodbury_topology_matrices` full-`Y_scaled` experiment path. The direct path is
timed once and reused as the accuracy reference for every IR step.

## Settings

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--batch-size` | 8 | `lax.map` only (no `vmap`) |
| Status suite | `changed_only` | Excludes base compact status; cycles if fewer than 8 unique changed samples |
| IR sweep | `0, 1, 2` | Fixed SMW-IR step counts; one JIT kernel per count (XLA unrolls at trace time) |
| `--warmup` | 3 | JIT compile included in warmup |
| `--repeats` | 10 | Timed repeats after warmup |
| `dtype` | float64 | `jax_enable_x64=True` |

## Grid presets (`--grid`)

| Grid | Dataset |
|------|---------|
| `13` | `datagen_topo/datasets/IEEE13Nodeckt/` |
| `123` | `datagen_topo/datasets/IEEE123Master/` |
| `240` | `datagen_topo/datasets/240bus_Master/` |
| `906` | `datagen_topo/datasets/906bus_Master/` |
| `8500` | `datagen_topo/datasets/8500_Master/` |

## Run tests

```bash
conda run -n new_opt pytest analysis/woodbury_vs_direct/test_benchmark.py -v
```

## Run benchmark (single grid)

```bash
conda activate new_opt
python analysis/woodbury_vs_direct/benchmark.py \
  --grid 906 --batch-size 8 \
  --out analysis/woodbury_vs_direct/results/results_906.json
```

## Run all grids (GPU)

```bash
bash analysis/woodbury_vs_direct/run_all.sh
```

Override grids: `bash analysis/woodbury_vs_direct/run_all.sh -- 13 906`

First grid run includes XLA compile time (covered by warmup). Running all five grids
typically takes a few minutes on GPU; the larger feeders (906, 8500) dominate.

## JSON schema

One file per grid, e.g. `results/results_906.json`:

```json
{
  "benchmark": "woodbury_vs_direct_inverse",
  "batch_size": 8,
  "kernel": "lax.map",
  "status_suite": "changed_only",
  "ir_steps_sweep": [0, 1, 2],
  "inverse_residual_tol": 1e-6,
  "warmup": 3,
  "repeats": 10,
  "device": "cuda:0",
  "dtype": "float64",
  "cases": {
    "906bus_Master": {
      "n_net": 2718,
      "n_changing_switches": 8,
      "r_max": 72,
      "n_unique_changed_statuses": 10,
      "batch_cycled": false,
      "woodbury": {
        "0": {
          "time_per_batch_ms": 27.18,
          "time_std_ms": 0.09,
          "time_per_sample_ms": 3.40,
          "speedup": 5.22,
          "max_Y_LL_inv_diff": 2.09e-11,
          "max_Y_LL_inv_diff_well_cond": 2.09e-11,
          "inverse_residual_max": 1.50e-10,
          "max_Y_LL_cond": 1.24e8,
          "elementwise_diff_trusted": true,
          "n_high_cond_samples": 0,
          "elementwise_checked_samples": 8
        },
        "1": { "time_per_batch_ms": 29.10, "speedup": 4.87, "inverse_residual_max": 1.20e-13, "...": "..." },
        "2": { "time_per_batch_ms": 31.05, "speedup": 4.57, "inverse_residual_max": 9.80e-14, "...": "..." }
      },
      "direct": {
        "time_per_batch_ms": 141.95,
        "time_std_ms": 0.07,
        "time_per_sample_ms": 17.74
      }
    }
  }
}
```

Skipped runs (missing HDF5 or no changed topologies) set top-level `"skipped": true`
and `"reason"`.

`woodbury` is keyed by IR-step string (`"0"`/`"1"`/`"2"`); each entry carries its
own timing block, `speedup`, and accuracy metrics. `direct` is a single timing
block (IR-independent). The per-IR `speedup` is `direct.time_per_batch_ms /
woodbury.<ir>.time_per_batch_ms`.

### Suggested result-table columns

| Column | JSON field |
|--------|------------|
| Grid | filename or `cases.*.n_net` |
| n_net | `cases.*.n_net` |
| IR Steps | `woodbury.<ir>` key |
| Woodbury (ms/sample) | `woodbury.<ir>.time_per_sample_ms` |
| Direct (ms/sample) | `direct.time_per_sample_ms` |
| Speedup | `woodbury.<ir>.speedup` |
| Accuracy (informational) | `woodbury.<ir>.max_Y_LL_inv_diff`, `woodbury.<ir>.inverse_residual_max` |

`max_Y_LL_inv_diff` is always the raw max `|Y_LL_inv_wb - Y_LL_inv_direct|` over
the batch, even when `cond(Y_LL)` is huge (123/240). In those cases the inverse is
non-unique in floating point, so the raw diff can be large while Woodbury is still
correct — check `elementwise_diff_trusted` and `max_Y_LL_cond`.

When `elementwise_diff_trusted` is `false`, prefer `inverse_residual_max` and
`max_Y_LL_inv_diff_well_cond` over the raw diff for interpretation.

IEEE123 is a known ill-conditioned case: unregularized delta transformer at bus 610 yields
`cond(Y_LL) ~ 2e12` while both Woodbury and direct satisfy
`‖Y Y^{-1} - I‖ ~ 1e-7`.

### Effect of iterative refinement

The IR sweep is the point of this benchmark: `ir=0` is the unrefined Woodbury
inverse; `ir=1`/`ir=2` apply one/two SMW-IR corrections (see
`_woodbury_inverse_with_ir` in `core/topology/woodbury.py`). Per-grid "sufficient"
step counts (where `inverse_residual_max < inverse_residual_tol = 1e-6`) are
**not** pre-decided — the sweep records raw residuals so the table shows the
effect directly:

- **13 / 123 / 240**: `ir=0` is insufficient; `ir=1` (13/123) or `ir=2` (240)
  reaches the tolerance (matches `EXPECTED_IR_STEPS` in
  `tests/test_woodbury_ir_steps.py`).
- **906 / 8500**: `ir=0` is already exact (low-rank update is well-conditioned);
  further IR steps add time without improving the residual.

So the table shows that IR is necessary on the smaller, stiffer grids and a
no-op (with a small time cost) on the larger, well-conditioned ones.

## Expected crossover

From earlier benchmark runs:

- **IEEE13** (`n_net≈38`): Woodbury **slower** than direct (`speedup < 1`) — fixed
  overhead dominates on tiny matrices.
- **906-bus** (`n_net≈2718`): Woodbury **~2.5× faster** with `lax.map` on
  `changed_only` batches.

123-bus may have only one unique training status (batch cycles base-only work);
240-bus follows the same production path as pytest but is registered only in this
driver (not default `CASE_ROOTS`).

## Accuracy

Accuracy metrics are reported **per IR step** (under `woodbury.<ir>`):

- `max_Y_LL_inv_diff`: max `|Y_LL_inv_woodbury - Y_LL_inv_direct|` over **all**
  batch samples (always reported).
- `max_Y_LL_inv_diff_well_cond`: same metric restricted to samples with
  `cond(Y_LL) ≤ 1e12`; `null` when every sample exceeds the threshold.
- `max_Y_LL_cond`: max `cond(Y_LL)` over the timed batch.
- `elementwise_diff_trusted`: `true` only when every sample passed the cond gate.
- `inverse_residual_max`: max `‖Y_LL Y_LL_inv - I‖_∞` using stamped `Y_LL` from
  the direct path.

Pytest checks (not enforced by the benchmark driver): element-wise `< 1e-8`,
residual non-increasing across the IR sweep with at least one step `< 1e-6`.

## Timing fairness

Both methods are timed with the **same protocol**:

1. **JIT kernels only** — Woodbury uses
   `jax.jit(jax.lax.map(woodbury_topology_matrices_slim_branchless, ir_steps=N))`
   for each `N` in `{0, 1, 2}` (one compiled kernel per IR count); direct uses
   `jax.jit(jax.lax.map(woodbury_topology_matrices_direct))`, timed once. Host-side
   `normalize_compact_status_matrix` is excluded from the timer (negligible on
   large feeders; ~1 ms on 13-bus).
2. **Identical input** — pre-built `batch_j` (`jnp.int32` on device) passed to
   both functions; no per-call NumPy→JAX conversion inside the timed region.
3. **Same warmup / repeats** — each method gets its own `warmup` untimed calls
   then `repeats` timed calls (default 3 + 10).
4. **Synchronization** — `jax.block_until_ready` after every warmup and timed call.

Metadata fields `timing_scope`, `woodbury_timed_fn`, `direct_timed_fn`, and
`timed_input` in the JSON document this protocol.
