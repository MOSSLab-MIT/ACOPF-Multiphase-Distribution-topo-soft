# Batch-size inference time (penalty_fs)

Paper-grade timing study: how **penalty_fs deploy pipeline wall time per batch**
grows with batch size.

## Metric

**Primary:** mean wall time **per batch** for the full deploy pipeline:

`raw` (NN apply + DER projection + power flow) + `post_process_slfs`

This matches `fs.prediction_time` / `avg_prediction_time_per_batch` in
[`methods/penalty_fs/trainer.py`](../../methods/penalty_fs/trainer.py).

**Excluded from the primary metric:** topology inverse (`precompute_topology`),
quality-metric power flows after the timed window.

Incomplete last batches are dropped (`drop_remainder`) so every timed batch has
exactly size `B` (avoids recompilation from a ragged last batch).

## Protocol

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--batch-sizes` | `1 2 4 8 16 32 64` | Powers of two |
| `--test-samples` | `256` | Rounded down to a multiple of `max(batch_sizes)` |
| `--warmup` | `3` | JIT / XLA covered here |
| `--repeats` | `10` | Timed passes over the fixed sample pool |
| Clock | `time.perf_counter` | + `block_until_ready` |
| Device caches | cleared between batch sizes | Fair per-`B` compile |

Same feasible test samples are regrouped for every `B`.

## Grid presets (`--grid`)

| Grid | Case dir | Default checkpoint resolve |
|------|----------|----------------------------|
| `13` | `IEEE13Nodeckt` | `…/models/last/` (final train save) |
| `123` | `IEEE123Master` | … |
| `240` | `240bus_Master` | … |
| `906` | `906bus_Master` | … |
| `8500` | `8500_Master` | … |

Default method suffix: `penalty_fs_rho1e5` (main paper config). Checkpoint discovery
prefers ``models/last/``; if missing, falls back to the highest ``checkpoint_epoch_*``.

## Run (single grid)

```bash
conda activate new_opt
python analysis/batch_size_inference_time/benchmark.py \
  --grid 13 \
  --results-root results/penalty_fs \
  --case-suffix penalty_fs_rho1e5 \
  --seed 1 \
  --out analysis/batch_size_inference_time/results/results_13.json
```

Or pass an explicit checkpoint:

```bash
python analysis/batch_size_inference_time/benchmark.py \
  --grid 13 \
  --checkpoint-path results/penalty_fs/IEEE13Nodeckt_penalty_fs_rho1e5/seed_1/models/checkpoint_epoch_10 \
  --out analysis/batch_size_inference_time/results/results_13.json
```

## Run all grids

```bash
bash analysis/batch_size_inference_time/run_all.sh
```

Override grids: `bash analysis/batch_size_inference_time/run_all.sh -- 13 123`

## Slurm

Submit one GPU job per grid (array tasks `0–4` → `13, 123, 240, 906, 8500`):

```bash
sbatch analysis/batch_size_inference_time/run_slurm.sh
```

Single grid only (array index: `0=13`, `1=123`, `2=240`, `3=906`, `4=8500`):

```bash
sbatch --array=2 analysis/batch_size_inference_time/run_slurm.sh
```

After all array tasks complete, build the combined figure:

```bash
python analysis/batch_size_inference_time/plot.py \
  --results-dir analysis/batch_size_inference_time/results \
  --output analysis/batch_size_inference_time/figures/batch_size_inference_time.pdf
```

Optional env overrides (same as `run_all.sh`): `BSIT_RESULTS_ROOT`, `BSIT_CASE_SUFFIX`,
`BSIT_SEED`, `BSIT_WARMUP`, `BSIT_REPEATS`, `BSIT_TEST_SAMPLES`, `BSIT_BATCH_SIZES`.

Logs: `logs/batch_size_inference_time/slurm_<jobid>_<task>.out`

## Plot

Paper figure (all grids on one axes, Okabe–Ito colors) — preferred for the paper:

```bash
python processing_results/plot_batch_size_inference_time.py \
  --results-dir analysis/batch_size_inference_time/results \
  --output-dir processing_results/figures
```

Multi-panel debug figure (one panel per grid):

```bash
python analysis/batch_size_inference_time/plot.py \
  --results-dir analysis/batch_size_inference_time/results \
  --output analysis/batch_size_inference_time/figures/batch_size_inference_time.pdf
```

## Tests

```bash
conda run -n new_opt pytest analysis/batch_size_inference_time/test_benchmark.py -v
```

## JSON schema

One file per grid, e.g. `results/results_906.json`:

```json
{
  "benchmark": "batch_size_inference_time",
  "method": "penalty_fs",
  "case_suffix": "penalty_fs_rho1e5",
  "grid": "906",
  "variant": "feasible",
  "warmup": 3,
  "repeats": 10,
  "test_samples": 256,
  "timed_scope": "raw_nn_project_pf_plus_post_process_slfs",
  "excludes": ["topology_inverse", "quality_metrics_pf"],
  "by_batch_size": {
    "1": {
      "pipeline_time_per_batch_mean_s": 0.0,
      "pipeline_time_per_batch_std_s": 0.0,
      "raw_time_per_batch_mean_s": 0.0,
      "fs_time_per_batch_mean_s": 0.0,
      "inverse_time_per_batch_mean_s": 0.0,
      "pipeline_time_per_sample_mean_s": 0.0
    }
  }
}
```
