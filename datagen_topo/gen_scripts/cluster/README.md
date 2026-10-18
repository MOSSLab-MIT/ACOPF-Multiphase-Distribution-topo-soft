# Datagen Cluster Launchers

Use `submit_datagen_matrix.sh` for normal submissions. It computes the Slurm array range automatically and submits `run_datagen_matrix.sh`.

## Grid Indexing


| Index | `--grid` | Feeder   |
| ----- | -------- | -------- |
| 0     | `13`     | 13 bus   |
| 1     | `123`    | 123 bus  |
| 2     | `240`    | 240 bus  |
| 3     | `906`    | 906 bus  |
| 4     | `8500`   | 8500 bus |


## Test Variants


| Variant set | Variants                                   | Jobs per grid |
| ----------- | ------------------------------------------ | ------------- |
| `core`      | base, low_load_high_der, high_load_low_der | 3             |
| `all`       | core + m01_std01 ... m05_std05             | 8             |


## Recommended Usage

Single feeder, core datasets only:

```bash
datagen_topo/gen_scripts/cluster/submit_datagen_matrix.sh \
  --grid 906 --variant-set core --mem 200G
```

Single feeder, core + shifted datasets:

```bash
datagen_topo/gen_scripts/cluster/submit_datagen_matrix.sh \
  --grid 906 --variant-set all --mem 200G
```

All feeders, core datasets only:

```bash
datagen_topo/gen_scripts/cluster/submit_datagen_matrix.sh \
  --variant-set core --mem 200G
```

Training all feeders:

```bash
datagen_topo/gen_scripts/cluster/submit_datagen_matrix.sh \
  --mode training --smin 1 --smax 200000
```

Add `--dry-run` to print the generated `sbatch` command without submitting.

## Split By Memory

Small feeders:

```bash
datagen_topo/gen_scripts/cluster/submit_datagen_matrix.sh \
  --grid 13 --variant-set all

datagen_topo/gen_scripts/cluster/submit_datagen_matrix.sh \
  --grid 123 --variant-set all

datagen_topo/gen_scripts/cluster/submit_datagen_matrix.sh \
  --grid 240 --variant-set core
```

Large feeders:

```bash
datagen_topo/gen_scripts/cluster/submit_datagen_matrix.sh \
  --grid 906 --variant-set all --mem 100G

datagen_topo/gen_scripts/cluster/submit_datagen_matrix.sh \
  --grid 8500 --variant-set core --mem 200G
```

## Out-of-training-load-range test sets (13 / 123 / 906)

Load scale `[1.5, 2.0]` — above standard training range `[0.2, 1.5]`. All other fields match `configs/testing_defaults.json`.

```bash
# All three grids (3 jobs)
datagen_topo/gen_scripts/cluster/submit_oor_load_testing.sh --mem 100G

# Single grid
datagen_topo/gen_scripts/cluster/submit_oor_load_testing.sh --grid 906 --mem 200G
```

Output: `<case>_test_feasible_oor_load.h5`.

## Raw `sbatch` Reference

If you call `run_datagen_matrix.sh` directly, you must provide both the array range and the variant set.

Single-grid mode: set `GRID` or `GRID_IDX`; `SLURM_ARRAY_TASK_ID` is interpreted as `variant_idx`.

```bash
# 906 bus, core + shifted variants in parallel
sbatch --array=0-7 --mem=200G \
  --export=MODE=testing,GRID=906,VARIANT_SET=all,NUM_SCENARIOS=20,FEASIBLE_SAMPLES_PER_SCENARIO=100 \
  datagen_topo/gen_scripts/cluster/run_datagen_matrix.sh
```

Full-matrix mode: if neither `GRID` nor `GRID_IDX` is set, testing uses a flat 2D array.

```text
task_id = grid_idx * n_variants + variant_idx
grid_idx = task_id // n_variants
variant_idx = task_id % n_variants
```


| `VARIANT_SET` | `n_variants` | Full matrix `--array` |
| ------------- | ------------ | --------------------- |
| `core`        | 3            | `0-14`                |
| `all`         | 8            | `0-39`                |


