# Slurm training jobs

One script per method, one YAML config per grid. Run a method script directly: it calls
`sbatch` on itself once per grid (one array task per seed) and exits.

```
scripts/slurm_runs/
├── proposed.sbatch                   # Proposed
├── direct_prediction.sbatch               # Direct Prediction        (knob: RHO=<rho_eq_init>, default 100)
├── proposed_regime_ablation.sbatch   # Proposed on <case>_training_<TRAIN_SUFFIX>.h5 (knob required)
├── common.sh                           # shared submit / job setup, sourced by every method script
├── grid_config.py                      # prints configs/<grid>.yaml as bash variables
└── configs/
    └── 13.yaml  123.yaml  240.yaml  906.yaml   # per-grid seeds, resources, args, test sets
```

- **Grid config** (`configs/<grid>.yaml`):
  ```yaml
  case_dir: IEEE13Nodeckt
  seed:         [1, 2, 3]                                       # one array task per seed
  slurm:        {time: "12:00:00", mem: 32G, partition: ...}   # -> --time=... --mem=...
  args:         {num_layers: 4, hidden_size: 512, ...}          # all methods
  method_args:  {proposed: {epochs: 10}, direct_prediction: {epochs: 20}}  # per family, overrides args
  train_set:    training                                        # -> <case_dir>_<train_set>.h5 (--train_h5)
  val_set:      feasible_validation                             # -> <case_dir>_test_<val_set>.h5 (--val_h5)
  test_sets:    {core: [feasible, ...], full: [feasible, ...]}  # first = reference test set
  ```
- **Method script** (`<method>.sbatch`): sets `METHOD`, `FAMILY`, the default `GRIDS`
  (defined in each script) and sources `common.sh`, then builds and runs `methods/<FAMILY>/train.py`.
  - Run from a login node: `common.sh` submits the script for each grid with the grid's seeds
    and Slurm options, job name `<method>_<grid>`, and logs in `logs/training/<method>/`.
  - Inside the job: `common.sh` loads the grid config (`CASE_DIR`, `DATA`, `GRID_ARGS`,
    `TRAIN_H5`, `TRAIN_TAG`, `VAL_H5`, `TEST_CORE_H5`, `TEST_FULL_H5`) and sets `SEED` from the array index.

Jobs activate `new_opt` themselves; so does the submit step (for PyYAML). Don't pass a
method script to `sbatch` yourself; it refuses to run that way.

## Submit

```bash
scripts/slurm_runs/proposed.sbatch                   # grid 906, seeds from its config
scripts/slurm_runs/proposed.sbatch 13                # just grid 13
DRY_RUN=1 scripts/slurm_runs/proposed.sbatch         # print sbatch + training commands only

# Direct Prediction rho sweep
for rho in 10 100 500 1000; do RHO=$rho scripts/slurm_runs/direct_prediction.sbatch; done


Options:

- Leading arguments are grids (`proposed.sbatch 13 906`); they replace the script's `GRIDS`.
- Change the default grids: edit `GRIDS=(...)` near the top of the method script.
- Change the seeds: edit `seed:` in `configs/<grid>.yaml`. For a one-off run,
  `SEEDS=1,4` or `SEEDS=1-5` overrides it for every grid. The regime ablation defaults to `SEEDS=1`.
- Options after the grids go to `sbatch` and override the grid defaults, e.g.
  `direct_prediction.sbatch 240 --time=24:00:00 --partition=pi_donti_gpu`.
- `--test-only` (as an sbatch option) asks Slurm to validate each job without submitting.

## Results and logs

- Checkpoints: `results/<method dir>/<CASE_DIR>_<suffix>/seed_<seed>/`, e.g.
  `results/proposed/IEEE13Nodeckt_proposed/seed_1/`,
  `results/direct_prediction/IEEE123Master_direct_prediction_rho500/seed_2/`,
  `results/proposed_regime_ablation/IEEE13Nodeckt_proposed_060_035_005/seed_1/`.
- Slurm logs: `logs/training/<method>/<method>_<grid>_seed<seed>_<jobid>.{out,err}`.
  The first line of each log is the results directory, the second the full command.

## Adding things

- **New grid**: copy `configs/13.yaml` to `configs/<grid>.yaml` and edit.
- **New method / ablation**: copy the closest `<method>.sbatch` to `<new>.sbatch`, set
  `METHOD=<new>`, and change its args, `CASE_NAME` and `RESULTS_DIR`. Distinct settings
  need distinct `CASE_NAME`s so checkpoints never collide.
- **Change a grid's training set**: set `train_set: training_<suffix>` in `configs/<grid>.yaml`
  to use `<case_dir>_training_<suffix>.h5`. `_<suffix>` is appended to `CASE_NAME`, so results
  go to e.g. `results/proposed/IEEE123Master_proposed_l02_u08/` and the `training`
  checkpoints stay intact. `proposed_test.sbatch` follows the same key, so keep `train_set`
  as it was at training time when testing. `proposed_regime_ablation.sbatch` ignores this
  key and uses `TRAIN_SUFFIX`.
- **Change a grid's hyperparameter**: edit `args` in `configs/<grid>.yaml` (every method)
  or `method_args.<family>` (one method family only).
