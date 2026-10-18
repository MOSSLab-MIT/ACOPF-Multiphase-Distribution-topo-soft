# Generated training jobs: slfs_fsnet

Total jobs: 30

## Results layout

Each job writes to `{results_dir}/{case_name}/seed_{seed}/`.
With `results_dir: results/slfs_fsnet`, example:
`results/slfs_fsnet/IEEE13Nodeckt_penalty_fs/seed_1/`.

## Submit to Slurm

Run `sbatch` from the **repository root** (Slurm log paths are relative to that directory):

```bash
cd /path/to/ACOPF-Multiphase-Distribution-topo
```

- `submit_large.sh` (6 jobs): `sbatch scripts/slurm_runs/slfs_fsnet/submit_large.sh`
- `submit_medium.sh` (12 jobs): `sbatch scripts/slurm_runs/slfs_fsnet/submit_medium.sh`
- `submit_small.sh` (12 jobs): `sbatch scripts/slurm_runs/slfs_fsnet/submit_small.sh`

Generated `submit.sh` discovers the repo root from its own location;
no machine-specific paths are embedded.

## Run one job locally

```bash
python scripts/slurm_runs/train_jobs.py run --jobs scripts/slurm_runs/slfs_fsnet/jobs.jsonl --index 0 --dry-run
```

## Regenerate

```bash
python scripts/slurm_runs/train_jobs.py generate --config scripts/slurm_runs/train_jobs_<method>.yaml
```

