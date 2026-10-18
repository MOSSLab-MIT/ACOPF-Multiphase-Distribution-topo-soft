# Generated training jobs: penalty_fs

Total jobs: 60

## Results layout

Each job writes to `{results_dir}/{case_name}/seed_{seed}/`.
With `results_dir: results/penalty_fs`, example:
`results/penalty_fs/IEEE13Nodeckt_penalty_fs/seed_1/`.

## Submit to Slurm

Run `sbatch` from the **repository root** (Slurm log paths are relative to that directory):

```bash
cd /path/to/ACOPF-Multiphase-Distribution-topo
```

- `submit_large.sh` (12 jobs): `sbatch scripts/slurm_runs/penalty_fs/submit_large.sh`
- `submit_medium.sh` (24 jobs): `sbatch scripts/slurm_runs/penalty_fs/submit_medium.sh`
- `submit_small.sh` (24 jobs): `sbatch scripts/slurm_runs/penalty_fs/submit_small.sh`

Generated `submit.sh` discovers the repo root from its own location;
no machine-specific paths are embedded.

## Run one job locally

```bash
python scripts/slurm_runs/train_jobs.py run --jobs scripts/slurm_runs/penalty_fs/jobs.jsonl --index 0 --dry-run
```

## Regenerate

```bash
python scripts/slurm_runs/train_jobs.py generate --config scripts/slurm_runs/train_jobs_<method>.yaml
```

