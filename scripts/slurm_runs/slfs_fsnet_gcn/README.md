# Generated training jobs: slfs_fsnet_gcn

Total jobs: 12

## Results layout

Each job writes to `{results_dir}/{case_name}/seed_{seed}/`.
With `results_dir: results/slfs_fsnet_gcn`, example:
`results/slfs_fsnet_gcn/IEEE13Nodeckt_penalty_fs/seed_1/`.

## Submit to Slurm

Run `sbatch` from the **repository root** (Slurm log paths are relative to that directory):

```bash
cd /path/to/ACOPF-Multiphase-Distribution-topo
```

- `submit_medium.sh` (6 jobs): `sbatch scripts/slurm_runs/slfs_fsnet_gcn/submit_medium.sh`
- `submit_small.sh` (6 jobs): `sbatch scripts/slurm_runs/slfs_fsnet_gcn/submit_small.sh`

Generated `submit.sh` discovers the repo root from its own location;
no machine-specific paths are embedded.

## Run one job locally

```bash
python scripts/slurm_runs/train_jobs.py run --jobs scripts/slurm_runs/slfs_fsnet_gcn/jobs.jsonl --index 0 --dry-run
```

## Regenerate and verify

```bash
python scripts/slurm_runs/train_jobs.py generate --config scripts/slurm_runs/train_jobs_<method>.yaml
python scripts/slurm_runs/train_jobs.py verify --jobs-dir scripts/slurm_runs/slfs_fsnet_gcn
```

Do not pass custom `sbatch --array=...` overrides; each submit script
embeds the jobs.jsonl indices it is allowed to run.

