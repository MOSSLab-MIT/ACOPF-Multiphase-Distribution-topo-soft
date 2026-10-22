# Generated training jobs: penalty_direct

Total jobs: 60

## Results layout

Each job writes to `{results_dir}/{case_name}/seed_{seed}/`, where `case_name` is
`<case_dir>_<case_suffix>` (e.g. `IEEE13Nodeckt_<case_suffix>`).
With `results_dir: results/penalty_direct`, example:
`results/penalty_direct/IEEE13Nodeckt_<case_suffix>/seed_1/`.

## Submit to Slurm

Run `sbatch` from the **repository root** (Slurm log paths are relative to that directory):

```bash
cd <path-to-repository-root>
```

- `submit_large.sh` (12 jobs): `sbatch scripts/slurm_runs/penalty_direct/submit_large.sh`
- `submit_medium.sh` (24 jobs): `sbatch scripts/slurm_runs/penalty_direct/submit_medium.sh`
- `submit_small.sh` (24 jobs): `sbatch scripts/slurm_runs/penalty_direct/submit_small.sh`

Generated `submit.sh` discovers the repo root from its own location;
no machine-specific paths are embedded.

## Run one job locally

```bash
python scripts/slurm_runs/train_jobs.py run --jobs scripts/slurm_runs/penalty_direct/jobs.jsonl --index 0 --dry-run
```

## Regenerate and verify

```bash
python scripts/slurm_runs/train_jobs.py generate --config scripts/slurm_runs/train_jobs_<method>.yaml
python scripts/slurm_runs/train_jobs.py verify --jobs-dir scripts/slurm_runs/penalty_direct
```

Do not pass custom `sbatch --array=...` overrides; each submit script
embeds the jobs.jsonl indices it is allowed to run.

