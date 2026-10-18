#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=train_slfs_fsnet
#SBATCH --output=logs/training/slfs_fsnet/%A_%a.out
#SBATCH --error=logs/training/slfs_fsnet/%A_%a.err
#SBATCH --array=12,13,14,27,28,29%20
#SBATCH --time=47:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH -G h200:1
#SBATCH --partition=pi_donti_gpu,mit_preemptable,mit_normal_gpu

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/slurm_runs/train_jobs.py" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "$SUBMIT_DIR/../../.." && pwd)"
fi
cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/scripts/slurm_runs/train_jobs.py" ]; then
  echo "ERROR: repository root not found at $PROJECT_DIR" >&2
  echo "Run sbatch from the repository root (see scripts/slurm_runs/README.md)." >&2
  exit 1
fi

set +u
if [ -f "${CONDA_EXE%/*}/../etc/profile.d/conda.sh" ]; then
  source "${CONDA_EXE%/*}/../etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2
  exit 1
fi
conda activate new_opt
set -u

JOBS_FILE="$PROJECT_DIR/scripts/slurm_runs/slfs_fsnet/jobs.jsonl"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

python "$PROJECT_DIR/scripts/slurm_runs/train_jobs.py" run --jobs "$JOBS_FILE" --index "$TASK_ID"

