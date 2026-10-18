#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=train_penalty_fs
#SBATCH --output=logs/training/penalty_fs/%A_%a.out
#SBATCH --error=logs/training/penalty_fs/%A_%a.err
#SBATCH --array=0,1,2,3,4,5,15,16,17,18,19,20,30,31,32,33,34,35,45,46,47,48,49,50%20
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -G h200:1
#SBATCH --partition=pi_donti_gpu,mit_preemptable

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

JOBS_FILE="$PROJECT_DIR/scripts/slurm_runs/penalty_fs/jobs.jsonl"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

python "$PROJECT_DIR/scripts/slurm_runs/train_jobs.py" run --jobs "$JOBS_FILE" --index "$TASK_ID"

