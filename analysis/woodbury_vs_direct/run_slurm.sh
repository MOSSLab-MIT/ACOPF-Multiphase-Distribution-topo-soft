#!/usr/bin/env bash
#SBATCH --job-name=woodbury_vs_direct
#SBATCH --output=logs/woodbury_vs_direct/slurm_%j.out
#SBATCH --error=logs/woodbury_vs_direct/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -G h200:1
#SBATCH --partition=pi_donti_gpu,mit_preemptable,mit_normal_gpu

set -euo pipefail

# sbatch copies this script to a spool path; BASH_SOURCE then points outside the repo.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/analysis/woodbury_vs_direct/run_slurm.sh" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p logs/woodbury_vs_direct analysis/woodbury_vs_direct/results

module load cuda/12.9.1

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

bash "$ROOT/analysis/woodbury_vs_direct/run_all.sh"
