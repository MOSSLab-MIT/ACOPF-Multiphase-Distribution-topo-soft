#!/usr/bin/env bash
# SLFS vs JAXopt voltage-LS benchmark (one grid per job, GPU).
#
# Submit (from repository root):
#   sbatch analysis/slfs_vs_voltage_ls/run_slfs_vs_voltage_ls.sh 13
#   sbatch analysis/slfs_vs_voltage_ls/run_slfs_vs_voltage_ls.sh 123
#   sbatch analysis/slfs_vs_voltage_ls/run_slfs_vs_voltage_ls.sh 240
#
# Grid 240 auto-selects newton_gmres for PF init (benchmark.py).

#SBATCH --requeue
#SBATCH --job-name=slfs_vs_lm
#SBATCH --output=logs/slfs_vs_voltage_ls/%j.out
#SBATCH --error=logs/slfs_vs_voltage_ls/%j.err
#SBATCH --partition=mit_preemptable,pi_donti_gpu,mit_normal_gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

GRID="${1:?usage: sbatch $0 <13|123|240>}"

case "$GRID" in
  13|123) BATCH_SIZE=4 ;;
  240)   BATCH_SIZE=4 ;;
  *)
    echo "ERROR: unknown grid '${GRID}'; expected 13, 123, or 240." >&2
    exit 1
    ;;
esac

# sbatch copies this script to a spool path; BASH_SOURCE then points outside the repo.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/analysis/slfs_vs_voltage_ls/run_slfs_vs_voltage_ls.sh" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_DIR"
mkdir -p logs/slfs_vs_voltage_ls analysis/slfs_vs_voltage_ls

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

python analysis/slfs_vs_voltage_ls/benchmark.py \
  --grid "$GRID" \
  --batch_size "$BATCH_SIZE" \
  --test_samples "${TEST_SAMPLES:-64}" \
  --seed "${SEED:-1}" \
  --out "analysis/slfs_vs_voltage_ls/results_${GRID}.json"
