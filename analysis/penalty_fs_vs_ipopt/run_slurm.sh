#!/usr/bin/env bash
#SBATCH --requeue
#SBATCH --job-name=dump_8500_penalty_fs
#SBATCH --output=logs/analysis/penalty_fs_vs_ipopt_%j.out
#SBATCH --error=logs/analysis/penalty_fs_vs_ipopt_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH -G h200:1
#SBATCH --partition=pi_donti_gpu,mit_preemptable,mit_normal_gpu
#
# Penalty+SLFS vs Ipopt dump for the 8500-node feeder (big host memory:
# the 8500 admittance and topology blocks need >100 GB; the earlier run peaked
# at ~128 GB RSS before the flow arrays were added).
#
# Submit from repository root:
#   sbatch analysis/penalty_fs_vs_ipopt/run_slurm.sh
#
# Other grids / batch size (batch 32 OOMs the 80+ GB H200 on 8500; 4 works):
#   DUMP_GRIDS="240 906" DUMP_BATCH_SIZE=32 sbatch analysis/penalty_fs_vs_ipopt/run_slurm.sh
#
# Then plot (login node is fine, only reads NPZ):
#   python analysis/penalty_fs_vs_ipopt/plot_hist.py --grids 8500

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/analysis/penalty_fs_vs_ipopt/run_slurm.sh" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs/analysis"

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

DUMP_GRIDS="${DUMP_GRIDS:-8500}"
DUMP_BATCH_SIZE="${DUMP_BATCH_SIZE:-4}"
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  grids=${DUMP_GRIDS} batch_size=${DUMP_BATCH_SIZE}"
# shellcheck disable=SC2086
python analysis/penalty_fs_vs_ipopt/dump.py --grids $DUMP_GRIDS --batch-size "$DUMP_BATCH_SIZE"
echo "Done grids ${DUMP_GRIDS}."
