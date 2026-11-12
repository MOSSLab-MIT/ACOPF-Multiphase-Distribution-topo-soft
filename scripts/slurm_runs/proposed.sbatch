#!/bin/bash
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH -G h200:1
#
# Proposed. Array index = seed.
# Submit:  scripts/slurm_runs/proposed.sbatch [grid ...] [extra sbatch options...]
#   No grid -> GRIDS below. Seeds: `seed` in configs/<grid>.yaml (SEEDS=1,4 overrides).
#   DRY_RUN=1 only prints the commands.

export PYTHONUNBUFFERED=1

set -euo pipefail
METHOD=proposed
FAMILY=proposed
GRIDS=(906)

# Repository root (sbatch is run from there, so inside a job it is the submit dir).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then cd "${SLURM_SUBMIT_DIR:-$PWD}"; else cd "$(dirname "${BASH_SOURCE[0]}")/../.."; fi
source scripts/slurm_runs/common.sh   # outside a job: submits one array job per grid, exits

CASE_NAME="${CASE_DIR}_proposed${TRAIN_TAG}"
RESULTS_DIR=results/proposed

CMD=(
  python methods/proposed/train.py
  --train_h5 "$TRAIN_H5"
  --data_json "$DATA/network_config.json"
  --case_name "$CASE_NAME"
  --results_dir "$RESULTS_DIR"
  --seed "$SEED"
  "${GRID_ARGS[@]}"
  --save_every 5
  --log_every 1
  --eval_every 1
  --woodbury_ir_steps auto
  --woodbury_batch_mode vmap
  --val_h5 "$VAL_H5"
  --test_h5_variants "${TEST_FULL_H5[@]}"
)

echo "results: $RESULTS_DIR/$CASE_NAME/seed_$SEED"
echo "${CMD[*]}"
[ "${DRY_RUN:-0}" = 1 ] && exit 0

exec "${CMD[@]}"
