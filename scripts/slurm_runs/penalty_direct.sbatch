#!/bin/bash
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH -G h200:1
#
# Penalty Direct with a fixed equality weight. Array index = seed.
# Submit:  RHO=500 scripts/slurm_runs/penalty_direct.sbatch [grid ...] [extra sbatch options...]
#   No grid -> GRIDS below. Seeds: `seed` in configs/<grid>.yaml (SEEDS=1,4 overrides).
#   DRY_RUN=1 only prints the commands.
# Knob: RHO = rho_eq_init (default 100); rho_rate=0, rho_decay_rate=1 keep it fixed.

set -euo pipefail
METHOD=penalty_direct
FAMILY=penalty_direct
GRIDS=(123 240 8500)   # submitted when no grid is given

# Repository root (sbatch is run from there, so inside a job it is the submit dir).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then cd "${SLURM_SUBMIT_DIR:-$PWD}"; else cd "$(dirname "${BASH_SOURCE[0]}")/../.."; fi
source scripts/slurm_runs/common.sh   # outside a job: submits one array job per grid, exits

RHO="${RHO:-100}"
CASE_NAME="${CASE_DIR}_penalty_direct_rho${RHO}"
RESULTS_DIR=results/penalty_direct

CMD=(
  python methods/penalty_direct/train.py
  --train_h5 "$DATA/${CASE_DIR}_training.h5"
  --data_json "$DATA/network_config.json"
  --case_name "$CASE_NAME"
  --results_dir "$RESULTS_DIR"
  --seed "$SEED"
  "${GRID_ARGS[@]}"
  --rho_eq_init "$RHO"
  --rho_rate 0
  --rho_decay_rate 1
  --rho_max 1.0e10
  --violation_threshold 1.0e-6
  --save_every 5
  --log_every 1
  --woodbury_ir_steps auto
  --woodbury_batch_mode vmap
  --val_h5 "$VAL_H5"
  --test_h5_variants "${TEST_FULL_H5[@]}"
)

echo "results: $RESULTS_DIR/$CASE_NAME/seed_$SEED"
echo "${CMD[*]}"
[ "${DRY_RUN:-0}" = 1 ] && exit 0

exec "${CMD[@]}"
