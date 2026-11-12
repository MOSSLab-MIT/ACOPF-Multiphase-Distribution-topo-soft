#!/bin/bash
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH -G h200:1
#
# Test-only rerun of trained Direct Prediction checkpoints (raw + PF-repaired metrics). Array index = seed.
# Submit:  scripts/slurm_runs/direct_prediction_test.sbatch [grid ...] [extra sbatch options...]
#   No grid -> GRIDS below. Seeds: `seed` in configs/<grid>.yaml (SEEDS=1,4 overrides).
#   TEST_SETS="name ..." -> test sets to run (default: the grid config's full test_sets).
#   RHO -> run name rho_eq_init (default 100), as in direct_prediction.sbatch.
#   Results go to testing/<set>/ next to the training results; a set that is rerun is overwritten.
#   DRY_RUN=1 only prints the commands.

set -euo pipefail
METHOD=direct_prediction_test
FAMILY=direct_prediction
GRIDS=(123 240 906)   # grids with trained checkpoints

# Repository root (sbatch is run from there, so inside a job it is the submit dir).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then cd "${SLURM_SUBMIT_DIR:-$PWD}"; else cd "$(dirname "${BASH_SOURCE[0]}")/../.."; fi
source scripts/slurm_runs/common.sh   # outside a job: submits one array job per grid, exits

# test.py rejects training flags, so take only what it needs from the grid config.
arg() {
  local i
  for ((i = 0; i < ${#GRID_ARGS[@]} - 1; i++)); do
    if [ "${GRID_ARGS[i]}" = "--$1" ]; then echo "${GRID_ARGS[i + 1]}"; return 0; fi
  done
  echo "ERROR: --$1 not in configs/$GRID.yaml" >&2
  return 1
}
EPOCHS="$(arg epochs)"
SOLVER="$(arg power_flow_solver)"   # not stored in the checkpoint; must match training
TRAIN_SAMPLES="$(arg train_samples)"
TEST_SAMPLES="$(arg test_samples)"

RHO="${RHO:-100}"
CASE_NAME="${CASE_DIR}_direct_prediction_rho${RHO}${TRAIN_TAG}"
RESULTS_DIR=results/direct_prediction
CKPT="$RESULTS_DIR/$CASE_NAME/seed_$SEED/models/checkpoint_epoch_$EPOCHS"

if [ -n "${TEST_SETS:-}" ]; then
  TEST_H5=()
  for name in $TEST_SETS; do
    TEST_H5+=("$DATA/${CASE_DIR}_test_$name.h5")
  done
else
  TEST_H5=("${TEST_FULL_H5[@]}")
fi

CMD=(
  python methods/direct_prediction/test.py
  --train_h5 "$TRAIN_H5"
  --data_json "$DATA/network_config.json"
  --checkpoint_path "$CKPT"
  --case_name "$CASE_NAME"
  --results_dir "$RESULTS_DIR"
  --seed "$SEED"
  --power_flow_solver "$SOLVER"
  --train_samples "$TRAIN_SAMPLES"
  --test_samples "$TEST_SAMPLES"
  --woodbury_ir_steps auto
  --woodbury_batch_mode vmap
  --test_h5_variants "${TEST_H5[@]}"
)

echo "results: $RESULTS_DIR/$CASE_NAME/seed_$SEED/testing"
echo "${CMD[*]}"
[ "${DRY_RUN:-0}" = 1 ] && exit 0

[ -d "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
for path in "${TEST_H5[@]}"; do
  [ -f "$path" ] || { echo "ERROR: test set not found: $path" >&2; exit 1; }
done
exec "${CMD[@]}"
