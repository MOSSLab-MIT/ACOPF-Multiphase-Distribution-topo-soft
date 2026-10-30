#!/bin/bash
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH -G h200:1
#
# Penalty PF trained on a suffixed training set <case>_training_<TRAIN_SUFFIX>.h5.
# Array index = seed.
# Submit:  TRAIN_SUFFIX=060_005_035 scripts/slurm_runs/penalty_pf_regime_ablation.sbatch [grid ...] [extra sbatch options...]
#   No grid -> GRIDS below. Seeds: 1 (SEEDS below; SEEDS=1,4 overrides), not the config.
#   DRY_RUN=1 only prints the commands.
# Knob (required): TRAIN_SUFFIX
#   060_005_035  W1: base/high/low = 0.6/0.05/0.35
#   060_035_005  W2: 0.6/0.35/0.05
#   060_020_020  W3: 0.6/0.2/0.2
#   l02_u08      OOR load cap: base load scale [0.2, 0.8]
# W0 baseline (unsuffixed training set) is the regular penalty_pf run.

set -euo pipefail
METHOD=penalty_pf_regime_ablation
FAMILY=penalty_pf
GRIDS=(123 240 8500)   # submitted when no grid is given
TRAIN_SUFFIX="${TRAIN_SUFFIX:?set TRAIN_SUFFIX (e.g. 060_005_035)}"
SEEDS="${SEEDS:-1}"    # one seed per ablation (overrides configs/<grid>.yaml seed)

# Repository root (sbatch is run from there, so inside a job it is the submit dir).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then cd "${SLURM_SUBMIT_DIR:-$PWD}"; else cd "$(dirname "${BASH_SOURCE[0]}")/../.."; fi
source scripts/slurm_runs/common.sh   # outside a job: submits one array job per grid, exits

CASE_NAME="${CASE_DIR}_penalty_pf_${TRAIN_SUFFIX}"
RESULTS_DIR=results/penalty_pf_regime_ablation

CMD=(
  python methods/penalty_pf/train.py
  --train_h5 "$DATA/${CASE_DIR}_training_${TRAIN_SUFFIX}.h5"
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
  --test_h5_variants "${TEST_CORE_H5[@]}"
)

echo "results: $RESULTS_DIR/$CASE_NAME/seed_$SEED"
echo "${CMD[*]}"
[ "${DRY_RUN:-0}" = 1 ] && exit 0

exec "${CMD[@]}"
