# Sourced by <method>.sbatch from the repository root with the script's own arguments, after
# it sets METHOD (its file name without .sbatch), FAMILY (methods/<FAMILY>/train.py and the
# method_args key) and GRIDS (default grids to submit).
#
# Outside a job (SLURM_ARRAY_TASK_ID unset): "$@" = [grid ...] [extra sbatch options...].
#   Submit scripts/slurm_runs/$METHOD.sbatch once per grid (the given grids, else GRIDS) as an
#   array job with one task per seed in configs/<grid>.yaml, then exit. Extra sbatch options
#   come after the grid defaults, so they override them. All grid configs are checked before
#   anything is submitted.
#   Env: SEEDS     Slurm array spec overriding the config's seeds, e.g. 1 or 1,4 or 1-5
#        DRY_RUN=1 print the sbatch commands and each seed's training command only
# Inside a job: "$@" = <grid>. Defines GRID, SEED, CASE_DIR, DATA, GRID_ARGS, TRAIN_H5, TRAIN_TAG,
#   VAL_H5, TEST_CORE_H5, TEST_FULL_H5.

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ] && [ "${ENVIRONMENT:-}" = BATCH ]; then
  echo "ERROR: run scripts/slurm_runs/$METHOD.sbatch directly (it calls sbatch itself), not via sbatch" >&2
  exit 1
fi

set +u   # conda activate hooks may read unset variables
source "${CONDA_EXE%/*}/../etc/profile.d/conda.sh"
conda activate new_opt   # grid_config.py needs PyYAML
set -u

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  SCRIPT="scripts/slurm_runs/$METHOD.sbatch"
  [ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found (METHOD must match the file name)" >&2; exit 1; }
  ARG_GRIDS=()
  while [ $# -gt 0 ] && [[ "$1" != -* ]]; do ARG_GRIDS+=("$1"); shift; done
  [ ${#ARG_GRIDS[@]} -gt 0 ] && GRIDS=("${ARG_GRIDS[@]}")
  for GRID in "${GRIDS[@]}"; do
    python scripts/slurm_runs/grid_config.py "$GRID" "$FAMILY" > /dev/null
  done
  LOG_DIR="logs/training/$METHOD"
  [ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$LOG_DIR"

  for GRID in "${GRIDS[@]}"; do
    GRID_VARS="$(python scripts/slurm_runs/grid_config.py "$GRID")"   # -> SEED_ARRAY, SBATCH_OPTS
    eval "$GRID_VARS"
    ARRAY="${SEEDS:-$SEED_ARRAY}"
    SBATCH=(
      sbatch
      --job-name="${METHOD}_${GRID}"
      --array="$ARRAY"
      --output="$LOG_DIR/%x_seed%a_%A.out"
      --error="$LOG_DIR/%x_seed%a_%A.err"
      "${SBATCH_OPTS[@]}"
      "$@"
      "$SCRIPT" "$GRID"
    )
    if [ "${DRY_RUN:-0}" = 1 ]; then
      echo "${SBATCH[*]}"
      for spec in ${ARRAY//,/ }; do
        spec="${spec%%%*}"
        for seed in $(seq "${spec%-*}" "${spec#*-}"); do
          SLURM_ARRAY_TASK_ID="$seed" SLURM_SUBMIT_DIR="$PWD" bash "$SCRIPT" "$GRID"
        done
      done
    else
      "${SBATCH[@]}"
    fi
  done
  exit 0
fi

GRID="${1:?usage: $METHOD.sbatch <grid> (inside a job)}"
SEED="$SLURM_ARRAY_TASK_ID"
# Grid config (configs/$GRID.yaml) -> CASE_DIR, DATA, GRID_ARGS, TRAIN_H5, TRAIN_TAG, VAL_H5, TEST_CORE_H5, TEST_FULL_H5
GRID_VARS="$(python scripts/slurm_runs/grid_config.py "$GRID" "$FAMILY")"
eval "$GRID_VARS"
