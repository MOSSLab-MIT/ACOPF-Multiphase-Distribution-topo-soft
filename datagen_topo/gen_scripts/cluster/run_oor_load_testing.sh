#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=oor_load_test
#SBATCH --output=/home/hoangh/Projects/ACOPF-Multiphase-Distribution-topo/logs/datagen/oor_load_test-%A_%a.out
#SBATCH --error=/home/hoangh/Projects/ACOPF-Multiphase-Distribution-topo/logs/datagen/oor_load_test-%A_%a.err
#SBATCH -p pi_donti,mit_preemptable
#SBATCH --time=35:50:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G

# Out-of-training-load-range test datasets for grids 13, 123, 906.
# Load scale [1.5, 2.0] — above standard training range [0.2, 1.5].
# All other sampling matches configs/testing_defaults.json.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

module load julia

JULIA_PROJECT="${JULIA_PROJECT:-datagen_topo/FeederFlow.jl}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

GRIDS=(
  "datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss"
  "datagen_topo/grid_data/123_bus/IEEE123Master.dss"
  "datagen_topo/grid_data/906_bus/906bus_Master.dss"
)

GRID_NAMES=("13" "123" "906")

CONFIG_PATH="datagen_topo/gen_scripts/configs/testing_oor_load.json"
OUTPUT_SUFFIX="oor_load"

NUM_SCENARIOS="${NUM_SCENARIOS:-10}"
FEASIBLE_SAMPLES_PER_SCENARIO="${FEASIBLE_SAMPLES_PER_SCENARIO:-100}"
MAX_ATTEMPTS_MULTIPLIER="${MAX_ATTEMPTS_MULTIPLIER:-5}"

resolve_requested_grid_idx() {
  local n_grids="${#GRIDS[@]}"

  if [[ -n "${GRID_IDX:-}" ]]; then
    if [[ ! "$GRID_IDX" =~ ^[0-9]+$ ]] || (( GRID_IDX < 0 || GRID_IDX >= n_grids )); then
      echo "ERROR: GRID_IDX must be an integer in [0,$((n_grids - 1))]; got '${GRID_IDX}'" >&2
      exit 1
    fi
    echo "$GRID_IDX"
    return
  fi

  if [[ -n "${GRID:-}" ]]; then
    local grid_name="${GRID,,}"
    grid_name="${grid_name#ieee}"
    grid_name="${grid_name%_bus}"
    grid_name="${grid_name%-bus}"
    grid_name="${grid_name%bus}"

    local idx
    for (( idx=0; idx<n_grids; idx++ )); do
      if [[ "$grid_name" == "${GRID_NAMES[$idx]}" ]]; then
        echo "$idx"
        return
      fi
    done

    echo "ERROR: GRID must be one of: ${GRID_NAMES[*]}; got '${GRID}'" >&2
    exit 1
  fi

  echo ""
}

n_grids="${#GRIDS[@]}"
requested_grid_idx="$(resolve_requested_grid_idx)"

if [[ -n "$requested_grid_idx" ]]; then
  grid_idx="$requested_grid_idx"
else
  if (( TASK_ID < 0 || TASK_ID >= n_grids )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,$((n_grids - 1))]" >&2
    exit 1
  fi
  grid_idx="$TASK_ID"
fi

grid="${GRIDS[$grid_idx]}"

cmd=(
  julia "--project=${JULIA_PROJECT}"
  datagen_topo/gen_scripts/create_testing_dataset.jl "$grid"
  --config "$CONFIG_PATH"
  --num_scenarios "$NUM_SCENARIOS"
  --feasible_samples_per_scenario "$FEASIBLE_SAMPLES_PER_SCENARIO"
  --max_attempts_multiplier "$MAX_ATTEMPTS_MULTIPLIER"
  --output_suffix "$OUTPUT_SUFFIX"
  --save_config false
)

echo "Running task[$TASK_ID] grid[$grid_idx]=$grid suffix=$OUTPUT_SUFFIX"
echo "CMD: ${cmd[*]}"
"${cmd[@]}"
