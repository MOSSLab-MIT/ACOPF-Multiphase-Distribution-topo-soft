#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=datagen
#SBATCH --output=/home/hoangh/Projects/ACOPF-Multiphase-Distribution-topo/logs/datagen/datagen-%A_%a.out
#SBATCH --error=/home/hoangh/Projects/ACOPF-Multiphase-Distribution-topo/logs/datagen/datagen-%A_%a.err
#SBATCH -p pi_donti,mit_preemptable
#SBATCH --time=35:50:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G


set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

module load julia

JULIA_PROJECT="${JULIA_PROJECT:-datagen_topo/FeederFlow.jl}"
MODE="${MODE:-testing}"  # testing | training
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

GRIDS=(
  "datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss"
  "datagen_topo/grid_data/123_bus/IEEE123Master.dss"
  "datagen_topo/grid_data/240_bus/240bus_Master.dss"
  "datagen_topo/grid_data/906_bus/906bus_Master.dss"
  "datagen_topo/grid_data/IEEE8500/8500_Master.dss"
)

GRID_NAMES=("13" "123" "240" "906" "8500")

# Base + extreme load/DER cases (run for every grid).
TEST_VARIANTS_CORE=(
  "base|"
  "low_load_high_der|--load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 --output_suffix low_load_high_der --save_config false"
  "high_load_low_der|--load_l 0.6 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.4 --output_suffix high_load_low_der --save_config false"
)

# Shifted distribution cases (optional second list).
TEST_VARIANTS_SHIFTED=(
  "m01_std01|--load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.1 --shift_der_mu 0.1 --output_suffix m01_std01 --save_config false"
  "m02_std02|--load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.2 --shift_der_mu 0.2 --output_suffix m02_std02 --save_config false"
  "m03_std03|--load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.3 --shift_der_mu 0.3 --output_suffix m03_std03 --save_config false"
  "m04_std04|--load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.4 --shift_der_mu 0.4 --output_suffix m04_std04 --save_config false"
  "m05_std05|--load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.5 --shift_der_mu 0.5 --output_suffix m05_std05 --save_config false"
)

# Testing uses a flat 2D array: task_id = grid_idx * n_variants + variant_idx.
build_test_variants() {
  local -n _out="$1"
  _out=("${TEST_VARIANTS_CORE[@]}")

  case "${VARIANT_SET:-core}" in
    core) ;;
    all) _out+=("${TEST_VARIANTS_SHIFTED[@]}") ;;
    *)
      echo "ERROR: VARIANT_SET must be 'core' or 'all'; got '${VARIANT_SET}'" >&2
      exit 1
      ;;
  esac
}

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

run_training() {
  local n_grids="${#GRIDS[@]}"
  if (( TASK_ID < 0 || TASK_ID >= n_grids )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,$((n_grids - 1))]" >&2
    exit 1
  fi

  local grid="${GRIDS[$TASK_ID]}"
  local smin="${SMIN:-1}"
  local smax="${SMAX:-200000}"
  local training_config="${TRAINING_CONFIG:-datagen_topo/gen_scripts/configs/training_defaults.json}"
  CMD=(julia "--project=${JULIA_PROJECT}" datagen_topo/gen_scripts/create_training_dataset.jl "$grid" --smin "$smin" --smax "$smax" --config "$training_config")
}

run_testing_variant() {
  local grid_idx="$1"
  local variant_idx="$2"
  local variant="$3"

  local grid="${GRIDS[$grid_idx]}"
  local variant_name="${variant%%|*}"
  local extra="${variant#*|}"
  local list_name="core"
  local num_scenarios="${NUM_SCENARIOS:-20}"
  local feasible_per_scenario="${FEASIBLE_SAMPLES_PER_SCENARIO:-100}"
  local max_attempts="${MAX_ATTEMPTS_MULTIPLIER:-3}"

  if (( variant_idx >= ${#TEST_VARIANTS_CORE[@]} )); then
    list_name="shifted"
  fi

  # shellcheck disable=SC2206
  local extra_args=($extra)
  local cmd=(
    julia "--project=${JULIA_PROJECT}" datagen_topo/gen_scripts/create_testing_dataset.jl "$grid"
    --num_scenarios "$num_scenarios"
    --feasible_samples_per_scenario "$feasible_per_scenario"
    --max_attempts_multiplier "$max_attempts"
    "${extra_args[@]}"
  )

  echo "Running task[$TASK_ID] grid[$grid_idx]=$grid variant[$variant_idx] list=$list_name name=$variant_name"
  echo "CMD: ${cmd[*]}"
  "${cmd[@]}"
}

run_testing() {
  local variants=()
  build_test_variants variants

  local n_variants="${#variants[@]}"
  local n_grids="${#GRIDS[@]}"
  local n_tasks=$(( n_grids * n_variants ))
  local requested_grid_idx
  requested_grid_idx="$(resolve_requested_grid_idx)"

  local grid_idx variant_idx
  if [[ -n "$requested_grid_idx" ]]; then
    if (( TASK_ID < 0 || TASK_ID >= n_variants )); then
      echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,$((n_variants - 1))] for GRID_IDX=${requested_grid_idx} (n_variants=${n_variants}, VARIANT_SET=${VARIANT_SET:-core})" >&2
      exit 1
    fi
    grid_idx="$requested_grid_idx"
    variant_idx="$TASK_ID"
  else
    if (( TASK_ID < 0 || TASK_ID >= n_tasks )); then
      echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,$((n_tasks - 1))] (n_grids=${n_grids}, n_variants=${n_variants}, VARIANT_SET=${VARIANT_SET:-core})" >&2
      exit 1
    fi
    grid_idx=$(( TASK_ID / n_variants ))
    variant_idx=$(( TASK_ID % n_variants ))
  fi

  local variant="${variants[$variant_idx]}"

  run_testing_variant "$grid_idx" "$variant_idx" "$variant"
}

case "$MODE" in
  training)
    run_training
    echo "Running MODE=${MODE} task ${TASK_ID}: ${CMD[*]}"
    "${CMD[@]}"
    ;;
  testing)
    echo "Running MODE=${MODE} task ${TASK_ID} (grid x variant), VARIANT_SET=${VARIANT_SET:-core}"
    run_testing
    ;;
  *)
    echo "ERROR: MODE must be 'training' or 'testing'; got '$MODE'" >&2
    exit 1
    ;;
esac
