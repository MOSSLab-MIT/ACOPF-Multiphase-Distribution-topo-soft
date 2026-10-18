#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=training_regime_ablation
#SBATCH --output=/home/hoangh/Projects/ACOPF-Multiphase-Distribution-topo/logs/datagen/training_regime_ablation-%A_%a.out
#SBATCH --error=/home/hoangh/Projects/ACOPF-Multiphase-Distribution-topo/logs/datagen/training_regime_ablation-%A_%a.err
#SBATCH -p pi_donti,mit_preemptable
#SBATCH --time=47:50:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --array=0-11

# Training ablation matrix: regime-weight mixes (W1–W3) plus load-cap OOR variant l02_u08.
#   W1–W3: configs/training_w060_*.json  -> suffix 060_005_035 / 060_035_005 / 060_020_020
#   OOR:   configs/training_l02_u08.json -> suffix l02_u08 (all-base load [0.2, 0.8])

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

module load julia

JULIA_PROJECT="${JULIA_PROJECT:-datagen_topo/FeederFlow.jl}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

GRIDS=(
  "datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss"
  "datagen_topo/grid_data/123_bus/IEEE123Master.dss"
  "datagen_topo/grid_data/240_bus/240bus_Master.dss"
)

GRID_NAMES=("13" "123" "240")

VARIANTS=(
  "060_005_035|datagen_topo/gen_scripts/configs/training_w060_005_035.json"
  "060_035_005|datagen_topo/gen_scripts/configs/training_w060_035_005.json"
  "060_020_020|datagen_topo/gen_scripts/configs/training_w060_020_020.json"
  "l02_u08|datagen_topo/gen_scripts/configs/training_l02_u08.json"
)

SMIN="${SMIN:-1}"
SMAX="${SMAX:-200000}"

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

resolve_requested_variant_idx() {
  local n_variants="${#VARIANTS[@]}"

  if [[ -n "${VARIANT_IDX:-}" ]]; then
    if [[ ! "$VARIANT_IDX" =~ ^[0-9]+$ ]] || (( VARIANT_IDX < 0 || VARIANT_IDX >= n_variants )); then
      echo "ERROR: VARIANT_IDX must be an integer in [0,$((n_variants - 1))]; got '${VARIANT_IDX}'" >&2
      exit 1
    fi
    echo "$VARIANT_IDX"
    return
  fi

  echo ""
}

n_variants="${#VARIANTS[@]}"
n_grids="${#GRIDS[@]}"
n_tasks=$(( n_grids * n_variants ))

requested_grid_idx="$(resolve_requested_grid_idx)"
requested_variant_idx="$(resolve_requested_variant_idx)"

if [[ -n "$requested_grid_idx" || -n "$requested_variant_idx" ]]; then
  if [[ -z "$requested_grid_idx" || -z "$requested_variant_idx" ]]; then
    echo "ERROR: GRID_IDX/GRID and VARIANT_IDX must be set together for single-job reruns." >&2
    exit 1
  fi
  grid_idx="$requested_grid_idx"
  variant_idx="$requested_variant_idx"
else
  if (( TASK_ID < 0 || TASK_ID >= n_tasks )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,$((n_tasks - 1))] (n_grids=${n_grids}, n_variants=${n_variants})" >&2
    exit 1
  fi
  grid_idx=$(( TASK_ID / n_variants ))
  variant_idx=$(( TASK_ID % n_variants ))
fi

grid="${GRIDS[$grid_idx]}"
variant="${VARIANTS[$variant_idx]}"
suffix="${variant%%|*}"
config_path="${variant#*|}"

cmd=(
  julia "--project=${JULIA_PROJECT}"
  datagen_topo/gen_scripts/create_training_dataset.jl "$grid"
  --smin "$SMIN"
  --smax "$SMAX"
  --config "$config_path"
  --output_suffix "$suffix"
)

echo "Running task[$TASK_ID] grid[$grid_idx]=$grid variant[$variant_idx] suffix=$suffix"
echo "CMD: ${cmd[*]}"
"${cmd[@]}"
