#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAUNCHER="$SCRIPT_DIR/run_datagen_matrix.sh"

GRID_NAMES=("13" "123" "240" "906" "8500")
N_GRIDS="${#GRID_NAMES[@]}"
CORE_VARIANT_COUNT=3
ALL_VARIANT_COUNT=8

usage() {
  cat <<'EOF'
Usage:
  submit_datagen_matrix.sh [options]

Options:
  --mode testing|training                 Default: testing
  --grid 13|123|240|906|8500              Run one feeder; testing array indexes variants
  --grid-idx 0|1|2|3|4                    Same as --grid, using grid index
  --variant-set core|all                  Testing only. core=3 variants, all=8 variants. Default: core
  --num-scenarios N                       Testing only. Default: 20
  --feasible-samples-per-scenario N       Testing only. Default: 100
  --max-attempts-multiplier N             Testing only. Default: launcher default
  --smin N                                Training only. Default: 1
  --smax N                                Training only. Default: 200000
  --config PATH                           Training only. Default: datagen_topo/gen_scripts/configs/training_defaults.json
  --mem VALUE                             Optional sbatch memory override, e.g. 200G
  --dry-run                               Print the sbatch command instead of submitting
  --help                                  Show this message
  --                                      Pass remaining arguments directly to sbatch

Examples:
  submit_datagen_matrix.sh --grid 906 --variant-set all --mem 200G
  submit_datagen_matrix.sh --grid 240 --variant-set core --dry-run
  submit_datagen_matrix.sh --variant-set core --mem 200G
EOF
}

normalize_grid() {
  local grid="$1"
  local grid_name="${grid,,}"
  grid_name="${grid_name#ieee}"
  grid_name="${grid_name%_bus}"
  grid_name="${grid_name%-bus}"
  grid_name="${grid_name%bus}"
  echo "$grid_name"
}

grid_to_idx() {
  local grid_name
  grid_name="$(normalize_grid "$1")"

  local idx
  for (( idx=0; idx<N_GRIDS; idx++ )); do
    if [[ "$grid_name" == "${GRID_NAMES[$idx]}" ]]; then
      echo "$idx"
      return
    fi
  done

  echo "ERROR: --grid must be one of: ${GRID_NAMES[*]}; got '$1'" >&2
  exit 1
}

append_export() {
  local item="$1"
  if [[ -z "$EXPORTS" ]]; then
    EXPORTS="$item"
  else
    EXPORTS="${EXPORTS},${item}"
  fi
}

MODE="testing"
GRID=""
GRID_IDX=""
VARIANT_SET="core"
NUM_SCENARIOS="20"
FEASIBLE_SAMPLES_PER_SCENARIO="100"
MAX_ATTEMPTS_MULTIPLIER=""
SMIN="1"
SMAX="200000"
TRAINING_CONFIG="datagen_topo/gen_scripts/configs/training_defaults.json"
MEM=""
DRY_RUN=0
SBATCH_ARGS=()

while (( $# > 0 )); do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --grid)
      GRID="$2"
      shift 2
      ;;
    --grid-idx)
      GRID_IDX="$2"
      shift 2
      ;;
    --variant-set)
      VARIANT_SET="$2"
      shift 2
      ;;
    --num-scenarios)
      NUM_SCENARIOS="$2"
      shift 2
      ;;
    --feasible-samples-per-scenario)
      FEASIBLE_SAMPLES_PER_SCENARIO="$2"
      shift 2
      ;;
    --max-attempts-multiplier)
      MAX_ATTEMPTS_MULTIPLIER="$2"
      shift 2
      ;;
    --smin)
      SMIN="$2"
      shift 2
      ;;
    --smax)
      SMAX="$2"
      shift 2
      ;;
    --config)
      TRAINING_CONFIG="$2"
      shift 2
      ;;
    --mem)
      MEM="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    --)
      shift
      SBATCH_ARGS+=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown option '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$GRID" && -n "$GRID_IDX" ]]; then
  echo "ERROR: use either --grid or --grid-idx, not both" >&2
  exit 1
fi

case "$MODE" in
  testing|training) ;;
  *)
    echo "ERROR: --mode must be 'testing' or 'training'; got '$MODE'" >&2
    exit 1
    ;;
esac

case "$VARIANT_SET" in
  core) N_VARIANTS="$CORE_VARIANT_COUNT" ;;
  all) N_VARIANTS="$ALL_VARIANT_COUNT" ;;
  *)
    echo "ERROR: --variant-set must be 'core' or 'all'; got '$VARIANT_SET'" >&2
    exit 1
    ;;
esac

if [[ -n "$GRID" ]]; then
  GRID_IDX="$(grid_to_idx "$GRID")"
fi

if [[ -n "$GRID_IDX" ]]; then
  if [[ ! "$GRID_IDX" =~ ^[0-9]+$ ]] || (( GRID_IDX < 0 || GRID_IDX >= N_GRIDS )); then
    echo "ERROR: --grid-idx must be an integer in [0,$((N_GRIDS - 1))]; got '$GRID_IDX'" >&2
    exit 1
  fi
fi

EXPORTS=""
append_export "MODE=$MODE"

case "$MODE" in
  training)
    if [[ -n "$GRID_IDX" ]]; then
      ARRAY_RANGE="$GRID_IDX"
    else
      ARRAY_RANGE="0-$((N_GRIDS - 1))"
    fi
    append_export "SMIN=$SMIN"
    append_export "SMAX=$SMAX"
    append_export "TRAINING_CONFIG=$TRAINING_CONFIG"
    ;;
  testing)
    if [[ -n "$GRID_IDX" ]]; then
      ARRAY_RANGE="0-$((N_VARIANTS - 1))"
      append_export "GRID_IDX=$GRID_IDX"
    else
      ARRAY_RANGE="0-$((N_GRIDS * N_VARIANTS - 1))"
    fi
    append_export "VARIANT_SET=$VARIANT_SET"
    append_export "NUM_SCENARIOS=$NUM_SCENARIOS"
    append_export "FEASIBLE_SAMPLES_PER_SCENARIO=$FEASIBLE_SAMPLES_PER_SCENARIO"
    if [[ -n "$MAX_ATTEMPTS_MULTIPLIER" ]]; then
      append_export "MAX_ATTEMPTS_MULTIPLIER=$MAX_ATTEMPTS_MULTIPLIER"
    fi
    ;;
esac

CMD=(sbatch "--array=$ARRAY_RANGE" "--export=$EXPORTS")
if [[ -n "$MEM" ]]; then
  CMD+=("--mem=$MEM")
fi
CMD+=("${SBATCH_ARGS[@]}" "$LAUNCHER")

cd "$PROJECT_DIR"
if (( DRY_RUN )); then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  "${CMD[@]}"
fi
