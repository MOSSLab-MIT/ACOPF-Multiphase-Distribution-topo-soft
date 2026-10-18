#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAUNCHER="$SCRIPT_DIR/run_oor_load_testing.sh"

GRID_NAMES=("13" "123" "906")
N_GRIDS="${#GRID_NAMES[@]}"

usage() {
  cat <<'EOF'
Usage:
  submit_oor_load_testing.sh [options]

Generate test datasets with load scale [1.5, 2.0] (above training range [0.2, 1.5]).
Grids: 13, 123, 906 (same set as shifted datasets).

Options:
  --grid 13|123|906                       Run one feeder
  --grid-idx 0|1|2                        Same as --grid, using grid index
  --num-scenarios N                       Default: 10
  --feasible-samples-per-scenario N       Default: 100
  --max-attempts-multiplier N             Default: 5
  --mem VALUE                             Optional sbatch memory override, e.g. 200G
  --dry-run                               Print the sbatch command instead of submitting
  --help                                  Show this message
  --                                      Pass remaining arguments directly to sbatch

Examples:
  submit_oor_load_testing.sh --grid 906 --mem 200G
  submit_oor_load_testing.sh --mem 100G
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

GRID=""
GRID_IDX=""
NUM_SCENARIOS="10"
FEASIBLE_SAMPLES_PER_SCENARIO="100"
MAX_ATTEMPTS_MULTIPLIER="5"
MEM=""
DRY_RUN=0
SBATCH_ARGS=()

while (( $# > 0 )); do
  case "$1" in
    --grid)
      GRID="$2"
      shift 2
      ;;
    --grid-idx)
      GRID_IDX="$2"
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
append_export "NUM_SCENARIOS=$NUM_SCENARIOS"
append_export "FEASIBLE_SAMPLES_PER_SCENARIO=$FEASIBLE_SAMPLES_PER_SCENARIO"
append_export "MAX_ATTEMPTS_MULTIPLIER=$MAX_ATTEMPTS_MULTIPLIER"

if [[ -n "$GRID_IDX" ]]; then
  ARRAY_RANGE="$GRID_IDX"
  append_export "GRID_IDX=$GRID_IDX"
else
  ARRAY_RANGE="0-$((N_GRIDS - 1))"
fi

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
