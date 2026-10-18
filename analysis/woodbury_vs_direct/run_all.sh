#!/usr/bin/env bash
set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/analysis/woodbury_vs_direct/run_all.sh" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p analysis/woodbury_vs_direct/results

if [ "${CONDA_DEFAULT_ENV:-}" != "new_opt" ]; then
  exec conda run --no-capture-output -n new_opt bash "$0" "$@"
fi

# Grids to run; override with: bash run_all.sh -- 13 123 240 906 8500
if [ $# -eq 0 ]; then
  GRIDS=(13 123 240 906 8500)
elif [ "${1:-}" = "--" ]; then
  shift
  GRIDS=("$@")
else
  GRIDS=("$@")
fi

DEFAULT_BATCH_SIZE=8

for g in "${GRIDS[@]}"; do
  bs="${WBVD_BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
  out="${WBVD_OUT:-analysis/woodbury_vs_direct/results/results_${g}.json}"
  echo "=== Grid ${g} (batch_size=${bs}) ==="
  if ! python analysis/woodbury_vs_direct/benchmark.py \
    --grid "$g" --batch-size "$bs" --out "$out"; then
    echo "Warning: benchmark failed for grid ${g}" >&2
    exit 1
  fi
done

echo "Done. Results in analysis/woodbury_vs_direct/results/"
