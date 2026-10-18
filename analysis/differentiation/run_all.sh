#!/usr/bin/env bash
set -euo pipefail

# When launched via sbatch, BASH_SOURCE may be a spool copy or a relative path.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/analysis/differentiation/run_all.sh" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p analysis/differentiation/results

if [ "${CONDA_DEFAULT_ENV:-}" != "new_opt" ]; then
  exec conda run --no-capture-output -n new_opt bash "$0" "$@"
fi

# Grids to run, override with: bash run_all.sh -- 13 123 906
GRIDS=("${@:-906}")
if [ "${GRIDS[0]}" = "--" ]; then
  shift
  GRIDS=("$@")
fi

for g in "${GRIDS[@]}"; do
  bs=4
  case "$g" in
    13|123) samples=256 ;;
    906)    samples=256 ;;
    8500)   samples=4; bs=1 ;;
    *)      samples=64 ;;
  esac

  # Optional overrides: PFJAC_SAMPLES, PFJAC_BATCH_SIZE, PFJAC_OUT
  if [ -n "${PFJAC_SAMPLES:-}" ]; then
    samples=$PFJAC_SAMPLES
  fi
  if [ -n "${PFJAC_BATCH_SIZE:-}" ]; then
    bs=$PFJAC_BATCH_SIZE
  fi
  out="${PFJAC_OUT:-analysis/differentiation/results/results_${g}.json}"

  echo "=== Grid ${g} (samples=${samples}, batch_size=${bs}) ==="
  python analysis/differentiation/compare_jacobians.py \
    --grid "$g" --samples "$samples" --batch-size "$bs" \
    --maxiter 12 --diffiters 1 2 3 4 \
    --out "$out"
done

echo "Done. Results in analysis/differentiation/results/"
