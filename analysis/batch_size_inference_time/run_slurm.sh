#!/usr/bin/env bash
#SBATCH --requeue
#SBATCH --job-name=bsit_penalty_fs
#SBATCH --output=logs/batch_size_inference_time/slurm_%A_%a.out
#SBATCH --error=logs/batch_size_inference_time/slurm_%A_%a.err
#SBATCH --array=0-4%5
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -G h200:1
#SBATCH --partition=pi_donti_gpu,mit_preemptable,mit_normal_gpu
#
# Batch-size inference-time benchmark (penalty_fs), one grid per array task.
#
# Submit from repository root:
#   sbatch analysis/batch_size_inference_time/run_slurm.sh
#
# Single grid (e.g. 240 → task 2):
#   sbatch --array=2 analysis/batch_size_inference_time/run_slurm.sh
#
# After all array tasks finish, plot:
#   python analysis/batch_size_inference_time/plot.py ...
#
# Override benchmark knobs via env (optional):
#   BSIT_CASE_SUFFIX=penalty_fs_rho1e5 BSIT_SEED=1 sbatch ...
#   BSIT_RESUME=0  — ignore partial results and rerun all batch sizes

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/analysis/batch_size_inference_time/run_slurm.sh" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

ANALYSIS_DIR="$ROOT/analysis/batch_size_inference_time"
LOG_DIR="$ROOT/logs/batch_size_inference_time"
RESULTS_DIR="$ANALYSIS_DIR/results"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

module load cuda/12.9.1

set +u
if [ -f "${CONDA_EXE%/*}/../etc/profile.d/conda.sh" ]; then
  source "${CONDA_EXE%/*}/../etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2
  exit 1
fi
conda activate new_opt
set -u

# Array index → grid preset (must stay aligned with GRID_PRESETS in benchmark.py).
GRIDS=(13 123 240 906 8500)

RESULTS_ROOT="${BSIT_RESULTS_ROOT:-results/penalty_fs}"
CASE_SUFFIX="${BSIT_CASE_SUFFIX:-penalty_fs_rho1e5}"
SEED="${BSIT_SEED:-1}"
WARMUP="${BSIT_WARMUP:-2}"
REPEATS="${BSIT_REPEATS:-3}"
TEST_SAMPLES="${BSIT_TEST_SAMPLES:-1024}"
BATCH_SIZES="${BSIT_BATCH_SIZES:-1 2 4 8 16 32 64}"

RESUME_ARGS=(--resume)
if [ "${BSIT_RESUME:-1}" = "0" ]; then
  RESUME_ARGS=(--no-resume)
fi

run_one_grid() {
  local grid="$1"
  local out="$RESULTS_DIR/results_${grid}.json"
  echo "=== Grid ${grid} (case_suffix=${CASE_SUFFIX} seed=${SEED}) ==="
  echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "Output: ${out} (incremental save after each batch size; resume=${BSIT_RESUME:-1})"
  # shellcheck disable=SC2086
  python "$ANALYSIS_DIR/benchmark.py" \
    --grid "$grid" \
    --results-root "$RESULTS_ROOT" \
    --case-suffix "$CASE_SUFFIX" \
    --seed "$SEED" \
    --warmup "$WARMUP" \
    --repeats "$REPEATS" \
    --test-samples "$TEST_SAMPLES" \
    --batch-sizes $BATCH_SIZES \
    --out "$out" \
    "${RESUME_ARGS[@]}"
}

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit with sbatch (SLURM_ARRAY_TASK_ID not set)}"
GRID="${GRIDS[$TASK_ID]}"
if [ -z "${GRID:-}" ]; then
  echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID} (expected 0-$(( ${#GRIDS[@]} - 1 )))" >&2
  exit 1
fi
run_one_grid "$GRID"
echo "Done grid ${GRID}."
