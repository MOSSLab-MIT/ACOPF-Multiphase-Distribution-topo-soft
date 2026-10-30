#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=retest_pd_123_rho100
#SBATCH --output=logs/testing/penalty_direct/%A_%a.out
#SBATCH --error=logs/testing/penalty_direct/%A_%a.err
#SBATCH --array=1-3
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -G h200:1
#SBATCH --partition=pi_donti_gpu,mit_preemptable
#
# Retest IEEE123Master penalty_direct_rho100 (seeds 1–3) for speedup timing.
# Array runs one seed at a time (%1) to reduce GPU contention.
#
# From repository root:
#   mkdir -p logs/testing/penalty_direct
#   sbatch scripts/slurm_runs/manual/retest_123_rho100_speedup.sh

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/methods/penalty_direct/test.py" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "$SUBMIT_DIR/../../.." && pwd)"
fi
cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/methods/penalty_direct/test.py" ]; then
  echo "ERROR: repository root not found at $PROJECT_DIR" >&2
  echo "Run sbatch from the repository root (see scripts/slurm_runs/README.md)." >&2
  exit 1
fi

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

SEED="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
CASE_NAME="IEEE123Master_penalty_direct_rho100"
CASE_DIR="datagen_topo/datasets/IEEE123Master"
CKPT="results/penalty_direct/${CASE_NAME}/seed_${SEED}/models/checkpoint_epoch_20"

if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

echo "Retesting ${CASE_NAME} seed=${SEED} on $(hostname)"
echo "Checkpoint: $CKPT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python methods/penalty_direct/test.py \
  --train_h5 "${CASE_DIR}/IEEE123Master_training.h5" \
  --data_json "${CASE_DIR}/network_config.json" \
  --checkpoint_path "$CKPT" \
  --case_name "$CASE_NAME" \
  --results_dir results/penalty_direct \
  --seed "$SEED" \
  --hidden_size 1024 \
  --num_layers 4 \
  --power_flow_solver fixed_point \
  --woodbury_ir_steps auto \
  --woodbury_batch_mode vmap \
  --test_batch_sizes 1 \
  --test_samples 2000 \
  --train_samples 200000 \
  --test_h5_variants \
    "${CASE_DIR}/IEEE123Master_test_feasible.h5"

echo "Done seed=${SEED}"
