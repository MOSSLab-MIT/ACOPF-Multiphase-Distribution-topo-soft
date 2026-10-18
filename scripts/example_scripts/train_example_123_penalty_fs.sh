#!/bin/bash
# Example training script for Penalty FS method on IEEE 123-bus case

TRAIN_H5="datagen_topo/datasets/IEEE123Master/IEEE123Master_training.h5"
VAL_H5="datagen_topo/datasets/IEEE123Master/IEEE123Master_test_feasible.h5"
TEST_H5_VARIANTS=(
    "datagen_topo/datasets/IEEE123Master/IEEE123Master_test_feasible.h5"
    "datagen_topo/datasets/IEEE123Master/IEEE123Master_test_feasible_low_load_high_der.h5"
    "datagen_topo/datasets/IEEE123Master/IEEE123Master_test_feasible_high_load_low_der.h5"
)

DATA_JSON="datagen_topo/datasets/IEEE123Master/network_config.json"
CASE_NAME="IEEE123Master_penalty_fs"

RESULTS_DIR="results_draft"

# Training parameters
BATCH_SIZE=32
EPOCHS=5
HIDDEN_SIZE=1024
NUM_LAYERS=4
LEARNING_RATE=0.5e-4

# Training and testing sample size
TRAIN_SAMPLES=200000
EVAL_SAMPLES=2000
TEST_SAMPLES=2000

# Penalty (fixed rho for entire run)
RHO=1e5

LOG_EVERY=1
EVAL_EVERY=1

# FS configuration
LIN_STEPS=2
DIFF_STEPS=0
FS_MAXITER=100
FS_TOL=1e-6
FS_RHO=0.0

echo "Starting training for $CASE_NAME..."
echo "Training data: $TRAIN_H5"
echo "Results dir: $RESULTS_DIR/$CASE_NAME"

module load cuda/12.9.1

python methods/penalty_fs/train.py \
    --train_h5 "$TRAIN_H5" \
    --val_h5 "$VAL_H5" \
    --test_h5_variants "${TEST_H5_VARIANTS[@]}" \
    --data_json "$DATA_JSON" \
    --case_name "$CASE_NAME" \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --train_samples $TRAIN_SAMPLES \
    --eval_samples $EVAL_SAMPLES \
    --test_samples $TEST_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --rho $RHO \
    --save_every 100 \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --results_dir "$RESULTS_DIR" \
    --seed 1 \
    --power_flow_solver fixed_point \
    --lin_steps $LIN_STEPS \
    --diff_steps $DIFF_STEPS \
    --fs_maxiter $FS_MAXITER \
    --fs_tol $FS_TOL \
    --fs_rho $FS_RHO \
    --woodbury_ir_steps auto \
    --woodbury_batch_mode vmap
