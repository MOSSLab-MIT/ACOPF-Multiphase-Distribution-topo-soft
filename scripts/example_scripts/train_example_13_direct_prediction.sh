#!/bin/bash
# Example training script for Direct Prediction method on IEEE 13-bus case

TRAIN_H5="datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5"
VAL_H5="datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5"
# First entry (bare path) is the reference test set; further entries: name=path.
TEST_H5_VARIANTS=(
    "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5"
    "feasible_high_load_low_der=datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_high_load_low_der.h5"
    "feasible_low_load_high_der=datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_low_load_high_der.h5"
)

DATA_JSON="datagen_topo/datasets/IEEE13Nodeckt/network_config.json"
CASE_NAME="IEEE13Nodeckt_direct_prediction"

RESULTS_DIR="results_draft"

# Training parameters
BATCH_SIZE=64
EPOCHS=50
HIDDEN_SIZE=512
NUM_LAYERS=4
LEARNING_RATE=0.5e-4

# Training and testing sample size
TRAIN_SAMPLES=200000
EVAL_SAMPLES=2000
TEST_SAMPLES=2000

# Adaptive power-flow equality penalty configuration
RHO_EQ_INIT=100
RHO_RATE=0
VIOLATION_THRESHOLD=1e-3
RHO_DECAY_RATE=1.0
RHO_MAX=1e3

LOG_EVERY=1

echo "Starting training for $CASE_NAME..."
echo "Training data: $TRAIN_H5"
echo "Results dir: $RESULTS_DIR/$CASE_NAME"

module load cuda/12.9.1

python methods/direct_prediction/train.py \
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
    --rho_eq_init $RHO_EQ_INIT \
    --rho_rate $RHO_RATE \
    --violation_threshold $VIOLATION_THRESHOLD \
    --rho_decay_rate $RHO_DECAY_RATE \
    --rho_max $RHO_MAX \
    --save_every 100 \
    --log_every $LOG_EVERY \
    --results_dir "$RESULTS_DIR" \
    --seed 42 \
    --woodbury_ir_steps auto \
    --woodbury_batch_mode vmap \
    --power_flow_solver fixed_point
