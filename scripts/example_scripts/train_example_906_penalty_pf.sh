#!/bin/bash
# Example training script for Penalty PF method on IEEE 123-bus case

TRAIN_H5="datagen_topo/datasets/906bus_Master/906bus_Master_training.h5"
VAL_H5="datagen_topo/datasets/906bus_Master/906bus_Master_test_feasible.h5"
TEST_H5_VARIANTS=(
    "datagen_topo/datasets/906bus_Master/906bus_Master_test_feasible.h5"
    "datagen_topo/datasets/906bus_Master/906bus_Master_test_feasible_high_load_low_der.h5"
    "datagen_topo/datasets/906bus_Master/906bus_Master_test_feasible_low_load_high_der.h5"
)

DATA_JSON="datagen_topo/datasets/906bus_Master/network_config.json"
CASE_NAME="906bus_penalty_pf"

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

LOG_EVERY=1
EVAL_EVERY=1

echo "Starting training for $CASE_NAME..."
echo "Training data: $TRAIN_H5"
echo "Results dir: $RESULTS_DIR/$CASE_NAME"

python methods/penalty_pf/train.py \
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
    --save_every 100 \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --results_dir "$RESULTS_DIR" \
    --seed 42 \
    --power_flow_solver fixed_point \
    --woodbury_ir_steps 0 \
    --woodbury_batch_mode vmap
