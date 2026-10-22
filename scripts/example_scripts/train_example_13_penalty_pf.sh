#!/bin/bash
# Example training script for Penalty PF method on IEEE 13-bus case

TRAIN_H5="datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5"
VAL_H5="datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5"
# Slots 1–2: bare paths → mandatory feasible_only + infeasible_only; further entries: name=path.
TEST_H5_VARIANTS=(
    "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5"
    "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_infeasible.h5"
    "feasible_high_load_low_der=datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_high_load_low_der.h5"
    "feasible_low_load_high_der=datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_low_load_high_der.h5"
    "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_shifted_m01_std01.h5"
    "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_shifted_m02_std02.h5"
    "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_shifted_m03_std03.h5"
    "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_shifted_m04_std04.h5"
    "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible_shifted_m05_std05.h5"
)

DATA_JSON="datagen_topo/datasets/IEEE13Nodeckt/network_config.json"
CASE_NAME="IEEE13Nodeckt_penalty_pf"

RESULTS_DIR="results_draft"

# Training parameters
BATCH_SIZE=32
EPOCHS=20
HIDDEN_SIZE=512
NUM_LAYERS=4
LEARNING_RATE=1e-4

# Training and testing sample size
TRAIN_SAMPLES=200000
EVAL_SAMPLES=500
TEST_SAMPLES=500

# Penalty (fixed rho for entire run)
RHO=1e5

LOG_EVERY=1
EVAL_EVERY=1

echo "Starting training for $CASE_NAME..."
echo "Training data: $TRAIN_H5"
echo "Results dir: $RESULTS_DIR/$CASE_NAME"

module load cuda/12.9.1

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
    --rho $RHO \
    --save_every 100 \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --results_dir "$RESULTS_DIR" \
    --seed 42 \
    --power_flow_solver fixed_point \
    --woodbury_ir_steps 0 \
    --woodbury_batch_mode vmap \
    # --model_type gcn