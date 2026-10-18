#!/bin/bash

TRAIN_H5="datagen_topo/datasets/IEEE123Master/IEEE123Master_training.h5"
VAL_H5="datagen_topo/datasets/IEEE123Master/IEEE123Master_test_feasible.h5"
TEST_H5="datagen_topo/datasets/IEEE123Master/IEEE123Master_test_feasible.h5"
TEST_FEASIBLE="datagen_topo/datasets/IEEE123Master/IEEE123Master_test_feasible.h5"
DATA_JSON="datagen_topo/datasets/IEEE123Master/network_config.json"
CASE_NAME="ieee123_fsnet"

RESULTS_DIR="results_draft"


# Training hyperparameters
EPOCHS=20
BATCH_SIZE=8
LEARNING_RATE=0.5e-4
HIDDEN_SIZE=1024
NUM_LAYERS=4

# Loss configuration (distance-based - recommended)
WEIGHT_DIST=0.01

# Solver configuration
LIN_STEPS=2
DIFF_STEPS=1
FS_MAXITER=30
FS_TOL=1e-6
FS_RHO=0.0

# Training and testing sample size
TRAIN_SAMPLES=200000
EVAL_SAMPLES=100
TEST_SAMPLES=1000

# module load cuda/12.9.1
# Run training
python methods/slfs_fsnet/train.py \
  --train_h5 "$TRAIN_H5" \
  --val_h5 "$VAL_H5" \
  --test_h5 "$TEST_H5" \
  --data_json "$DATA_JSON" \
  --case_name "$CASE_NAME" \
  --results_dir "$RESULTS_DIR" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --train_samples $TRAIN_SAMPLES \
  --eval_samples $EVAL_SAMPLES \
  --test_samples $TEST_SAMPLES \
  --learning_rate $LEARNING_RATE \
  --hidden_size $HIDDEN_SIZE \
  --num_layers $NUM_LAYERS \
  --weight_dist $WEIGHT_DIST \
  --lin_steps $LIN_STEPS \
  --diff_steps $DIFF_STEPS \
  --fs_maxiter $FS_MAXITER \
  --fs_tol $FS_TOL \
  --fs_rho $FS_RHO \
  --power_flow_solver fixed_point \
  --save_every 100 \
  --log_every 1 \
  --woodbury_ir_steps auto \
  --woodbury_batch_mode vmap \
  --seed 1 \