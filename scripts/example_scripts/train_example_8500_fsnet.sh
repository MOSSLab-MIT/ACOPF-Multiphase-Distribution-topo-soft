#!/bin/bash
TRAIN_H5="datagen_topo/datasets/8500_Master/8500_Master_training.h5"
VAL_H5="datagen_topo/datasets/8500_Master/8500_Master_test_feasible.h5"
TEST_H5_VARIANTS=(
    "datagen_topo/datasets/8500_Master/8500_Master_test_feasible.h5"
    "datagen_topo/datasets/8500_Master/8500_Master_test_infeasible.h5"
    "datagen_topo/datasets/8500_Master/8500_Master_test_feasible_low_load_high_der.h5"
    "datagen_topo/datasets/8500_Master/8500_Master_test_feasible_high_load_low_der.h5"
)

DATA_JSON="datagen_topo/datasets/8500_Master/network_config.json"
CASE_NAME="8500_fsnet"

RESULTS_DIR="results_draft"


# Training hyperparameters
EPOCHS=5
BATCH_SIZE=4
LEARNING_RATE=5e-6
HIDDEN_SIZE=2048
NUM_LAYERS=4

# Loss configuration (distance-based - recommended)
WEIGHT_DIST=0.01

# Solver configuration
LIN_STEPS=2
DIFF_STEPS=1
FS_MAXITER=50
FS_TOL=1e-6
FS_RHO=0.0

# Training and testing sample size
TRAIN_SAMPLES=200000
EVAL_SAMPLES=100
TEST_SAMPLES=2000

# module load cuda/12.9.1
# Run training
python methods/slfs_fsnet/train.py \
  --train_h5 "$TRAIN_H5" \
  --val_h5 "$VAL_H5" \
  --test_h5_variants "${TEST_H5_VARIANTS[@]}" \
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
  --woodbury_ir_steps 0 \
  --woodbury_batch_mode vmap \
  --seed 1 \