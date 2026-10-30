TRAIN_H5="datagen_topo/datasets/8500_Master/8500_Master_training.h5"
VAL_H5="datagen_topo/datasets/8500_Master/8500_Master_test_feasible_validation.h5"
TEST_H5_VARIANTS=(
    "datagen_topo/datasets/8500_Master/8500_Master_test_feasible.h5"
    "datagen_topo/datasets/8500_Master/8500_Master_test_feasible_low_load_high_der.h5"
    "datagen_topo/datasets/8500_Master/8500_Master_test_feasible_high_load_low_der.h5"
)

DATA_JSON="datagen_topo/datasets/8500_Master/network_config.json"
CASE_NAME="8500_penalty_pf"

RESULTS_DIR="results_draft"

# Training parameters
BATCH_SIZE=4
EPOCHS=20
HIDDEN_SIZE=2048
NUM_LAYERS=4
# LEARNING_RATE=0.03e-4 # this works well for this case : 0.03e-4
LEARNING_RATE=0.5e-5     #5e-6 before using layernorm

# Training and testing sample size
# Cost norm reads meta/primal_objective_value from --val_h5 (no topology precompute).
TRAIN_SAMPLES=200000
EVAL_SAMPLES=100
TEST_SAMPLES=2000

# Logging / eval (eval_every defaults to log_every in penalty_pf/train.py)
LOG_EVERY=1
EVAL_EVERY=1

echo "Starting training for $CASE_NAME..."
echo "Training data: $TRAIN_H5"
echo "Results dir: $RESULTS_DIR/$CASE_NAME"

module load cuda/12.9.1

source /home/hoangh/miniconda3/etc/profile.d/conda.sh
PROJECT_DIR="/home/hoangh/Projects/ACOPF-Multiphase-Distribution-topo"
cd $PROJECT_DIR
conda activate new_opt

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
    --warmup_epochs 1 \
    --save_every 100 \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --results_dir "$RESULTS_DIR" \
    --seed 42 \
    --power_flow_solver fixed_point \
    --woodbury_ir_steps 0 \
    --woodbury_batch_mode vmap

