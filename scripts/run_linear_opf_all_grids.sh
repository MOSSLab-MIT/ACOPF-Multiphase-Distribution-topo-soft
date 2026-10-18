#!/usr/bin/env bash
# Submit parallel Slurm jobs for fixed-reference linear OPF on standard grids.
#
# Submit (from repository root):
#   bash scripts/run_linear_opf_all_grids.sh
#   bash scripts/run_linear_opf_all_grids.sh 13 906
#   NUM_SHARDS=8 bash scripts/run_linear_opf_all_grids.sh 8500
#
# Worker mode (invoked by sbatch; do not call directly):
#   bash scripts/run_linear_opf_all_grids.sh --worker 13 feasible 0 1
#   bash scripts/run_linear_opf_all_grids.sh --merge 8500 feasible
#
# Environment overrides (passed through sbatch --export=ALL):
#   RESULTS_DIR      default: results/linear_opf
#   TEST_SAMPLES     default: 2000
#   SEED             default: 1
#   NUM_SHARDS       default: 1 for grids 13/123, 6 for 240/906/8500
#   CONDA_ENV        default: new_opt
#   SLURM_PARTITION  default: pi_donti

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RESULTS_DIR="${RESULTS_DIR:-results/linear_opf}"
TEST_SAMPLES="${TEST_SAMPLES:-2000}"
SEED="${SEED:-1}"
CONDA_ENV="${CONDA_ENV:-new_opt}"
SLURM_PARTITION="${SLURM_PARTITION:-pi_donti,mit_preemptable}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-logs/linear_opf}"

# Linear OPF is CPU-only (CVXPY + Clarabel).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export LINEAR_OPF_RSS_LOG="${LINEAR_OPF_RSS_LOG:-1}"

DATASET_ROOT="datagen_topo/datasets"
ALL_GRIDS=(13 123 240 906 8500)

activate_conda() {
  set +u
  module load cuda/12.9.1
  if [ -f "${CONDA_EXE%/*}/../etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_EXE%/*}/../etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda.sh not found; activate ${CONDA_ENV} manually and re-run." >&2
    exit 1
  fi
  conda activate "$CONDA_ENV"
  set -u
}

grid_resources() {
  # Echo: time mem cpus (cpus fixed at 24 for all grids).
  case "$1" in
    13|123) echo "12:00:00 32G 24" ;;
    240|906) echo "165:00:00 128G 24" ;;
    8500)  echo "7-00:00:00 100G 24" ;;
    *)
      echo "ERROR: unknown grid key '${1}'; expected one of 13, 123, 240, 906, 8500." >&2
      return 1
      ;;
  esac
}

merge_resources() {
  echo "00:30:00 4G 1"
}

grid_num_shards() {
  local grid_key="$1"
  if [ -n "${NUM_SHARDS:-}" ]; then
    echo "$NUM_SHARDS"
    return 0
  fi
  case "$grid_key" in
    13|123|240) echo "1" ;;
    906|8500) echo "8" ;;
    *)
      echo "ERROR: unknown grid key '${grid_key}'." >&2
      return 1
      ;;
  esac
}

grid_case_dir() {
  case "$1" in
    13) echo "IEEE13Nodeckt" ;;
    123) echo "IEEE123Master" ;;
    240) echo "240bus_Master" ;;
    906) echo "906bus_Master" ;;
    8500) echo "8500_Master" ;;
    *)
      echo "ERROR: unknown grid key '${1}'." >&2
      return 1
      ;;
  esac
}

grid_power_flow_solver() {
  case "$1" in
    240) echo "newton_gmres" ;;
    13|123|906|8500) echo "fixed_point" ;;
    *)
      echo "ERROR: unknown grid key '${1}'." >&2
      return 1
      ;;
  esac
}

list_variant_names() {
  local grid_key="$1"
  case "$grid_key" in
    13|123|240|906|8500)
      # echo "feasible"
      echo "infeasible"
      echo "feasible_high_load_low_der"
      echo "feasible_low_load_high_der"
      ;;
    *)
      echo "ERROR: unknown grid key '${grid_key}'." >&2
      return 1
      ;;
  esac
}

resolve_variant_h5() {
  local grid_key="$1"
  local variant="$2"
  local case_dir
  case_dir="$(grid_case_dir "$grid_key")"
  echo "${DATASET_ROOT}/${case_dir}/${case_dir}_test_${variant}.h5"
}

validate_grid_variant() {
  local grid_key="$1"
  local variant="$2"
  local case_dir data_json h5_path

  case_dir="$(grid_case_dir "$grid_key")"
  data_json="${DATASET_ROOT}/${case_dir}/network_config.json"
  h5_path="$(resolve_variant_h5 "$grid_key" "$variant")"

  if [ ! -f "$data_json" ]; then
    echo "ERROR: missing network config for grid ${grid_key}: $data_json" >&2
    return 1
  fi
  if [ ! -f "$h5_path" ]; then
    echo "ERROR: missing test HDF5 for grid ${grid_key} variant ${variant}: $h5_path" >&2
    return 1
  fi
}

variant_results_dir() {
  local grid_key="$1"
  local variant="$2"
  local case_dir
  case_dir="$(grid_case_dir "$grid_key")"
  echo "${RESULTS_DIR}/${case_dir}_linear_opf/seed_${SEED}/testing/${variant}"
}

run_worker() {
  local grid_key="$1"
  local variant="$2"
  local shard_id="$3"
  local num_shards="$4"
  local case_dir data_json h5_path power_flow_solver case_name

  validate_grid_variant "$grid_key" "$variant"

  case_dir="$(grid_case_dir "$grid_key")"
  data_json="${DATASET_ROOT}/${case_dir}/network_config.json"
  h5_path="$(resolve_variant_h5 "$grid_key" "$variant")"
  power_flow_solver="$(grid_power_flow_solver "$grid_key")"
  case_name="${case_dir}_linear_opf"

  echo "================================================================================"
  echo "Linear OPF | grid=${grid_key} | variant=${variant} | shard=${shard_id}/${num_shards} | samples=${TEST_SAMPLES}"
  echo "================================================================================"

  # 8500: Woodbury IR-step auto-probe segfaults on dense 8525x8525 matrices.
  # Use ir_steps=0 (matches experiment default for this grid).
  local woodbury_args=()
  if [[ "$grid_key" == "8500" ]]; then
    woodbury_args+=(--woodbury_ir_steps 0)
  fi

  python methods/linear_opf/test.py \
    --data_json "$data_json" \
    --test_h5 "$h5_path" \
    --test_samples "$TEST_SAMPLES" \
    --shard_id "$shard_id" \
    --num_shards "$num_shards" \
    --test_batch_sizes 1 \
    --case_name "$case_name" \
    --results_dir "$RESULTS_DIR" \
    --seed "$SEED" \
    --power_flow_solver "$power_flow_solver" \
    "${woodbury_args[@]}"
}

run_merge() {
  local grid_key="$1"
  local variant="$2"
  local variant_dir

  validate_grid_variant "$grid_key" "$variant"
  variant_dir="$(variant_results_dir "$grid_key" "$variant")"

  echo "================================================================================"
  echo "Linear OPF merge | grid=${grid_key} | variant=${variant}"
  echo "================================================================================"

  python methods/linear_opf/merge_shard_results.py \
    --variant_dir "$variant_dir" \
    --batch_size 1
}

submit_grids() {
  local -a grids=("$@")
  local grid variant num_shards shard_id time mem cpus merge_time merge_mem merge_cpus
  local -a shard_job_ids=()
  local shard_job_id merge_job_id variant_dir dependency

  if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found; run on a Slurm login node." >&2
    exit 1
  fi

  mkdir -p "$PROJECT_DIR/$SLURM_LOG_DIR"

  echo "Project root: $PROJECT_DIR"
  echo "Results dir:  $RESULTS_DIR"
  echo "Log dir:      $SLURM_LOG_DIR"
  echo "Grids:        ${grids[*]}"
  echo

  for grid in "${grids[@]}"; do
    num_shards="$(grid_num_shards "$grid")"
    read -r time mem cpus < <(grid_resources "$grid")
    read -r merge_time merge_mem merge_cpus < <(merge_resources)

    while IFS= read -r variant; do
      validate_grid_variant "$grid" "$variant"
      shard_job_ids=()

      for ((shard_id = 0; shard_id < num_shards; shard_id++)); do
        shard_job_id="$(
          sbatch --parsable \
            --requeue \
            --job-name="lop_${grid}_${variant}_s${shard_id}" \
            --output="${SLURM_LOG_DIR}/grid_${grid}_${variant}_s${shard_id}_%j.out" \
            --error="${SLURM_LOG_DIR}/grid_${grid}_${variant}_s${shard_id}_%j.err" \
            --time="$time" \
            --mem="$mem" \
            --cpus-per-task="$cpus" \
            --partition="$SLURM_PARTITION" \
            --export=ALL \
            "$SCRIPT_PATH" --worker "$grid" "$variant" "$shard_id" "$num_shards"
        )"
        shard_job_ids+=("$shard_job_id")
        echo "Submitted grid ${grid} variant ${variant} shard ${shard_id}/${num_shards}: job ${shard_job_id}"
      done

      if [ "$num_shards" -gt 1 ]; then
        dependency="afterok:$(IFS=,:; echo "${shard_job_ids[*]}")"
        merge_job_id="$(
          sbatch --parsable \
            --requeue \
            --job-name="lop_${grid}_${variant}_merge" \
            --output="${SLURM_LOG_DIR}/grid_${grid}_${variant}_merge_%j.out" \
            --error="${SLURM_LOG_DIR}/grid_${grid}_${variant}_merge_%j.err" \
            --time="$merge_time" \
            --mem="$merge_mem" \
            --cpus-per-task="$merge_cpus" \
            --partition="$SLURM_PARTITION" \
            --dependency="$dependency" \
            --export=ALL \
            "$SCRIPT_PATH" --merge "$grid" "$variant"
        )"
        echo "Submitted merge for grid ${grid} variant ${variant}: job ${merge_job_id} (depends on ${shard_job_ids[*]})"
      fi
    done < <(list_variant_names "$grid")
  done

  echo
  echo "All jobs submitted."
  echo "Shard logs: ${SLURM_LOG_DIR}/grid_<grid>_<variant>_s<shard>_<jobid>.{out,err}"
  echo "Merge logs: ${SLURM_LOG_DIR}/grid_<grid>_<variant>_merge_<jobid>.{out,err}"
  echo "Results: ${RESULTS_DIR}/<case_name>/seed_${SEED}/testing/<variant>/"
}

if [[ "${1:-}" == "--worker" ]]; then
  if [ "$#" -ne 5 ]; then
    echo "ERROR: --worker requires <grid> <variant> <shard_id> <num_shards>." >&2
    exit 1
  fi

  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/run_linear_opf_all_grids.sh" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
  fi
  cd "$PROJECT_DIR"
  activate_conda
  run_worker "$2" "$3" "$4" "$5"
  exit 0
fi

if [[ "${1:-}" == "--merge" ]]; then
  if [ "$#" -ne 3 ]; then
    echo "ERROR: --merge requires <grid> <variant>." >&2
    exit 1
  fi

  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/run_linear_opf_all_grids.sh" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
  fi
  cd "$PROJECT_DIR"
  activate_conda
  run_merge "$2" "$3"
  exit 0
fi

if [ "$#" -gt 0 ]; then
  GRIDS=("$@")
else
  GRIDS=("${ALL_GRIDS[@]}")
fi

cd "$PROJECT_DIR"
submit_grids "${GRIDS[@]}"
