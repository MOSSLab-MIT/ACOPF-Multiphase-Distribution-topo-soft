#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=datagen
#SBATCH --output=/home/hoangh/orcd/scratch/Projects/ACOPF-Multiphase-Distribution-topo-soft/logs/datagen/datagen-%A_%a.out
#SBATCH --error=/home/hoangh/orcd/scratch/Projects/ACOPF-Multiphase-Distribution-topo-soft/logs/datagen/datagen-%A_%a.err
#SBATCH -p pi_donti,mit_preemptable
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G

# Edit JOBS below, then submit all of them as one Slurm array:
#   bash datagen_topo/gen_scripts/cluster/run_datagen.sh [extra sbatch flags, e.g. --mem=200G]
#
# One job per line: "<training|testing> <grid> <config in configs/> [julia flags...]"
# Grids: 13 | 123 | 240 | 906 | 8500. Julia flags override the config.

JOBS=(
  # "training 13 training_defaults.json"
  # "testing 13 testing_defaults.json"
  # "testing 13 val_defaults.json"
  # "testing 13 testing_high_load_low_der.json"
  # "testing 13 testing_low_load_high_der.json"

  # "training 123 training_defaults.json"
  # "training 123 training_w060_020_020.json"
  # "testing 123 testing_defaults.json"
  # "testing 123 val_defaults.json"
  # "testing 123 testing_high_load_low_der.json"
  # "testing 123 testing_low_load_high_der.json"
  # "testing 123 testing_shifted_m01_std01.json"
  # "testing 123 testing_shifted_m02_std02.json"
  # "testing 123 testing_shifted_m03_std03.json"
  # "testing 123 testing_shifted_m04_std04.json"
  # "testing 123 testing_shifted_m05_std05.json"

  # "training 240 training_defaults.json"
  # "training 240 training_w060_020_020.json"
  # "testing 240 testing_defaults.json"
  # "testing 240 testing_high_load_low_der.json"
  # "testing 240  val_defaults.json"
  # "testing 240 testing_low_load_high_der.json"
  # "testing 240 testing_shifted_m01_std01.json"
  # "testing 240 testing_shifted_m02_std02.json"
  # "testing 240 testing_shifted_m03_std03.json"
  # "testing 240 testing_shifted_m04_std04.json"
  # "testing 240 testing_shifted_m05_std05.json"

  # "training 8500 training_defaults.json"
  "training 8500 training_w060_020_020.json"
  "testing 8500 testing_defaults.json"
  "testing 8500  val_defaults.json"
  "testing 8500 testing_high_load_low_der.json"
  "testing 8500 testing_low_load_high_der.json"
  "testing 8500 testing_shifted_m01_std01.json"
  "testing 8500 testing_shifted_m02_std02.json"
  "testing 8500 testing_shifted_m03_std03.json"
  # "testing 8500 testing_shifted_m04_std04.json"
  # "testing 8500 testing_shifted_m05_std05.json"
)

set -euo pipefail

declare -A GRIDS=(
  [13]="datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss"
  [123]="datagen_topo/grid_data/123_bus/IEEE123Master.dss"
  [240]="datagen_topo/grid_data/240_bus/240bus_Master.dss"
  [906]="datagen_topo/grid_data/906_bus/906bus_Master.dss"
  [8500]="datagen_topo/grid_data/IEEE8500/8500_Master.dss"
)

# Outside Slurm: submit this script as an array with one task per JOBS entry.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
  exec sbatch --array=0-$(( ${#JOBS[@]} - 1 )) "$@" "datagen_topo/gen_scripts/cluster/run_datagen.sh"
fi

# Inside Slurm: run the JOBS entry selected by the array index.
cd "$SLURM_SUBMIT_DIR"
read -r -a JOB <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
MODE="${JOB[0]}"
[[ -n "${GRIDS[${JOB[1]}]:-}" ]] || { echo "ERROR: unknown grid '${JOB[1]}'; use one of: ${!GRIDS[*]}" >&2; exit 1; }
GRID="${GRIDS[${JOB[1]}]}"
CONFIG="datagen_topo/gen_scripts/configs/${JOB[2]}"

# `module load julia` is unreliable in batch shells; the default HSL_jll artifact is a
# placeholder (Ipopt -> INVALID_OPTION -> segfault), so point to the licensed libhsl.
export PATH="/orcd/software/core/001/pkg/julia/1.12.6/bin:$PATH"
export ACOPF_HSL_PATH="${ACOPF_HSL_PATH:-$HOME/Packages/HSL_jll.jl.v2025.7.21/override/lib/x86_64-linux-gnu-libgfortran5/libhsl.so}"

CMD=(
  julia --project=datagen_topo/FeederFlow.jl
  "datagen_topo/gen_scripts/create_${MODE}_dataset.jl" "$GRID"
  --config "$CONFIG"
  "${JOB[@]:3}"
)

echo "CMD: ${CMD[*]}"
"${CMD[@]}"
