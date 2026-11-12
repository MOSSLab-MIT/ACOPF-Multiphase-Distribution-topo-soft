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
#   bash datagen_topo/gen_scripts/cluster/run_datagen_240.sh [extra sbatch flags, e.g. --mem=200G]
#
# One job per line: "<training|testing> <grid> <config in configs/> [julia flags...]"
# Grids: 13 | 123 | 240 | 906. Julia flags override the config.

# Price for this grid (pu, [quad, lin, const]): 2x the default sub [1,100,0] / DER [0.1,5,0]. Because base MVA of 240 is 2x that of 123, that make the substation cost per MVA the same.
# Every job of one grid must use the same price (the Python loader checks it).
PRICE="--sub_cost 2,200,0 --der_cost 0.2,10,0"

JOBS=(
  "training 240 training_defaults.yaml $PRICE"
  "training 240 training_w060_020_020.yaml $PRICE"
  "testing 240 testing_defaults.yaml $PRICE"
  "testing 240 testing_high_load_low_der.yaml $PRICE"
  "testing 240  val_defaults.yaml $PRICE"
  "testing 240 testing_low_load_high_der.yaml $PRICE"
  "testing 240 testing_shifted_m005_std005.yaml $PRICE"
  "testing 240 testing_shifted_m01_std01.yaml $PRICE"
  "testing 240 testing_shifted_m015_std015.yaml $PRICE"
  "testing 240 testing_shifted_m02_std02.yaml $PRICE"

)

set -euo pipefail

declare -A GRIDS=(
  [13]="datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss"
  [123]="datagen_topo/grid_data/123_bus/IEEE123Master.dss"
  [240]="datagen_topo/grid_data/240_bus/240bus_Master.dss"
  [906]="datagen_topo/grid_data/906_bus/906bus_Master.dss"
)

# Outside Slurm: submit this script as an array with one task per JOBS entry.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
  exec sbatch --array=0-$(( ${#JOBS[@]} - 1 )) "$@" "datagen_topo/gen_scripts/cluster/$(basename "${BASH_SOURCE[0]}")"
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
  "datagen_topo/src/generation/${MODE}.jl" "$GRID"
  --config "$CONFIG"
  "${JOB[@]:3}"
)

echo "CMD: ${CMD[*]}"
"${CMD[@]}"
