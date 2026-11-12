# Self-Supervised Learning for Soft-Constrained AC-OPF in Multiphase Distribution Systems

Abstract: This paper presents a self-supervised learning framework for fast, soft-constrained AC optimal power flow (OPF) in unbalanced multiphase distribution systems.
The soft AC-OPF objective balances operating cost, renewable energy curtailment, and penalties for violating operating limits.
A neural network predicts distributed energy resource (DER) setpoints conditioned on operating conditions, switch configurations, and objective weights. The predicted setpoints are then processed through a DER projection and differentiable AC power flow to enforce device limits and network equalities. The framework is trained by directly minimizing the AC-OPF objective, without optimal-solution labels.
Experiments on 123-, 240-, and 906-bus systems yield mean objective gaps of 0.32%, 0.92%, and 0.56%, respectively, relative to IPOPT, across wide operating ranges.
Across these systems and the evaluated high-load/low-generation and low-load/high-generation scenarios, the largest mean gap is 1.04%, with mean speedups of 58--5272 times relative to IPOPT.

## Environment setup

Create the Python environment from the repo root:

```bash
conda env create -f environment.yml
conda activate new_opt
```

For **GPU training**, ensure JAX sees your CUDA build (the default pip JAX may be CPU-only):

```bash
pip install -U "jax[cuda12]"   # adjust for your CUDA version; see JAX install docs
```

Julia datagen uses a separate repo to read the OpenDSS file: https://github.com/hoang-tn/FeederFlow.jl


## Usage

Run from the repository root with `conda activate new_opt`. These launchers submit
Slurm jobs themselves. Supported grids: `13`, `123`, `240`, `906`.

### Generate data
```bash
mkdir -p logs/datagen
bash datagen_topo/gen_scripts/cluster/run_datagen.sh
```

Data goes to `datagen_topo/datasets/<case>/`. Generate the training, validation, and
additional test sets specified in [`configs/<grid>.yaml`](scripts/slurm_runs/configs/).
See the [data generation guide](datagen_topo/gen_scripts/USAGE.md) for configs and variants.

### Train

```bash
bash scripts/slurm_runs/proposed.sbatch 123
RHO=100 bash scripts/slurm_runs/direct_prediction.sbatch 123
```

Replace `123` with one or more grid keys, e.g. `240 906`. Seeds, resources, and hyperparameters come from the per-grid YAML configs. Training includes post-training tests; checkpoints and metrics go to `results/`, logs to `logs/training/`.

### Test saved checkpoints

```bash
TEST_SETS="feasible" bash scripts/slurm_runs/proposed_test.sbatch 13
RHO=100 TEST_SETS="feasible" bash scripts/slurm_runs/direct_prediction_test.sbatch 13
```

Keep the training config, seeds, and `RHO` consistent with the saved checkpoints.
`TEST_SETS` accepts space-separated test-set names. See the
[Slurm job guide](scripts/slurm_runs/README.md) for more options.
