# Dataset generation (`datagen_topo/gen_scripts`)

Generation entrypoints under `datagen_topo/src/generation/` build HDF5 datasets for Python training; configs and Slurm launchers live here. Run from the **repository root** with the FeederFlow project:

```bash
julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl [grid.dss] [flags...]
```

Use `testing.jl` in the same directory to generate OPF labels.

Grids live under `datagen_topo/grid_data/` (e.g. `13_bus/IEEE13Nodeckt.dss`, `123_bus/IEEE123Master.dss`, `906_bus/906bus_Master.dss`). Outputs go to `datagen_topo/datasets/<case_name>/`.

## Scripts

| Script | Purpose |
|--------|---------|
| [`training.jl`](../src/generation/training.jl) | Training inputs: loads, DER availability, topology switches, objectives (no OPF solve) |
| [`testing.jl`](../src/generation/testing.jl) | Test data: scenario topologies + Ipopt OPF labels (feasible / infeasible); `--variant base` or `--variant shifted` |
| [`../datagen_topo.jl`](../datagen_topo.jl) | Shared include aggregator for datagen code (not a Julia package) |
| [`configs/`](configs/) | YAML configs for training/testing generation (`--config <file>.yaml`; JSON configs are rejected). Needs `YAML.jl` in the Julia environment. |
| [`cluster/run_datagen.sh`](cluster/run_datagen.sh) | Slurm launcher (array index = grid, parameters from config); see [`cluster/README.md`](cluster/README.md) |


---

## Load and DER sampling

Training, validation and test data all use the same two samplers. Each sample draws one feeder-wide factor plus per-element noise:

```text
load:  pd_i = α·η_i·pd_ref_i           α ~ U[l, u]              η_i ~ U[1-σ, 1+σ], σ = load.sigma (configs: 0.3)
       qd_i = α·η'_i·qd_ref_i          (same α, independent η')
DER:   avail_i = min(α_d·η_i, 1)·rated_i   α_d ~ U[frac_l, frac_u]   η_i ~ U[1-σ, 1+σ], σ = der.sigma (configs: 0.5)
```

- `load.l/u` and `der.frac_l/frac_u` bound the feeder-wide factor, not each element. A load can reach `u·(1+σ)`; a DER unit can exceed `frac_u` by up to `(1+σ)`, and only its rating clips it.
- Clipping lowers the feeder-wide DER total at high `α_d`: at `α_d = 1` the expected total is `(1 - σ/4)` of rated, or `0.875` for `σ = 0.5`.
- Per-element variation around the feeder-wide level is `σ/√3`, about 17% for load and 29% for DER.

Datasets generated before this sampler (independent uniform DER draws per unit, load `σ = 0.1`) have different distributions. Archive old datasets and results before regenerating: file names are unchanged, so new files overwrite old ones and results reuse the same case names.

---

## Training data

### What it does

- Samples `smin:smax` seeds (one column per seed).
- Mixes **load/DER regimes** defined by script defaults or `--config configs/training_defaults.yaml` (default mix: base `1.0`, high_load_low_der `0.0`, low_load_high_der `0.0`); schedule is round-robin by allocated counts.
- Per sample: random connected topology (`TopologySampler`), `changing_switch_status`, loads, DER availability, objective weights.
- Writes **inputs only** to `<case>_training.h5`, or `<case>_training_<suffix>.h5` when `output_suffix` is set in the YAML config or CLI. Generate or reuse `network_config.json` separately for Python loaders.

### CLI

| Flag | Meaning |
|------|---------|
| `--smin`, `--smax` | Inclusive seed range (default `1:1000` in script config) |
| `--load_l`, `--load_u` | Override load scale bounds on **all** regimes |
| `--der_frac_l`, `--der_frac_u` | Override the feeder-wide DER factor bounds on **all** regimes |
| `--flip_probability`, `--radial_ratio` | Topology sampler |
| `--floating_point_type` | `float32` or `float64` (default `float64`) |
| `--config` | Optional YAML config; see [`configs/training_defaults.yaml`](configs/training_defaults.yaml) |
| `--output_suffix` | Append `_<suffix>` to the output filename; overrides config `output_suffix` (default `""`). `training_w060_020_020.yaml` sets `060_020_020`. |

Positional: `[grid.dss]` or legacy `[grid.dss smin smax]` (overridden by `--smin` / `--smax` when set).

### Example (IEEE 13-bus)

```bash
# Smoke
julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --smin 1 --smax 300

# Full (200k samples)
julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --smin 1 --smax 200000
```

**Output:** `datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5`

**HDF5 groups:** `input/` (`seed`, `regime_id`, `pd`, `qd`, `pder_avai`, `changing_switch_status`, `is_radial`, `obj_weights`), `metadata/` (dimensions, regime JSON, topology settings).

---

## Test data (scenario-based)

### Scenario model

A **scenario** is:

- one fixed topology (`changing_switch_status` / compact switch vector), and  
- many load/DER profiles solved with **Ipopt** (Feasible samples are resampled until the target count or the attempt cap).

The generator writes **combined HDF5 files** (`*_test_feasible*.h5`, `*_test_infeasible*.h5`).

`scenario_mode=false` is **not** supported.

### Main CLI flags

| Flag | Default (in script) | Meaning |
|------|------------------------|---------|
| `--variant` | config `shift.enabled` (`false` → `base`) | `base` or `shifted` (shifted applies Gaussian load/DER perturbation) |
| `--num_scenarios` | `10` | Number of distinct topologies |
| `--feasible_samples_per_scenario` | `100` | Target feasible OPF samples per scenario |
| `--max_attempts_multiplier` | `5` | `attempt_limit = feasible_samples_per_scenario × multiplier` |
| `--seed_base` | `1234` | RNG for topology plan (topologies only) |
| `--seed_start` | `2000001` | First profile seed (`randomness.smin`): seeds loads, DER availability, objective weights |
| `--output_suffix` | config `output_suffix` (`""`) | Appends `_<suffix>` to output filenames |
| `--save_config` | `true` | Write `network_config.json` (or `network_config_<suffix>.json`) |
| `--load_l`, `--load_u` | script `config` | Load scale bounds |
| `--der_frac_l`, `--der_frac_u` | script `config` | Bounds of the feeder-wide DER factor `α` (see [Load and DER sampling](#load-and-der-sampling)) |
| `--flip_probability`, `--radial_ratio` | script `config` | Topology sampling |
| `--floating_point_type` | `float64` | Export precision |
| `--shift_mu`, `--shift_der_mu` | script `config` | Shifted variant only: shift magnitude on loads / DER (see `random_sign`) |
| `--shift_random_sign` | `true` | When `true` (default), each load/DER element gets `±(mu + sigma·|Z|)` with independent random sign; when `false`, legacy signed Gaussian `mu + sigma·Z` |
| `--config` | [`configs/testing_defaults.yaml`](configs/testing_defaults.yaml) | Optional YAML config merged before CLI overrides |

Config keys `shift/sigma_rel`, `shift/der_sigma`, `shift/random_sign`, etc. are represented in [`configs/testing_defaults.yaml`](configs/testing_defaults.yaml).

With the default `random_sign=true`, `--shift_mu` and `--shift_der_mu` set the **magnitude** of the relative shift (not a one-sided upward bias). Pass `--shift_random_sign false` to recover the previous signed-Gaussian behavior.

### Feasible resampling

Per scenario:

```text
attempt_limit = feasible_samples_per_scenario × max_attempts_multiplier
```

Each attempt runs OPF; **feasible** solves (`LOCALLY_SOLVED` / `ALMOST_LOCALLY_SOLVED`) count toward the feasible target. If the cap is hit early, the run logs a partial summary. Scenario 1’s first attempt is a warm-up and is not stored.

### Outputs

Under `datagen_topo/datasets/<case_name>/`:

| File | Content |
|------|---------|
| `<case>_test_feasible.h5` | Feasible OPF samples (all scenarios), `--variant base` |
| `<case>_test_infeasible.h5` | Infeasible samples (if any collected) |
| `<case>_test_feasible_<suffix>.h5` | Regime variant (`--output_suffix`) |
| `<case>_test_feasible_shifted.h5` | Shifted feasible samples (`--variant shifted`) |
| `<case>_test_infeasible_shifted.h5` | Shifted infeasible samples (if any collected) |

### Examples (IEEE 13-bus)

**Base feasible test set** (matches [README](../../README.md) quick start):

```bash
julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/testing.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss \
  --num_scenarios 10 --feasible_samples_per_scenario 100 \
  --max_attempts_multiplier 5
```

**Load/DER regime variants** (for multi-split eval in training scripts):

```bash
julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/testing.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss \
  --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 \
  --load_l 0.2 --load_u 0.4 --der_frac_l 0.7 --der_frac_u 1.0 \
  --output_suffix low_load_high_der --save_config false

julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/testing.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss \
  --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 \
  --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 \
  --output_suffix high_load_low_der --save_config false
```

**Shifted test** (default `random_sign=true` → `1 ± (mu + sigma·|Z|)` per element):

```bash
julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/testing.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss \
  --variant shifted --num_scenarios 10 --feasible_samples_per_scenario 100 \
  --max_attempts_multiplier 5 --shift_mu 0.1 --shift_der_mu 0.2
```

After the shift, each load is clamped between `0` and `u·(1+σ)` times its reference (`1.95×` for standard configs, either sign), and DER availability is clamped to `[0, rated]`. Unshifted base samples are never clipped by these caps.

**Legacy signed-Gaussian shift** (no forced random sign):

```bash
julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/testing.jl \
  datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss \
  --variant shifted --num_scenarios 10 --feasible_samples_per_scenario 100 \
  --max_attempts_multiplier 5 --shift_mu 0.1 --shift_der_mu 0.2 --shift_random_sign false
```

---

## HDF5 schema (test / scenario files)

**`input/`:** `seed`, `scenario_index`, `pd`, `qd`, `pder_avai`, `changing_switch_status`, `is_radial`, `obj_weights` (6 columns: `[w_cost, w_curtail, ρ_V, ρ_I, ρ_θ, ρ_sub]`)  

**`primal/`:** `vr`, `vi`, `pder`, `qder`, `psub`, `qsub`, `psub_phase`, `qsub_phase` (OPF solution; scalar `psub`/`qsub` are phase sums)  

**`network_config.json`:** `sub_smax_phase` — per-phase substation apparent-power limits (p.u.), length `n_slk` (soft: `|S_k|² ≤ smax_k² + s_sub,k`).

**`metadata/`:** sizes, topology sampler settings, and the price used (`sub_cost_lin`, `sub_cost_quad`, `der_cost_lin`, `der_cost_quad`, pu).

**`meta/`:** Ipopt termination, timings, objectives (`primal_objective_value` is the soft objective, penalties included)  

Python loaders expand compact `changing_switch_status` to full `branch_status` via `network_config.json` ([`core/grids/data.py`](../../core/grids/data.py)).

---

## Using outputs in Python

Training / test entrypoints expect HDF5 paths, for example:

- `*_training.h5`
- `*_test_feasible.h5` (mandatory eval slots in example scripts)
- `*_test_infeasible.h5` (second mandatory slot can repeat feasible path or point to infeasible file)
- `*_test_feasible_<suffix>.h5` via `name=path` in `--test_h5_variants`
- `*_test_feasible_shifted.h5` for shifted evaluation (`--variant shifted`)

---

## Legacy positional CLI

Still accepted for test scripts:

```bash
julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/testing.jl \
  <grid.dss> <smin> <smax>
```

Maps to `num_scenarios=1`, `feasible_samples_per_scenario = smax - smin + 1` (unless scenario flags override).

Shifted legacy (`--variant shifted`): `<grid> <smin> <smax> [shift_mu] [shift_der_mu] [output_suffix]`.

Prefer explicit `--num_scenarios` and `--feasible_samples_per_scenario` for new work.

---

## Customizing behavior

- **Training regimes:** default mix is all-base (`training_defaults.yaml`, weights `1.0/0.0/0.0`). Previous default mix (`0.6/0.05/0.35`) lives in `configs/training_w060_005_035.yaml` with output suffix `060_005_035`. Load-cap OOR training (all-base load `[0.2, 0.8]`) uses `configs/training_l02_u08.yaml` with output suffix `l02_u08` (launcher: `cluster/run_datagen.sh`, see `cluster/README.md`).
- **Disjoint splits:** profile seed `s` gives the same load/DER draws in every split, so each split uses its own seed block: training `smin:smax` (`1:200000`), validation `1000001+` ([`configs/val_defaults.yaml`](configs/val_defaults.yaml)), test `2000001+` (all `testing_*.yaml`; test variants share the block so they are paired). A test run uses at most `num_scenarios × feasible_samples_per_scenario × max_attempts_multiplier + 1` seeds. Keep training `smax < 1000001`. `seed_base` does not separate splits: it only seeds topologies.
- **Test sampling defaults:** use `--config datagen_topo/gen_scripts/configs/testing_defaults.yaml` as the editable template.
- **Soft V / I / θ / substation limits:** the OPF is soft-only. Voltage magnitude, branch current, angle-difference and per-phase substation apparent-power limits get nonnegative slacks with an L1 penalty `ρ_V·Σs_V + ρ_I·Σs_I + ρ_θ·Σs_θ + ρ_sub·Σs_sub` in raw pu² units (DER limits and power balance stay hard). The `penalty` config block (`rho_min`, `rho_max`, order `[ρ_V, ρ_I, ρ_θ, ρ_sub]`; configs use `[1, 1, 1, 100]` / `[100, 100, 100, 200]`, code default `[100, 100, 100, 100]` / `[1000, 1000, 1000, 200]`) sets the per-sample ρ, sampled log-uniformly; `obj` still samples `[w_cost, w_curtail]` uniformly. Legacy 3-entry penalty blocks are rejected. Infeasibility now only arises from power balance (loadability), so `*_test_infeasible.h5` is normally empty and not written. Julia tests: `tests/test_soft_constraints.jl`.
- **Price:** the `cost` block (`sub_cost`, `der_cost`, each `[quad, lin, const]`, default `[1.0, 100.0, 0.0]` / `[0.1, 5.0, 0.0]`) sets the substation and DER cost `quad·p² + lin·p` on **per-unit** power, so the price per MW scales with each feeder's `Sbase`. `der_cost` applies to every DER; the constant term must be 0 (it is ignored). Labels and `network_config.json` both carry the price, so every config used for one case (training, validation, test, shifted) must share the same `cost` block. Each HDF5 records it in `metadata/{sub,der}_cost_{lin,quad}`, and the Python loader rejects a file whose price differs from `network_config.json`. Julia tests: `tests/test_config_cost.jl`.
- **Topology / OPF:** shared code now loads through `datagen_topo/datagen_topo.jl`, with implementation files under `datagen_topo/src/`.

See also the repository [README](../../README.md) and [AGENTS.md](../../AGENTS.md).
