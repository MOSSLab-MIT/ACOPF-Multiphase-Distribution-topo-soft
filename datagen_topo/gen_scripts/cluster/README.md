# Datagen Cluster Launcher

One script, [`run_datagen.sh`](run_datagen.sh). Edit the `JOBS` list at the top, then submit:

```bash
bash datagen_topo/gen_scripts/cluster/run_datagen.sh              # all JOBS as one Slurm array
bash datagen_topo/gen_scripts/cluster/run_datagen.sh --mem=200G   # extra flags go to sbatch
```

Each `JOBS` line is `"<training|testing> <grid> <config> [julia flags...]"`:

- grid: `13 | 123 | 240 | 906`
- config: a file name in `datagen_topo/gen_scripts/configs/`; sampling parameters live there
- julia flags: forwarded to `src/generation/<mode>.jl`, override the config

## Example JOBS lines

```bash
JOBS=(
  # Training
  "training 906 training_defaults.yaml"
  "training 13  training_w060_005_035.yaml --output_suffix 060_005_035"
  "training 13  training_l02_u08.yaml      --output_suffix l02_u08"

  # Testing: one config per case (variant and output_suffix are read from the config)
  "testing 906 testing_defaults.yaml"            # base
  "testing 906 testing_low_load_high_der.yaml"
  "testing 906 testing_high_load_low_der.yaml"
  "testing 906 testing_shifted_m01_std01.yaml"   # ... m02 ... m05
  "testing 906 testing_oor_load.yaml"
)
```

| Testing config | Load scale | DER frac | Shift (`mu_rel` = `der_mu`) | Output |
|---|---|---|---|---|
| `testing_defaults.yaml` | [0.2, 1.5] | [0.0, 1.0] | – | `<case>_test_feasible.h5` |
| `testing_low_load_high_der.yaml` | [0.2, 0.5] | [0.6, 1.0] | – | `..._low_load_high_der.h5` |
| `testing_high_load_low_der.yaml` | [0.6, 1.0] | [0.0, 0.4] | – | `..._high_load_low_der.h5` |
| `testing_shifted_m0k_std0k.yaml` | [0.2, 1.5] | [0.0, 1.0] | 0.k | `..._shifted_m0k_std0k.h5` |
| `testing_oor_load.yaml` | [0.5, 2.0] | [0.0, 1.0] | – | `..._oor_load.h5` |

All tasks in one submission share the same Slurm resources. Submit feeders with different memory requirements separately using `--mem`.
