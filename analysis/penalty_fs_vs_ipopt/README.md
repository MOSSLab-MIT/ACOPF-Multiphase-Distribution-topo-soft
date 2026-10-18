# Penalty FS vs Ipopt dispatch / voltage analysis

Compare DER active/reactive setpoints and voltage magnitudes from
`penalty_fs` (before and after deploy-time SLFS) against Ipopt labels in the
feasible test HDF5.

## Scope

- Checkpoints: `results/penalty_fs/{grid}_penalty_fs_rho1e5/seed_1/models/last`
- Dataset: `{grid}_test_feasible.h5`
- Grids: 13, 123, 240, 906, 8500

## Run (repo root, conda `new_opt`)

```bash
# Full dump (GPU recommended; 240 uses newton_gmres)
python analysis/penalty_fs_vs_ipopt/dump.py --grids 13 123 240 906 8500

# 8500 needs >128 GB host RAM: submit the big-memory job instead of running interactively
sbatch analysis/penalty_fs_vs_ipopt/run_slurm.sh

# Smoke (few samples)
python analysis/penalty_fs_vs_ipopt/dump.py --grids 13 --test-samples 64 --batch-size 32

# Plots + error table
python analysis/penalty_fs_vs_ipopt/plot.py --grids 13 123 240 906 8500

# Histograms of |V|, substation P/Q, branch |I|/P/Q (Penalty+SLFS vs Ipopt)
python analysis/penalty_fs_vs_ipopt/plot_hist.py --grids 13 123 240 906 8500
```

## Outputs

- `results/{case_dir}/seed_1_feasible.npz` — arrays
- `results/{case_dir}/seed_1_feasible_flows.npz` — Penalty+SLFS and Ipopt substation per-phase P/Q and
  branch |I|, P, Q (valid phase-terms only, float32); helpers in `flows.py`
- `results/{case_dir}/seed_1_feasible_summary.json` — MAE/RMSE/max + compute_norms
- `results/{case_dir}/seed_1_feasible_flows.npz` — substation per-phase P/Q and branch |I|, P, Q
  (valid phase-terms only, float32) for Penalty+SLFS (`*_fs`) and Ipopt (`*_ipopt`); computed from
  voltages by `flows.py` (series admittance only, same as the `|I| <= imax` constraint)
- `figures/{case_dir}_histograms.pdf` — 2x3 histograms (`plot_hist.py`)
- `figures/{case_dir}_dispatch_voltage_scatter.pdf` — scatters
- `figures/{case_dir}_voltage_profile_sample{N}.pdf` — |V| profile by phase (Ipopt / pred / FS)
- `figures/{case_dir}_voltage_difference_sample{N}.pdf` — |V| errors by phase
- `figures/{case_dir}_der_setpoints_sample{N}.pdf` — DER p/q by phase
- `figures/{case_dir}_der_difference_sample{N}.pdf` — DER p/q errors by phase
- `figures/{case_dir}_histograms.pdf` — 2x3 density histograms, Ipopt vs Penalty+SLFS
- `figures/profile_sample_selection.json` — auto-picked sample indices
- `figures/error_summary.md` / `.tex`

Profile sample (default): median after-FS DER L2 error among samples where SLFS
changed the DER setpoints. Override with `--sample-idx N`.

```bash
# Profiles only (reuse existing NPZ); PDF only by default
python analysis/penalty_fs_vs_ipopt/plot.py --no-scatters

# Optional PNG copies
python analysis/penalty_fs_vs_ipopt/plot.py --png
```

## Flow definitions (`flows.py`)

Same maths as the branch-current constraint `|I| <= branch_imax` (series admittance only,
system p.u., switches excluded): `I[b,r] = sum_c y_series[b,r,c] (v_f[c] - v_t[c])`,
`S[b,r] = v_f[r] conj(I[b,r])`. Substation power is `calc_substation_power_per_phase`.
