#!/usr/bin/env python3
"""Histograms of |V|, substation power and branch flows: Penalty+SLFS vs Ipopt.

Reads the NPZ dumps from ``dump.py`` (``seed_N_feasible.npz`` for |V| and
``seed_N_feasible_flows.npz`` for flows) and writes one 2x3 figure per grid:

    |V|            substation P (total)   substation Q (total)
    branch |I|     branch P                branch Q

Ipopt is a filled blue histogram, Penalty+SLFS a red step outline, both density
normalised on shared bins. |V| and branch panels use a log y-axis so tails stay
visible; substation totals are linear.
Substation values are summed over phases (one value per sample); branch values
are pooled over samples x valid branch phase-terms (from-end, system p.u.).

Example (repo root, conda ``new_opt``)::

    python analysis/penalty_fs_vs_ipopt/plot_hist.py --grids 13 123 240 906 8500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.batch_size_inference_time.benchmark import GRID_PRESETS, resolve_grid
from analysis.penalty_fs_vs_ipopt.plot import (
    COLOR_FS,
    COLOR_IPOPT,
    DEFAULT_FIGURE_DIR,
    DEFAULT_GRIDS,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SEED,
    DEFAULT_TEST_VARIANT,
    GRID_DISPLAY,
    LABEL_FS,
    _load_dump,
    _resolve_path,
    _save_fig,
)

PAPER_RC = {
    "font.size": 14.0,
    "axes.titlesize": 15.0,
    "axes.labelsize": 15.0,
    "xtick.labelsize": 14.0,
    "ytick.labelsize": 14.0,
    "legend.fontsize": 14.0,
    "figure.titlesize": 15.0,
}
plt.rcParams.update(PAPER_RC)

N_BINS = 60
HIST_FIG_SIZE = (13.0, 7.0)
IPOPT_ALPHA = 0.45
LABEL_IPOPT = "Ipopt"

# (key, title, unit, log_y); keys select the arrays in ``_collect_quantities``.
# Substation totals have one value per sample and no heavy tails -> linear y.
PANELS: tuple[tuple[str, str, str, bool], ...] = (
    ("vmag", "Voltage magnitude $|V|$", "p.u.", True),
    ("psub", "Substation $P$ (3-phase total)", "p.u.", False),
    ("qsub", "Substation $Q$ (3-phase total)", "p.u.", False),
    ("branch_i", "Branch current $|I|$", "p.u.", True),
    ("branch_p", "Branch $P$", "p.u.", True),
    ("branch_q", "Branch $Q$", "p.u.", True),
)


def _load_flows(results_root: Path, case_dir: str, seed: int) -> dict[str, np.ndarray] | None:
    path = results_root / case_dir / f"seed_{seed}_{DEFAULT_TEST_VARIANT}_flows.npz"
    if not path.exists():
        print(f"  [warn] missing {path}; flow panels skipped (re-run dump.py)")
        return None
    return {k: np.asarray(v) for k, v in np.load(path).items()}


def _collect_quantities(
    arrays: dict[str, np.ndarray],
    flows: dict[str, np.ndarray] | None,
) -> dict[str, dict[str, np.ndarray]]:
    """Return ``{panel_key: {"ipopt": values, "fs": values}}`` as flat 1-D arrays."""
    out: dict[str, dict[str, np.ndarray]] = {
        "vmag": {s: arrays[f"vmag_{s}"].ravel() for s in ("ipopt", "fs")}
    }
    if flows is None:
        return out
    for s in ("ipopt", "fs"):
        # Phase-sum of the per-phase substation power -> one value per sample.
        out.setdefault("psub", {})[s] = flows[f"psub_phase_{s}"].sum(axis=1)
        out.setdefault("qsub", {})[s] = flows[f"qsub_phase_{s}"].sum(axis=1)
        for key in ("branch_i", "branch_p", "branch_q"):
            out.setdefault(key, {})[s] = flows[f"{key}_{s}"].ravel()
    return out


def _hist_panel(
    ax: plt.Axes, ipopt: np.ndarray, fs: np.ndarray, title: str, unit: str, log_y: bool
) -> None:
    """Overlay Ipopt / Penalty+SLFS density histograms on shared bin edges."""
    pooled = np.concatenate([ipopt, fs]).astype(np.float64)
    lo, hi = float(pooled.min()), float(pooled.max())
    if hi <= lo:  # degenerate (constant) data: give the bins a finite width
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, N_BINS + 1)
    ax.hist(ipopt, bins=edges, density=True, color=COLOR_IPOPT, alpha=IPOPT_ALPHA,
            label=LABEL_IPOPT)
    ax.hist(fs, bins=edges, density=True, histtype="step", color=COLOR_FS,
            linewidth=1.6, label=LABEL_FS)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(f"{title} ({unit})")
    ax.set_ylabel("Density")


def plot_grid_histograms(
    *,
    grid: str,
    arrays: dict[str, np.ndarray],
    flows: dict[str, np.ndarray] | None,
    figure_dir: Path,
    save_png: bool = False,
) -> Path:
    """Write the 2x3 histogram figure for one grid; return the PDF path."""
    data = _collect_quantities(arrays, flows)
    fig, axes = plt.subplots(2, 3, figsize=HIST_FIG_SIZE, constrained_layout=True)
    display = GRID_DISPLAY.get(grid, grid)
    # fig.suptitle(f"{LABEL_FS} vs Ipopt — {display} (feasible)")

    for ax, (key, title, unit, log_y) in zip(axes.ravel(), PANELS):
        if key not in data:
            ax.set_axis_off()
            continue
        _hist_panel(ax, data[key]["ipopt"], data[key]["fs"], title, unit, log_y)
    handles, labels = next(a for a in axes.ravel() if a.has_data()).get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)

    case_dir = resolve_grid(grid)["case_dir"]
    stem = figure_dir / f"{case_dir}_histograms"
    _save_fig(fig, stem, save_png=save_png)
    return stem.with_suffix(".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Histograms of |V|, substation power and branch flows (Penalty+SLFS vs Ipopt)"
    )
    parser.add_argument("--grids", nargs="+", default=list(DEFAULT_GRIDS),
                        choices=sorted(GRID_PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--png", action="store_true", help="Also write PNG copies")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = _resolve_path(Path(args.results_root))
    figure_dir = _resolve_path(Path(args.figure_dir))
    for grid in args.grids:
        case_dir = resolve_grid(grid)["case_dir"]
        arrays, _ = _load_dump(results_root, case_dir, args.seed)
        flows = _load_flows(results_root, case_dir, args.seed)
        plot_grid_histograms(
            grid=grid, arrays=arrays, flows=flows, figure_dir=figure_dir, save_png=args.png
        )


if __name__ == "__main__":
    main()
