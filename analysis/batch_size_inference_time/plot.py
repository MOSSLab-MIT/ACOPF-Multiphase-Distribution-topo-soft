#!/usr/bin/env python3
"""Paper figure: penalty_fs pipeline wall time per batch vs batch size.

Reads ``results/results_*.json`` from this analysis folder and writes a multi-panel
PDF (one panel per grid) with mean ± std error bars.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.batch_size_inference_time.benchmark import (  # noqa: E402
    BENCHMARK_NAME,
    GRID_PRESETS,
)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "figures" / "batch_size_inference_time.pdf"
)

PAPER_RC = {
    "font.size": 8.5,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 8.0,
}
plt.rcParams.update(PAPER_RC)

GRID_ORDER = ("13", "123", "240", "906", "8500")
GRID_TITLES = {
    "13": "IEEE 13",
    "123": "IEEE 123",
    "240": "240-bus",
    "906": "906-bus",
    "8500": "8500-bus",
}
LINE_COLOR = "#0072B2"
MARKER_SIZE = 4.5
LINE_WIDTH = 1.6
FIG_WIDTH_IN = 7.1
PANEL_HEIGHT_IN = 2.2


def load_result_files(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load ``results_*.json`` keyed by grid id."""
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("results_*.json")):
        with open(path) as f:
            payload = json.load(f)
        if payload.get("benchmark") != BENCHMARK_NAME:
            continue
        grid = str(payload.get("grid", path.stem.replace("results_", "")))
        payloads[grid] = payload
    return payloads


def _extract_curve(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_b = payload["by_batch_size"]
    sizes = sorted(int(k) for k in by_b)
    means = np.array(
        [by_b[str(b)]["pipeline_time_per_batch_mean_s"] for b in sizes],
        dtype=np.float64,
    )
    stds = np.array(
        [by_b[str(b)]["pipeline_time_per_batch_std_s"] for b in sizes],
        dtype=np.float64,
    )
    return np.asarray(sizes, dtype=np.float64), means, stds


def plot_batch_size_curves(
    payloads: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write a multi-panel PDF of pipeline batch wall time vs batch size."""
    grids = [g for g in GRID_ORDER if g in payloads]
    extras = sorted(set(payloads) - set(grids))
    grids.extend(extras)
    if not grids:
        raise FileNotFoundError("No matching results_*.json payloads to plot")

    n = len(grids)
    fig_w = FIG_WIDTH_IN
    fig_h = PANEL_HEIGHT_IN * max(1, (n + 2) // 3) if n > 3 else PANEL_HEIGHT_IN
    ncols = min(n, 3) if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, PANEL_HEIGHT_IN * nrows),
        sharex=False,
        squeeze=False,
    )

    for idx, grid in enumerate(grids):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sizes, means, stds = _extract_curve(payloads[grid])
        ax.errorbar(
            sizes,
            means,
            yerr=stds,
            fmt="-o",
            color=LINE_COLOR,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            capsize=2.5,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(int(s)) for s in sizes])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Pipeline time / batch (s)")
        ax.set_title(GRID_TITLES.get(grid, grid))
        ax.grid(True, which="both", alpha=0.3)

    # Hide unused axes.
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "penalty_fs deploy latency (NN+PF+SLFS; topology inverse excluded)",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot penalty_fs batch-size inference-time curves"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory with results_*.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = load_result_files(args.results_dir)
    missing = [g for g in GRID_PRESETS if g not in payloads]
    if missing:
        print(f"Note: missing grids (not plotted): {missing}")
    plot_batch_size_curves(payloads, args.output)


if __name__ == "__main__":
    main()
