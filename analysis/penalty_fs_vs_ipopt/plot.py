#!/usr/bin/env python3
"""Scatter and profile plots: penalty_fs pred / after-SLFS vs Ipopt.

Reads NPZ dumps from ``dump.py`` and writes:
  - per-grid scatter figures (pder / qder / |V|)
  - per-grid DER + voltage profile figures for one picked sample
  - error-summary markdown/tex table

Profile sample selection (default): among samples where SLFS changed DER
setpoints, pick the median after-FS DER L2 error vs Ipopt (representative,
not worst-case). Override with ``--sample-idx``. Profile figures omit
Pred (before SLFS) unless ``--plot-pred`` is set.

Example (repo root, conda ``new_opt``)::

    python analysis/penalty_fs_vs_ipopt/plot.py
    python analysis/penalty_fs_vs_ipopt/plot.py --no-scatters --profiles
"""

from __future__ import annotations

import argparse
import ast
import json
import mmap
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.batch_size_inference_time.benchmark import GRID_PRESETS, resolve_grid

DEFAULT_GRIDS = ("13", "123", "240", "906", "8500")
DEFAULT_SEED = 1
DEFAULT_TEST_VARIANT = "feasible"
DEFAULT_RESULTS_ROOT = Path("analysis/penalty_fs_vs_ipopt/results")
DEFAULT_FIGURE_DIR = Path("analysis/penalty_fs_vs_ipopt/figures")
MAX_SCATTER_POINTS = 50_000
SCATTER_SEED = 0
SLFS_CHANGE_TOL = 1e-8

PAPER_RC = {
    "font.size": 14.0,
    "axes.titlesize": 15.0,
    "axes.labelsize": 15.0,
    "xtick.labelsize": 13.0,
    "ytick.labelsize": 13.0,
    "legend.fontsize": 12.0,
    "figure.titlesize": 15.0,
}
plt.rcParams.update(PAPER_RC)

# pred = orange, fs = red, ipopt = blue
COLOR_PRED = "#E69F00"
COLOR_FS = "#D62728"
COLOR_IPOPT = "#0072B2"
LABEL_FS = "Penalty+SLFS"
COLOR_IDENTITY = "#333333"
COLOR_LIMITS = "lightgray"

PROFILE_FIG_DPI = 300
PROFILE_FIG_SIZE_V = (10.0, 7.4)
PROFILE_FIG_SIZE_VERR = (12.0, 5.6)
PROFILE_FIG_SIZE_VERR_NO_PRED = (12.0, 3.2)
PROFILE_FIG_SIZE_DER = (12.0, 5.6)

# OpenDSS phase ids in ``all_index``: 1 = A, 2 = B, 3 = C.
PHASE_IDS: tuple[int, ...] = (1, 2, 3)
PHASE_LABELS: dict[int, str] = {1: "A", 2: "B", 3: "C"}

QUANTITY_SPECS: tuple[tuple[str, str, str], ...] = (
    ("pder", "Active DER $p$", "p.u."),
    ("qder", "Reactive DER $q$", "p.u."),
    ("vmag", "Voltage magnitude $|V|$", "p.u."),
)

GRID_DISPLAY: dict[str, str] = {
    "13": "13-bus",
    "123": "123-bus",
    "240": "240-bus",
    "906": "906-bus",
    "8500": "8500-node",
}

# Profile plots need only these network-config fields. The rest of the file is
# the bus admittance, which is about 11 GB for the 8500-node feeder.
_PROFILE_CFG_KEYS = frozenset(
    {
        "all_index",
        "all_is_slack",
        "all_to_slack_local",
        "der_node",
        "der_phase_to_unit",
        "n_net",
        "v_scale_slack",
        "v_sub_ref",
        "vmax",
        "vmin",
    }
)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (_ROOT / path).resolve()


def _load_dump(results_root: Path, case_dir: str, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = results_root / case_dir / f"seed_{seed}_{DEFAULT_TEST_VARIANT}.npz"
    summary_path = results_root / case_dir / f"seed_{seed}_{DEFAULT_TEST_VARIANT}_summary.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dump NPZ: {npz_path}. Run dump.py first.")
    arrays = {k: np.asarray(v) for k, v in np.load(npz_path).items()}
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return arrays, summary


def _flatten_pair(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(pred, dtype=np.float64).ravel(), np.asarray(target, dtype=np.float64).ravel()


def _subsample_xy(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    if n <= max_points:
        return x, y
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def _error_stats(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _scatter_panel(
    ax: plt.Axes,
    x_ipopt: np.ndarray,
    y_model: np.ndarray,
    *,
    color: str,
    title: str,
    xlabel: str,
    ylabel: str,
    max_points: int,
    rng: np.random.Generator,
) -> None:
    xs, ys = _subsample_xy(x_ipopt, y_model, max_points, rng)
    ax.scatter(xs, ys, s=4, alpha=0.25, c=color, linewidths=0, rasterized=True)
    lo = float(min(np.min(x_ipopt), np.min(y_model)))
    hi = float(max(np.max(x_ipopt), np.max(y_model)))
    pad = 0.02 * (hi - lo + 1e-12)
    lo_p, hi_p = lo - pad, hi + pad
    ax.plot([lo_p, hi_p], [lo_p, hi_p], color=COLOR_IDENTITY, linewidth=0.8, linestyle="--")
    ax.set_xlim(lo_p, hi_p)
    ax.set_ylim(lo_p, hi_p)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3, alpha=0.4)


def pick_profile_sample(
    arrays: dict[str, np.ndarray],
    sample_idx: int | None = None,
) -> tuple[int, dict[str, float]]:
    """Pick one sample index for profile plots.

    Default: median after-FS DER L2 error among samples where SLFS changed
    ``(pder, qder)``; if none changed, use the overall median.
    """
    n = int(arrays["pder_pred"].shape[0])
    if sample_idx is not None:
        if not (0 <= sample_idx < n):
            raise ValueError(f"sample_idx={sample_idx} out of range [0, {n})")
        i = int(sample_idx)
        meta = _sample_error_meta(arrays, i)
        meta["selection"] = "manual"
        return i, meta

    dp = arrays["pder_fs"] - arrays["pder_ipopt"]
    dq = arrays["qder_fs"] - arrays["qder_ipopt"]
    der_l2 = np.sqrt(np.sum(dp ** 2 + dq ** 2, axis=1))
    dx = np.sqrt(
        np.sum(
            (arrays["pder_fs"] - arrays["pder_pred"]) ** 2
            + (arrays["qder_fs"] - arrays["qder_pred"]) ** 2,
            axis=1,
        )
    )
    changed = dx > SLFS_CHANGE_TOL
    if np.any(changed):
        cand = np.where(changed)[0]
        order = np.argsort(der_l2[cand])
        i = int(cand[order[len(order) // 2]])
        selection = "median_der_l2_among_slfs_changed"
    else:
        i = int(np.argsort(der_l2)[len(der_l2) // 2])
        selection = "median_der_l2_all"
    meta = _sample_error_meta(arrays, i)
    meta["selection"] = selection
    meta["num_slfs_changed"] = int(np.sum(changed))
    return i, meta


def _sample_error_meta(arrays: dict[str, np.ndarray], i: int) -> dict[str, float]:
    dp = arrays["pder_fs"][i] - arrays["pder_ipopt"][i]
    dq = arrays["qder_fs"][i] - arrays["qder_ipopt"][i]
    dx = np.concatenate(
        [
            arrays["pder_fs"][i] - arrays["pder_pred"][i],
            arrays["qder_fs"][i] - arrays["qder_pred"][i],
        ]
    )
    dv = arrays["vmag_fs"][i] - arrays["vmag_ipopt"][i]
    return {
        "sample_idx": int(i),
        "der_l2_fs_vs_ipopt": float(np.linalg.norm(np.concatenate([dp, dq]))),
        "vmag_l2_fs_vs_ipopt": float(np.linalg.norm(dv)),
        "der_l2_fs_vs_pred": float(np.linalg.norm(dx)),
    }


def _nonslack_phase_ids(cfg: dict[str, Any]) -> np.ndarray:
    """OpenDSS phase id for each non-slack node, in voltage-vector order.

    ``all_index`` maps ``("bus", phase)`` to a 1-based position in the full
    node list. Non-slack nodes occupy ``0 .. n_net-1`` of that list, which is
    the order of ``vmag``.
    """
    n_net = int(cfg["n_net"])
    slack = cfg["all_is_slack"]
    phase_of = np.full(n_net, -1, dtype=np.int64)
    for key, value in cfg["all_index"].items():
        pos = int(value) - 1
        if pos < 0 or pos >= len(slack) or bool(slack[pos]):
            continue
        if pos >= n_net:
            raise ValueError(
                f"Non-slack node {key} has full index {pos}, past n_net={n_net}."
            )
        _bus, phase = ast.literal_eval(key)
        phase_of[pos] = int(phase)
    missing = int(np.sum(phase_of < 0))
    if missing:
        raise ValueError(f"{missing} non-slack nodes have no phase in all_index.")
    unknown = sorted(set(phase_of.tolist()) - set(PHASE_IDS))
    if unknown:
        raise ValueError(f"Unexpected OpenDSS phase ids: {unknown}.")
    return phase_of


def _json_complex(value: Any) -> np.ndarray:
    """Parse JSON complexes stored as ``{"re", "im"}``."""

    def convert(item: Any) -> Any:
        if isinstance(item, dict) and "re" in item and "im" in item:
            return complex(float(item["re"]), float(item["im"]))
        if isinstance(item, list):
            return [convert(child) for child in item]
        return item

    return np.asarray(convert(value), dtype=np.complex128)


def _slack_by_phase(cfg: dict[str, Any]) -> dict[int, dict[str, float]]:
    """Fixed substation ``|V|`` and limits, keyed by OpenDSS phase.

    Local slack voltage is ``v_sub_ref / v_scale_slack``, in ``slack_order``.
    ``all_to_slack_local`` is the 1-based index into that vector (0 off the slack).
    """
    v_ref = _json_complex(cfg["v_sub_ref"])
    v_scale = np.asarray(cfg["v_scale_slack"], dtype=np.float64)
    if v_ref.ndim != 1 or v_ref.shape != v_scale.shape:
        raise ValueError(
            f"v_sub_ref shape {v_ref.shape} != v_scale_slack shape {v_scale.shape}."
        )
    if np.any(v_scale <= 0.0):
        raise ValueError("v_scale_slack must be positive.")
    vmag = np.abs(v_ref / v_scale)
    vmin = np.asarray(cfg["vmin"], dtype=np.float64)
    vmax = np.asarray(cfg["vmax"], dtype=np.float64)
    slack_local = np.asarray(cfg["all_to_slack_local"], dtype=np.int64)
    if vmin.shape != vmax.shape or vmin.shape != slack_local.shape:
        raise ValueError(
            "vmin, vmax, and all_to_slack_local must have the same length."
        )

    by_phase: dict[int, dict[str, float]] = {}
    for key, value in cfg["all_index"].items():
        pos = int(value) - 1
        if pos < 0 or pos >= slack_local.shape[0]:
            raise ValueError(f"all_index entry {key} -> {value} is outside the node list.")
        local = int(slack_local[pos])
        if local <= 0:
            continue
        if local > vmag.shape[0]:
            raise ValueError(
                f"Slack node {key} has local index {local}, past n_slk={vmag.shape[0]}."
            )
        _bus, phase = ast.literal_eval(key)
        phase = int(phase)
        if phase in by_phase:
            raise ValueError(f"More than one slack node on phase {phase}.")
        by_phase[phase] = {
            "vmag": float(vmag[local - 1]),
            "vmin": float(vmin[pos]),
            "vmax": float(vmax[pos]),
        }
    missing = [phase for phase in PHASE_IDS if phase not in by_phase]
    if missing:
        raise ValueError(f"No slack voltage for OpenDSS phases {missing}.")
    return by_phase


def _prepend_slack(values: np.ndarray, sel: np.ndarray, slack_value: float) -> np.ndarray:
    """Phase-node series in the current order, with the slack value at index 0."""
    tail = np.asarray(values, dtype=np.float64)[sel]
    series = np.empty(tail.size + 1, dtype=np.float64)
    series[0] = float(slack_value)
    series[1:] = tail
    return series


def _json_value_end(buf: mmap.mmap, start: int) -> int:
    """Index just past the JSON value that begins at ``start``."""
    kind = buf[start]
    if kind == ord('"'):
        i = start + 1
        while True:
            quote = buf.find(b'"', i)
            if quote < 0:
                raise ValueError("Unterminated string in network_config.json.")
            slashes = 0
            j = quote - 1
            while j >= start and buf[j] == ord("\\"):
                slashes += 1
                j -= 1
            if slashes % 2 == 0:
                return quote + 1
            i = quote + 1
    if kind not in b"{[":
        i = start + 1
        while i < len(buf) and buf[i] not in b",}] \t\r\n":
            i += 1
        return i

    depth = 0
    i = start
    in_string = False
    while i < len(buf):
        ch = buf[i]
        if in_string:
            if ch == ord("\\"):
                i += 2
                continue
            if ch == ord('"'):
                in_string = False
            i += 1
            continue
        if ch == ord('"'):
            in_string = True
        elif ch in b"{[":
            depth += 1
        elif ch in b"}]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("Unterminated array or object in network_config.json.")


def _load_profile_config(path: Path) -> dict[str, Any]:
    """Load the network-config fields used by profile plots.

    Top-level keys are indented by two spaces. Scanning for that prefix skips
    the admittance blocks without parsing them.
    """
    found: dict[str, Any] = {}
    needle = b'\n  "'
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            pos = 0
            while len(found) < len(_PROFILE_CFG_KEYS):
                hit = buf.find(needle, pos)
                if hit < 0:
                    break
                key_start = hit + len(needle)
                key_end = buf.find(b'"', key_start)
                key = buf[key_start:key_end].decode("ascii")
                if key not in _PROFILE_CFG_KEYS:
                    pos = key_end
                    continue
                colon = buf.find(b":", key_end, key_end + 4)
                if colon < 0:
                    raise ValueError(f"Malformed key {key!r} in {path}.")
                val_start = colon + 1
                while val_start < len(buf) and buf[val_start] in b" \t\r\n":
                    val_start += 1
                val_end = _json_value_end(buf, val_start)
                found[key] = json.loads(buf[val_start:val_end])
                pos = val_end
    missing = _PROFILE_CFG_KEYS - found.keys()
    if missing:
        raise ValueError(f"{path} is missing profile fields: {sorted(missing)}.")
    return found


def _load_profile_aux(grid: str, sample_idx: int) -> dict[str, Any]:
    """Load vmin/vmax, phase ids, pder_avai (phase), and scenario_index."""
    preset = resolve_grid(grid)
    data_json = _ROOT / preset["data_json"]
    test_h5 = _ROOT / preset["test_h5"]
    cfg = _load_profile_config(data_json)
    n_net = int(cfg["n_net"])
    vmin = np.asarray(cfg["vmin"][:n_net], dtype=np.float64)
    vmax = np.asarray(cfg["vmax"][:n_net], dtype=np.float64)
    phase_id = _nonslack_phase_ids(cfg)
    slack = _slack_by_phase(cfg)
    der_phase_id = _der_phase_ids(cfg, phase_id)
    der_phase_to_unit = np.asarray(cfg["der_phase_to_unit"], dtype=np.int64)
    # Config is 1-based unit ids in some dumps; normalize to 0-based gather.
    if der_phase_to_unit.min() >= 1:
        der_phase_to_unit = der_phase_to_unit - 1

    with h5py.File(test_h5, "r") as f:
        pder_avai_unit = np.asarray(f["input/pder_avai"][sample_idx], dtype=np.float64)
        scenario_index = (
            int(f["input/scenario_index"][sample_idx])
            if "scenario_index" in f["input"]
            else None
        )
    pder_avai_phase = pder_avai_unit[der_phase_to_unit]
    return {
        "vmin": vmin,
        "vmax": vmax,
        "phase_id": phase_id,
        "slack": slack,
        "der_phase_id": der_phase_id,
        "pder_avai_phase": pder_avai_phase,
        "scenario_index": scenario_index,
        "n_net": n_net,
    }


def _phase_slices(phase_id: np.ndarray) -> dict[int, np.ndarray]:
    """Indices for each OpenDSS phase, preserving the parent-array order."""
    return {phase: np.flatnonzero(phase_id == phase) for phase in PHASE_IDS}


def _der_phase_ids(cfg: dict[str, Any], node_phase_id: np.ndarray) -> np.ndarray:
    """OpenDSS phase id of each DER phase.

    ``der_node`` is a 1-based index into the non-slack node list, the same
    order as ``node_phase_id`` and ``vmag``.
    """
    nodes = np.asarray(cfg["der_node"], dtype=np.int64) - 1
    n_net = int(node_phase_id.shape[0])
    if nodes.size and (int(nodes.min()) < 0 or int(nodes.max()) >= n_net):
        raise ValueError(
            f"der_node maps outside non-slack nodes [0, {n_net}): "
            f"min={int(nodes.min())}, max={int(nodes.max())}."
        )
    phase_id = node_phase_id[nodes]
    unknown = sorted(set(phase_id.tolist()) - set(PHASE_IDS))
    if unknown:
        raise ValueError(f"DER phases have unexpected OpenDSS phase ids: {unknown}.")
    return phase_id


def _save_fig(fig: plt.Figure, stem: Path, *, save_png: bool = False) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    if save_png:
        fig.savefig(stem.with_suffix(".png"), dpi=PROFILE_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem.with_suffix('.pdf')}")


def plot_grid_profiles(
    *,
    grid: str,
    arrays: dict[str, np.ndarray],
    figure_dir: Path,
    sample_idx: int | None,
    save_png: bool = False,
    plot_pred: bool = False,
) -> dict[str, Any]:
    """Notebook-style voltage/DER profile + error figures for one sample.

    ``plot_pred`` adds the pre-SLFS prediction. The default figures compare
    Ipopt with the voltage and setpoints after SLFS.
    """
    i, meta = pick_profile_sample(arrays, sample_idx=sample_idx)
    aux = _load_profile_aux(grid, i)
    display = GRID_DISPLAY.get(grid, grid)
    case_dir = resolve_grid(grid)["case_dir"]
    title_suffix = f"sample {i}"
    if aux["scenario_index"] is not None:
        title_suffix += f", scenario {aux['scenario_index']}"

    n_net = int(arrays["vmag_ipopt"].shape[1])
    n_der = int(arrays["pder_ipopt"].shape[1])
    if int(aux["phase_id"].shape[0]) != n_net:
        raise ValueError(
            f"phase map length {aux['phase_id'].shape[0]} != voltage length {n_net}."
        )
    phase_slices = _phase_slices(aux["phase_id"])
    if int(aux["der_phase_id"].shape[0]) != n_der:
        raise ValueError(
            f"DER phase map length {aux['der_phase_id'].shape[0]} != "
            f"setpoint length {n_der}."
        )
    der_slices = _phase_slices(aux["der_phase_id"])

    vm_ip = arrays["vmag_ipopt"][i]
    vm_pred = arrays["vmag_pred"][i]
    vm_fs = arrays["vmag_fs"][i]
    p_ip = arrays["pder_ipopt"][i]
    p_pred = arrays["pder_pred"][i]
    p_fs = arrays["pder_fs"][i]
    q_ip = arrays["qder_ipopt"][i]
    q_pred = arrays["qder_pred"][i]
    q_fs = arrays["qder_fs"][i]

    # A: voltage magnitude profile, one panel per phase (A/B/C).
    # Index 0 is that phase's fixed substation voltage. The remaining nodes
    # stay in voltage-vector order.
    fig, axes = plt.subplots(len(PHASE_IDS), 1, figsize=PROFILE_FIG_SIZE_V, sharey=True)
    for row, phase in enumerate(PHASE_IDS):
        ax = axes[row]
        sel = phase_slices[phase]
        slack = aux["slack"][phase]
        x_v = np.arange(sel.size + 1)
        ax.fill_between(
            x_v,
            _prepend_slack(aux["vmin"], sel, slack["vmin"]),
            _prepend_slack(aux["vmax"], sel, slack["vmax"]),
            color=COLOR_LIMITS,
            alpha=0.35,
            label="[vmin, vmax]",
        )
        ax.plot(
            x_v,
            _prepend_slack(vm_ip, sel, slack["vmag"]),
            "-",
            color=COLOR_IPOPT,
            linewidth=1.6,
            label="Ipopt",
        )
        if plot_pred:
            ax.plot(
                x_v,
                _prepend_slack(vm_pred, sel, slack["vmag"]),
                "--",
                color=COLOR_PRED,
                linewidth=1.1,
                label="Pred (before SLFS)",
            )
        ax.plot(
            x_v,
            _prepend_slack(vm_fs, sel, slack["vmag"]),
            "-.",
            color=COLOR_FS,
            linewidth=1.7,
            label=LABEL_FS,
        )
        ax.set_ylabel(f"Phase {PHASE_LABELS[phase]}\n" + r"$|V|$ (p.u.)")
        ax.set_xlim(-0.5, max(sel.size + 0.5, 0.5))
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc="upper right")
    # Shared y across phases, so one limit applies to every panel.
    axes[0].set_ylim(0.93, 1.07)
    axes[-1].set_xlabel("Node index")
    # fig.suptitle(f"{display} voltage magnitude — {title_suffix}", y=1.01)
    fig.tight_layout()
    _save_fig(fig, figure_dir / f"{case_dir}_voltage_profile_sample{i}", save_png=save_png)

    # B: voltage magnitude errors. Columns are phases. The pred row is optional.
    n_err_rows = 2 if plot_pred else 1
    verr_size = PROFILE_FIG_SIZE_VERR if plot_pred else PROFILE_FIG_SIZE_VERR_NO_PRED
    fig, axes = plt.subplots(
        n_err_rows,
        len(PHASE_IDS),
        figsize=verr_size,
        sharey=True,
        squeeze=False,
    )
    pred_err = vm_pred - vm_ip
    fs_err = vm_fs - vm_ip
    fs_row = 1 if plot_pred else 0
    for col, phase in enumerate(PHASE_IDS):
        sel = phase_slices[phase]
        # Substation voltage is fixed, so the error at index 0 is zero.
        x_v = np.arange(sel.size + 1)
        if plot_pred:
            axes[0, col].bar(
                x_v, _prepend_slack(pred_err, sel, 0.0), width=0.8, color=COLOR_PRED, alpha=0.85
            )
        axes[fs_row, col].bar(
            x_v, _prepend_slack(fs_err, sel, 0.0), width=0.8, color=COLOR_FS, alpha=0.85
        )
        for row in range(n_err_rows):
            axes[row, col].axhline(0.0, color="k", linewidth=0.8)
            axes[row, col].grid(True, axis="y", alpha=0.3)
            axes[row, col].set_xlim(-0.5, max(sel.size + 0.5, 0.5))
        axes[0, col].set_title(f"Phase {PHASE_LABELS[phase]}")
        axes[-1, col].set_xlabel("Node index")
    if plot_pred:
        axes[0, 0].set_ylabel(r"$|V|_{\mathrm{pred}} - |V|_{\mathrm{Ipopt}}$")
    axes[fs_row, 0].set_ylabel(r"$|V|_{\mathrm{FS}} - |V|_{\mathrm{Ipopt}}$")
    fig.suptitle(f"Voltage magnitude error — {title_suffix}", y=1.02)
    fig.tight_layout()
    _save_fig(fig, figure_dir / f"{case_dir}_voltage_difference_sample{i}", save_png=save_png)

    # C: DER setpoints, columns = phases, rows = p / q.
    fig, axes = plt.subplots(
        2,
        len(PHASE_IDS),
        figsize=PROFILE_FIG_SIZE_DER,
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    p_avai = aux["pder_avai_phase"]
    for col, phase in enumerate(PHASE_IDS):
        sel = der_slices[phase]
        x_der = np.arange(sel.size)
        axes[0, col].fill_between(
            x_der,
            0.0,
            p_avai[sel],
            color=COLOR_LIMITS,
            alpha=0.35,
            label="pder available",
        )
        axes[0, col].plot(
            x_der, p_ip[sel], "o-", color=COLOR_IPOPT, markersize=5, linewidth=1.4, label="Ipopt"
        )
        if plot_pred:
            axes[0, col].plot(
                x_der, p_pred[sel], "s--", color=COLOR_PRED, markersize=5, linewidth=1.2, label="Pred"
            )
        axes[0, col].plot(
            x_der,
            p_fs[sel],
            "^-.",
            color=COLOR_FS,
            markersize=5,
            linewidth=1.5,
            label=LABEL_FS,
        )
        axes[1, col].plot(
            x_der, q_ip[sel], "o-", color=COLOR_IPOPT, markersize=5, linewidth=1.4, label="Ipopt"
        )
        if plot_pred:
            axes[1, col].plot(
                x_der, q_pred[sel], "s--", color=COLOR_PRED, markersize=5, linewidth=1.5, label="Pred"
            )
        axes[1, col].plot(
            x_der,
            q_fs[sel],
            "^-.",
            color=COLOR_FS,
            markersize=5,
            linewidth=1.7,
            label=LABEL_FS,
        )
        for row in (0, 1):
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlim(-0.5, max(sel.size - 0.5, 0.5))
        axes[0, col].set_title(f"Phase {PHASE_LABELS[phase]}")
        axes[1, col].set_xlabel("DER index")
    axes[0, 0].set_ylabel(r"$p_{\mathrm{der}}$ (p.u.)")
    axes[1, 0].set_ylabel(r"$q_{\mathrm{der}}$ (p.u.)")
    axes[0, 0].legend(loc="upper left")
    # fig.suptitle(f"{display} DER setpoints — {title_suffix}", y=1.02)
    fig.tight_layout()
    _save_fig(fig, figure_dir / f"{case_dir}_der_setpoints_sample{i}", save_png=save_png)

    # D: DER errors, same phase columns. Pred is drawn beside after-SLFS when requested.
    fig, axes = plt.subplots(
        2,
        len(PHASE_IDS),
        figsize=PROFILE_FIG_SIZE_DER,
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    w = 0.35
    for col, phase in enumerate(PHASE_IDS):
        sel = der_slices[phase]
        x_der = np.arange(sel.size)
        if plot_pred:
            p_pred_x, p_fs_x = x_der - w / 2, x_der + w / 2
            bar_w = w
            axes[0, col].bar(
                p_pred_x, (p_pred - p_ip)[sel], width=bar_w, color=COLOR_PRED, alpha=0.85, label="Pred"
            )
            axes[1, col].bar(
                p_pred_x, (q_pred - q_ip)[sel], width=bar_w, color=COLOR_PRED, alpha=0.85, label="Pred"
            )
        else:
            p_fs_x = x_der
            bar_w = 0.8
        axes[0, col].bar(
            p_fs_x,
            (p_fs - p_ip)[sel],
            width=bar_w,
            color=COLOR_FS,
            alpha=0.85,
            label=LABEL_FS,
        )
        axes[1, col].bar(
            p_fs_x,
            (q_fs - q_ip)[sel],
            width=bar_w,
            color=COLOR_FS,
            alpha=0.85,
            label=LABEL_FS,
        )
        for row in (0, 1):
            axes[row, col].axhline(0.0, color="k", linewidth=0.8)
            axes[row, col].grid(True, axis="y", alpha=0.3)
            axes[row, col].set_xlim(-0.5, max(sel.size - 0.5, 0.5))
        axes[0, col].set_title(f"Phase {PHASE_LABELS[phase]}")
        axes[1, col].set_xlabel("DER index")
    axes[0, 0].set_ylabel(r"$\Delta p_{\mathrm{der}}$ vs Ipopt")
    axes[1, 0].set_ylabel(r"$\Delta q_{\mathrm{der}}$ vs Ipopt")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"DER setpoint error — {title_suffix}", y=1.02)
    fig.tight_layout()
    _save_fig(fig, figure_dir / f"{case_dir}_der_difference_sample{i}", save_png=save_png)

    meta.update(
        {
            "grid": grid,
            "case_dir": case_dir,
            "title_suffix": title_suffix,
            "scenario_index": aux["scenario_index"],
            "plot_pred": plot_pred,
        }
    )
    print(
        f"[{grid}] profile sample={i} selection={meta['selection']} "
        f"der_l2_fs={meta['der_l2_fs_vs_ipopt']:.4e} "
        f"dx_slfs={meta['der_l2_fs_vs_pred']:.4e}"
    )
    return meta


def plot_grid_scatters(
    *,
    grid: str,
    arrays: dict[str, np.ndarray],
    figure_dir: Path,
    max_points: int,
    save_png: bool = False,
) -> Path:
    """Write one 3x2 scatter figure for a grid; return PDF path."""
    rng = np.random.default_rng(SCATTER_SEED)
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 8.4), constrained_layout=True)
    display = GRID_DISPLAY.get(grid, grid)
    fig.suptitle(f"Penalty FS vs Ipopt — {display} (feasible)")

    for row, (key, label, unit) in enumerate(QUANTITY_SPECS):
        x_ip = arrays[f"{key}_ipopt"].ravel()
        y_pred = arrays[f"{key}_pred"].ravel()
        y_fs = arrays[f"{key}_fs"].ravel()
        xlabel = f"Ipopt {label} ({unit})"
        _scatter_panel(
            axes[row, 0],
            x_ip,
            y_pred,
            color=COLOR_PRED,
            title=f"{label} — before SLFS",
            xlabel=xlabel,
            ylabel=f"Pred {label} ({unit})",
            max_points=max_points,
            rng=rng,
        )
        _scatter_panel(
            axes[row, 1],
            x_ip,
            y_fs,
            color=COLOR_FS,
            title=f"{label} — {LABEL_FS}",
            xlabel=xlabel,
            ylabel=f"FS {label} ({unit})",
            max_points=max_points,
            rng=rng,
        )

    case_dir = resolve_grid(grid)["case_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / f"{case_dir}_dispatch_voltage_scatter.pdf"
    fig.savefig(pdf_path)
    if save_png:
        fig.savefig(pdf_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"[{grid}] wrote {pdf_path}")
    return pdf_path


def build_error_table(
    *,
    grids: list[str],
    results_root: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], str, str]:
    """Return row dicts plus markdown and LaTeX table strings."""
    rows: list[dict[str, Any]] = []
    for grid in grids:
        case_dir = resolve_grid(grid)["case_dir"]
        arrays, summary = _load_dump(results_root, case_dir, seed)
        err = summary.get("error_summary")
        if err is None:
            err = {
                "pred": {
                    k: _error_stats(arrays[f"{k}_pred"], arrays[f"{k}_ipopt"])
                    for k, _, _ in QUANTITY_SPECS
                },
                "fs": {
                    k: _error_stats(arrays[f"{k}_fs"], arrays[f"{k}_ipopt"])
                    for k, _, _ in QUANTITY_SPECS
                },
            }
        row = {
            "grid": grid,
            "display": GRID_DISPLAY.get(grid, grid),
            "num_samples": int(arrays["pder_pred"].shape[0]),
            "error": err,
        }
        rows.append(row)

    # Markdown
    md_lines = [
        "# Penalty FS vs Ipopt error summary",
        "",
        f"Seed `{seed}`, feasible test set. Errors are elementwise over all samples × DER phases / non-slack nodes.",
        "",
        "| Grid | Stage | Quantity | MAE | RMSE | Max abs |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        for stage in ("pred", "fs"):
            stage_label = "before SLFS" if stage == "pred" else LABEL_FS
            for key, label, _ in QUANTITY_SPECS:
                e = row["error"][stage][key]
                md_lines.append(
                    f"| {row['display']} | {stage_label} | {label} | "
                    f"{e['mae']:.4e} | {e['rmse']:.4e} | {e['max_abs']:.4e} |"
                )
    md = "\n".join(md_lines) + "\n"

    # Compact LaTeX: one row per grid×stage with p/q/|V| MAE columns
    tex_lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Grid & Stage & MAE $p$ & MAE $q$ & MAE $|V|$ \\",
        r"\midrule",
    ]
    for row in rows:
        for stage in ("pred", "fs"):
            stage_label = "pred" if stage == "pred" else "FS"
            ep = row["error"][stage]["pder"]["mae"]
            eq = row["error"][stage]["qder"]["mae"]
            ev = row["error"][stage]["vmag"]["mae"]
            tex_lines.append(
                f"{row['display']} & {stage_label} & "
                f"{ep:.3e} & {eq:.3e} & {ev:.3e} \\\\"
            )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex = "\n".join(tex_lines)
    return rows, md, tex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot penalty_fs vs Ipopt dispatch/voltage scatters and profiles"
    )
    parser.add_argument(
        "--grids",
        nargs="+",
        default=list(DEFAULT_GRIDS),
        choices=sorted(GRID_PRESETS.keys()),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--max-points",
        type=int,
        default=MAX_SCATTER_POINTS,
        help="Max scatter points per panel (stats still use all samples)",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Fixed sample index for profile plots (default: auto median among SLFS-changed)",
    )
    parser.add_argument(
        "--no-scatters",
        action="store_true",
        help="Skip scatter figures",
    )
    parser.add_argument(
        "--no-profiles",
        action="store_true",
        help="Skip DER/voltage profile figures",
    )
    parser.add_argument(
        "--plot-pred",
        action="store_true",
        help="Include Pred (before SLFS) on the profile figures (default: omit it)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also write PNG copies (PDF only by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = _resolve_path(Path(args.results_root))
    figure_dir = _resolve_path(Path(args.figure_dir))
    do_scatters = not args.no_scatters
    do_profiles = not args.no_profiles
    if not do_scatters and not do_profiles:
        raise SystemExit("Nothing to do: both --no-scatters and --no-profiles set")

    profile_meta: list[dict[str, Any]] = []
    for grid in args.grids:
        case_dir = resolve_grid(grid)["case_dir"]
        arrays, _ = _load_dump(results_root, case_dir, args.seed)
        if do_scatters:
            plot_grid_scatters(
                grid=grid,
                arrays=arrays,
                figure_dir=figure_dir,
                max_points=args.max_points,
                save_png=args.png,
            )
        if do_profiles:
            meta = plot_grid_profiles(
                grid=grid,
                arrays=arrays,
                figure_dir=figure_dir,
                sample_idx=args.sample_idx,
                save_png=args.png,
                plot_pred=args.plot_pred,
            )
            profile_meta.append(meta)

    if do_scatters:
        _, md, tex = build_error_table(
            grids=list(args.grids),
            results_root=results_root,
            seed=args.seed,
        )
        figure_dir.mkdir(parents=True, exist_ok=True)
        md_path = figure_dir / "error_summary.md"
        tex_path = figure_dir / "error_summary.tex"
        md_path.write_text(md, encoding="utf-8")
        tex_path.write_text(tex, encoding="utf-8")
        print(f"Wrote {md_path}")
        print(f"Wrote {tex_path}")

    if profile_meta:
        figure_dir.mkdir(parents=True, exist_ok=True)
        meta_path = figure_dir / "profile_sample_selection.json"
        meta_path.write_text(json.dumps(profile_meta, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()