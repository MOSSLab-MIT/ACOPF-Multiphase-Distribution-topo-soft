import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import tensorflow as tf
import jax

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    # Safe fallback if another import initialized the runtime first.
    pass


ROOT = Path(__file__).resolve().parents[1]
# ACOPF_DATASET_ROOT points the tests at an alternative datasets/ tree (e.g. freshly regenerated data).
DATASET_ROOT = Path(os.environ.get("ACOPF_DATASET_ROOT", ROOT / "datagen_topo" / "datasets"))
CASE_ROOTS = {
    "IEEE13Nodeckt": DATASET_ROOT / "IEEE13Nodeckt",
    "IEEE123Master": DATASET_ROOT / "IEEE123Master",
    "240bus_Master": DATASET_ROOT / "240bus_Master",
    "906bus_Master": DATASET_ROOT / "906bus_Master",
}


os.environ.setdefault("MPLCONFIGDIR", "/tmp/acopf_mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

jax.config.update("jax_enable_x64", True)


def case_paths(case_name: str) -> dict:
    case_root = CASE_ROOTS[case_name]
    train_path = case_root / f"{case_name}_training.h5"
    test_path = case_root / f"{case_name}_test.h5"
    if not train_path.exists():
        train_path = test_path
    return {
        "root": case_root,
        "json": case_root / "network_config.json",
        "train": train_path,
        "test": test_path,
        "test_feasible": case_root / f"{case_name}_test_feasible.h5",
    }


def load_case_config(case_name: str) -> dict:
    return json.loads(case_paths(case_name)["json"].read_text())


def parse_json_tree(node):
    if isinstance(node, dict) and set(node.keys()) == {"re", "im"}:
        return complex(node["re"], node["im"])
    if isinstance(node, list):
        return np.array([parse_json_tree(item) for item in node])
    return node


def load_h5_sample(case_name: str, split: str = "test_feasible", index: int = 0) -> dict:
    path = case_paths(case_name)[split]
    sample = {"input": {}, "primal": {}, "metadata": {}}
    with h5py.File(path, "r") as f:
        first_input = next(iter(f["input"].values()))
        if index >= first_input.shape[0]:
            pytest.skip(f"{path.name} has only {first_input.shape[0]} samples; index {index} unavailable")
        for key in f["input"].keys():
            sample["input"][key] = np.asarray(f["input"][key][index])
        if "primal" in f:
            for key in f["primal"].keys():
                sample["primal"][key] = np.asarray(f["primal"][key][index])
        if "meta" in f:
            for key in f["meta"].keys():
                sample["metadata"][key] = np.asarray(f["meta"][key][index])
        if "metadata" in f:
            for key in f["metadata"].keys():
                sample["metadata"][key] = np.asarray(f["metadata"][key][()])
    return sample


def expand_compact_status(cfg: dict, compact_status) -> np.ndarray:
    base = np.asarray(cfg["branch_status"], dtype=bool).copy()
    compact = np.asarray(compact_status, dtype=bool).reshape(-1)
    cols = cfg.get("changing_switches", [])
    switchable = np.asarray(cols, dtype=np.int64) - 1
    if compact.size == 0:
        return base
    full = base.copy()
    full[switchable] = compact
    return full


def stamp_switch_topology(cfg: dict, compact_status) -> np.ndarray:
    y_scaled = np.asarray(parse_json_tree(cfg["Y_scaled_base"]), dtype=np.complex128).copy()
    base_branch_status = np.asarray(cfg["branch_status"], dtype=bool)
    changing_switches = np.asarray(cfg.get("changing_switches", []), dtype=np.int64) - 1
    switch_base = np.asarray(cfg.get("switch_is_closed_base", []), dtype=bool)
    full_status = expand_compact_status(cfg, compact_status)

    switch_from_idx = [
        np.asarray(parse_json_tree(row), dtype=np.int64) - 1 for row in cfg.get("switch_from_idx", [])
    ]
    switch_to_idx = [
        np.asarray(parse_json_tree(row), dtype=np.int64) - 1 for row in cfg.get("switch_to_idx", [])
    ]
    switch_valid = [
        np.asarray(parse_json_tree(row), dtype=bool) for row in cfg.get("switch_valid", [])
    ]
    switch_y_series = [
        np.asarray(parse_json_tree(mat), dtype=np.complex128) for mat in cfg.get("switch_y_series", [])
    ]
    switch_y_shunt = [
        np.asarray(parse_json_tree(mat), dtype=np.complex128) for mat in cfg.get("switch_y_shunt", [])
    ]
    v_scale_full = np.asarray(cfg["v_scale_full"], dtype=np.float64)

    switch_idx = np.asarray(cfg.get("switch_idx", []), dtype=np.int64) - 1
    switch_pos_by_line = {int(line_idx): pos for pos, line_idx in enumerate(switch_idx)}

    for line_idx in changing_switches:
        if line_idx < 0 or line_idx >= base_branch_status.size:
            continue
        s = switch_pos_by_line[int(line_idx)]
        current_closed = bool(full_status[line_idx])
        if current_closed == bool(switch_base[s]):
            continue
        stamp_sign = 1 if current_closed else -1
        from_idx = switch_from_idx[s]
        to_idx = switch_to_idx[s]
        valid = switch_valid[s]
        y_series = switch_y_series[s]
        y_shunt = switch_y_shunt[s]

        for r in range(len(from_idx)):
            i_from = int(from_idx[r])
            i_to = int(to_idx[r])
            if not valid[r] or i_from < 0 or i_to < 0:
                continue
            for c in range(len(from_idx)):
                j_from = int(from_idx[c])
                j_to = int(to_idx[c])
                if not valid[c] or j_from < 0 or j_to < 0:
                    continue
                self_val = y_series[r, c] + 0.5 * y_shunt[r, c]
                mutual_val = -y_series[r, c]
                scale_ff = v_scale_full[i_from] * v_scale_full[j_from]
                scale_tt = v_scale_full[i_to] * v_scale_full[j_to]
                scale_ft = v_scale_full[i_from] * v_scale_full[j_to]
                scale_tf = v_scale_full[i_to] * v_scale_full[j_from]
                y_scaled[i_from, j_from] += stamp_sign * self_val * scale_ff
                y_scaled[i_to, j_to] += stamp_sign * self_val * scale_tt
                y_scaled[i_from, j_to] += stamp_sign * mutual_val * scale_ft
                y_scaled[i_to, j_from] += stamp_sign * mutual_val * scale_tf

    return y_scaled


def power_flow_residual(u, s, Y_LL, y_ns, v_slack_local):
    return u * np.conjugate(Y_LL @ u + y_ns @ v_slack_local) - s


def local_voltage_vector(cfg: dict, vr: np.ndarray, vi: np.ndarray) -> np.ndarray:
    v_slack_local = np.asarray(parse_json_tree(cfg["v_sub_ref"]), dtype=np.complex128) / np.asarray(
        cfg["v_scale_slack"], dtype=np.float64
    )
    return np.concatenate([vr + 1j * vi, v_slack_local])


def system_voltage_vector(cfg: dict, v_local: np.ndarray) -> np.ndarray:
    return v_local * np.asarray(cfg["v_scale_full"], dtype=np.float64)


def load_case_matrices(case_name: str):
    cfg = load_case_config(case_name)
    y_scaled = np.asarray(parse_json_tree(cfg["Y_scaled"]), dtype=np.complex128)
    Y_LL = np.asarray(parse_json_tree(cfg["Y_nn"]), dtype=np.complex128)
    y_ns = np.asarray(parse_json_tree(cfg["Y_ns"]), dtype=np.complex128)
    if y_ns.shape == (Y_LL.shape[0], cfg["n_slk"]):
        pass
    elif y_ns.shape == (cfg["n_slk"], Y_LL.shape[0]):
        y_ns = y_ns.T
    return cfg, y_scaled, Y_LL, y_ns
