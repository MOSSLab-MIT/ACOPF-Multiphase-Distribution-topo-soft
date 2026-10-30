#!/usr/bin/env python3
"""Benchmark Jacobian accuracy and cost for power-flow autodiff methods.

Compares full unroll, truncated unroll (production-style ``diffiter``), and a
local fixed-point implicit adjoint (GMRES) w.r.t. net injection ``s``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
except Exception:
    tf = None  # noqa: N816

from core.grids.data import OPFData
from core.grids.power_flow import (
    solve_power_flow_fixed_point,
    solve_power_flow_fixed_point_full,
)
from core.utils.runtime import clear_gpu_memory, get_gpu_memory_usage
from core.utils.batch import prepare_batch_data

from analysis.differentiation.implicit_solver import (
    make_solve_power_flow_fixed_point_implicit,
)

GRID_PRESETS: dict[str, dict[str, str]] = {
    "13": {
        "train_h5": "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5",
        "test_h5": "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5",
        "data_json": "datagen_topo/datasets/IEEE13Nodeckt/network_config.json",
    },
    "123": {
        "train_h5": "datagen_topo/datasets/IEEE123Master/IEEE123Master_training.h5",
        "test_h5": "datagen_topo/datasets/IEEE123Master/IEEE123Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/IEEE123Master/network_config.json",
    },
    "906": {
        "train_h5": "datagen_topo/datasets/906bus_Master/906bus_Master_training.h5",
        "test_h5": "datagen_topo/datasets/906bus_Master/906bus_Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/906bus_Master/network_config.json",
    },
    "8500": {
        "train_h5": "datagen_topo/datasets/8500_Master/8500_Master_training.h5",
        "test_h5": "datagen_topo/datasets/8500_Master/8500_Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/8500_Master/network_config.json",
    },
}

@dataclass
class TimingStats:
    total_s: float
    per_batch_s: float


@dataclass
class MethodStats:
    error_max: float
    error_mean: float
    error_rel_fro: float
    time: TimingStats
    peak_mem_kb: int
    peak_gpu_mem_gb: Optional[float]
    peak_gpu_mem_method_only_gb: Optional[float]
    peak_gpu_mem_end_to_end_gb: Optional[float]
    batch_size: int
    benchmark_mode: str
    # Per-device JAX allocator counters (GB) at the method-only / end-to-end peak.
    # Keys: device name -> {in_use, peak_in_use, reserved, peak_reserved}.
    peak_gpu_mem_method_only_breakdown: Optional[Dict[str, Dict[str, float]]] = None
    peak_gpu_mem_end_to_end_breakdown: Optional[Dict[str, Dict[str, float]]] = None


def get_test_feasible_cases(datasets_root: str) -> List[Tuple[str, str, str]]:
    cases = []
    for name in sorted(os.listdir(datasets_root)):
        case_dir = os.path.join(datasets_root, name)
        if not os.path.isdir(case_dir):
            continue
        h5 = os.path.join(case_dir, f"{name}_test_feasible.h5")
        cfg = os.path.join(case_dir, "network_config.json")
        if os.path.exists(h5) and os.path.exists(cfg):
            train_h5 = os.path.join(case_dir, f"{name}_training.h5")
            if not os.path.exists(train_h5):
                train_h5 = h5
            cases.append((name, train_h5, h5, cfg))
    return cases


def _host_array(x) -> np.ndarray:
    """Copy a JAX or NumPy array to a host-side NumPy buffer."""
    return np.asarray(x)


def batch_to_device(
    s_net: np.ndarray,
    v_init: np.ndarray,
    y_inv: np.ndarray,
    w: np.ndarray,
    start: int,
    end: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Transfer one sample slice from host memory to the default JAX device."""
    return (
        jax.device_put(s_net[start:end]),
        jax.device_put(v_init[start:end]),
        jax.device_put(y_inv[start:end]),
        jax.device_put(w[start:end]),
    )


def load_batch_tensors(
    data: OPFData, test_h5_path: str, samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load net injection, voltage init, per-sample Y_inv and w (host NumPy arrays)."""
    with h5py.File(test_h5_path, "r") as f:
        inputs = np.concatenate(
            [np.asarray(f["input"][k]) for k in data.input_keys], axis=1
        )[:samples]
        targets = np.concatenate(
            [np.asarray(f["primal"][k]) for k in data.output_keys], axis=1
        )[:samples]

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        batch = prepare_batch_data(inputs, targets, data)
        _, y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)
        s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
        s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(
            batch.s_load, s_der
        )

    return (
        _host_array(s_net),
        _host_array(batch.v_target),
        _host_array(y_inv),
        _host_array(w),
    )


def fixed_point_residual_norm(v, s, y_inv, w):
    """||v - G(v, s)|| for Z-bus fixed-point map G."""
    g_val = w + y_inv @ jnp.conj(s / v)
    return jnp.linalg.norm(v - g_val)


def verify_forward_convergence(
    s_net: np.ndarray,
    v_init: np.ndarray,
    y_inv: np.ndarray,
    w: np.ndarray,
    maxiter: int,
    batch_size: int,
    *,
    rtol: float = 1e-6,
) -> float:
    """Return max residual norm over samples after full-unroll forward solve.

    Processes samples in batches to avoid materializing the full dataset on GPU
    at once (important for large networks with many samples).
    """
    actual_samples = int(s_net.shape[0])
    curr_bs = max(1, min(batch_size, actual_samples))
    num_batches = (actual_samples + curr_bs - 1) // curr_bs

    solver = jax.jit(
        jax.vmap(
            lambda s, v, y, ww: solve_power_flow_fixed_point_full(
                s, v, y, ww, maxiter=maxiter
            ),
            in_axes=(0, 0, 0, 0),
        )
    )
    residual_fn = jax.jit(
        jax.vmap(fixed_point_residual_norm, in_axes=(0, 0, 0, 0))
    )

    max_res = 0.0
    for b in range(num_batches):
        start = b * curr_bs
        end = min((b + 1) * curr_bs, actual_samples)
        if end <= start:
            continue
        s_b, v_b, y_b, w_b = batch_to_device(s_net, v_init, y_inv, w, start, end)

        v_sol = solver(s_b, v_b, y_b, w_b)
        v_sol.block_until_ready()
        res = residual_fn(v_sol, s_b, y_b, w_b)
        res.block_until_ready()
        max_res = max(max_res, float(jnp.max(res)))
        del v_sol, res

    if max_res > rtol:
        print(
            f"  WARNING: forward fixed-point residual max={max_res:.3e} "
            f"(threshold {rtol:.1e}); Jacobian comparisons may be unreliable."
        )
    else:
        print(f"  Forward residual max={max_res:.3e} (ok).")
    return max_res


def compute_jacobian_real_single(s, v_init, y_inv, w, solver_fn):
    def pf_wrapper(s_flat):
        n = s_flat.shape[0] // 2
        s_curr = s_flat[:n] + 1j * s_flat[n:]
        v_final = solver_fn(s_curr, v_init, y_inv, w)
        return jnp.concatenate([jnp.real(v_final), jnp.imag(v_final)])

    s_flat = jnp.concatenate([jnp.real(s), jnp.imag(s)])
    return jax.jacfwd(pf_wrapper)(s_flat)


def compute_jacobian_real_single_rev(s, v_init, y_inv, w, solver_fn):
    def pf_wrapper(s_flat):
        n = s_flat.shape[0] // 2
        s_curr = s_flat[:n] + 1j * s_flat[n:]
        v_final = solver_fn(s_curr, v_init, y_inv, w)
        return jnp.concatenate([jnp.real(v_final), jnp.imag(v_final)])

    s_flat = jnp.concatenate([jnp.real(s), jnp.imag(s)])
    return jax.jacrev(pf_wrapper)(s_flat)


def make_batched_jacobian(solver_fn, *, use_rev: bool = False):
    if use_rev:
        single = lambda s, v_init, y, ww: compute_jacobian_real_single_rev(
            s, v_init, y, ww, solver_fn
        )
    else:
        single = lambda s, v_init, y, ww: compute_jacobian_real_single(
            s, v_init, y, ww, solver_fn
        )
    return jax.jit(jax.vmap(single, in_axes=(0, 0, 0, 0)))


def jacobian_error_stats(j_method: jnp.ndarray, j_full: jnp.ndarray) -> Tuple[float, float, float]:
    diff = jnp.abs(j_method - j_full)
    err_max = float(jnp.max(diff))
    err_mean = float(jnp.mean(diff))
    fro_rel = float(
        jnp.linalg.norm(diff) / jnp.maximum(jnp.linalg.norm(j_full), 1e-18)
    )
    return err_max, err_mean, fro_rel


def peak_mem_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def peak_gpu_mem_gb(stats: Dict) -> Optional[float]:
    """Canonical scalar GPU peak (GiB) since last reset.

    Uses ``peak_bytes_reserved_gb`` (allocator reservations, the better OOM proxy)
    and falls back to ``peak_bytes_in_use_gb`` then a current ``bytes_in_use_gb``
    snapshot when the peak counters stay at zero (common on small grids).
    """
    if not stats:
        return None
    peaks_reserved = []
    peaks_in_use = []
    snaps_in_use = []
    for val in stats.values():
        if not isinstance(val, dict):
            continue
        if val.get("peak_bytes_reserved_gb") is not None:
            peaks_reserved.append(val["peak_bytes_reserved_gb"])
        if val.get("peak_bytes_in_use_gb") is not None:
            peaks_in_use.append(val["peak_bytes_in_use_gb"])
        if val.get("bytes_in_use_gb") is not None:
            snaps_in_use.append(val["bytes_in_use_gb"])
    if peaks_reserved and max(peaks_reserved) > 0:
        return max(peaks_reserved)
    if peaks_in_use and max(peaks_in_use) > 0:
        return max(peaks_in_use)
    if snaps_in_use:
        return max(snaps_in_use)
    return None


def gpu_mem_breakdown_gb(stats: Dict) -> Dict[str, Dict[str, float]]:
    """Per-device JAX allocator counters (GiB) snapshot.

    Captures all four counters (current + peak, in-use + reserved) so callers can
    see exactly which term dominates, rather than only a single scalar.
    """
    out: Dict[str, Dict[str, float]] = {}
    for device_name, val in stats.items():
        if not isinstance(val, dict) or "bytes_in_use_gb" not in val:
            continue
        out[device_name] = {
            "in_use_gb": float(val.get("bytes_in_use_gb") or 0.0),
            "peak_in_use_gb": float(val.get("peak_bytes_in_use_gb") or 0.0),
            "reserved_gb": float(val.get("bytes_reserved_gb") or 0.0),
            "peak_reserved_gb": float(val.get("peak_bytes_reserved_gb") or 0.0),
        }
    return out


def gpu_mem_in_use_gb(stats: Dict) -> Optional[float]:
    """Current GPU memory in use (GiB) across devices — snapshot, not cumulative peak."""
    if not stats:
        return None
    in_use = []
    for val in stats.values():
        if not isinstance(val, dict):
            continue
        if val.get("bytes_in_use_gb") is not None:
            in_use.append(val["bytes_in_use_gb"])
    return max(in_use) if in_use else None


def reset_gpu_peak_stats() -> None:
    """Reset JAX per-device peak counters (after untimed warmup ops)."""
    try:
        for device in jax.local_devices():
            if device.platform == "gpu":
                device.clear_memory_stats()
    except AttributeError:
        pass


def record_method_gpu_peak(
    gpu_peak: Optional[float], since_reset: bool = True
) -> Tuple[Optional[float], Dict[str, Dict[str, float]]]:
    """Update running GPU peak for timed method eval.

    Prefer cumulative peak since ``reset_gpu_peak_stats``; fall back to in-use
    snapshot when counters stay at zero (common on small grids).

    Returns ``(updated_peak_scalar, per_device_breakdown_at_this_sample)``.
    """
    stats = get_gpu_memory_usage()
    breakdown = gpu_mem_breakdown_gb(stats)
    if since_reset:
        peak = peak_gpu_mem_gb(stats)
        if peak and peak > 0:
            return max(gpu_peak or 0.0, peak), breakdown
    snap = gpu_mem_in_use_gb(stats)
    if snap is not None:
        return max(gpu_peak or 0.0, snap), breakdown
    return gpu_peak, breakdown


def merge_gpu_breakdown(
    running: Optional[Dict[str, Dict[str, float]]],
    sample: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Keep the per-device, per-counter maximum across all sampled batches."""
    out = {dev: dict(counters) for dev, counters in (running or {}).items()}
    for device_name, counters in sample.items():
        acc = out.setdefault(device_name, {})
        for key, value in counters.items():
            acc[key] = max(acc.get(key, 0.0), float(value))
    return out


def run_case(
    case_name: str,
    train_h5: str,
    test_h5: str,
    cfg_json: str,
    samples: int,
    batch_size: int,
    maxiter: int,
    diffiters: List[int],
) -> Dict[str, MethodStats]:
    data = OPFData(train_h5_path=train_h5, test_h5_path=test_h5, json_path=cfg_json)
    s_net, v_init, y_inv, w = load_batch_tensors(data, test_h5, samples)
    actual_samples = int(s_net.shape[0])
    if actual_samples == 0:
        raise RuntimeError(f"No samples found in {test_h5}")

    verify_forward_convergence(s_net, v_init, y_inv, w, maxiter, batch_size)

    def run_full_only(curr_bs: int) -> Dict[str, MethodStats]:
        clear_gpu_memory()
        num_batches = (actual_samples + curr_bs - 1) // curr_bs

        full_solver = jax.jit(
            lambda s, v, y, ww: solve_power_flow_fixed_point_full(
                s, v, y, ww, maxiter=maxiter
            )
        )
        j_full_fn = make_batched_jacobian(full_solver, use_rev=False)
        s_warm, v_warm, y_warm, w_warm = batch_to_device(
            s_net, v_init, y_inv, w, 0, curr_bs
        )
        j_full_fn(s_warm, v_warm, y_warm, w_warm).block_until_ready()

        full_time = 0.0
        method_only_gpu_peak = None
        method_only_gpu_breakdown: Dict[str, Dict[str, float]] = {}
        end_to_end_gpu_peak = None
        end_to_end_gpu_breakdown: Dict[str, Dict[str, float]] = {}
        for b in range(num_batches):
            start = b * curr_bs
            end = min((b + 1) * curr_bs, actual_samples)
            if end <= start:
                continue
            s_b, v_b, y_b, w_b = batch_to_device(s_net, v_init, y_inv, w, start, end)

            reset_gpu_peak_stats()
            t0 = time.perf_counter()
            j_full = j_full_fn(s_b, v_b, y_b, w_b)
            j_full.block_until_ready()
            t1 = time.perf_counter()
            full_time += t1 - t0

            method_only_gpu_peak, bd = record_method_gpu_peak(method_only_gpu_peak)
            method_only_gpu_breakdown = merge_gpu_breakdown(
                method_only_gpu_breakdown, bd
            )
            end_to_end_gpu_peak, bd = record_method_gpu_peak(end_to_end_gpu_peak)
            end_to_end_gpu_breakdown = merge_gpu_breakdown(
                end_to_end_gpu_breakdown, bd
            )
            del j_full

        peak_kb = peak_mem_kb()
        return {
            "full": MethodStats(
                error_max=0.0,
                error_mean=0.0,
                error_rel_fro=0.0,
                time=TimingStats(total_s=full_time, per_batch_s=full_time / num_batches),
                peak_mem_kb=peak_kb,
                peak_gpu_mem_gb=method_only_gpu_peak,
                peak_gpu_mem_method_only_gb=method_only_gpu_peak,
                peak_gpu_mem_end_to_end_gb=end_to_end_gpu_peak,
                batch_size=curr_bs,
                benchmark_mode="full_reference",
                peak_gpu_mem_method_only_breakdown=method_only_gpu_breakdown,
                peak_gpu_mem_end_to_end_breakdown=end_to_end_gpu_breakdown,
            )
        }

    def run_method(
        curr_bs: int, method_name: str, method_fn: Callable
    ) -> Dict[str, MethodStats]:
        clear_gpu_memory()
        print(f"  Running method '{method_name}' with batch_size={curr_bs}...")
        num_batches = (actual_samples + curr_bs - 1) // curr_bs

        full_solver = jax.jit(
            lambda s, v, y, ww: solve_power_flow_fixed_point_full(
                s, v, y, ww, maxiter=maxiter
            )
        )
        j_full_fn = make_batched_jacobian(full_solver, use_rev=False)
        method_batched = make_batched_jacobian(
            method_fn,
            use_rev=(method_name == "implicit"),
        )

        s_warm, v_warm, y_warm, w_warm = batch_to_device(
            s_net, v_init, y_inv, w, 0, curr_bs
        )
        method_batched(s_warm, v_warm, y_warm, w_warm).block_until_ready()

        total_time = 0.0
        err_max = 0.0
        err_mean = 0.0
        err_rel_fro = 0.0
        method_only_gpu_peak = None
        method_only_gpu_breakdown: Dict[str, Dict[str, float]] = {}
        end_to_end_gpu_peak = None
        end_to_end_gpu_breakdown: Dict[str, Dict[str, float]] = {}

        for b in range(num_batches):
            start = b * curr_bs
            end = min((b + 1) * curr_bs, actual_samples)
            if end <= start:
                continue
            s_b, v_b, y_b, w_b = batch_to_device(s_net, v_init, y_inv, w, start, end)

            reset_gpu_peak_stats()
            t0 = time.perf_counter()
            j_m_timed = method_batched(s_b, v_b, y_b, w_b)
            j_m_timed.block_until_ready()
            t1 = time.perf_counter()
            total_time += t1 - t0
            method_only_gpu_peak, bd = record_method_gpu_peak(method_only_gpu_peak)
            method_only_gpu_breakdown = merge_gpu_breakdown(
                method_only_gpu_breakdown, bd
            )
            del j_m_timed

        j_full_fn(s_warm, v_warm, y_warm, w_warm).block_until_ready()

        for b in range(num_batches):
            start = b * curr_bs
            end = min((b + 1) * curr_bs, actual_samples)
            if end <= start:
                continue
            s_b, v_b, y_b, w_b = batch_to_device(s_net, v_init, y_inv, w, start, end)

            reset_gpu_peak_stats()
            j_full = j_full_fn(s_b, v_b, y_b, w_b)
            j_full.block_until_ready()
            j_m = method_batched(s_b, v_b, y_b, w_b)
            j_m.block_until_ready()
            b_max, b_mean, b_fro = jacobian_error_stats(j_m, j_full)
            err_max = max(err_max, b_max)
            err_mean = max(err_mean, b_mean)
            err_rel_fro = max(err_rel_fro, b_fro)

            end_to_end_gpu_peak, bd = record_method_gpu_peak(end_to_end_gpu_peak)
            end_to_end_gpu_breakdown = merge_gpu_breakdown(
                end_to_end_gpu_breakdown, bd
            )
            del j_full, j_m

        peak_kb = peak_mem_kb()
        return {
            method_name: MethodStats(
                error_max=err_max,
                error_mean=err_mean,
                error_rel_fro=err_rel_fro,
                time=TimingStats(total_s=total_time, per_batch_s=total_time / num_batches),
                peak_mem_kb=peak_kb,
                peak_gpu_mem_gb=method_only_gpu_peak,
                peak_gpu_mem_method_only_gb=method_only_gpu_peak,
                peak_gpu_mem_end_to_end_gb=end_to_end_gpu_peak,
                batch_size=curr_bs,
                benchmark_mode="dense_jacobian",
                peak_gpu_mem_method_only_breakdown=method_only_gpu_breakdown,
                peak_gpu_mem_end_to_end_breakdown=end_to_end_gpu_breakdown,
            )
        }

    results: Dict[str, MethodStats] = {}
    results.update(run_full_only(batch_size))

    for diffiter in diffiters:
        solver = jax.jit(
            partial(solve_power_flow_fixed_point, maxiter=maxiter, diffiter=diffiter)
        )
        name = f"truncated_diffiter_{diffiter}"
        results.update(run_method(batch_size, name, solver))

    implicit_solver = make_solve_power_flow_fixed_point_implicit(maxiter)
    results.update(run_method(batch_size, "implicit", implicit_solver))

    return results


def resolve_cases_from_h5(h5_paths: List[str]) -> List[Tuple[str, str, str, str]]:
    cases: List[Tuple[str, str, str, str]] = []
    for h5 in h5_paths:
        h5_path = os.path.abspath(h5)
        if not os.path.exists(h5_path):
            raise FileNotFoundError(h5_path)
        case_dir = os.path.dirname(h5_path)
        case_name = os.path.basename(case_dir)
        cfg = os.path.join(case_dir, "network_config.json")
        if not os.path.exists(cfg):
            raise FileNotFoundError(cfg)
        train_h5 = os.path.join(case_dir, f"{case_name}_training.h5")
        if not os.path.exists(train_h5):
            train_h5 = h5_path
        cases.append((case_name, train_h5, h5_path, cfg))
    return cases


def resolve_grid_case(grid: str) -> Tuple[str, str, str, str]:
    if grid not in GRID_PRESETS:
        raise ValueError(f"Unknown --grid {grid!r}; choose from {sorted(GRID_PRESETS)}")
    preset = GRID_PRESETS[grid]
    case_name = Path(preset["test_h5"]).parent.name
    return (
        case_name,
        preset["train_h5"],
        preset["test_h5"],
        preset["data_json"],
    )


def method_stats_to_dict(stats: MethodStats) -> dict:
    return {
        "error_max": stats.error_max,
        "error_mean": stats.error_mean,
        "error_rel_fro": stats.error_rel_fro,
        "time_total_s": stats.time.total_s,
        "time_per_batch_s": stats.time.per_batch_s,
        "peak_mem_kb": stats.peak_mem_kb,
        "peak_gpu_mem_gb": stats.peak_gpu_mem_gb,
        "peak_gpu_mem_method_only_gb": stats.peak_gpu_mem_method_only_gb,
        "peak_gpu_mem_end_to_end_gb": stats.peak_gpu_mem_end_to_end_gb,
        "peak_gpu_mem_method_only_breakdown": stats.peak_gpu_mem_method_only_breakdown,
        "peak_gpu_mem_end_to_end_breakdown": stats.peak_gpu_mem_end_to_end_breakdown,
        "batch_size": stats.batch_size,
        "benchmark_mode": stats.benchmark_mode,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare PF Jacobian autodiff: full unroll, truncated, implicit."
    )
    parser.add_argument("--datasets-root", default="datagen_topo/datasets")
    parser.add_argument("--grid", choices=sorted(GRID_PRESETS), default=None)
    parser.add_argument("--h5-paths", nargs="*", default=None)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--diffiters", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--out",
        default="analysis/differentiation/results/jacobian_compare.json",
    )
    args = parser.parse_args()

    if args.grid:
        cases = [resolve_grid_case(args.grid)]
    elif args.h5_paths:
        cases = resolve_cases_from_h5(args.h5_paths)
    else:
        cases = get_test_feasible_cases(args.datasets_root)
        if not cases:
            raise RuntimeError("No *_test_feasible.h5 datasets found.")

    all_results = {
        "samples": args.samples,
        "batch_size": args.batch_size,
        "maxiter": args.maxiter,
        "diffiters": args.diffiters,
        "cases": {},
    }

    for case_name, train_h5, test_h5, cfg_json in cases:
        print(f"Running case: {case_name}")
        stats = run_case(
            case_name,
            train_h5,
            test_h5,
            cfg_json,
            args.samples,
            args.batch_size,
            args.maxiter,
            args.diffiters,
        )
        all_results["cases"][case_name] = {
            k: method_stats_to_dict(v) for k, v in stats.items()
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
