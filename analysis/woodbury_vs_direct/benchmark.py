#!/usr/bin/env python3
"""Benchmark production Woodbury (slim + lax.map) vs dense JAX ``solve(Y_LL, I)``.

Times identical stamped topologies on GPU, checks inverse accuracy, and writes
structured JSON for downstream result tables.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_ROOT = Path(__file__).resolve().parents[2]
_TESTS = _ROOT / "tests"
for path in (_ROOT, _TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import conftest

# Large feeders not in default pytest CASE_ROOTS; register before importing test helpers.
conftest.CASE_ROOTS["240bus_Master"] = conftest.DATASET_ROOT / "240bus_Master"
conftest.CASE_ROOTS["8500_Master"] = conftest.DATASET_ROOT / "8500_Master"

from core.topology.utils import Y_LL_L0_from_scaled
from core.topology.woodbury import (
    WoodburyTopologyBackend,
    woodbury_static_from_net,
    woodbury_topology_matrices_slim_branchless,
)
from test_topology_woodbury_jax import (
    COND_SKIP_ELEMENTWISE_INV,
    _build_status_suite,
    _host_cond,
    _inverse_residual,
    _load_compact_status_matrix,
    _load_data,
    _suite_limits,
    woodbury_topology_matrices_direct,
)

BENCHMARK_NAME = "woodbury_vs_direct_inverse_ir"
DEFAULT_BATCH_SIZE = 8
DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10
DEFAULT_KERNEL = "lax.map"
DEFAULT_STATUS_SUITE = "changed_only"

# Iterative-refinement sweep: each grid is timed and accuracy-checked at every
# fixed IR-step count. ir=0 is the unrefined Woodbury inverse; ir=1/2 apply one
# / two SMW-IR corrections (see ``_woodbury_inverse_with_ir`` in core/topology).
# Per-grid "sufficient" step counts (where residual < tol) are NOT pre-decided:
# the sweep records raw residuals so the table shows the effect of IR.
IR_STEPS: tuple[int, ...] = (0, 1, 2)
INVERSE_RESIDUAL_TOL = 1e-6

GRID_PRESETS: dict[str, str] = {
    "13": "IEEE13Nodeckt",
    "123": "IEEE123Master",
    "240": "240bus_Master",
    "906": "906bus_Master",
    "8500": "8500_Master",
}


def resolve_grid_case(grid: str) -> str:
    """Map a ``--grid`` key (13, 123, 240, 906, 8500) to its dataset case name."""
    if grid not in GRID_PRESETS:
        raise ValueError(f"Unknown grid {grid!r}; choose from {sorted(GRID_PRESETS)}")
    return GRID_PRESETS[grid]


def _jax_device_string() -> str:
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if gpus:
        return str(gpus[0])
    return str(jax.devices()[0])


def _timed_batch(
    fn,
    arg,
    *,
    n_warmup: int,
    n_repeats: int,
) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for one batched call after warmup."""
    for _ in range(n_warmup):
        out = fn(arg)
        jax.block_until_ready(out)

    times: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn(arg)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times) * 1e3
    return float(arr.mean()), float(arr.std())


def build_changed_only_batch(
    statuses: list[np.ndarray],
    base_status: np.ndarray,
    batch_size: int,
) -> tuple[Optional[np.ndarray], bool, int]:
    """Build a batch of changed-only compact statuses (cycle when needed).

    Returns ``(batch, batch_cycled, n_unique_changed)``. ``batch`` is ``None``
    when no topology differs from the base compact status.
    """
    status_arrays = [np.asarray(status, dtype=np.int32) for status in statuses]
    changed = [
        status
        for status in status_arrays
        if not np.array_equal(status, base_status)
    ]
    n_unique_changed = len(changed)
    if not changed:
        return None, False, 0

    batch_cycled = n_unique_changed < batch_size
    idx = np.arange(batch_size, dtype=np.int64) % n_unique_changed
    batch = np.asarray([changed[int(i)] for i in idx], dtype=np.int32)
    return batch, batch_cycled, n_unique_changed


def measure_accuracy(
    out_woodbury: tuple[Any, ...],
    out_direct: tuple[Any, ...],
    *,
    n_net: int,
) -> dict[str, Any]:
    """Accuracy metrics on a timed batch (informational; no pass/fail in benchmark)."""
    batch_size = int(out_direct[1].shape[0])
    max_inv_diff_all = 0.0
    max_inv_diff_well_cond = 0.0
    max_residual = 0.0
    max_cond = 0.0
    n_elementwise = 0
    n_high_cond = 0

    for k in range(batch_size):
        Y_LL, _ = Y_LL_L0_from_scaled(out_direct[0][k], n_net)
        cond = _host_cond(Y_LL)
        max_cond = max(max_cond, cond)
        inv_wb = out_woodbury[1][k]
        inv_direct = out_direct[1][k]

        diff = jnp.max(jnp.abs(inv_wb - inv_direct))
        jax.block_until_ready(diff)
        max_inv_diff_all = max(max_inv_diff_all, float(diff))

        if cond > COND_SKIP_ELEMENTWISE_INV:
            n_high_cond += 1
        else:
            max_inv_diff_well_cond = max(max_inv_diff_well_cond, float(diff))
            n_elementwise += 1

        res = _inverse_residual(Y_LL, inv_wb)
        max_residual = max(max_residual, res)

    return {
        "max_Y_LL_inv_diff": max_inv_diff_all,
        "max_Y_LL_inv_diff_well_cond": (
            None if n_elementwise == 0 else max_inv_diff_well_cond
        ),
        "inverse_residual_max": max_residual,
        "max_Y_LL_cond": max_cond,
        "elementwise_diff_trusted": n_elementwise == batch_size,
        "n_high_cond_samples": n_high_cond,
        "elementwise_checked_samples": n_elementwise,
    }


def _timing_block(mean_ms: float, std_ms: float, batch_size: int) -> dict[str, float]:
    return {
        "time_per_batch_ms": mean_ms,
        "time_std_ms": std_ms,
        "time_per_sample_ms": mean_ms / batch_size,
    }


def run_benchmark(
    grid: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Run Woodbury vs direct-inverse timing and accuracy for one grid preset."""
    case_name = resolve_grid_case(grid)
    net, paths = _load_data(case_name)
    train_h5 = paths["train"]
    if not train_h5.exists():
        return {
            "benchmark": BENCHMARK_NAME,
            "grid": grid,
            "skipped": True,
            "reason": f"training HDF5 not found: {train_h5}",
        }

    s = woodbury_static_from_net(net)
    # ``WoodburyTopologyBackend`` is used only to obtain the static buffers and the
    # status suite; we never time its adaptive batch kernel. Building it once (and
    # dropping the reference before the IR timing loop) avoids a second base-inverse
    # solve and keeps peak VRAM at one topology per IR step on 8500-bus.
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    max_status = max(_suite_limits(case_name), batch_size + 1)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=max_status)
    base_status = np.asarray(jax_tm.base_compact_status, dtype=np.int32)

    batch_np, batch_cycled, n_unique_changed = build_changed_only_batch(
        statuses, base_status, batch_size
    )
    if batch_np is None:
        return {
            "benchmark": BENCHMARK_NAME,
            "grid": grid,
            "skipped": True,
            "reason": "no changed topologies in dataset (changed_only suite empty)",
            "cases": {
                case_name: {
                    "n_net": int(s.n_net),
                    "n_changing_switches": int(net.n_changing_switches),
                    "n_unique_changed_statuses": 0,
                }
            },
        }

    batch_j = jnp.asarray(batch_np, dtype=jnp.int32)

    # Direct dense ``solve(Y_LL, I)`` is IR-independent: time it once and reuse the
    # output as the accuracy reference for every IR step.
    direct_fn = jax.jit(
        lambda status_batch: jax.lax.map(
            functools.partial(woodbury_topology_matrices_direct, s), status_batch
        )
    )
    d_mean, d_std = _timed_batch(
        direct_fn, batch_j, n_warmup=warmup, n_repeats=repeats
    )
    out_d = direct_fn(batch_j)

    # Sweep IR steps 0/1/2. Each step gets its own JIT'd ``lax.map`` over the
    # branchless slim fn (``ir_steps`` is a static int, so XLA unrolls the IR
    # loop at trace time -- one compiled kernel per step count).
    woodbury_results: dict[str, Any] = {}
    for ir in IR_STEPS:
        fn_slim = functools.partial(
            woodbury_topology_matrices_slim_branchless, ir_steps=ir
        )
        # Default-arg capture of ``fn_slim`` avoids late-binding the loop variable.
        woodbury_fn = jax.jit(
            lambda status_batch, _fn=fn_slim: jax.lax.map(
                lambda c: _fn(s, c), status_batch
            )
        )

        w_mean, w_std = _timed_batch(
            woodbury_fn, batch_j, n_warmup=warmup, n_repeats=repeats
        )
        out_w = woodbury_fn(batch_j)
        accuracy = measure_accuracy(out_w, out_d, n_net=int(s.n_net))
        speedup = d_mean / w_mean if w_mean > 0 else float("inf")

        woodbury_results[str(ir)] = {
            **_timing_block(w_mean, w_std, batch_size),
            "speedup": speedup,
            **accuracy,
        }

    case_result = {
        "n_net": int(s.n_net),
        "n_changing_switches": int(net.n_changing_switches),
        "r_max": int(s.n_changing_switches * 3 * s.max_switch_terms),
        "n_unique_changed_statuses": n_unique_changed,
        "batch_cycled": batch_cycled,
        "woodbury": woodbury_results,
        "direct": _timing_block(d_mean, d_std, batch_size),
    }

    return {
        "benchmark": BENCHMARK_NAME,
        "batch_size": batch_size,
        "kernel": DEFAULT_KERNEL,
        "status_suite": DEFAULT_STATUS_SUITE,
        "ir_steps_sweep": list(IR_STEPS),
        "inverse_residual_tol": INVERSE_RESIDUAL_TOL,
        "warmup": warmup,
        "repeats": repeats,
        "device": _jax_device_string(),
        "dtype": "float64",
        "timing_scope": "jit_kernel",
        "woodbury_timed_fn": (
            "jax.jit(jax.lax.map(woodbury_topology_matrices_slim_branchless"
            ", ir_steps=N))"
        ),
        "direct_timed_fn": "jax.jit(jax.lax.map(woodbury_topology_matrices_direct))",
        "timed_input": "jnp.int32 on default device (pre-built before timer)",
        "cases": {case_name: case_result},
    }


def write_results(payload: dict[str, Any], out_path: Path) -> None:
    """Write the benchmark results dict as JSON, creating parent directories."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    """CLI entrypoint: benchmark one grid and write the results JSON."""
    parser = argparse.ArgumentParser(
        description="Benchmark Woodbury slim batch vs dense JAX inverse."
    )
    parser.add_argument(
        "--grid",
        choices=sorted(GRID_PRESETS),
        required=True,
        help="Feeder preset (13, 123, 240, 906, 8500).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: analysis/woodbury_vs_direct/results/results_{grid}.json).",
    )
    args = parser.parse_args()

    batch_size = (
        args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
    )

    out = args.out
    if out is None:
        out = (
            _ROOT
            / "analysis"
            / "woodbury_vs_direct"
            / "results"
            / f"results_{args.grid}.json"
        )

    print(
        f"Running {BENCHMARK_NAME}: grid={args.grid} batch={batch_size} "
        f"warmup={args.warmup} repeats={args.repeats} device={_jax_device_string()}"
    )
    payload = run_benchmark(
        args.grid,
        batch_size=batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    if payload.get("skipped"):
        print(f"Skipped: {payload.get('reason', 'unknown')}")
    else:
        case_name = resolve_grid_case(args.grid)
        case = payload["cases"][case_name]
        d_ms = case["direct"]["time_per_batch_ms"]
        print(f"  {case_name}: n_net={case['n_net']} direct={d_ms:.3f} ms/batch")
        for ir in IR_STEPS:
            w = case["woodbury"][str(ir)]
            print(
                f"    ir={ir}: woodbury={w['time_per_batch_ms']:.3f} ms/batch "
                f"speedup={w['speedup']:.2f}x "
                f"residual={w['inverse_residual_max']:.3e}"
            )

    write_results(payload, out)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
