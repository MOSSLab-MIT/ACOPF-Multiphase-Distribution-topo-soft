"""Tests for Woodbury vs direct-inverse benchmark (analysis/woodbury_vs_direct/)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TESTS = _ROOT / "tests"
for path in (_ROOT, _TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.woodbury_vs_direct.benchmark import (
    BENCHMARK_NAME,
    GRID_PRESETS,
    IR_STEPS,
    build_changed_only_batch,
    measure_accuracy,
    resolve_grid_case,
    run_benchmark,
    write_results,
)
from test_topology_woodbury_jax import (
    INVERSE_RESIDUAL_TOL,
    STRICT_MAX_ABS_TOL,
    _load_data,
    woodbury_topology_matrices_direct,
)

IEEE13_CASE = GRID_PRESETS["13"]
_TRAIN_H5 = (
    _ROOT / "datagen_topo" / "datasets" / "IEEE13Nodeckt" / "IEEE13Nodeckt_training.h5"
)

# Experiment-derived per-case IR-step needs (tol = 1e-6); see
# ``tests/test_woodbury_ir_steps.py::EXPECTED_IR_STEPS``. IEEE13 requires >=1 IR
# step, so the ir=0 row of the sweep is expected to be insufficient there.
_IEEE13_REQUIRED_IR_STEPS = 1


def _assert_ir_block_accuracy(ir_block: dict) -> None:
    """Pytest tolerances for one IR-step accuracy block.

    The strict element-wise diff against direct only holds once the inverse is
    actually accurate (residual < tol). For insufficient IR steps the large diff
    is the point of the sweep, so we skip the element-wise check there.
    """
    res = ir_block["inverse_residual_max"]
    if res >= INVERSE_RESIDUAL_TOL:
        return  # insufficient IR step: large diff is expected, not a failure

    well_cond_diff = ir_block.get("max_Y_LL_inv_diff_well_cond")
    if well_cond_diff is not None:
        assert well_cond_diff < STRICT_MAX_ABS_TOL, (
            f"max_Y_LL_inv_diff_well_cond {well_cond_diff:.3e} exceeds "
            f"{STRICT_MAX_ABS_TOL:.1e}"
        )


def _assert_ir_monotonic(case: dict) -> None:
    """Inverse residual is non-increasing across the IR sweep; at least one step < tol."""
    residuals = [
        case["woodbury"][str(ir)]["inverse_residual_max"] for ir in IR_STEPS
    ]
    for a, b in zip(residuals, residuals[1:]):
        assert b <= a + 1e-15, (
            f"inverse_residual_max not non-increasing across IR sweep: {residuals}"
        )
    assert min(residuals) < INVERSE_RESIDUAL_TOL, (
        f"no IR step reached residual < {INVERSE_RESIDUAL_TOL:.1e}: {residuals}"
    )

_REQUIRED_TOP_KEYS = {
    "benchmark",
    "batch_size",
    "kernel",
    "status_suite",
    "ir_steps_sweep",
    "inverse_residual_tol",
    "warmup",
    "repeats",
    "device",
    "dtype",
    "cases",
}
_REQUIRED_CASE_KEYS = {
    "n_net",
    "n_changing_switches",
    "r_max",
    "n_unique_changed_statuses",
    "batch_cycled",
    "woodbury",
    "direct",
}
_REQUIRED_IR_KEYS = {
    "time_per_batch_ms",
    "time_std_ms",
    "time_per_sample_ms",
    "speedup",
    "max_Y_LL_inv_diff",
    "inverse_residual_max",
}


def _assert_json_schema(payload: dict) -> None:
    assert payload["benchmark"] == BENCHMARK_NAME
    assert set(payload) >= _REQUIRED_TOP_KEYS
    assert payload["kernel"] == "lax.map"
    assert payload["status_suite"] == "changed_only"
    assert payload["ir_steps_sweep"] == list(IR_STEPS)
    assert payload["inverse_residual_tol"] == INVERSE_RESIDUAL_TOL
    for case_name, case in payload["cases"].items():
        assert set(case) >= _REQUIRED_CASE_KEYS
        # ``woodbury`` is keyed by IR-step string; ``direct`` is a single block.
        assert set(case["woodbury"]) == {str(ir) for ir in IR_STEPS}
        for ir in IR_STEPS:
            ir_block = case["woodbury"][str(ir)]
            assert set(ir_block) >= _REQUIRED_IR_KEYS
        direct = case["direct"]
        assert {
            "time_per_batch_ms",
            "time_std_ms",
            "time_per_sample_ms",
        } <= set(direct)
        assert case["woodbury"]["0"]["time_per_sample_ms"] == pytest.approx(
            case["woodbury"]["0"]["time_per_batch_ms"] / payload["batch_size"]
        )


def test_build_changed_only_batch_cycles_when_needed():
    base = np.array([0, 1], dtype=np.int32)
    changed = [np.array([1, 0], dtype=np.int32)]
    batch, cycled, n_unique = build_changed_only_batch(
        [base, changed[0]], base, batch_size=4
    )
    assert batch is not None
    assert cycled is True
    assert n_unique == 1
    assert batch.shape == (4, 2)


def test_build_changed_only_batch_empty_when_only_base():
    base = np.array([0, 1], dtype=np.int32)
    batch, cycled, n_unique = build_changed_only_batch([base], base, batch_size=8)
    assert batch is None
    assert cycled is False
    assert n_unique == 0


@pytest.mark.skipif(not _TRAIN_H5.exists(), reason="IEEE13 training HDF5 required")
def test_run_benchmark_ieee13_schema_and_accuracy():
    payload = run_benchmark("13", batch_size=8, warmup=1, repeats=2)
    assert not payload.get("skipped"), payload.get("reason")
    _assert_json_schema(payload)

    case_name = resolve_grid_case("13")
    case = payload["cases"][case_name]
    _assert_ir_monotonic(case)
    for ir in IR_STEPS:
        ir_block = case["woodbury"][str(ir)]
        _assert_ir_block_accuracy(ir_block)
        assert ir_block["max_Y_LL_inv_diff"] is not None
        assert ir_block["speedup"] > 0 and ir_block["speedup"] < float("inf")

    # IEEE13 requires >=1 IR step for residual < tol: the ir=0 row is the
    # unrefined Woodbury inverse and is expected to be insufficient there.
    res0 = case["woodbury"]["0"]["inverse_residual_max"]
    res1 = case["woodbury"]["1"]["inverse_residual_max"]
    assert res0 >= INVERSE_RESIDUAL_TOL, (
        f"IEEE13 ir=0 residual {res0:.3e} expected >= {INVERSE_RESIDUAL_TOL:.1e} "
        f"(required IR steps = {_IEEE13_REQUIRED_IR_STEPS})"
    )
    assert res1 < INVERSE_RESIDUAL_TOL, (
        f"IEEE13 ir=1 residual {res1:.3e} expected < {INVERSE_RESIDUAL_TOL:.1e}"
    )


@pytest.mark.skipif(not _TRAIN_H5.exists(), reason="IEEE13 training HDF5 required")
def test_write_results_roundtrip(tmp_path):
    payload = run_benchmark("13", batch_size=8, warmup=1, repeats=1)
    out = tmp_path / "results_13.json"
    write_results(payload, out)
    loaded = json.loads(out.read_text())
    _assert_json_schema(loaded)


@pytest.mark.skipif(not _TRAIN_H5.exists(), reason="IEEE13 training HDF5 required")
def test_measure_accuracy_matches_direct_inverse():
    import functools

    import jax
    import jax.numpy as jnp

    from core.topology.woodbury import (
        WoodburyTopologyBackend,
        woodbury_static_from_net,
        woodbury_topology_matrices_slim_branchless,
    )

    net, paths = _load_data(IEEE13_CASE)
    del paths
    s = woodbury_static_from_net(net)
    jax_tm = WoodburyTopologyBackend(net)
    compact = jnp.asarray(jax_tm.base_compact_status, dtype=jnp.int32).reshape(1, -1)

    # ir=1 is sufficient for IEEE13 (EXPECTED_IR_STEPS["IEEE13Nodeckt"] = 1): the
    # strict-tolerance assertions below require an accurate inverse.
    sufficient_ir = _IEEE13_REQUIRED_IR_STEPS
    woodbury_fn = jax.jit(
        functools.partial(
            woodbury_topology_matrices_slim_branchless, ir_steps=sufficient_ir
        )
    )
    direct_fn = jax.jit(
        lambda status_batch: jax.lax.map(
            functools.partial(woodbury_topology_matrices_direct, s), status_batch
        )
    )
    out_w = jax.lax.map(
        lambda c: woodbury_fn(s, c), compact
    )
    out_d = direct_fn(compact)
    metrics = measure_accuracy(out_w, out_d, n_net=int(s.n_net))

    assert metrics["max_Y_LL_inv_diff"] is not None
    assert metrics["max_Y_LL_inv_diff"] >= 0.0
    if metrics["max_Y_LL_inv_diff_well_cond"] is not None:
        assert metrics["max_Y_LL_inv_diff_well_cond"] < STRICT_MAX_ABS_TOL
    assert metrics["inverse_residual_max"] < INVERSE_RESIDUAL_TOL

    # ir=0 (unrefined) residual must be >= ir=1 (sufficient) residual on IEEE13,
    # i.e. iterative refinement strictly improves the inverse here.
    fn_ir0 = jax.jit(
        functools.partial(woodbury_topology_matrices_slim_branchless, ir_steps=0)
    )
    out_w0 = jax.lax.map(lambda c: fn_ir0(s, c), compact)
    metrics0 = measure_accuracy(out_w0, out_d, n_net=int(s.n_net))
    assert metrics0["inverse_residual_max"] >= metrics["inverse_residual_max"], (
        f"ir=0 residual {metrics0['inverse_residual_max']:.3e} expected >= "
        f"ir=1 residual {metrics['inverse_residual_max']:.3e} on IEEE13"
    )
