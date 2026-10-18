"""Tests for PF Jacobian differentiation benchmark (analysis/differentiation/)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.differentiation.compare_jacobians import (
    GRID_PRESETS,
    compute_jacobian_real_single,
    compute_jacobian_real_single_rev,
    load_batch_tensors,
    run_case,
)
from analysis.differentiation.implicit_solver import (
    make_solve_power_flow_fixed_point_implicit,
)
from core.grids.data import OPFData
from core.grids.power_flow import (
    solve_power_flow_fixed_point,
    solve_power_flow_fixed_point_full,
)

IEEE13_PRESET = GRID_PRESETS["13"]
_MAXITER = 12
_ATOL_FORWARD = 1e-10
_ATOL_JACOBIAN = 1e-8


def _load_ieee13_tensors(n: int = 4):
    data = OPFData(
        train_h5_path=IEEE13_PRESET["train_h5"],
        test_h5_path=IEEE13_PRESET["test_h5"],
        json_path=IEEE13_PRESET["data_json"],
    )
    return data, load_batch_tensors(data, IEEE13_PRESET["test_h5"], n)


def test_implicit_forward_matches_full_unroll_ieee13():
    """Implicit solver forward must match full fixed-point unroll."""
    _, (s_net, v_init, y_inv, w) = _load_ieee13_tensors(n=1)
    implicit = make_solve_power_flow_fixed_point_implicit(_MAXITER)
    v_implicit = implicit(s_net[0], v_init[0], y_inv[0], w[0])
    v_full = solve_power_flow_fixed_point_full(
        s_net[0], v_init[0], y_inv[0], w[0], maxiter=_MAXITER
    )
    np.testing.assert_allclose(
        np.asarray(v_implicit), np.asarray(v_full), atol=_ATOL_FORWARD, rtol=0.0
    )


def test_implicit_jacobian_matches_full_unroll_ieee13():
    """Implicit adjoint Jacobian must match jacfwd full-unroll reference."""
    _, (s_net, v_init, y_inv, w) = _load_ieee13_tensors(n=4)
    implicit = make_solve_power_flow_fixed_point_implicit(_MAXITER)
    full_solver = jax.jit(
        lambda s, v, y, ww: solve_power_flow_fixed_point_full(
            s, v, y, ww, maxiter=_MAXITER
        )
    )

    err_max = 0.0
    for i in range(int(s_net.shape[0])):
        j_full = compute_jacobian_real_single(
            s_net[i], v_init[i], y_inv[i], w[i], full_solver
        )
        j_implicit = compute_jacobian_real_single_rev(
            s_net[i], v_init[i], y_inv[i], w[i], implicit
        )
        err_max = max(err_max, float(jnp.abs(j_implicit - j_full).max()))

    assert err_max < _ATOL_JACOBIAN, f"implicit Jacobian error_max={err_max:.3e}"


def test_truncated_matches_full_when_diffiter_equals_maxiter():
    """When diffiter=maxiter, truncated primal and Jacobian match full unroll."""
    _, (s_net, v_init, y_inv, w) = _load_ieee13_tensors(n=2)
    truncated = jax.jit(
        lambda s, v, y, ww: solve_power_flow_fixed_point(
            s, v, y, ww, maxiter=_MAXITER, diffiter=_MAXITER
        )
    )
    full_solver = jax.jit(
        lambda s, v, y, ww: solve_power_flow_fixed_point_full(
            s, v, y, ww, maxiter=_MAXITER
        )
    )

    for i in range(int(s_net.shape[0])):
        v_trunc = truncated(s_net[i], v_init[i], y_inv[i], w[i])
        v_full = full_solver(s_net[i], v_init[i], y_inv[i], w[i])
        np.testing.assert_allclose(
            np.asarray(v_trunc), np.asarray(v_full), atol=1e-12, rtol=1e-12
        )

        j_trunc = compute_jacobian_real_single(
            s_net[i], v_init[i], y_inv[i], w[i], truncated
        )
        j_full = compute_jacobian_real_single(
            s_net[i], v_init[i], y_inv[i], w[i], full_solver
        )
        np.testing.assert_allclose(
            np.asarray(j_trunc), np.asarray(j_full), atol=1e-11, rtol=1e-11
        )


def test_run_case_smoke_ieee13():
    """End-to-end benchmark driver completes on a tiny IEEE13 batch."""
    preset = IEEE13_PRESET
    case_name = Path(preset["test_h5"]).parent.name
    stats = run_case(
        case_name,
        preset["train_h5"],
        preset["test_h5"],
        preset["data_json"],
        samples=4,
        batch_size=2,
        maxiter=_MAXITER,
        diffiters=[1, 2],
    )
    assert "full" in stats
    assert "implicit" in stats
    assert stats["implicit"].error_max < _ATOL_JACOBIAN
