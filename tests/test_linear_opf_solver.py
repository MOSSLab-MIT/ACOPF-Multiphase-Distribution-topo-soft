"""Tests for linearized AC OPF CVXPY solver (Julia-equivalent DER constraints)."""

from pathlib import Path
from typing import Any, NamedTuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from conftest import load_h5_sample
from core.grids.data import OPFData
from core.grids.linear_opf_solver import (
    LinearizationPowerFlowError,
    affine_voltage,
    build_linear_opf_cache,
    solve_linear_opf,
)
from core.grids.metrics import calc_ac_metrics_pf
from core.topology.batch import build_topology_batch_jit, precompute_topology
from core.topology.utils import make_topology_data

CASE = "IEEE13Nodeckt"
CASE_DPP_MATCH = "IEEE123Master"
SAMPLE_INDICES = [0, 1]
DPP_MATCH_INDICES = list(range(5))
DER_TOL = 1e-6
AC_VIO_TOL = 1e-3
LINEAR_TOL = 1e-10
FINITE_DIFF_STEP = 1e-6
IPOPT_TARGET_AC_VIO_TOL = 1e-4
COST_RATIO_LIMIT = 2.0
COST_ABS_SLACK = 1.0


class LinearOPFSample(NamedTuple):
    data: OPFData
    s_load: jax.Array
    pder_avai: jax.Array
    obj_weights: jax.Array
    pder_target: jax.Array
    qder_target: jax.Array
    v_target: jax.Array
    topo: Any


class LinearOPFSolution(NamedTuple):
    sample: LinearOPFSample
    cache: Any
    pder: np.ndarray
    qder: np.ndarray
    info: dict[str, Any]


def _case_root(case: str = CASE) -> Path:
    return Path(__file__).resolve().parents[1] / "datagen_topo" / "datasets" / case


def _load_data(case: str = CASE) -> OPFData:
    root = _case_root(case)
    train_path = root / f"{case}_training.h5"
    return OPFData(
        str(train_path),
        str(root / f"{case}_test_feasible.h5"),
        str(root / "network_config.json"),
    )


def _load_sample(
    index: int = 0, *, case: str = CASE, data: OPFData | None = None
) -> LinearOPFSample:
    if data is None:
        data = _load_data(case)
    sample = load_h5_sample(case, split="test_feasible", index=index)
    s_load = jnp.asarray(sample["input"]["pd"], dtype=jnp.float64) + 1j * jnp.asarray(
        sample["input"]["qd"], dtype=jnp.float64
    )
    pder_avai = jnp.asarray(sample["input"]["pder_avai"], dtype=jnp.float64)
    obj_weights = jnp.asarray(sample["input"]["obj_weights"], dtype=jnp.float64)
    pder_target = jnp.asarray(sample["primal"]["pder"], dtype=jnp.float64)
    qder_target = jnp.asarray(sample["primal"]["qder"], dtype=jnp.float64)
    vr = jnp.asarray(sample["primal"]["vr"], dtype=jnp.float64)
    vi = jnp.asarray(sample["primal"]["vi"], dtype=jnp.float64)
    v_target = vr + 1j * vi

    compact = np.asarray(sample["input"][data.topology_input_key], dtype=np.int32)
    topo_batch = build_topology_batch_jit(compact[None, :], data)
    topo = make_topology_data(
        Y_0row=topo_batch.Y_0row[0],
        Y_LL_inv=topo_batch.Y_LL_inv[0],
        w=topo_batch.w[0],
        v_init=topo_batch.w[0],
        branch_status=topo_batch.branch_status[0],
    )
    return LinearOPFSample(
        data=data,
        s_load=s_load,
        pder_avai=pder_avai,
        obj_weights=obj_weights,
        pder_target=pder_target,
        qder_target=qder_target,
        v_target=v_target,
        topo=topo,
    )


def _solve_sample(index: int = 0) -> LinearOPFSolution:
    sample = _load_sample(index)
    cache = build_linear_opf_cache(sample.data.net, sample.topo, dict(sample.data.config))
    pder, qder, info = solve_linear_opf(
        cache,
        np.asarray(sample.s_load),
        np.asarray(sample.pder_avai),
        np.asarray(sample.obj_weights),
        net=sample.data.net,
        topo=sample.topo,
    )
    return LinearOPFSolution(
        sample=sample,
        cache=cache,
        pder=pder,
        qder=qder,
        info=info,
    )


def test_solve_linear_opf_skips_when_linearization_pf_fails(monkeypatch):
    """PF non-convergence at the linearization point must skip, not abort the batch."""
    sample = _load_sample(0)
    cache = build_linear_opf_cache(sample.data.net, sample.topo, dict(sample.data.config))

    def _boom(*_args, **_kwargs):
        raise LinearizationPowerFlowError(
            "Power flow at the linearization point did not converge "
            "(residual 6.128e-01); linear model would be invalid."
        )

    monkeypatch.setattr(
        "core.grids.linear_opf_solver._sample_linearization",
        _boom,
    )
    with pytest.warns(UserWarning, match="Skipping linear OPF sample"):
        pder, qder, info = solve_linear_opf(
            cache,
            np.asarray(sample.s_load),
            np.asarray(sample.pder_avai),
            np.asarray(sample.obj_weights),
            net=sample.data.net,
            topo=sample.topo,
        )

    n_der = sample.data.net.n_der_phase
    assert info["skipped"] is True
    assert info["solved"] is False
    assert info["status"] == "linearization_pf_failed"
    assert np.allclose(pder, 0.0) and pder.shape == (n_der,)
    assert np.allclose(qder, 0.0) and qder.shape == (n_der,)
    np.testing.assert_allclose(
        info["u_linear"], np.asarray(sample.topo.v_init, dtype=np.complex128).reshape(-1)
    )
    assert info["solve_time"] == 0.0
    assert info["qp_wall_time"] == 0.0
    assert "residual" in info["skip_reason"]


def _compare_dpp_and_ignore_dpp(sample: LinearOPFSample) -> None:
    cache = build_linear_opf_cache(sample.data.net, sample.topo, dict(sample.data.config))
    solve_kw = dict(
        s_load=np.asarray(sample.s_load),
        pder_avai=np.asarray(sample.pder_avai),
        obj_weights=np.asarray(sample.obj_weights),
        net=sample.data.net,
        topo=sample.topo,
    )
    orig_kwargs = dict(cache._solve_kwargs)
    try:
        cache._solve_kwargs = {**orig_kwargs, "ignore_dpp": False}
        pder_dpp, qder_dpp, info_dpp = solve_linear_opf(cache, **solve_kw)
        cache._solve_kwargs = orig_kwargs
        pder_ign, qder_ign, info_ign = solve_linear_opf(cache, **solve_kw)
    finally:
        cache._solve_kwargs = orig_kwargs

    assert info_dpp["solved"] and info_ign["solved"]
    np.testing.assert_allclose(pder_dpp, pder_ign, rtol=DER_TOL, atol=DER_TOL)
    np.testing.assert_allclose(qder_dpp, qder_ign, rtol=DER_TOL, atol=DER_TOL)
    np.testing.assert_allclose(
        info_dpp["objective"], info_ign["objective"], rtol=DER_TOL, atol=DER_TOL
    )


def test_linear_opf_dpp_and_ignore_dpp_match_ieee123_five_samples():
    """ignore_dpp=True must match DPP on IEEE123 across five feasible samples."""
    data = _load_data(CASE_DPP_MATCH)
    for sample_idx in DPP_MATCH_INDICES:
        _compare_dpp_and_ignore_dpp(_load_sample(sample_idx, case=CASE_DPP_MATCH, data=data))


@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_linear_opf_dpp_and_ignore_dpp_match(sample_idx):
    """ignore_dpp=True must match the DPP canonicalization path on small grids."""
    _compare_dpp_and_ignore_dpp(_load_sample(sample_idx))


@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_linear_opf_solve_feasible_status(sample_idx):
    solution = _solve_sample(sample_idx)
    data = solution.sample.data
    pder_avai = solution.sample.pder_avai
    cache = solution.cache
    pder = solution.pder
    qder = solution.qder
    info = solution.info
    assert info["solved"] is True
    assert info["status"] in {"optimal", "optimal_inaccurate"}
    assert pder.shape == (data.net.n_der_phase,)
    assert qder.shape == (data.net.n_der_phase,)
    pmin = np.asarray(data.net.pdermin)
    pmax = np.asarray(data.net.pdermax)
    qmin = np.asarray(data.net.qdermin)
    qmax = np.asarray(data.net.qdermax)
    smax = np.asarray(data.net.sdermax)
    assert np.all(pder >= pmin - DER_TOL)
    assert np.all(pder <= pmax + DER_TOL)
    assert np.all(qder >= qmin - DER_TOL)
    assert np.all(qder <= qmax + DER_TOL)
    assert np.all(pder**2 + qder**2 <= smax**2 + DER_TOL)
    unit_p = cache.unit_matrix @ pder
    assert np.all(unit_p <= np.asarray(pder_avai) + DER_TOL)


def test_affine_voltage_matches_frozen_z_formulation():
    solution = _solve_sample(0)
    sample = solution.sample
    cache = solution.cache
    pder = solution.pder
    qder = solution.qder
    x = np.concatenate([pder, qder])
    s_der = pder + 1j * qder
    s_net = np.asarray(
        sample.data.net.compute_net_injection(jnp.asarray(sample.s_load), jnp.asarray(s_der))
    )
    expected = np.asarray(sample.topo.w) + np.asarray(sample.topo.Y_LL_inv) @ np.conj(
        s_net / cache.u_hat
    )
    np.testing.assert_allclose(affine_voltage(cache, x), expected, atol=LINEAR_TOL)


def test_substation_real_power_affine_matches_finite_difference():
    solution = _solve_sample(0)
    data = solution.sample.data
    topo = solution.sample.topo
    cache = solution.cache
    rng = np.random.default_rng(0)
    dx = rng.normal(size=2 * data.net.n_der_phase)
    dx /= np.linalg.norm(dx)
    du = cache.U_net @ dx
    y_0row = np.asarray(topo.Y_0row)
    v_slack = np.asarray(data.net.v_slack_local)

    def psub_from_u(u_net):
        v_full = np.concatenate([u_net, v_slack])
        return float(np.real(np.sum(v_slack * np.conj(y_0row @ v_full))))

    finite_diff = (
        psub_from_u(cache.u_hat + FINITE_DIFF_STEP * du) - psub_from_u(cache.u_hat)
    ) / FINITE_DIFF_STEP
    linearized = float(cache.a_psub_param.value @ dx)
    np.testing.assert_allclose(linearized, finite_diff, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_linear_opf_metrics_vs_ipopt_target(sample_idx):
    solution = _solve_sample(sample_idx)
    sample = solution.sample
    data = sample.data
    h5_path = _case_root() / f"{CASE}_test_feasible.h5"
    inputs_np = np.asarray(data._load_inputs(h5_path, samples=sample_idx + 1))[
        sample_idx : sample_idx + 1
    ]
    topo_batch = precompute_topology(data, inputs_np)
    u_linear = jnp.asarray(solution.info["u_linear"], dtype=jnp.complex128)[None, :]
    pred_metrics = calc_ac_metrics_pf(
        data.net,
        jnp.asarray(sample.s_load)[None, :],
        jnp.asarray(solution.pder)[None, :],
        jnp.asarray(solution.qder)[None, :],
        jnp.asarray(sample.pder_avai)[None, :],
        jnp.asarray(sample.obj_weights)[None, :],
        topo_batch,
        v_init=u_linear,
    )
    target_metrics = calc_ac_metrics_pf(
        data.net,
        jnp.asarray(sample.s_load)[None, :],
        sample.pder_target[None, :],
        sample.qder_target[None, :],
        jnp.asarray(sample.pder_avai)[None, :],
        jnp.asarray(sample.obj_weights)[None, :],
        topo_batch,
        v_init=sample.v_target[None, :],
    )
    vio_tgt = float(np.asarray(target_metrics.violation).reshape(-1)[0])
    cost_tgt = float(np.asarray(target_metrics.cost).reshape(-1)[0])
    cost_pred = float(np.asarray(pred_metrics.cost).reshape(-1)[0])
    vio_pred = float(np.asarray(pred_metrics.violation).reshape(-1)[0])
    assert vio_tgt < IPOPT_TARGET_AC_VIO_TOL
    assert cost_tgt > 0.0
    assert vio_pred < AC_VIO_TOL
    assert cost_pred < COST_RATIO_LIMIT * cost_tgt + COST_ABS_SLACK
