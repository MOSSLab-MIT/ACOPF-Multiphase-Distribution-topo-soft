"""Soft V / I / θ / substation limits: Python soft objective vs Julia, penalty math, hard-only violation."""

import os
import sys

import h5py
import numpy as np
import pytest
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.utils.runtime import configure_tensorflow_for_data_loading

configure_tensorflow_for_data_loading()
jax.config.update("jax_enable_x64", True)

from conftest import case_paths
from core.grids.data import OPFData
from core.grids.metrics import (
    HARD_VIOLATION_KEYS,
    N_OBJ_WEIGHTS,
    SOFT_LIMIT_EXCEEDANCE_KEYS,
    calc_cost,
    calc_ineq_violation,
    calc_soft_penalties,
    calc_substation_power_per_phase,
    calc_violation_components_ac,
    compute_ac_metrics_from_voltage,
    with_slack_voltage,
)
from core.topology.batch import precompute_topology
from core.utils.batch import prepare_batch_data

CASE = "IEEE13Nodeckt"
N_SAMPLES = 1000
# Julia objective is evaluated at Ipopt's primal point; Python recomputes it from the stored
# (vr, vi, pder, qder) with psub from S = u conj(Y u). They differ only by Ipopt's primal constraint
# violation (constr_viol_tol = 1e-6 on the balance rows, ~1e-9 in practice at convergence).
OBJ_RTOL = 1e-6
OBJ_ATOL = 1e-8
# Penalty sums are evaluated in float64 by two implementations of the same formula.
PEN_RTOL = 1e-10
PEN_ATOL = 1e-12
ACTIVE_PENALTY = 1e-6


def _load(case_name: str, n_samples: int):
    paths = case_paths(case_name)
    h5_path = paths["test_feasible"]
    if not h5_path.exists():
        pytest.skip(f"No feasible test HDF5 for {case_name}")
    data = OPFData(str(paths["train"]), str(h5_path), str(paths["json"]))
    inputs = np.asarray(data._load_inputs(h5_path, samples=n_samples))
    targets = data._load_targets(h5_path, samples=n_samples)
    topo = precompute_topology(data, inputs)
    batch = prepare_batch_data(jnp.asarray(inputs), jnp.asarray(targets), data, topo=topo)
    with h5py.File(h5_path, "r") as f:
        z_julia = np.asarray(f["meta"]["primal_objective_value"][:n_samples], dtype=np.float64)
        s_sub_julia = np.asarray(f["primal"]["psub_phase"][:n_samples]) + 1j * np.asarray(
            f["primal"]["qsub_phase"][:n_samples]
        )
    return data, batch, topo, z_julia, s_sub_julia


@pytest.fixture(scope="module")
def feasible():
    return _load(CASE, N_SAMPLES)


def _reference_penalties(net, v_full: np.ndarray, y_0row: np.ndarray) -> np.ndarray:
    """Loop reference of Julia's slack sums (model.jl), one sample, independent of the JAX code."""
    v_full = np.asarray(v_full)
    y_0row = np.asarray(y_0row)
    vmin = np.asarray(net.vmin)
    vmax = np.asarray(net.vmax)
    pen_v = 0.0
    for i in range(net.n_net):
        m2 = v_full[i].real ** 2 + v_full[i].imag ** 2
        pen_v += max(vmin[i] ** 2 - m2, 0.0) + max(m2 - vmax[i] ** 2, 0.0)

    from_idx = np.asarray(net.branch_from_idx)
    to_idx = np.asarray(net.branch_to_idx)
    valid = np.asarray(net.branch_valid)
    y_series = np.asarray(net.branch_y_series)
    imax = np.asarray(net.branch_imax)
    scale = np.asarray(net.v_scale_full)
    dvamax = float(net.dvamax)
    pen_i = 0.0
    pen_theta = 0.0
    for line in range(from_idx.shape[0]):
        for r in range(from_idx.shape[1]):
            if not valid[line, r]:
                continue
            # Current: I = Σ_c y[r, c] (V_f,c - V_t,c) over valid terminals c, system voltages.
            current = 0.0 + 0.0j
            for c in range(from_idx.shape[1]):
                if valid[line, c]:
                    f, t = from_idx[line, c], to_idx[line, c]
                    current += y_series[line, r, c] * (scale[f] * v_full[f] - scale[t] * v_full[t])
            pen_i += max(abs(current) ** 2 - imax[line] ** 2, 0.0)
            # Angle: cross = vr_f vi_t - vi_f vr_t, dot = vr_f vr_t + vi_f vi_t (local pu).
            uf, ut = v_full[from_idx[line, r]], v_full[to_idx[line, r]]
            dot = uf.real * ut.real + uf.imag * ut.imag
            cross = uf.real * ut.imag - uf.imag * ut.real
            pen_theta += max(cross - dvamax * dot, 0.0) + max(-cross - dvamax * dot, 0.0)

    # Substation: S_k = u_slk,k conj(Σ_m Y0[k, m] u_m) (Julia p_sub_k + j q_sub_k), slack phases k.
    v_slack = np.asarray(net.v_slack_local)
    smax = np.asarray(net.sub_smax_phase)
    pen_sub = 0.0
    for k in range(net.n_slk):
        i_k = sum(y_0row[k, m] * v_full[m] for m in range(v_full.shape[0]))
        s_k = v_slack[k] * np.conj(i_k)
        pen_sub += max(s_k.real**2 + s_k.imag**2 - smax[k] ** 2, 0.0)
    return np.array([pen_v, pen_i, pen_theta, pen_sub])


def test_obj_weights_have_penalty_columns(feasible):
    _, batch, _, _, _ = feasible
    w = np.asarray(batch.obj_weights)
    assert w.shape == (N_SAMPLES, N_OBJ_WEIGHTS)
    # gen_scripts/configs/testing_defaults.yaml: ρ_V, ρ_I, ρ_θ ~ logU[1, 100], ρ_sub ~ logU[100, 200].
    assert np.all((w[:, 2:5] >= 1.0) & (w[:, 2:5] <= 100.0))
    assert np.all((w[:, 5] >= 100.0) & (w[:, 5] <= 200.0))


def test_python_soft_objective_matches_julia(feasible):
    """compute_ac_metrics_from_voltage at the stored OPF solution reproduces Julia's soft objective."""
    data, batch, topo, z_julia, _ = feasible
    v_full = with_slack_voltage(data.net, batch.v_target)
    metrics = compute_ac_metrics_from_voltage(
        data.net, batch.s_load, batch.pder_target, batch.qder_target, batch.pder_avai,
        v_full, batch.obj_weights, topo,
    )
    cost = np.asarray(metrics.cost)
    np.testing.assert_allclose(cost, z_julia, rtol=OBJ_RTOL, atol=OBJ_ATOL)

    # The penalty branch must actually be exercised: some stored optima violate hard V/I/θ limits.
    pen = np.asarray(metrics.soft_penalty)
    n_active = int(np.sum(pen.sum(axis=1) > ACTIVE_PENALTY))
    print(f"samples with active soft penalty: {n_active}/{N_SAMPLES}")
    assert n_active > 0
    weighted = np.sum(np.asarray(batch.obj_weights)[:, 2:] * pen, axis=1)
    assert np.all(weighted[pen.sum(axis=1) > ACTIVE_PENALTY] > 0.0)


def test_substation_power_matches_julia(feasible):
    """Per-phase S_k = u_slk conj(Y_0row u) (the quantity pen_sub prices) == Julia's stored p_sub_k + j q_sub_k."""
    data, batch, topo, _, s_sub_julia = feasible
    v_full = with_slack_voltage(data.net, batch.v_target)
    s_sub = np.asarray(calc_substation_power_per_phase(data.net, v_full, topo.Y_0row))
    # Julia's p_sub_k / q_sub_k equalities hold to Ipopt's constr_viol_tol.
    np.testing.assert_allclose(s_sub, s_sub_julia, rtol=0.0, atol=1e-6)


def test_soft_penalties_match_loop_reference(feasible):
    """calc_soft_penalties == independent loop reference, at stored and perturbed voltages.

    Perturbations push every family past its limit: radial scaling (V, I and substation power), and a
    phase rotation on half the nodes (θ and I).
    """
    data, batch, topo, _, _ = feasible
    net = data.net
    v_full = np.asarray(with_slack_voltage(net, batch.v_target))[:6]
    y_0row = np.asarray(topo.Y_0row)[:6]
    n_net = net.n_net
    scaled = v_full.copy()
    scaled[:, :n_net] *= 1.12
    rotated = v_full.copy()
    rotated[:, : n_net // 2] *= np.exp(1j * 1.3)
    cases = np.concatenate([v_full, scaled, rotated], axis=0)
    y_cases = np.concatenate([y_0row, y_0row, y_0row], axis=0)

    got = np.asarray(calc_soft_penalties(net, jnp.asarray(cases), jnp.asarray(y_cases)))
    ref = np.stack([_reference_penalties(net, v, y) for v, y in zip(cases, y_cases)])
    np.testing.assert_allclose(got, ref, rtol=PEN_RTOL, atol=PEN_ATOL)
    # Each family is exercised by at least one perturbed case.
    assert np.all(ref.max(axis=0) > ACTIVE_PENALTY)


def test_calc_cost_decomposition_and_validation(feasible):
    data, batch, topo, _, _ = feasible
    # Scaled voltages so every penalty family (incl. substation) is active and the ρ_sub column matters.
    v_full = with_slack_voltage(data.net, batch.v_target)
    v_full = v_full.at[:, : data.net.n_net].multiply(1.12)
    psub = jnp.zeros(batch.obj_weights.shape[0])
    w = batch.obj_weights
    w_econ = w.at[:, 2:].set(0.0)
    cost = calc_cost(data.net, psub, batch.pder_target, batch.pder_avai, w, v_full, topo.Y_0row)
    econ = calc_cost(data.net, psub, batch.pder_target, batch.pder_avai, w_econ, v_full, topo.Y_0row)
    pen = calc_soft_penalties(data.net, v_full, topo.Y_0row)
    assert np.all(np.asarray(pen)[:, 3] > ACTIVE_PENALTY)
    np.testing.assert_allclose(
        np.asarray(cost - econ), np.asarray(jnp.sum(w[:, 2:] * pen, axis=1)), rtol=1e-12, atol=1e-12
    )
    # ρ_sub enters linearly: doubling it adds exactly ρ_sub · pen_sub.
    w_sub2 = w.at[:, 5].multiply(2.0)
    cost_sub2 = calc_cost(data.net, psub, batch.pder_target, batch.pder_avai, w_sub2, v_full, topo.Y_0row)
    np.testing.assert_allclose(
        np.asarray(cost_sub2 - cost), np.asarray(w[:, 5] * pen[:, 3]), rtol=1e-10, atol=1e-12
    )
    with pytest.raises(ValueError, match="obj_weights must have 6 columns"):
        calc_cost(data.net, psub, batch.pder_target, batch.pder_avai, w[:, :5], v_full, topo.Y_0row)


def test_ineq_violation_is_hard_only(feasible):
    """Soft V/I/θ/substation exceedance must not enter calc_ineq_violation; it is reported as diagnostics only."""
    data, batch, topo, _, _ = feasible
    net = data.net
    v_full = with_slack_voltage(net, batch.v_target)
    # Over-voltage on every non-slack node: soft limit broken, hard rows evaluated at the same point.
    v_bad = v_full.at[:, : net.n_net].multiply(1.12)
    args = (net, batch.pder_target, batch.qder_target, batch.pder_avai)
    hard = np.asarray(calc_ineq_violation(*args, v_bad, Y_0row=topo.Y_0row))
    comps = calc_violation_components_ac(*args, v_bad, Y_0row=topo.Y_0row)

    hard_max = np.max(np.stack([np.asarray(comps[k]) for k in HARD_VIOLATION_KEYS]), axis=0)
    np.testing.assert_allclose(hard, hard_max, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(comps["total_violation"]), hard, rtol=0.0, atol=0.0)
    assert np.all(np.asarray(comps["voltage_magnitude_violation"]) > 0.0)
    assert np.all(np.asarray(comps["substation_violation"]) > 0.0)
    assert "substation_violation" in SOFT_LIMIT_EXCEEDANCE_KEYS
    assert "substation_violation" not in HARD_VIOLATION_KEYS
    pen_bad = np.asarray(calc_soft_penalties(net, v_bad, topo.Y_0row))
    assert np.all(pen_bad[:, 0] > 0.0) and np.all(pen_bad[:, 3] > 0.0)
    # Hard violation depends only on the DER setpoints, not on voltages: identical at v_full and v_bad.
    hard_ok = np.asarray(calc_ineq_violation(*args, v_full, Y_0row=topo.Y_0row))
    np.testing.assert_array_equal(hard, hard_ok)
