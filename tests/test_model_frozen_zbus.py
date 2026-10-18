"""Rigorous correctness tests for ``core.grids.linear_model`` frozen Z-bus linearization.

Validates the SLFS affine model
    f(x) ≈ f0 + A (x - x0),  c = f0 - A @ x0
against:
  - direct constraint evaluation at the expansion point,
  - manual / finite-difference frozen-denominator voltage sensitivity,
  - JAX autodiff of the frozen constraint map (gold standard for A),
  - first-order remainder scaling (quadratic in ||dx||),
  - ``create_models`` reduced-system assembly.

Imports only the linear-model module, not the SLFS/training loops.
"""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import case_paths, load_h5_sample
from core.grids.data import OPFData
from core.grids.metrics import (
    constraint_bounds,
    constraint_row_layout,
    constraint_values_from_voltage,
)
from core.grids.linear_model import (
    _full_local_voltage,
    _voltage_control_sensitivity,
    create_linear_model,
    create_models,
    create_models_batch,
)
from core.grids.power_flow import power_flow_residual, solve_power_flow_fixed_point
from core.topology.utils import expand_topology_batch, make_topology_data, Y_LL_L0_from_scaled, y_0row_from_scaled


jax.config.update("jax_enable_x64", True)

# Cases with new-schema ``network_config.json`` and feasible HDF5 in-repo.
CASES = ["IEEE13Nodeckt", "IEEE123Master"]
CASE_KEYS = {
    "IEEE13Nodeckt": 11,
    "IEEE123Master": 123,
}
SAMPLE_INDICES = [0, 1]
DIRECTION_SEEDS = [0, 1]

# Tight analytic checks (float64).
ATOL_F0 = 1e-10
ATOL_A = 1e-8
ATOL_C = 1e-12
ATOL_U = 1e-12
ATOL_FD = 1e-5
PF_MAXITER = 80
PF_RES_TOL = 5e-8


def _load_data(case_name: str) -> OPFData:
    paths = case_paths(case_name)
    return OPFData(
        str(paths["train"]),
        str(paths["test_feasible"]),
        str(paths["json"]),
    )


def _compact_branch_status(sample_input: dict, net) -> np.ndarray:
    """Resolve topology switch vector from HDF5 (new or legacy key)."""
    if "changing_switch_status" in sample_input:
        return np.asarray(sample_input["changing_switch_status"], dtype=np.int32)
    if "branch_status" in sample_input:
        return np.asarray(sample_input["branch_status"], dtype=np.int32)
    return np.asarray(net.base_branch_status, dtype=np.int32)


def _topology_from_compact(net, compact_status: np.ndarray):
    """Scalar ``TopologyData`` (no batch axis) for one switch/status vector."""
    batched = _topology_batch_from_compact(net, np.asarray(compact_status, dtype=np.int32)[None, :])
    return make_topology_data(
        Y_0row=batched.Y_0row[0],
        Y_LL_inv=batched.Y_LL_inv[0],
        w=batched.w[0],
        v_init=batched.v_init[0],
        branch_status=batched.branch_status[0],
    )


def _topology_batch_from_compact(net, compact_status_batch: np.ndarray):
    """Batched ``TopologyData`` with batch axis 0 on every field."""
    compact = jnp.asarray(compact_status_batch, dtype=jnp.int32)
    Y_scaled, Y_LL_inv, w, branch_status = net.topology_matrices_batch(compact)
    return make_topology_data(
        Y_0row=y_0row_from_scaled(Y_scaled, net.n_net, net.n_total),
        Y_LL_inv=Y_LL_inv,
        w=w,
        v_init=w,
        branch_status=branch_status,
    )


def _load_case_point(case_name: str, index: int = 0):
    data = _load_data(case_name)
    sample = load_h5_sample(case_name, split="test_feasible", index=index)

    s_load = jnp.asarray(sample["input"]["pd"], dtype=jnp.float64) + 1j * jnp.asarray(
        sample["input"]["qd"], dtype=jnp.float64
    )
    s_der0 = jnp.asarray(sample["primal"]["pder"], dtype=jnp.float64) + 1j * jnp.asarray(
        sample["primal"]["qder"], dtype=jnp.float64
    )
    pder_avai = jnp.asarray(sample["input"]["pder_avai"], dtype=jnp.float64)

    compact_status = _compact_branch_status(sample["input"], data.net)
    topo = _topology_from_compact(data.net, compact_status)
    return data, s_load, s_der0, pder_avai, topo, compact_status


def _reference_pf_u(net, s_load, s_der, topo, compact_status):
    """High-iteration fixed-point solve used as reference ``u_hat``."""
    s_net = net.compute_net_injection(s_load, s_der)
    u = solve_power_flow_fixed_point(
        s_net,
        topo.v_init,
        topo.Y_LL_inv,
        topo.w,
        maxiter=PF_MAXITER,
        diffiter=1,
    )
    Y_full, _, _, _ = net.topology_matrices(jnp.asarray(compact_status, dtype=jnp.int32))
    Y_LL, y_ns = Y_LL_L0_from_scaled(Y_full, net.n_net)
    res = power_flow_residual(u, s_net, Y_LL, y_ns, net.v_slack_local)
    return u, s_net, res


def _frozen_voltage_map(net, topo, u_hat, x0, x):
    """u(x) under frozen denominator 1/conj(u_hat)."""
    M = topo.Y_LL_inv * (1.0 / jnp.conj(u_hat))[None, :]
    U_p = M[:, net.der_node]
    U_q = -1j * U_p
    U_net = jnp.concatenate([U_p, U_q], axis=1)
    return u_hat + U_net @ (x - x0)


def _frozen_constraint_values(net, topo, u_hat, x0, x):
    u = _frozen_voltage_map(net, topo, u_hat, x0, x)
    v_full = _full_local_voltage(net, u)
    return constraint_values_from_voltage(net, x, v_full, topo.Y_0row)


def _manual_voltage_sensitivity(net, topo, u_hat):
    M = topo.Y_LL_inv * (1.0 / jnp.conj(u_hat))[None, :]
    U_p = M[:, net.der_node]
    U_q = -1j * U_p
    return jnp.concatenate([U_p, U_q], axis=1)


def _finite_difference_frozen_voltage(net, topo, s_load, u_hat, x0, eps=1e-7):
    n = 2 * net.n_der_phase
    U_fd = jnp.zeros((net.n_net, n), dtype=jnp.complex128)

    def u_at(dx):
        pder = x0[: net.n_der_phase] + dx[: net.n_der_phase]
        qder = x0[net.n_der_phase :] + dx[net.n_der_phase :]
        s_der = pder + 1j * qder
        s_net = net.compute_net_injection(s_load, s_der)
        return topo.w + topo.Y_LL_inv @ jnp.conj(s_net / u_hat)

    for k in range(n):
        dx = jnp.zeros(n).at[k].set(eps)
        U_fd = U_fd.at[:, k].set((u_at(dx) - u_at(jnp.zeros(n))) / eps)
    return U_fd


def _invalid_branch_row_indices(net):
    invalid = ~np.asarray(net.branch_valid)
    invalid_flat = np.flatnonzero(invalid.reshape(-1))
    if invalid_flat.size == 0:
        return None

    layout = constraint_row_layout(net)
    n_v = layout["n_v"]
    n_terms = layout["n_angle"] // 2
    substation_row = layout["substation_row"]
    angle_rows = np.concatenate(
        [n_v + invalid_flat, n_v + n_terms + invalid_flat],
    )
    branch_rows = substation_row + layout["n_sub"] + invalid_flat
    return angle_rows, branch_rows


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_power_flow_residual_at_linearization_point(case_name, sample_idx):
    data, s_load, s_der, _pder_avai, topo, compact = _load_case_point(case_name, sample_idx)
    u_hat, _s_net, res = _reference_pf_u(data.net, s_load, s_der, topo, compact)
    assert float(jnp.max(jnp.abs(res))) <= PF_RES_TOL


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_u_hat_matches_high_iteration_fixed_point_solver(case_name, sample_idx):
    data, s_load, s_der, _pder_avai, topo, compact = _load_case_point(case_name, sample_idx)
    u_ref, _, _ = _reference_pf_u(data.net, s_load, s_der, topo, compact)

    lm = create_linear_model(
        data.net, s_load, s_der, topo.v_init, topo, u_hat=u_ref
    )
    np.testing.assert_allclose(np.asarray(lm["u_hat"]), np.asarray(u_ref), atol=ATOL_F0, rtol=ATOL_F0)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_voltage_control_sensitivity_matches_manual_formula(case_name, sample_idx):
    data, s_load, s_der, _pder_avai, topo, compact = _load_case_point(case_name, sample_idx)
    u_hat, _, _ = _reference_pf_u(data.net, s_load, s_der, topo, compact)

    U_func = _voltage_control_sensitivity(data.net, u_hat, topo)
    U_manual = _manual_voltage_sensitivity(data.net, topo, u_hat)
    np.testing.assert_allclose(
        np.asarray(U_func), np.asarray(U_manual), atol=ATOL_U, rtol=0.0
    )


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_voltage_sensitivity_finite_difference_frozen_map(case_name, sample_idx):
    data, s_load, s_der, _pder_avai, topo, compact = _load_case_point(case_name, sample_idx)
    u_hat, _, _ = _reference_pf_u(data.net, s_load, s_der, topo, compact)
    x0 = jnp.concatenate([s_der.real, s_der.imag])

    U_analytic = _voltage_control_sensitivity(data.net, u_hat, topo)
    U_fd = _finite_difference_frozen_voltage(data.net, topo, s_load, u_hat, x0)
    np.testing.assert_allclose(
        np.asarray(U_analytic), np.asarray(U_fd), atol=ATOL_FD, rtol=0.0
    )


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_f0_matches_direct_constraint_evaluation(case_name, sample_idx):
    data, s_load, s_der, _pder_avai, topo, compact = _load_case_point(case_name, sample_idx)
    u_hat, _, _ = _reference_pf_u(data.net, s_load, s_der, topo, compact)

    lm = create_linear_model(
        data.net, s_load, s_der, topo.v_init, topo, u_hat=u_hat
    )
    x0 = lm["x0"]
    v_full = _full_local_voltage(data.net, u_hat)
    f0_direct = constraint_values_from_voltage(
        data.net, x0, v_full, topo.Y_0row
    )
    np.testing.assert_allclose(np.asarray(lm["f0"]), np.asarray(f0_direct), atol=ATOL_F0, rtol=0.0)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_affine_offset_c_equals_f0_minus_A_x0(case_name, sample_idx):
    data, s_load, s_der, _pder_avai, topo, compact = _load_case_point(case_name, sample_idx)
    u_hat, _, _ = _reference_pf_u(data.net, s_load, s_der, topo, compact)

    lm = create_linear_model(
        data.net, s_load, s_der, topo.v_init, topo, u_hat=u_hat
    )
    c_expected = lm["f0"] - lm["A"] @ lm["x0"]
    np.testing.assert_allclose(np.asarray(lm["c"]), np.asarray(c_expected), atol=ATOL_C, rtol=0.0)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_A_matches_jax_jacfwd_of_frozen_constraint_map(case_name, sample_idx):
    data, s_load, s_der, _pder_avai, topo, compact = _load_case_point(case_name, sample_idx)
    u_hat, _, _ = _reference_pf_u(data.net, s_load, s_der, topo, compact)

    lm = create_linear_model(
        data.net, s_load, s_der, topo.v_init, topo, u_hat=u_hat
    )
    x0 = lm["x0"]

    frozen_at_x = lambda x: _frozen_constraint_values(data.net, topo, u_hat, x0, x)
    f0_frozen = frozen_at_x(x0)
    A_ref = jax.jacfwd(frozen_at_x)(x0)

    np.testing.assert_allclose(np.asarray(lm["f0"]), np.asarray(f0_frozen), atol=ATOL_F0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(lm["A"]), np.asarray(A_ref), atol=ATOL_A, rtol=ATOL_A)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
@pytest.mark.parametrize("direction_seed", DIRECTION_SEEDS)
def test_first_order_remainder_decreases_quadratically(case_name, sample_idx, direction_seed):
    data, s_load, s_der, _pder_avai, topo, compact = _load_case_point(case_name, sample_idx)
    u_hat, _, _ = _reference_pf_u(data.net, s_load, s_der, topo, compact)

    lm = create_linear_model(
        data.net, s_load, s_der, topo.v_init, topo, u_hat=u_hat
    )
    x0 = lm["x0"]
    frozen_at_x = lambda x: _frozen_constraint_values(data.net, topo, u_hat, x0, x)

    key = jax.random.PRNGKey(CASE_KEYS[case_name] + 97 * sample_idx + direction_seed)
    direction = jax.random.normal(key, x0.shape, dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)

    epsilons = [1e-3, 5e-4, 2.5e-4]
    errors = []
    for eps in epsilons:
        delta = eps * direction
        f_true = frozen_at_x(x0 + delta)
        f_lin = lm["f0"] + lm["A"] @ delta
        errors.append(float(jnp.linalg.norm(f_true - f_lin, ord=2)))

    assert errors[2] < errors[1] < errors[0], (
        f"Frozen linearization error did not decrease for {case_name}: errors={errors}"
    )
    ratio_12 = errors[0] / errors[1]
    ratio_23 = errors[1] / errors[2]
    assert ratio_12 > 2.7, f"Expected ~quadratic decay (ratio_12), got {ratio_12:.3f}"
    assert ratio_23 > 2.7, f"Expected ~quadratic decay (ratio_23), got {ratio_23:.3f}"


@pytest.mark.parametrize("case_name", CASES)
def test_invalid_branch_rows_are_zero(case_name):
    data, s_load, s_der, _pder_avai, topo, compact = _load_case_point(case_name, index=0)
    row_idx = _invalid_branch_row_indices(data.net)
    if row_idx is None:
        pytest.skip(f"No padded/invalid branch terms for {case_name}")

    lm = create_linear_model(data.net, s_load, s_der, topo.v_init, topo)
    f0 = np.asarray(lm["f0"])
    A = np.asarray(lm["A"])
    angle_rows, branch_rows = row_idx

    assert float(np.max(np.abs(f0[angle_rows]))) < 1e-14
    assert float(np.max(np.abs(f0[branch_rows]))) < 1e-14
    assert float(np.max(np.abs(A[angle_rows, :]))) < 1e-14
    assert float(np.max(np.abs(A[branch_rows, :]))) < 1e-14


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def testconstraint_bounds_one_sided_rows_use_neg_inf(case_name, sample_idx):
    """Angle / |s_k| / |I| rows are g(x) <= u only; lowers must be -inf.

    Unit available-power is enforced in primal projection, not as linear rows.
    Feasible angle margins are often strictly negative.  A finite lower bound of 0
    would incorrectly report violations on OPF-feasible samples.
    """
    data, s_load, s_der, pder_avai, topo, compact = _load_case_point(case_name, sample_idx)
    net = data.net
    l, u = constraint_bounds(net)
    layout = constraint_row_layout(net)

    assert l.shape[0] == u.shape[0] == (
        layout["n_v"] + layout["n_angle"] + layout["n_sub"] + layout["n_branch_i"]
    )
    assert jnp.all(jnp.isfinite(l[layout["sl_v"]]))
    assert jnp.all(jnp.isinf(l[layout["sl_angle"]]) & (l[layout["sl_angle"]] < 0))
    assert jnp.all(jnp.isinf(l[layout["sl_sub"]]) & (l[layout["sl_sub"]] < 0))
    assert jnp.all(jnp.isinf(l[layout["sl_branch"]]) & (l[layout["sl_branch"]] < 0))

    assert jnp.allclose(u[layout["sl_angle"]], 0.0)
    np.testing.assert_allclose(
        np.asarray(u[layout["sl_sub"]]), np.asarray(net.sub_smax_phase), rtol=0.0, atol=0.0
    )

    solve_pf = partial(solve_power_flow_fixed_point, maxiter=PF_MAXITER, diffiter=1)
    models = create_models(
        net, s_load, s_der, pder_avai, topo.v_init, topo, solve_pf_fn=solve_pf
    )
    x = jnp.concatenate([s_der.real, s_der.imag])
    v_full = _full_local_voltage(net, models["v_hat"])
    vals = constraint_values_from_voltage(net, x, v_full, topo.Y_0row)
    angle_vals = vals[layout["sl_angle"]]

    assert float(jnp.min(angle_vals)) < -1e-6, (
        f"{case_name}: expected some strictly negative angle margins at the feasible point"
    )
    assert float(jnp.max(jnp.maximum(l[layout["sl_angle"]] - angle_vals, 0.0))) < 1e-6
    false_vio = jnp.maximum(jnp.zeros_like(angle_vals) - angle_vals, 0.0)
    assert float(jnp.max(false_vio)) > 1e-6, (
        "l=0 on angle rows would falsely flag this feasible sample"
    )


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_create_models_reduced_system_matches_bounds_minus_c(case_name, sample_idx):
    data, s_load, s_der, pder_avai, topo, compact = _load_case_point(case_name, sample_idx)
    solve_pf = partial(
        solve_power_flow_fixed_point,
        maxiter=PF_MAXITER,
        diffiter=1,
    )
    models = create_models(
        data.net,
        s_load,
        s_der,
        pder_avai,
        topo.v_init,
        topo,
        solve_pf_fn=solve_pf,
    )
    net = data.net
    l, u = constraint_bounds(net)
    c = models["linear_model"]["c"]
    n_angle = 2 * net.n_branch_constraints * net.max_branch_terms
    n_branch_i = net.n_branch_constraints * net.max_branch_terms
    n_linear_rows = net.n_net + n_angle + 1 + n_branch_i

    l_expected = l - c
    u_expected = u - c
    rs = models["reduced_system"]
    assert "pder_avai" in rs
    np.testing.assert_allclose(
        np.asarray(rs["pder_avai"]),
        np.asarray(pder_avai),
        rtol=0.0,
        atol=0.0,
    )
    assert rs["A_reduced"].shape[0] == l_expected.shape[0] == n_linear_rows

    l_finite = jnp.isfinite(l_expected) & jnp.isfinite(rs["l_reduced"])
    u_finite = jnp.isfinite(u_expected) & jnp.isfinite(rs["u_reduced"])
    if jnp.any(l_finite):
        np.testing.assert_allclose(
            np.asarray(rs["l_reduced"][l_finite]),
            np.asarray(l_expected[l_finite]),
            atol=ATOL_C,
            rtol=0.0,
        )
    if jnp.any(u_finite):
        np.testing.assert_allclose(
            np.asarray(rs["u_reduced"][u_finite]),
            np.asarray(u_expected[u_finite]),
            atol=ATOL_C,
            rtol=0.0,
        )

    l_neg_inf = jnp.isinf(l_expected) & (l_expected < 0)
    assert jnp.all(l_neg_inf == (jnp.isinf(rs["l_reduced"]) & (rs["l_reduced"] < 0)))
    assert jnp.all(jnp.isinf(u_expected) == jnp.isinf(rs["u_reduced"]))

    np.testing.assert_allclose(
        np.asarray(rs["A_reduced"]), np.asarray(models["linear_model"]["A"]), atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("case_name", CASES)
def test_create_models_batch_matches_single_sample(case_name):
    data, s_load, s_der, pder_avai, topo, compact = _load_case_point(case_name, index=0)
    n_batch = 2
    s_load_b = jnp.stack([s_load] * n_batch)
    s_der_b = jnp.stack([s_der] * n_batch)
    pder_avai_b = jnp.stack([pder_avai] * n_batch)
    v_init_b = jnp.stack([topo.v_init] * n_batch)

    solve_pf = partial(
        solve_power_flow_fixed_point,
        maxiter=PF_MAXITER,
        diffiter=1,
    )
    single = create_models(
        data.net, s_load, s_der, pder_avai, topo.v_init, topo, solve_pf_fn=solve_pf
    )
    # Batched topo: leading axis on every field (same as trainer / topology_matrices_batch).
    topo_b = expand_topology_batch(topo, n_batch)
    batch = create_models_batch(
        data.net,
        s_load_b,
        s_der_b,
        pder_avai_b,
        v_init_b,
        topo_b,
        solve_pf_fn=solve_pf,
    )

    for i in range(n_batch):
        np.testing.assert_allclose(
            np.asarray(batch["v_hat"][i]),
            np.asarray(single["v_hat"]),
            atol=1e-9,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(batch["linear_model"]["A"][i]),
            np.asarray(single["linear_model"]["A"]),
            atol=ATOL_A,
            rtol=ATOL_A,
        )
        # Batched path is JIT-compiled; allow tiny solver/aggregation drift vs scalar.
        np.testing.assert_allclose(
            np.asarray(batch["linear_model"]["c"][i]),
            np.asarray(single["linear_model"]["c"]),
            atol=1e-9,
            rtol=1e-9,
        )
