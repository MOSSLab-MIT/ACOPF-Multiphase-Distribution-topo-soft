from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import load_h5_sample
import core.grids.linear_model as linear_model
from core.grids.data import OPFData
from core.grids.power_flow import power_flow_residual, solve_power_flow_fixed_point as solve_power_flow
from core.topology.batch import build_topology_batch_jit
from core.topology.utils import make_topology_data, Y_LL_L0_from_scaled


jax.config.update("jax_enable_x64", True)


CASES = ["IEEE13Nodeckt", "IEEE123Master"]
CASE_KEYS = {
    "IEEE13Nodeckt": 11,
    "IEEE123Master": 123,
}
SAMPLE_INDICES = [0, 1]
DIRECTION_SEEDS = [0, 1]


def _case_root(case_name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "datagen_topo" / "datasets" / case_name


def _load_data(case_name: str) -> OPFData:
    case_root = _case_root(case_name)
    train_path = case_root / f"{case_name}_training.h5"
    if not train_path.exists():
        # IEEE123Master currently ships test files only.
        train_path = case_root / f"{case_name}_test.h5"
    return OPFData(
        str(train_path),
        str(case_root / f"{case_name}_test_feasible.h5"),
        str(case_root / "network_config.json"),
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

    compact_status = np.asarray(sample["input"][data.topology_input_key], dtype=np.int32)
    topo_batch = build_topology_batch_jit(compact_status[None, :], data)
    topo = make_topology_data(
        Y_0row=topo_batch.Y_0row[0],
        Y_LL_inv=topo_batch.Y_LL_inv[0],
        w=topo_batch.w[0],
        v_init=topo_batch.w[0],
        branch_status=topo_batch.branch_status[0],
    )
    return data, s_load, s_der0, topo, compact_status


def _full_local_voltage(net, u_net):
    return jnp.concatenate([u_net, net.v_slack_local])


def _reference_substation_power_per_phase(net, v_full_local, y_matrix):
    slack_rows = (
        y_matrix[net.n_net : net.n_total, :]
        if y_matrix.shape[0] == net.n_total
        else y_matrix
    )
    i_slack = slack_rows @ v_full_local
    return net.v_slack_local * jnp.conj(i_slack)


def _reference_angle_values(net, v_full_local):
    vf = v_full_local[net.branch_from_idx]
    vt = v_full_local[net.branch_to_idx]
    mask = net.branch_valid.astype(v_full_local.real.dtype)
    dot = vf.real * vt.real + vf.imag * vt.imag
    cross = vf.real * vt.imag - vf.imag * vt.real
    upper = (cross - net.dvamax * dot) * mask
    lower = (-cross - net.dvamax * dot) * mask
    return jnp.concatenate([upper.reshape(-1), lower.reshape(-1)])


def _reference_branch_current_magnitudes(net, v_full_local):
    v_system = net.v_scale_full * v_full_local
    vf = v_system[net.branch_from_idx]
    vt = v_system[net.branch_to_idx]
    dv = vf - vt
    mask = net.branch_valid.astype(v_full_local.real.dtype)
    currents = jnp.sum(net.branch_y_series * dv[:, None, :], axis=2)
    return (jnp.abs(currents) * mask).reshape(-1)


def _reference_group_phase_sum(values, groups, mask):
    gathered = values[groups] * mask.astype(values.dtype)
    return jnp.sum(gathered, axis=1)


def _reference_constraint_values_from_voltage(net, x, v_full_local, y_scaled):
    del x  # unit availability is enforced in primal projection, not linear rows
    vmag = jnp.abs(v_full_local[: net.n_net])
    angle = _reference_angle_values(net, v_full_local)
    substation = jnp.abs(_reference_substation_power_per_phase(net, v_full_local, y_scaled))
    branch_i = _reference_branch_current_magnitudes(net, v_full_local)
    return jnp.concatenate([vmag, angle, substation, branch_i])


def _frozen_constraint_values(net, topo, u_hat, x0, x):
    M = topo.Y_LL_inv * (1.0 / jnp.conj(u_hat))[None, :]
    U_p = M[:, net.der_node]
    U_q = -1j * U_p
    U_net = jnp.concatenate([U_p, U_q], axis=1)
    u = u_hat + U_net @ (x - x0)
    return _reference_constraint_values_from_voltage(net, x, _full_local_voltage(net, u), topo.Y_0row)


def _reference_u_hat(net, s_load, s_der0, topo, compact_status, *, maxiter: int = 8):
    s_net0 = net.compute_net_injection(s_load, s_der0)
    u_hat = solve_power_flow(
        s_net0,
        topo.v_init,
        topo.Y_LL_inv,
        topo.w,
        maxiter=maxiter,
        diffiter=2,
    )
    Y_full, _, _, _ = net.topology_matrices(jnp.asarray(compact_status, dtype=jnp.int32))
    Y_LL, y_ns = Y_LL_L0_from_scaled(Y_full, net.n_net)
    residual = power_flow_residual(u_hat, s_net0, Y_LL, y_ns, net.v_slack_local)
    return u_hat, residual


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_create_linear_model_default_u_hat_matches_solver(case_name, sample_idx):
    data, s_load, s_der0, topo, compact = _load_case_point(case_name, index=sample_idx)

    lm = linear_model.create_linear_model(data.net, s_load, s_der0, topo.v_init, topo)
    u_hat_ref, residual = _reference_u_hat(data.net, s_load, s_der0, topo, compact, maxiter=7)

    np.testing.assert_allclose(np.asarray(lm["u_hat"]), np.asarray(u_hat_ref), atol=1e-10, rtol=1e-10)
    assert float(jnp.max(jnp.abs(residual))) <= 5e-8


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
def test_create_linear_model_matches_frozen_jacobian(case_name, sample_idx):
    data, s_load, s_der0, topo, compact = _load_case_point(case_name, index=sample_idx)
    u_hat_ref, _ = _reference_u_hat(data.net, s_load, s_der0, topo, compact, maxiter=40)

    lm = linear_model.create_linear_model(
        data.net,
        s_load,
        s_der0,
        topo.v_init,
        topo,
        u_hat=u_hat_ref,
    )
    x0 = lm["x0"]

    frozen_values = lambda x: _frozen_constraint_values(data.net, topo, u_hat_ref, x0, x)

    f0_ref = _reference_constraint_values_from_voltage(
        data.net,
        x0,
        _full_local_voltage(data.net, u_hat_ref),
        topo.Y_0row,
    )
    f0_frozen_x0 = frozen_values(x0)
    A_ref = jax.jacfwd(frozen_values)(x0)

    np.testing.assert_allclose(np.asarray(lm["f0"]), np.asarray(f0_ref), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(lm["f0"]), np.asarray(f0_frozen_x0), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(lm["A"]), np.asarray(A_ref), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sample_idx", SAMPLE_INDICES)
@pytest.mark.parametrize("direction_seed", DIRECTION_SEEDS)
def test_create_linear_model_first_order_remainder_is_quadratic(case_name, sample_idx, direction_seed):
    data, s_load, s_der0, topo, compact = _load_case_point(case_name, index=sample_idx)
    u_hat_ref, _ = _reference_u_hat(data.net, s_load, s_der0, topo, compact, maxiter=40)

    lm = linear_model.create_linear_model(
        data.net,
        s_load,
        s_der0,
        topo.v_init,
        topo,
        u_hat=u_hat_ref,
    )
    x0 = lm["x0"]
    frozen_values = lambda x: _frozen_constraint_values(data.net, topo, u_hat_ref, x0, x)

    key = jax.random.PRNGKey(CASE_KEYS[case_name] + 97 * sample_idx + direction_seed)
    direction = jax.random.normal(key, x0.shape, dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)

    epsilons = [1e-3, 5e-4, 2.5e-4]
    errors = []
    for eps in epsilons:
        delta = eps * direction
        f_true = frozen_values(x0 + delta)
        f_lin = lm["f0"] + lm["A"] @ delta
        errors.append(float(jnp.linalg.norm(f_true - f_lin, ord=2)))

    assert errors[2] < errors[1] < errors[0], (
        f"Linearization remainder is not decreasing for {case_name}: "
        f"errors={errors}"
    )

    ratio_12 = errors[0] / errors[1]
    ratio_23 = errors[1] / errors[2]
    assert ratio_12 > 2.7, f"Expected near-quadratic decay (ratio_12), got {ratio_12:.3f}"
    assert ratio_23 > 2.7, f"Expected near-quadratic decay (ratio_23), got {ratio_23:.3f}"


@pytest.mark.parametrize("case_name", CASES)
def test_create_linear_model_zero_rows_for_invalid_branch_terms(case_name):
    data, s_load, s_der0, topo, compact = _load_case_point(case_name, index=0)

    lm = linear_model.create_linear_model(data.net, s_load, s_der0, topo.v_init, topo)
    f0 = np.asarray(lm["f0"])
    A = np.asarray(lm["A"])

    invalid = ~np.asarray(data.net.branch_valid)
    invalid_flat_idx = np.flatnonzero(invalid.reshape(-1))
    if invalid_flat_idx.size == 0:
        pytest.skip(f"No padded/invalid branch terms for {case_name}")

    from core.grids.metrics import constraint_row_layout

    layout = constraint_row_layout(data.net)
    n_v = layout["n_v"]
    n_terms = layout["n_angle"] // 2
    substation_row = layout["substation_row"]

    angle_rows = np.concatenate(
        [
            n_v + invalid_flat_idx,
            n_v + n_terms + invalid_flat_idx,
        ]
    )
    branch_rows = substation_row + layout["n_sub"] + invalid_flat_idx

    assert float(np.max(np.abs(f0[angle_rows]))) < 1e-14
    assert float(np.max(np.abs(f0[branch_rows]))) < 1e-14
    assert float(np.max(np.abs(A[angle_rows, :]))) < 1e-14
    assert float(np.max(np.abs(A[branch_rows, :]))) < 1e-14
