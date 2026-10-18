from pathlib import Path

import h5py
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from conftest import case_paths


CASES = ["IEEE13Nodeckt"]


def _load_data_and_batch(case_name: str, n: int = 2):
    from core.grids.data import OPFData
    from core.utils.workflow import prepare_batch_data
    from core.topology.utils import Y_LL_L0_from_scaled

    paths = case_paths(case_name)
    data = OPFData(str(paths["train"]), str(paths["test_feasible"]), str(paths["json"]))
    with h5py.File(paths["test_feasible"], "r") as f:
        inputs = np.concatenate([np.asarray(f["input"][k]) for k in data.input_keys], axis=1)[:n]
        targets = np.concatenate(
            [np.asarray(f["primal"][k]) for k in data.output_keys]
            + [np.asarray(f["meta"]["solve_time"])[:, None]],
            axis=1,
        )[:n]
    return data, prepare_batch_data(jnp.asarray(inputs), jnp.asarray(targets), data)


@pytest.mark.parametrize("case_name", CASES)
def test_residual_helper_matches_formula(case_name):
    from core.grids.power_flow import power_flow_residual

    data, batch = _load_data_and_batch(case_name, n=1)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    actual = np.asarray(power_flow_residual(batch.v_target[0], s_net[0], Y_LL[0], Y_L0[0], data.v_slack_local))
    expected = np.asarray(
        batch.v_target[0] * np.conjugate(Y_LL[0] @ batch.v_target[0] + Y_L0[0] @ data.v_slack_local)
        - s_net[0]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("case_name", CASES)
def test_fixed_point_solution_residual_is_small(case_name):
    from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow, power_flow_residual_batch

    data, batch = _load_data_and_batch(case_name, n=2)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow(s, v0, Y, w0, maxiter=80, diffiter=2)
    )
    u = solve_many(s_net, w, Y_inv, w)
    residual = power_flow_residual_batch(u, s_net, Y_LL, Y_L0, data.v_slack_local)
    # Inputs originate from float32 HDF5 datasets; the Newton residual should
    # still land near double-precision tolerance after casting to x64.
    assert np.max(np.abs(np.asarray(residual))) <= 5e-8


@pytest.mark.parametrize("case_name", CASES)
def test_power_flow_warm_start_consistency(case_name):
    from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow, power_flow_residual_batch

    data, batch = _load_data_and_batch(case_name, n=2)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow(s, v0, Y, w0, maxiter=80, diffiter=2)
    )
    u_from_w = solve_many(s_net, w, Y_inv, w)
    u_from_sample = solve_many(s_net, batch.v_target, Y_inv, w)
    residual_w = power_flow_residual_batch(u_from_w, s_net, Y_LL, Y_L0, data.v_slack_local)
    residual_sample = power_flow_residual_batch(u_from_sample, s_net, Y_LL, Y_L0, data.v_slack_local)
    assert np.max(np.abs(np.asarray(residual_w))) <= 5e-8
    assert np.max(np.abs(np.asarray(residual_sample))) <= 5e-8


@pytest.mark.parametrize("case_name", ["IEEE13Nodeckt"])
def test_voltage_from_dataset_pder_qder_matches_presolved(case_name):
    from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow

    data, batch = _load_data_and_batch(case_name, n=None)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow(s, v0, Y, w0, maxiter=80, diffiter=2)
    )
    v_pred = solve_many(s_net, batch.v_init, Y_inv, w)
    err = jnp.max(jnp.abs(v_pred - batch.v_target), axis=1)
    assert bool(jnp.all(err < 1e-5))
