"""Tests for ``core.grids.power_flow`` multi-solver API (fixed-point, NR, GMRES)."""

import h5py
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from conftest import case_paths
from core.topology.utils import Y_LL_L0_from_scaled


CASES = ["IEEE13Nodeckt"]


def _load_data_and_batch(case_name: str, n: int = 2):
    from core.grids.data import OPFData
    from core.utils.batch import prepare_batch_data

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
def test_fixed_point_residual_small(case_name):
    from core.grids.power_flow import solve_power_flow_fixed_point, power_flow_residual_batch

    data, batch = _load_data_and_batch(case_name, n=2)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_fixed_point(
            s, v0, Y, w0, maxiter=80, diffiter=2
        )
    )
    u = solve_many(s_net, w, Y_inv, w)
    residual = power_flow_residual_batch(u, s_net, Y_LL, Y_L0, data.v_slack_local)
    assert np.max(np.abs(np.asarray(residual))) <= 5e-8


@pytest.mark.parametrize("case_name", CASES)
def test_nr_residual_small(case_name):
    from core.grids.power_flow import solve_power_flow_nr, power_flow_residual_batch

    data, batch = _load_data_and_batch(case_name, n=2)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_nr(s, v0, Y, w0, maxiter=25, diffiter=3)
    )
    u = solve_many(s_net, w, Y_inv, w)
    residual = power_flow_residual_batch(u, s_net, Y_LL, Y_L0, data.v_slack_local)
    assert np.max(np.abs(np.asarray(residual))) <= 5e-8


@pytest.mark.parametrize("case_name", CASES)
def test_gmres_residual_small(case_name):
    from core.grids.power_flow import solve_power_flow_gmres, power_flow_residual_batch

    data, batch = _load_data_and_batch(case_name, n=2)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_gmres(s, v0, Y, w0, maxiter=25, diffiter=3)
    )
    u = solve_many(s_net, w, Y_inv, w)
    residual = power_flow_residual_batch(u, s_net, Y_LL, Y_L0, data.v_slack_local)
    assert np.max(np.abs(np.asarray(residual))) <= 5e-8


@pytest.mark.parametrize("case_name", ["IEEE13Nodeckt"])
def test_newton_variants_agree_with_fixed_point(case_name):
    from core.grids.power_flow import (
        solve_power_flow_fixed_point,
        solve_power_flow_nr,
        solve_power_flow_gmres,
    )

    data, batch = _load_data_and_batch(case_name, n=2)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    u_fp = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_fixed_point(
            s, v0, Y, w0, maxiter=80, diffiter=2
        )
    )(s_net, w, Y_inv, w)
    u_nr = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_nr(s, v0, Y, w0, maxiter=25, diffiter=3)
    )(s_net, w, Y_inv, w)
    u_gm = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_gmres(s, v0, Y, w0, maxiter=25, diffiter=3)
    )(s_net, w, Y_inv, w)

    np.testing.assert_allclose(u_nr, u_fp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(u_gm, u_fp, rtol=1e-5, atol=1e-5)


def test_synthetic_toy_network_all_solvers():
    """Tiny hand-built network: target injection implied by a chosen `u`, then recover `u`."""
    from core.grids.power_flow import (
        power_flow_residual,
        solve_power_flow_fixed_point,
        solve_power_flow_gmres,
        solve_power_flow_nr,
    )

    n = 3
    # Diagonal-dominant admittance (well-conditioned); small fixed coupling.
    Y_LL = (6.0 + 6.0j) * jnp.eye(n, dtype=jnp.complex128)
    Y_LL = Y_LL.at[0, 1].set(0.2 + 0.1j).at[1, 0].set(0.2 + 0.1j)
    Y_L0 = ((0.1 + 0.05j) * jnp.ones((n, 1))).astype(jnp.complex128)
    v_slack_local = jnp.array([1.0 + 0.0j], dtype=jnp.complex128)
    Y_LL_inv = jnp.linalg.inv(Y_LL)
    w = -Y_LL_inv @ (Y_L0 @ v_slack_local)

    u_star = jnp.array([1.0 + 0.01j, 1.02 - 0.02j, 0.99 + 0.01j], dtype=jnp.complex128)
    s = u_star * jnp.conj(Y_LL @ u_star + Y_L0 @ v_slack_local)
    # Use u_star as initial guess for better convergence
    v_init = u_star

    for solve in (
        lambda: solve_power_flow_fixed_point(s, v_init, Y_LL_inv, w, maxiter=100, diffiter=3),
        lambda: solve_power_flow_nr(s, v_init, Y_LL_inv, w, maxiter=20, diffiter=3),
        lambda: solve_power_flow_gmres(s, v_init, Y_LL_inv, w, maxiter=20, diffiter=3),
    ):
        u = solve()
        r = power_flow_residual(u, s, Y_LL, Y_L0, v_slack_local)
        assert float(jnp.max(jnp.abs(r))) <= 1e-10
