"""Fixed-point PF vs dataset v_target: residual convergence and root selection.

On large feeders (IEEE123), many phase nodes have zero *net* injection (DER - load).
At those buses the power-balance residual can be tiny for many voltage values; the
Z-bus map F(v)=w+Y^{-1}conj(s/v) picks a different root than the stored OPF v_target.
On buses with nonzero net injection, the solver and v_target agree to ~1e-9.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from conftest import case_paths


def _load_data_and_batch(case_name: str, n: int | None = 2):
    from core.grids.data import OPFData
    from core.utils.workflow import prepare_batch_data
    from core.topology.utils import Y_LL_L0_from_scaled

    paths = case_paths(case_name)
    data = OPFData(str(paths["train"]), str(paths["test_feasible"]), str(paths["json"]))
    inputs = data._load_inputs(paths["test_feasible"], samples=n)
    targets = data._load_targets(paths["test_feasible"], samples=n)
    return data, prepare_batch_data(jnp.asarray(inputs), jnp.asarray(targets), data)


@pytest.mark.parametrize("case_name", ["IEEE13Nodeckt", "IEEE123Master"])
def test_fixed_point_residual_at_solution(case_name):
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


@pytest.mark.parametrize("case_name", ["IEEE13Nodeckt", "IEEE123Master"])
def test_vtarget_satisfies_power_flow_residual(case_name):
    from core.grids.power_flow import power_flow_residual_batch

    data, batch = _load_data_and_batch(case_name, n=2)
    Y_scaled, _, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    res = power_flow_residual_batch(
        batch.v_target, s_net, Y_LL, Y_L0, data.v_slack_local
    )
    assert np.max(np.abs(np.asarray(res))) <= 5e-8


@pytest.mark.parametrize("case_name", ["IEEE13Nodeckt"])
def test_fixed_point_matches_vtarget_all_buses(case_name):
    from core.grids.power_flow import solve_power_flow_fixed_point

    data, batch = _load_data_and_batch(case_name, n=None)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_fixed_point(
            s, v0, Y, w0, maxiter=80, diffiter=2
        )
    )
    u = solve_many(s_net, batch.v_init, Y_inv, w)
    err = jnp.max(jnp.abs(u - batch.v_target), axis=1)
    assert bool(jnp.all(err < 1e-5))


@pytest.mark.parametrize("case_name", ["IEEE123Master"])
def test_fixed_point_matches_vtarget_on_nonzero_injection_buses(case_name):
    """IEEE123: FP root differs on zero-injection buses; agree where s_net != 0."""
    from core.grids.power_flow import solve_power_flow_fixed_point

    data, batch = _load_data_and_batch(case_name, n=5)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_fixed_point(
            s, v0, Y, w0, maxiter=80, diffiter=2
        )
    )
    u = solve_many(s_net, batch.v_init, Y_inv, w)

    active = jnp.abs(s_net) > 1e-12
    err_active = jnp.max(jnp.where(active, jnp.abs(u - batch.v_target), 0.0), axis=1)
    assert bool(jnp.all(err_active < 1e-5))

    # Documented: zero-injection buses can differ O(0.1 pu) between OPF v and Z-bus root
    err_all = jnp.max(jnp.abs(u - batch.v_target), axis=1)
    assert float(jnp.max(err_all)) > 0.1


@pytest.mark.parametrize("case_name", ["IEEE123Master"])
def test_vtarget_not_zbus_fixed_point_on_zero_injection(case_name):
    """v_target satisfies s=u*conj(Yu) but F(v_target) != v_target on zero-s buses."""
    data, batch = _load_data_and_batch(case_name, n=1)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)[0]

    v = batch.v_target[0]
    Fv = w[0] + Y_inv[0] @ jnp.conj(s_net / v)
    zero_s = jnp.abs(s_net) < 1e-14

    defect = jnp.abs(v - Fv)
    assert float(jnp.max(jnp.where(zero_s, defect, 0.0))) > 0.1
    assert float(jnp.max(jnp.where(~zero_s, defect, 0.0))) < 1e-5
