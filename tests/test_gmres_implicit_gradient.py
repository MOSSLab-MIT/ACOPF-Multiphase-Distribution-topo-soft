import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tests.test_power_flow_gradient import (
    _load_data_and_batch,
    _pack_complex,
    _unpack_complex,
    _solver_scalar_output,
    _assert_directional_derivatives_match,
)
from core.grids.power_flow import solve_power_flow_gmres_implicit
from core.topology.utils import Y_LL_L0_from_scaled

CASES = ["IEEE13Nodeckt", "240bus_Master"]

@pytest.mark.parametrize("case_name", CASES)
def test_gmres_implicit_gradient_wrt_v_init(case_name):
    data, batch = _load_data_and_batch(case_name, n=1)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    s0 = jnp.asarray(s_net[0], dtype=jnp.complex128)
    y_inv0 = jnp.asarray(Y_inv[0], dtype=jnp.complex128)
    w0 = jnp.asarray(w[0], dtype=jnp.complex128)
    v_init0 = jnp.asarray(batch.v_target[0], dtype=jnp.complex128)
    z0 = _pack_complex(v_init0)

    def solve_fn_real(z):
        v_init = _unpack_complex(z, v_init0.shape)
        v_out = solve_power_flow_gmres_implicit(s0, v_init, y_inv0, w0)
        return _solver_scalar_output(v_out)

    _assert_directional_derivatives_match(
        solve_fn_real,
        z0,
        eps=1e-6,
        n_dirs=4,
        atol=2e-5,
        rtol=2e-4,
    )

@pytest.mark.parametrize("case_name", CASES)
def test_gmres_implicit_gradient_wrt_injection(case_name):
    data, batch = _load_data_and_batch(case_name, n=1)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    s0 = jnp.asarray(s_net[0], dtype=jnp.complex128)
    y_inv0 = jnp.asarray(Y_inv[0], dtype=jnp.complex128)
    w0 = jnp.asarray(w[0], dtype=jnp.complex128)
    v_init0 = jnp.asarray(batch.v_target[0], dtype=jnp.complex128)
    z0 = _pack_complex(s0)

    def solve_fn_real(z):
        s = _unpack_complex(z, s0.shape)
        v_out = solve_power_flow_gmres_implicit(s, v_init0, y_inv0, w0)
        return _solver_scalar_output(v_out)

    _assert_directional_derivatives_match(
        solve_fn_real,
        z0,
        eps=1e-6,
        n_dirs=4,
        atol=2e-5,
        rtol=2e-4,
    )


CASES_UNROLL = ["IEEE13Nodeckt"]

@pytest.mark.parametrize("case_name", CASES_UNROLL)
def test_gmres_implicit_matches_unrolled(case_name):
    """Test that the implicit gradient matches the fully unrolled GMRES gradient."""
    from core.grids.power_flow import make_solve_power_flow_gmres_implicit, solve_power_flow_gmres
    data, batch = _load_data_and_batch(case_name, n=1)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    s0 = jnp.asarray(s_net[0], dtype=jnp.complex128)
    y_inv0 = jnp.asarray(Y_inv[0], dtype=jnp.complex128)
    w0 = jnp.asarray(w[0], dtype=jnp.complex128)
    v_init0 = jnp.asarray(batch.v_target[0], dtype=jnp.complex128)

    maxiter = 10
    implicit_solver = make_solve_power_flow_gmres_implicit(maxiter=maxiter, diffiter=3)

    def loss_implicit_s(z):
        s = _unpack_complex(z, s0.shape)
        v_out = implicit_solver(s, v_init0, y_inv0, w0)
        return _solver_scalar_output(v_out)

    def loss_unrolled_s(z):
        s = _unpack_complex(z, s0.shape)
        v_out = solve_power_flow_gmres(s, v_init0, y_inv0, w0, maxiter=maxiter, diffiter=maxiter)
        return _solver_scalar_output(v_out)

    def loss_implicit_v(z):
        v = _unpack_complex(z, v_init0.shape)
        v_out = implicit_solver(s0, v, y_inv0, w0)
        return _solver_scalar_output(v_out)

    def loss_unrolled_v(z):
        v = _unpack_complex(z, v_init0.shape)
        v_out = solve_power_flow_gmres(s0, v, y_inv0, w0, maxiter=maxiter, diffiter=maxiter)
        return _solver_scalar_output(v_out)

    z_s = _pack_complex(s0)
    z_v = _pack_complex(v_init0)

    g_imp_s = jax.grad(loss_implicit_s)(z_s)
    g_unr_s = jax.grad(loss_unrolled_s)(z_s)
    
    g_imp_v = jax.grad(loss_implicit_v)(z_v)
    g_unr_v = jax.grad(loss_unrolled_v)(z_v)

    np.testing.assert_allclose(np.asarray(g_imp_s), np.asarray(g_unr_s), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(g_imp_v), np.asarray(g_unr_v), atol=1e-10, rtol=1e-10)

