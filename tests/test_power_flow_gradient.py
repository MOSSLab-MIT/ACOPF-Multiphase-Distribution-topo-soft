"""Gradient tests for power flow solver."""
import h5py
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tests.conftest import case_paths


CASES_FD = ["IEEE13Nodeckt"]
CASES_UNROLL = ["IEEE13Nodeckt", "IEEE123Master"]


def _load_data_and_batch(case_name: str, n: int = 1):
    from core.grids.data import OPFData
    from core.utils.batch import prepare_batch_data
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


def _prepare_solver_inputs(case_name: str):
    """Create deterministic single-sample tensors for solver comparison tests."""
    data, batch = _load_data_and_batch(case_name, n=1)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)
    return (
        jnp.asarray(s_net[0], dtype=jnp.complex128),
        jnp.asarray(batch.v_target[0], dtype=jnp.complex128),
        jnp.asarray(w[0], dtype=jnp.complex128),
        jnp.asarray(Y_inv[0], dtype=jnp.complex128),
    )


def _solver_scalar_output(v: jnp.ndarray) -> jnp.ndarray:
    """Real scalar objective used to compare complex gradients consistently."""
    return jnp.sum(v.real) + jnp.sum(v.imag)


def _pack_complex(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([jnp.ravel(x.real), jnp.ravel(x.imag)])


def _unpack_complex(z: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
    n = int(np.prod(shape))
    real = z[:n].reshape(shape)
    imag = z[n:].reshape(shape)
    return real + 1j * imag


def _assert_directional_derivatives_match(
    loss_fn,
    x0: jnp.ndarray,
    *,
    eps: float,
    n_dirs: int,
    atol: float,
    rtol: float,
):
    """Validate grad(loss_fn) using central finite differences in random directions."""
    grad = jax.grad(loss_fn)(x0)
    key = jax.random.PRNGKey(0)
    dirs = jax.random.normal(key, (n_dirs,) + x0.shape, dtype=x0.dtype)
    flat_dirs = dirs.reshape((n_dirs, -1))
    norms = jnp.linalg.norm(flat_dirs, axis=1, keepdims=True)
    dirs = (flat_dirs / norms).reshape((n_dirs,) + x0.shape)

    for direction in dirs:
        numerical = (loss_fn(x0 + eps * direction) - loss_fn(x0 - eps * direction)) / (2 * eps)
        analytical = jnp.vdot(grad, direction)
        np.testing.assert_allclose(
            float(np.asarray(numerical)),
            float(np.asarray(analytical).real),
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("case_name", CASES_FD)
def test_solve_power_flow_gradient_wrt_v_init(case_name):
    """Test gradient of power flow solution w.r.t. initial voltage."""
    from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow

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
        v_out = solve_power_flow(s0, v_init, y_inv0, w0, maxiter=5, diffiter=1)
        return _solver_scalar_output(v_out)

    _assert_directional_derivatives_match(
        solve_fn_real,
        z0,
        eps=1e-6,
        n_dirs=4,
        atol=2e-5,
        rtol=2e-4,
    )


@pytest.mark.parametrize("case_name", CASES_FD)
def test_solve_power_flow_gradient_wrt_injection(case_name):
    """Test gradient of power flow solution w.r.t. power injection.
    
    For non-holomorphic complex-to-complex functions, we use jax.vjp 
    and extract the real gradient properly.
    """
    from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow

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
        v_out = solve_power_flow(s, v_init0, y_inv0, w0, maxiter=5, diffiter=1)
        return _solver_scalar_output(v_out)

    _assert_directional_derivatives_match(
        solve_fn_real,
        z0,
        eps=1e-6,
        n_dirs=4,
        atol=2e-5,
        rtol=2e-4,
    )


@pytest.mark.parametrize("case_name", CASES_FD)
def test_solve_power_flow_gradient_wrt_Y_inv(case_name):
    """Test gradient of power flow solution w.r.t. Y_inv matrix."""
    from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow

    data, batch = _load_data_and_batch(case_name, n=1)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    s0 = jnp.asarray(s_net[0], dtype=jnp.complex128)
    y_inv0 = jnp.asarray(Y_inv[0], dtype=jnp.complex128)
    w0 = jnp.asarray(w[0], dtype=jnp.complex128)
    v_init0 = jnp.asarray(batch.v_target[0], dtype=jnp.complex128)

    rows = min(3, y_inv0.shape[0])
    cols = min(3, y_inv0.shape[1])
    y_block0 = y_inv0[:rows, :cols]
    z0 = _pack_complex(y_block0)

    def solve_fn_real(z):
        y_block = _unpack_complex(z, y_block0.shape)
        y_curr = y_inv0.at[:rows, :cols].set(y_block)
        v_out = solve_power_flow(s0, v_init0, y_curr, w0, maxiter=5, diffiter=1)
        return _solver_scalar_output(v_out)

    _assert_directional_derivatives_match(
        solve_fn_real,
        z0,
        eps=1e-6,
        n_dirs=4,
        atol=3e-5,
        rtol=3e-4,
    )


@pytest.mark.parametrize("case_name", CASES_FD)
def test_power_flow_residual_gradient(case_name):
    """Test gradient of power flow residual.
    
    For complex inputs with real outputs, the gradient has both real and imaginary parts.
    We test the gradient of a real-valued loss function w.r.t. complex voltage u.
    """
    from core.grids.power_flow import power_flow_residual

    data, batch = _load_data_and_batch(case_name, n=1)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    u = batch.v_target[0]
    Y_LL_val = Y_LL[0]
    Y_L0_val = Y_L0[0]
    v_slack = data.v_slack_local

    u0 = jnp.asarray(u, dtype=jnp.complex128)
    s0 = jnp.asarray(s_net[0], dtype=jnp.complex128)
    Y_LL0 = jnp.asarray(Y_LL_val, dtype=jnp.complex128)
    y_ns0 = jnp.asarray(Y_L0_val, dtype=jnp.complex128)
    v_slack0 = jnp.asarray(v_slack, dtype=jnp.complex128)
    z0 = _pack_complex(u0)

    def residual_fn_real(z):
        u_val = _unpack_complex(z, u0.shape)
        res = power_flow_residual(u_val, s0, Y_LL0, y_ns0, v_slack0)
        return _solver_scalar_output(res)

    _assert_directional_derivatives_match(
        residual_fn_real,
        z0,
        eps=1e-6,
        n_dirs=4,
        atol=2e-5,
        rtol=2e-4,
    )


@pytest.mark.parametrize("case_name", CASES_UNROLL)
def test_solve_power_flow_matches_full_unroll_when_diffiter_equals_maxiter(case_name):
    """When diffiter=maxiter, truncated solver must match full-unroll primal output."""
    from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow, solve_power_flow_fixed_point_full as solve_power_flow_full

    s_net, v_init, w_ref, Y_inv = _prepare_solver_inputs(case_name)
    maxiter = 8

    v_truncated = solve_power_flow(
        s_net,
        v_init,
        Y_inv,
        w_ref,
        maxiter=maxiter,
        diffiter=maxiter,
    )
    v_full = solve_power_flow_full(
        s_net,
        v_init,
        Y_inv,
        w_ref,
        maxiter=maxiter,
    )

    np.testing.assert_allclose(np.asarray(v_truncated), np.asarray(v_full), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("case_name", CASES_UNROLL)
def test_solve_power_flow_gradient_matches_full_unroll_when_diffiter_equals_maxiter(case_name):
    """When diffiter=maxiter, gradients must equal the fully unrolled reference."""
    from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow, solve_power_flow_fixed_point_full as solve_power_flow_full

    s_net, v_init, w_ref, Y_inv = _prepare_solver_inputs(case_name)
    maxiter = 8

    def loss_truncated_wrt_injection(s):
        v = solve_power_flow(
            s,
            v_init,
            Y_inv,
            w_ref,
            maxiter=maxiter,
            diffiter=maxiter,
        )
        return _solver_scalar_output(v)

    def loss_full_wrt_injection(s):
        v = solve_power_flow_full(
            s,
            v_init,
            Y_inv,
            w_ref,
            maxiter=maxiter,
        )
        return _solver_scalar_output(v)

    def loss_truncated_wrt_vinit(v0):
        v = solve_power_flow(
            s_net,
            v0,
            Y_inv,
            w_ref,
            maxiter=maxiter,
            diffiter=maxiter,
        )
        return _solver_scalar_output(v)

    def loss_full_wrt_vinit(v0):
        v = solve_power_flow_full(
            s_net,
            v0,
            Y_inv,
            w_ref,
            maxiter=maxiter,
        )
        return _solver_scalar_output(v)

    grad_s_truncated = jax.grad(loss_truncated_wrt_injection)(s_net)
    grad_s_full = jax.grad(loss_full_wrt_injection)(s_net)
    grad_v_truncated = jax.grad(loss_truncated_wrt_vinit)(v_init)
    grad_v_full = jax.grad(loss_full_wrt_vinit)(v_init)

    np.testing.assert_allclose(
        np.asarray(grad_s_truncated),
        np.asarray(grad_s_full),
        atol=1e-11,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(grad_v_truncated),
        np.asarray(grad_v_full),
        atol=1e-11,
        rtol=1e-11,
    )


@pytest.mark.parametrize("case_name", CASES_UNROLL)
def test_truncated_gradient_has_positive_alignment_with_full_unroll(case_name):
    """Truncated gradient should stay finite and aligned with full unrolling."""
    from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow, solve_power_flow_fixed_point_full as solve_power_flow_full

    s_net, v_init, w_ref, Y_inv = _prepare_solver_inputs(case_name)
    maxiter = 8
    diffiter = 2

    def loss_truncated_wrt_injection(s):
        v = solve_power_flow(
            s,
            v_init,
            Y_inv,
            w_ref,
            maxiter=maxiter,
            diffiter=diffiter,
        )
        return _solver_scalar_output(v)

    def loss_full_wrt_injection(s):
        v = solve_power_flow_full(
            s,
            v_init,
            Y_inv,
            w_ref,
            maxiter=maxiter,
        )
        return _solver_scalar_output(v)

    grad_truncated = jax.grad(loss_truncated_wrt_injection)(s_net)
    grad_full = jax.grad(loss_full_wrt_injection)(s_net)

    norm_truncated = jnp.linalg.norm(grad_truncated)
    norm_full = jnp.linalg.norm(grad_full)
    denom = jnp.maximum(norm_truncated * norm_full, 1e-18)
    cosine_similarity = jnp.real(jnp.vdot(grad_truncated, grad_full)) / denom
    rel_error = jnp.linalg.norm(grad_truncated - grad_full) / jnp.maximum(norm_full, 1e-18)

    assert bool(jnp.all(jnp.isfinite(grad_truncated)))
    assert bool(jnp.all(jnp.isfinite(grad_full)))
    assert float(norm_truncated) > 0.0
    assert float(norm_full) > 0.0
    assert float(cosine_similarity) > 0.90
    assert float(rel_error) < 0.50
