"""Rigorous tests for implicit differentiation of solve_power_flow_gmres."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from core.grids.power_flow import solve_power_flow_gmres, make_solve_power_flow_gmres_implicit


def test_forward_pass():
    """Test that forward pass works and gives same result as explicit version."""
    key = jax.random.PRNGKey(42)
    n = 5

    # Create realistically conditioned system
    Y_LL_inv = jnp.eye(n, dtype=jnp.complex128) * 1.0
    s = jnp.ones(n, dtype=jnp.complex128) * 0.001
    v_init = jnp.ones(n, dtype=jnp.complex128)
    w = jnp.ones(n, dtype=jnp.complex128)

    # Create implicit solver with same parameters
    eps, maxiter, diffiter = 1e-7, 15, 5
    solve_implicit = make_solve_power_flow_gmres_implicit(eps, maxiter, diffiter)

    # Both should give same result
    v_gmres = solve_power_flow_gmres(s, v_init, Y_LL_inv, w, eps, maxiter, diffiter)
    v_implicit = solve_implicit(s, v_init, Y_LL_inv, w)

    assert jnp.allclose(v_gmres, v_implicit, atol=1e-10), \
        f"Solutions don't match: max diff = {jnp.max(jnp.abs(v_gmres - v_implicit))}"


def test_solution_is_fixed_point():
    """Test that the solution is actually a fixed point (residual near zero)."""
    n = 5
    Y_LL_inv = jnp.eye(n, dtype=jnp.complex128) * 1.0
    s = jnp.ones(n, dtype=jnp.complex128) * 0.001
    v_init = jnp.ones(n, dtype=jnp.complex128)
    w = jnp.ones(n, dtype=jnp.complex128)

    solve_implicit = make_solve_power_flow_gmres_implicit(maxiter=20, diffiter=5)
    v = solve_implicit(s, v_init, Y_LL_inv, w)

    # Compute residual
    def fixed_point(v):
        return w + Y_LL_inv @ jnp.conj(s / v)

    r = v - fixed_point(v)
    residual_norm = jnp.linalg.norm(r)

    assert residual_norm < 1e-5, \
        f"Residual norm too large: {residual_norm}"


def test_gradient_vs_finite_differences():
    """Rigorous test: compare implicit gradient with finite differences."""
    n = 5
    Y_LL_inv = jnp.eye(n, dtype=jnp.complex128) * 1.0
    s = jnp.ones(n, dtype=jnp.complex128) * 0.001
    v_init = jnp.ones(n, dtype=jnp.complex128)
    w = jnp.ones(n, dtype=jnp.complex128)

    solve_implicit = make_solve_power_flow_gmres_implicit(maxiter=15, diffiter=5)

    # Loss function: sum of squared voltage magnitudes
    def loss_fn(s):
        v = solve_implicit(s, v_init, Y_LL_inv, w)
        return jnp.sum(jnp.abs(v)**2)

    # Gradient via implicit differentiation
    grad_implicit = jax.grad(loss_fn)(s)

    # Gradient via finite differences
    eps = 1e-6
    grad_fd = jnp.zeros_like(s, dtype=jnp.complex128)

    for i in range(n):
        # Real part
        s_plus = s.at[i].set(s[i] + eps)
        s_minus = s.at[i].set(s[i] - eps)
        grad_real = (loss_fn(s_plus) - loss_fn(s_minus)) / (2 * eps)

        # Imag part (complex step)
        s_plus = s.at[i].set(s[i] + eps * 1j)
        s_minus = s.at[i].set(s[i] - eps * 1j)
        grad_imag = (loss_fn(s_plus) - loss_fn(s_minus)) / (2 * eps)

        grad_fd = grad_fd.at[i].set(grad_real + 1j * grad_imag)

    # Compare gradients
    max_diff = jnp.max(jnp.abs(grad_implicit - grad_fd))
    rel_error = max_diff / (jnp.linalg.norm(grad_implicit) + 1e-10)

    assert max_diff < 0.1, \
        f"Gradient absolute difference too large: {max_diff}"
    assert rel_error < 0.05, \
        f"Gradient relative error too large: {rel_error}"

    print(f"  Max absolute diff: {max_diff}")
    print(f"  Relative error: {rel_error}")


def test_gradient_direction_is_ascent():
    """Test that gradient points in direction of steepest ascent."""
    n = 5
    Y_LL_inv = jnp.eye(n, dtype=jnp.complex128) * 1.0
    s = jnp.ones(n, dtype=jnp.complex128) * 0.001
    v_init = jnp.ones(n, dtype=jnp.complex128)
    w = jnp.ones(n, dtype=jnp.complex128)

    solve_implicit = make_solve_power_flow_gmres_implicit()

    def loss_fn(s):
        v = solve_implicit(s, v_init, Y_LL_inv, w)
        return jnp.sum(jnp.abs(v)**2)

    grad = jax.grad(loss_fn)(s)
    loss_0 = loss_fn(s)

    # Gradient points in direction of steepest ASCENT
    s_perturbed = s + 1e-6 * grad  # Positive gradient = should increase loss
    loss_1 = loss_fn(s_perturbed)

    assert loss_1 >= loss_0 - 1e-10, \
        f"Gradient direction wrong: loss went from {loss_0} to {loss_1}"


def test_jit_compatibility():
    """Test that the function works under JIT."""
    n = 5
    Y_LL_inv = jnp.eye(n, dtype=jnp.complex128) * 1.0
    s = jnp.ones(n, dtype=jnp.complex128) * 0.001
    v_init = jnp.ones(n, dtype=jnp.complex128)
    w = jnp.ones(n, dtype=jnp.complex128)

    solve_implicit = make_solve_power_flow_gmres_implicit()
    jit_solve = jax.jit(solve_implicit)
    jit_grad = jax.jit(jax.grad(lambda s: jnp.sum(jnp.abs(jit_solve(s, v_init, Y_LL_inv, w))**2)))

    # Should work under JIT
    v = jit_solve(s, v_init, Y_LL_inv, w)
    assert v.shape == (n,), f"Expected shape ({n},), got {v.shape}"

    grad = jit_grad(s)
    assert grad.shape == (n,), f"Expected gradient shape ({n},), got {grad.shape}"
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"


def test_batch_support():
    """Test batched version works correctly."""
    n = 5
    batch_size = 3
    Y_LL_inv = jnp.eye(n, dtype=jnp.complex128) * 1.0
    s_batch = jnp.ones((batch_size, n), dtype=jnp.complex128) * 0.001
    v_init_batch = jnp.ones((batch_size, n), dtype=jnp.complex128)
    w = jnp.ones(n, dtype=jnp.complex128)

    solve_implicit = make_solve_power_flow_gmres_implicit()
    batched_solve = jax.jit(
        jax.vmap(solve_implicit, in_axes=(0, 0, None, None))
    )

    v_batch = batched_solve(s_batch, v_init_batch, Y_LL_inv, w)
    assert v_batch.shape == (batch_size, n), \
        f"Expected shape ({batch_size}, {n}), got {v_batch.shape}"


if __name__ == "__main__":
    print("Running test_forward_pass...")
    test_forward_pass()
    print("PASSED")

    print("Running test_solution_is_fixed_point...")
    test_solution_is_fixed_point()
    print("PASSED")

    print("Running test_gradient_vs_finite_differences...")
    test_gradient_vs_finite_differences()
    print("PASSED")

    print("Running test_gradient_direction_is_ascent...")
    test_gradient_direction_is_ascent()
    print("PASSED")

    print("Running test_jit_compatibility...")
    test_jit_compatibility()
    print("PASSED")

    print("Running test_batch_support...")
    test_batch_support()
    print("PASSED")

    print("\nAll rigorous tests passed!")
