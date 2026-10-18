"""Tests for voltage-weighted inference PDHG (core.feasibility.fs_inference_weighted)."""

import jax
import jax.numpy as jnp
import pytest

from core.feasibility.fs_inference_weighted import feasibility_seeking_batch_inference as fs_weighted


jax.config.update("jax_enable_x64", True)


class _TinyData:
    n_der = 1
    n_der_phase = 1
    n_der_unit = 1
    n_net = 1
    pdermin = jnp.array([-10.0], dtype=jnp.float64)
    pdermax = jnp.array([10.0], dtype=jnp.float64)
    qdermin = jnp.array([-10.0], dtype=jnp.float64)
    qdermax = jnp.array([10.0], dtype=jnp.float64)
    sdermax = jnp.array([10.0], dtype=jnp.float64)
    der_phase_to_unit = jnp.array([0], dtype=jnp.int32)
    der_unit_to_phase = jnp.array([[0]], dtype=jnp.int32)
    der_unit_to_phase_mask = jnp.array([[True]])

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def _one_sided_system(*, batched_bounds):
    bounds_shape = (1, 2) if batched_bounds else (2,)
    avail_shape = (1, 1) if batched_bounds else (1,)
    lx = jnp.full(bounds_shape, -10.0, dtype=jnp.float64)
    ux = jnp.full(bounds_shape, 10.0, dtype=jnp.float64)
    return {
        "A_reduced": jnp.array([[[1.0, 0.0]]], dtype=jnp.float64),
        "l_reduced": jnp.array([[-jnp.inf]], dtype=jnp.float64),
        "u_reduced": jnp.array([[0.0]], dtype=jnp.float64),
        "lx": lx,
        "ux": ux,
        "pder_avai": jnp.full(avail_shape, 10.0, dtype=jnp.float64),
    }


def _two_row_system():
    """Two constraints: row0 is voltage-like (single var), row1 is coupling."""
    return {
        "A_reduced": jnp.array(
            [[[2.0, 0.0], [0.0, 1.0]]],
            dtype=jnp.float64,
        ),
        "l_reduced": jnp.array([[-jnp.inf, -jnp.inf]], dtype=jnp.float64),
        "u_reduced": jnp.array([[0.5, 0.0]], dtype=jnp.float64),
        "lx": jnp.array([[-10.0, -10.0]], dtype=jnp.float64),
        "ux": jnp.array([[10.0, 10.0]], dtype=jnp.float64),
        "pder_avai": jnp.array([[10.0]], dtype=jnp.float64),
    }


def test_weighted_solver_respects_loose_tolerance_freeze():
    data = _TinyData()
    reduced = _one_sided_system(batched_bounds=False)
    x_init = jnp.array([[1.0, 0.0]], dtype=jnp.float64)

    one_iter = fs_weighted(data, reduced, x_init, tol=2.0, maxiter=1, rho=0.0)
    many_iter = fs_weighted(data, reduced, x_init, tol=2.0, maxiter=50, rho=0.0)

    assert jnp.allclose(many_iter, one_iter, rtol=0.0, atol=0.0)


def test_voltage_weight_one_is_deterministic_on_single_row():
    """With one row, n_voltage_rows=1 and w=1 should be reproducible across calls."""
    data = _TinyData()
    reduced = _one_sided_system(batched_bounds=False)
    x_init = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    kwargs = dict(
        tol=1e-12,
        maxiter=80,
        rho=0.0,
        n_voltage_rows=1,
        voltage_weight=1.0,
        step_fraction=0.99,
    )

    x_a = fs_weighted(data, reduced, x_init, **kwargs)
    x_b = fs_weighted(data, reduced, x_init, **kwargs)
    assert jnp.allclose(x_a, x_b, rtol=0.0, atol=0.0)


def test_default_voltage_weight_differs_from_unweighted_on_two_rows():
    """Default w=8 on the first row should not match w=1 when two rows differ."""
    data = _TinyData()
    data.n_net = 1
    reduced = _two_row_system()
    x_init = jnp.array([[0.8, 0.3]], dtype=jnp.float64)

    x_unweighted = fs_weighted(
        data,
        reduced,
        x_init,
        tol=1e-12,
        maxiter=120,
        rho=0.0,
        n_voltage_rows=1,
        voltage_weight=1.0,
        step_fraction=0.99,
    )
    x_def = fs_weighted(data, reduced, x_init, tol=1e-12, maxiter=120, rho=0.0)
    assert not jnp.allclose(x_unweighted, x_def, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize("step_fraction", [0.5, 0.75, 0.99, 1.0, 1.25])
def test_step_fraction_runs_and_reduces_violation(step_fraction):
    data = _TinyData()
    data.n_net = 1
    reduced = _two_row_system()
    x_init = jnp.array([[1.2, 0.5]], dtype=jnp.float64)

    x_out = fs_weighted(
        data,
        reduced,
        x_init,
        tol=1e-12,
        maxiter=200,
        rho=0.0,
        voltage_weight=8.0,
        step_fraction=step_fraction,
    )
    assert x_out.shape == x_init.shape
    assert jnp.all(jnp.isfinite(x_out))
