"""Optimality gap uses (f - f*) / max(|f*|, 1)."""

import jax.numpy as jnp
import numpy as np

from core.grids.metrics import compute_optimality_metrics


def test_gap_large_target_is_relative():
    cost = jnp.array([110.0, 90.0, -220.0])
    target = jnp.array([100.0, 100.0, -200.0])
    err, abs_err, gap = compute_optimality_metrics(cost, target)
    np.testing.assert_allclose(err, [10.0, -10.0, -20.0])
    np.testing.assert_allclose(abs_err, [10.0, 10.0, 20.0])
    np.testing.assert_allclose(gap, [0.10, -0.10, -0.10])


def test_gap_small_target_uses_unit_floor():
    cost = jnp.array([0.7, 0.25, -0.3, 0.5])
    target = jnp.array([0.5, 0.0, -0.5, 1.0])
    _, _, gap = compute_optimality_metrics(cost, target)
    # denominators: max(0.5,1)=1, max(0,1)=1, max(0.5,1)=1, max(1,1)=1
    np.testing.assert_allclose(gap, [0.2, 0.25, 0.2, -0.5], atol=1e-7)


def test_gap_zero_target_is_finite():
    _, _, gap = compute_optimality_metrics(jnp.array([2.0]), jnp.array([0.0]))
    np.testing.assert_allclose(gap, [2.0])
