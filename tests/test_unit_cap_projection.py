"""Focused tests for per-unit active-power projection (metadata-only cap)."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from core.grids.der_projection import (
    project_der_controls_batch,
    project_der_controls_from_x_batch,
    unit_active_power_violation_batch,
    unit_phase_sum_batch,
)


class _MultiPhaseUnitData:
    """Two-phase unit 0 + single-phase unit 1 (same layout as penalty projection tests)."""

    def __init__(self) -> None:
        self.n_der_phase = 3
        self.pdermin = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)
        self.pdermax = jnp.asarray([5.0, 5.0, 5.0], dtype=jnp.float64)
        self.qdermin = jnp.asarray([-5.0, -5.0, -5.0], dtype=jnp.float64)
        self.qdermax = jnp.asarray([5.0, 5.0, 5.0], dtype=jnp.float64)
        self.sdermax = jnp.asarray([5.0, 5.0, 5.0], dtype=jnp.float64)
        self.der_unit_to_phase = jnp.asarray([[0, 1], [2, 0]], dtype=jnp.int32)
        self.der_unit_to_phase_mask = jnp.asarray([[True, True], [True, False]])
        self.der_phase_to_unit = jnp.asarray([0, 0, 1], dtype=jnp.int32)


def test_unit_projection_saturates_cap_preserves_q_and_disk():
    """Phase-wise feasible (box + disk) but unit-0 sum exceeds cap; projection fixes P only."""
    data = _MultiPhaseUnitData()
    eps = jnp.asarray(1e-12, dtype=jnp.float64)

    # Unit 0: p=2+3=5 > cap 2; each phase inside [0,5] and |s|<=5 with small q.
    p = jnp.asarray([[2.0, 3.0, 1.0]], dtype=jnp.float64)
    q = jnp.asarray([[0.1, -0.2, 0.3]], dtype=jnp.float64)
    pder_avai = jnp.asarray([[2.0, 3.0]], dtype=jnp.float64)

    p_proj, q_proj = project_der_controls_batch(data, p, q, pder_avai, eps)

    # Common per-unit scale: unit 0 uses factor 2/5 on phases [2, 3].
    np.testing.assert_allclose(
        np.asarray(p_proj), np.asarray([[0.8, 1.2, 1.0]]), atol=1e-6, rtol=0.0
    )
    np.testing.assert_allclose(np.asarray(q_proj), np.asarray(q), atol=1e-12, rtol=0.0)

    unit_sum = np.asarray(
        unit_phase_sum_batch(
            p_proj, data.der_unit_to_phase, data.der_unit_to_phase_mask
        )
    )
    np.testing.assert_allclose(unit_sum[0, 0], 2.0, atol=1e-9, rtol=0.0)
    assert unit_sum[0, 1] <= pder_avai[0, 1] + 1e-9

    disk_vio = float(jnp.max(p_proj**2 + q**2 - data.sdermax**2))
    assert disk_vio <= 1e-12
    assert float(unit_active_power_violation_batch(data, p_proj, pder_avai)[0]) <= 1e-12


def test_concatenated_projection_matches_split_and_enforces_unit_cap():
    """``[p; q]`` projection: q unchanged (inside disk), unit-0 sum saturates at its cap."""
    data = _MultiPhaseUnitData()
    eps = jnp.asarray(1e-12, dtype=jnp.float64)
    x = jnp.asarray([[2.0, 3.0, 1.0, 0.1, -0.2, 0.3]], dtype=jnp.float64)
    pder_avai = jnp.asarray([[2.0, 3.0]], dtype=jnp.float64)

    x_proj = project_der_controls_from_x_batch(data, x, pder_avai, eps)
    p_proj = np.asarray(x_proj[:, : data.n_der_phase])
    q_proj = np.asarray(x_proj[:, data.n_der_phase :])
    q_raw = np.asarray(x[:, data.n_der_phase :])

    # Identical to the split (p, q) interface.
    p_split, q_split = project_der_controls_batch(
        data, x[:, : data.n_der_phase], x[:, data.n_der_phase :], pder_avai, eps
    )
    np.testing.assert_array_equal(p_proj, np.asarray(p_split))
    np.testing.assert_array_equal(q_proj, np.asarray(q_split))

    np.testing.assert_allclose(q_proj, q_raw, atol=1e-12, rtol=0.0)
    unit_sum = np.asarray(
        unit_phase_sum_batch(
            jnp.asarray(p_proj),
            data.der_unit_to_phase,
            data.der_unit_to_phase_mask,
        )
    )
    assert np.all(unit_sum <= np.asarray(pder_avai) + 1e-9)
    np.testing.assert_allclose(unit_sum[0, 0], 2.0, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(
        p_proj, np.asarray([[0.8, 1.2, 1.0]]), atol=1e-9, rtol=0.0
    )


def _assert_der_feasible(data, p, q, caps):
    p, q = np.asarray(p), np.asarray(q)
    assert np.all(p >= np.asarray(data.pdermin) - 1e-12)
    assert np.all(p <= np.asarray(data.pdermax) + 1e-12)
    assert np.all(q >= np.asarray(data.qdermin) - 1e-12)
    assert np.all(q <= np.asarray(data.qdermax) + 1e-12)
    assert np.all(p**2 + q**2 <= np.asarray(data.sdermax)**2 + 1e-12)
    assert np.all(_unit_sums_numpy(data, p) <= np.asarray(caps) + 1e-12)


def _unit_sums_numpy(data, p):
    # Independent of the JAX production reduction.
    groups = np.asarray(data.der_unit_to_phase)
    mask = np.asarray(data.der_unit_to_phase_mask)
    return np.stack([p[:, g[m]].sum(axis=1) for g, m in zip(groups, mask)], axis=1)


def test_projection_enforces_concentrated_phase_limits_and_reactive_boxes():
    data = _MultiPhaseUnitData()
    data.sdermax = jnp.asarray([1.0, 2.0, 3.0])
    data.pdermax = jnp.asarray([0.8, 2.5, 3.0])
    data.qdermin = jnp.asarray([-0.3, -0.4, -0.5])
    data.qdermax = jnp.asarray([0.2, 0.6, 0.7])
    p = jnp.asarray([[100.0, 0.0, -1.0], [0.0, 100.0, 100.0]])
    q = jnp.asarray([[20.0, -20.0, 20.0], [-20.0, 20.0, -20.0]])
    caps = jnp.asarray([[2.0, 3.0], [2.0, 3.0]])
    project = jax.jit(lambda pi, qi, ai: project_der_controls_batch(data, pi, qi, ai, 1e-12))
    p_out, q_out = project(p, q, caps)
    _assert_der_feasible(data, p_out, q_out, caps)
    np.testing.assert_allclose(p_out, [[0.8, 0.0, 0.0], [0.0, 2.0, 3.0]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(q_out, [[0.2, -0.4, 0.7], [-0.3, 0.0, 0.0]], atol=1e-12, rtol=0)


def test_disk_boundary_has_zero_reactive_headroom_and_finite_gradients():
    data = _MultiPhaseUnitData()
    caps = jnp.asarray([[5.0, 5.0]])
    x = jnp.asarray([[5.0, 0.0, 5.0, 1.0, 10.0, -1.0]])
    project = jax.jit(lambda xi: project_der_controls_from_x_batch(data, xi, caps, 1e-12))
    out = project(x)
    np.testing.assert_array_equal(out, [[5.0, 0.0, 5.0, 0.0, 5.0, 0.0]])
    assert np.isfinite(np.asarray(jax.jacfwd(project)(x))).all()
    assert np.isfinite(np.asarray(jax.jacrev(project)(x))).all()


def test_projection_respects_positive_active_lower_bounds():
    data = _MultiPhaseUnitData()
    data.pdermin = jnp.asarray([0.2, 0.4, 0.1])
    p = jnp.asarray([[0.0, 10.0, 10.0]])
    q = jnp.zeros_like(p)
    caps = jnp.asarray([[0.7, 0.1]])
    p_out, q_out = project_der_controls_batch(data, p, q, caps, 1e-12)
    _assert_der_feasible(data, p_out, q_out, caps)
    np.testing.assert_allclose(p_out, [[0.2, 0.5, 0.1]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(_unit_sums_numpy(data, np.asarray(p_out)), caps, atol=1e-12, rtol=0)


def test_projection_reserves_headroom_for_reactive_boxes_excluding_zero():
    data = _MultiPhaseUnitData()
    data.sdermax = jnp.asarray([1.0, 2.0, 3.0])
    data.qdermin = jnp.asarray([0.6, -2.0, 0.0])
    data.qdermax = jnp.asarray([0.8, -1.2, 0.4])
    caps = jnp.asarray([[10.0, 10.0]])
    p_out, q_out = project_der_controls_batch(
        data, jnp.full((1, 3), 100.0), jnp.zeros((1, 3)), caps, 1e-12
    )
    _assert_der_feasible(data, p_out, q_out, caps)
    np.testing.assert_allclose(p_out, [[0.8, 1.6, 3.0]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(q_out, [[0.6, -1.2, 0.0]], atol=1e-12, rtol=0)


def test_feasible_tiny_controls_are_unchanged():
    data = _MultiPhaseUnitData()
    p = jnp.asarray([[1e-15, 1e-15, 0.0]])
    q = jnp.asarray([[0.1, -0.2, 0.3]])
    # Doubling is exact in binary, so the cap equals the represented phase sum.
    caps = jnp.asarray([[2e-15, 0.0]])
    p_out, q_out = project_der_controls_batch(data, p, q, caps, 1e-12)
    np.testing.assert_array_equal(p_out, p)
    np.testing.assert_array_equal(q_out, q)


def test_disk_gradient_matches_analytic_derivative_away_from_boundary():
    data = _MultiPhaseUnitData()
    caps = jnp.asarray([[10.0, 10.0]])

    def projected_q0(p0):
        p = jnp.array([[p0, 0.0, 0.0]])
        q = jnp.array([[10.0, 0.0, 0.0]])
        return project_der_controls_batch(data, p, q, caps, 1e-12)[1][0, 0]

    # At p=3, smax=5: q=sqrt(25-9)=4 and dq/dp=-3/4.
    np.testing.assert_allclose(jax.jit(projected_q0)(3.0), 4.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(jax.jit(jax.grad(projected_q0))(3.0), -0.75, atol=1e-12, rtol=0)


def test_zero_availability_has_finite_gradients_for_controls_and_caps():
    data = _MultiPhaseUnitData()
    x = jnp.asarray([[0.0, 0.0, 0.0, 0.1, -0.2, 0.3],
                     [10.0, -1.0, 20.0, 10.0, -20.0, 30.0]])
    caps = jnp.zeros((2, 2))
    project = jax.jit(lambda xi, ai: project_der_controls_from_x_batch(data, xi, ai, 1e-12))
    out = project(x, caps)
    np.testing.assert_array_equal(out[:, :3], np.zeros((2, 3)))
    _assert_der_feasible(data, out[:, :3], out[:, 3:], caps)
    _, gradients = jax.value_and_grad(lambda xi, ai: jnp.sum(project(xi, ai)), argnums=(0, 1))(x, caps)
    assert all(np.isfinite(np.asarray(g)).all() for g in gradients)


@pytest.mark.parametrize("batch_caps", [False, True])
def test_random_projection_is_feasible_and_idempotent(batch_caps):
    data = _MultiPhaseUnitData()
    data.pdermax = jnp.asarray([2.0, 4.0, 6.0])
    data.qdermin = jnp.asarray([-0.3, -2.0, -4.0])
    data.qdermax = jnp.asarray([0.7, 1.0, 2.0])
    rng = np.random.default_rng(42)
    p = jnp.asarray(rng.uniform(-20, 20, (1024, 3)))
    q = jnp.asarray(rng.uniform(-20, 20, (1024, 3)))
    caps = jnp.asarray(rng.uniform(0, 10, (1024, 2)) if batch_caps else [3.0, 4.0])
    project = jax.jit(lambda pi, qi: project_der_controls_batch(data, pi, qi, caps, 1e-12))
    p_out, q_out = project(p, q)
    _assert_der_feasible(data, p_out, q_out, caps)
    p_twice, q_twice = project(p_out, q_out)
    np.testing.assert_allclose(p_twice, p_out, atol=1e-12, rtol=0)
    np.testing.assert_allclose(q_twice, q_out, atol=1e-12, rtol=0)
