"""Focused tests for per-unit active-power projection (metadata-only cap)."""

import numpy as np
import jax.numpy as jnp

from core.feasibility.fs_common import (
    project_der_controls_batch,
    project_primal_controls_batch,
    unit_active_power_violation_batch,
    unit_phase_sum_batch,
)


class _MultiPhaseUnitData:
    """Two-phase unit 0 + single-phase unit 1 (same layout as penalty projection tests)."""

    def __init__(self) -> None:
        self.n_der_phase = 3
        self.pdermin = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)
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


def test_primal_projection_order_box_disk_then_unit_cap():
    """Full control projection leaves q unchanged and enforces unit sum after box/disk."""
    data = _MultiPhaseUnitData()
    eps = jnp.asarray(1e-12, dtype=jnp.float64)
    lx = jnp.concatenate([data.pdermin, -data.sdermax])
    ux = jnp.concatenate([data.sdermax, data.sdermax])
    x = jnp.asarray([[2.0, 3.0, 1.0, 0.1, -0.2, 0.3]], dtype=jnp.float64)
    pder_avai = jnp.asarray([[2.0, 3.0]], dtype=jnp.float64)

    x_proj = project_primal_controls_batch(data, x, lx, ux, pder_avai, eps)
    p_proj = np.asarray(x_proj[:, : data.n_der_phase])
    q_proj = np.asarray(x_proj[:, data.n_der_phase :])
    q_raw = np.asarray(x[:, data.n_der_phase :])

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
