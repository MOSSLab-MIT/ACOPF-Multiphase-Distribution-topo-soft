import numpy as np
import jax.numpy as jnp
import pytest

from core.grids.metrics import ac_constraint_violation_vector
from methods.proposed import trainer as proposed_trainer
from methods.direct_prediction import trainer as direct_prediction_trainer


class _DummyData:
    def __init__(self) -> None:
        self.n_der_phase = 3
        self.n_total = 4
        self.n_net = self.n_total - 1
        self.pdermin = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)
        self.pdermax = jnp.asarray([5.0, 5.0, 5.0], dtype=jnp.float64)
        self.qdermin = jnp.asarray([-5.0, -5.0, -5.0], dtype=jnp.float64)
        self.qdermax = jnp.asarray([5.0, 5.0, 5.0], dtype=jnp.float64)
        self.sdermax = jnp.asarray([5.0, 5.0, 5.0], dtype=jnp.float64)
        # Unit 0 has phases [0, 1], unit 1 has phase [2].
        self.der_unit_to_phase = jnp.asarray([[0, 1], [2, 0]], dtype=jnp.int32)
        self.der_unit_to_phase_mask = jnp.asarray([[True, True], [True, False]])
        self.der_phase_to_unit = jnp.asarray([0, 0, 1], dtype=jnp.int32)


def _unit_phase_sum(p_phase: np.ndarray, data: _DummyData) -> np.ndarray:
    groups = np.asarray(data.der_unit_to_phase)
    mask = np.asarray(data.der_unit_to_phase_mask, dtype=np.float64)
    gathered = p_phase[:, groups] * mask[None, :, :]
    return np.sum(gathered, axis=2)


def test_proposed_projection_applies_common_per_unit_scaling():
    data = _DummyData()
    x_pred_raw = jnp.asarray([[1.0, 3.0, 4.0, 0.1, -0.2, 0.3]], dtype=jnp.float64)
    pder_avai = jnp.asarray([[2.0, 3.0]], dtype=jnp.float64)

    x_proj = proposed_trainer._project_predictions(x_pred_raw, pder_avai, data)
    p_proj = np.asarray(x_proj[:, : data.n_der_phase])

    expected = np.asarray([[0.5, 1.5, 3.0]], dtype=np.float64)
    np.testing.assert_allclose(p_proj, expected, atol=1e-12, rtol=1e-12)

    raw = np.asarray(x_pred_raw[:, : data.n_der_phase])
    ratio_u0 = p_proj[0, :2] / raw[0, :2]
    np.testing.assert_allclose(ratio_u0, np.asarray([0.5, 0.5]), atol=1e-12, rtol=1e-12)


def test_direct_prediction_projection_scales_p_and_preserves_voltage_parts():
    data = _DummyData()
    p_raw = [1.0, 3.0, 4.0]
    q_raw = [0.2, -0.4, 0.6]
    vr_raw = [0.98, 1.01, 1.02]
    vi_raw = [0.01, -0.02, 0.03]
    x_pred_raw = jnp.asarray([p_raw + q_raw + vr_raw + vi_raw], dtype=jnp.float64)
    pder_avai = jnp.asarray([[2.0, 3.0]], dtype=jnp.float64)

    x_proj = direct_prediction_trainer._project_predictions(data, x_pred_raw, pder_avai)
    p_proj = np.asarray(x_proj[:, : data.n_der_phase])
    tail_proj = np.asarray(x_proj[:, 2 * data.n_der_phase :])
    tail_raw = np.asarray(x_pred_raw[:, 2 * data.n_der_phase :])

    np.testing.assert_allclose(p_proj, np.asarray([[0.5, 1.5, 3.0]]), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(tail_proj, tail_raw, atol=1e-12, rtol=1e-12)


def test_both_penalty_projections_enforce_unit_availability_with_even_scaling():
    data = _DummyData()
    rng = np.random.default_rng(7)
    batch_size = 8

    p_raw = rng.uniform(0.0, 8.0, size=(batch_size, data.n_der_phase))
    q_raw = rng.uniform(-2.0, 2.0, size=(batch_size, data.n_der_phase))
    pder_avai = rng.uniform(0.2, 6.0, size=(batch_size, 2))

    x_proposed = jnp.asarray(np.concatenate([p_raw, q_raw], axis=1), dtype=jnp.float64)
    x_direct = jnp.asarray(
        np.concatenate(
            [
                p_raw,
                q_raw,
                rng.uniform(0.95, 1.05, size=(batch_size, data.n_total - 1)),
                rng.uniform(-0.05, 0.05, size=(batch_size, data.n_total - 1)),
            ],
            axis=1,
        ),
        dtype=jnp.float64,
    )
    pder_avai_jnp = jnp.asarray(pder_avai, dtype=jnp.float64)

    p_proposed = np.asarray(
        proposed_trainer._project_predictions(x_proposed, pder_avai_jnp, data)[:, : data.n_der_phase]
    )
    p_direct = np.asarray(
        direct_prediction_trainer._project_predictions(data, x_direct, pder_avai_jnp)[:, : data.n_der_phase]
    )
    np.testing.assert_allclose(p_proposed, p_direct, atol=1e-12, rtol=1e-12)

    unit_sum = _unit_phase_sum(p_proposed, data)
    assert np.all(unit_sum <= pder_avai + 1e-12)

    groups = np.asarray(data.der_unit_to_phase)
    mask = np.asarray(data.der_unit_to_phase_mask)
    for b in range(batch_size):
        for unit_idx in range(groups.shape[0]):
            phase_ids = groups[unit_idx][mask[unit_idx]]
            if phase_ids.size <= 1:
                continue
            # Common scaling preserves ratios after enforcing per-phase bounds.
            raw = np.clip(p_raw[b, phase_ids], 0.0, np.asarray(data.pdermax)[phase_ids])
            out = p_proposed[b, phase_ids]
            active = raw > 1e-10
            if np.count_nonzero(active) <= 1:
                continue
            ratios = out[active] / raw[active]
            assert np.max(ratios) - np.min(ratios) <= 1e-12


@pytest.mark.parametrize("method", ["proposed", "direct_prediction"])
def test_training_projection_enforces_all_der_constraints(method):
    data = _DummyData()
    data.pdermax = jnp.asarray([0.8, 2.5, 3.0])
    data.sdermax = jnp.asarray([1.0, 2.0, 3.0])
    data.qdermin = jnp.asarray([-0.3, -0.4, -0.5])
    data.qdermax = jnp.asarray([0.2, 0.6, 0.7])
    x = jnp.asarray([[100.0, 0.0, -1.0, 20.0, -20.0, 20.0]])
    caps = jnp.asarray([[2.0, 3.0]])
    if method == "proposed":
        x_out = proposed_trainer._project_predictions(x, caps, data)
    else:
        voltages = jnp.asarray([[1.0, 0.98, 1.02, 0.0, 0.01, -0.01]])
        full_out = direct_prediction_trainer._project_predictions(
            data, jnp.concatenate([x, voltages], axis=1), caps
        )
        np.testing.assert_array_equal(full_out[:, 6:], voltages)
        x_out = full_out[:, :6]
    violations = ac_constraint_violation_vector(data, x_out[0], None, None, caps[0])
    assert np.all(np.asarray(violations) <= 1e-12)
    np.testing.assert_allclose(x_out, [[0.8, 0.0, 0.0, 0.2, -0.4, 0.7]], atol=1e-12, rtol=0)
