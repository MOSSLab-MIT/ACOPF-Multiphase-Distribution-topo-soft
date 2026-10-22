"""Closed-form projection of DER controls onto their local feasible set.

The NN predicts raw per-phase DER setpoints ``(pder, qder)``. Before power flow,
they are mapped to a set that respects the DER device limits (which are hard,
unlike the soft network voltage/current/angle limits):

1. ``p >= pdermin`` (per phase),
2. per-unit active-power cap ``sum_{phases in unit} p <= pder_avai`` enforced by
   one common scale factor for all phases of a unit (preserves phase ratios),
3. apparent-power disk ``p^2 + q^2 <= sdermax^2`` (per phase) by clipping ``q``
   at the projected ``p``.

Step 2 only shrinks ``p``, so the disk in step 3 always has a non-empty ``q``
interval provided ``pdermin <= sdermax``.
"""

import jax.numpy as jnp


def unit_phase_sum_batch(values, der_unit_to_phase, der_unit_to_phase_mask):
    """Sum per-phase values into per-unit totals.

    Args:
        values: Per-phase values ``(B, n_der_phase)``.
        der_unit_to_phase: Padded phase indices per unit ``(n_der_unit, max_phases)``.
        der_unit_to_phase_mask: Validity mask for the padded indices.

    Returns:
        Per-unit sums ``(B, n_der_unit)``.
    """
    gathered = values[:, der_unit_to_phase]
    mask = der_unit_to_phase_mask[None, :, :].astype(values.dtype)
    return jnp.sum(gathered * mask, axis=2)


def project_der_controls_batch(data, p, q, pder_avai, eps):
    """Project DER ``(p, q)`` onto box, per-unit availability, and apparent-power disk.

    Args:
        data: ``NetworkParams`` (or any object with ``pdermin``, ``sdermax``,
            ``der_unit_to_phase``, ``der_unit_to_phase_mask``, ``der_phase_to_unit``).
        p, q: Per-phase active / reactive setpoints ``(B, n_der_phase)``.
        pder_avai: Per-unit active-power caps ``(B, n_der_unit)`` or ``(n_der_unit,)``.
        eps: Small constant for safe divisions / square roots.

    Returns:
        ``(p_proj, q_proj)``, each ``(B, n_der_phase)``.
    """
    p_der = jnp.maximum(p, data.pdermin)
    caps = _broadcast_batch(pder_avai.astype(p.dtype), p.shape[0])
    unit_p = unit_phase_sum_batch(
        p_der,
        data.der_unit_to_phase,
        data.der_unit_to_phase_mask,
    )
    # Common per-unit scale s_u = min(1, cap_u / sum_u p); units with zero
    # (or negative) total active power are left unchanged.
    safe_unit_p = jnp.maximum(unit_p, eps)
    unit_scale = jnp.where(
        unit_p > 0.0,
        jnp.minimum(1.0, jnp.maximum(caps, 0.0) / safe_unit_p),
        1.0,
    )
    p_der = p_der * unit_scale[:, data.der_phase_to_unit]
    # Disk |s| <= sdermax at the projected p:  |q| <= sqrt(sdermax^2 - p^2).
    q_der_max = jnp.sqrt(jnp.maximum(data.sdermax**2 - p_der**2, eps))
    q_der = jnp.clip(q, -q_der_max, q_der_max)
    return p_der, q_der


def project_der_controls_from_x_batch(data, x, pder_avai, eps):
    """Project concatenated ``x = [pder; qder]`` controls ``(B, 2 * n_der_phase)``."""
    n = data.n_der_phase
    p, q = project_der_controls_batch(
        data, x[:, :n], x[:, n : 2 * n], pder_avai, eps
    )
    return jnp.concatenate([p, q], axis=1)


def unit_active_power_violation_batch(data, p, pder_avai):
    """Max per-unit active-power cap violation ``max_u (sum_u p - cap_u)_+`` ``(B,)``."""
    caps = _broadcast_batch(pder_avai.astype(p.dtype), p.shape[0])
    unit_p = unit_phase_sum_batch(
        p,
        data.der_unit_to_phase,
        data.der_unit_to_phase_mask,
    )
    return jnp.max(jnp.maximum(unit_p - caps, 0.0), axis=1)


def _broadcast_batch(values, batch_size: int):
    """Broadcast an unbatched 1-D array to ``(batch_size, n)``; pass 2-D through."""
    if values.ndim == 1:
        return jnp.broadcast_to(values[None, :], (batch_size, values.shape[0]))
    return values
