"""Closed-form projection of DER controls onto their local feasible set.

The NN predicts raw per-phase DER setpoints ``(pder, qder)``. Before power flow,
they are mapped to a set that respects the DER device limits (which are hard,
unlike the soft network voltage/current/angle limits):

1. per-phase active-power bounds, also leaving enough apparent-power capacity
   for the reactive-power box,
2. per-unit active-power cap ``sum_{phases in unit} p <= pder_avai`` enforced by
   one common scale factor for the power above each phase's lower bound
   (preserves clipped phase ratios when ``pdermin == 0``),
3. the intersection of the reactive-power box and apparent-power disk
   ``p^2 + q^2 <= sdermax^2`` (per phase) by clipping ``q`` at the projected ``p``.

Requires nonnegative active-power lower bounds and a nonempty feasible set:
each phase's box must intersect its disk, and each device's availability must
cover the sum of its active-power lower bounds. The data loader uses zero
active-power lower bounds. This sequential map is not a Euclidean projection.
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
        data: ``NetworkParams`` (or any object with ``pdermin``, ``pdermax``,
            ``qdermin``, ``qdermax``, ``sdermax``,
            ``der_unit_to_phase``, ``der_unit_to_phase_mask``, ``der_phase_to_unit``).
        p, q: Per-phase active / reactive setpoints ``(B, n_der_phase)``.
        pder_avai: Per-unit active-power caps ``(B, n_der_unit)`` or ``(n_der_unit,)``.
        eps: Positive constant used only in inactive division / square-root
            branches; it does not enlarge the disk or shrink feasible controls.

    Returns:
        ``(p_proj, q_proj)``, each ``(B, n_der_phase)``.
    """
    # The point in the q box closest to zero needs the least apparent power.
    # Reserving this headroom also supports q boxes that exclude zero.
    q_nearest_zero = jnp.clip(jnp.zeros_like(data.qdermin), data.qdermin, data.qdermax)
    p_disk_max = _nonnegative_sqrt(data.sdermax**2 - q_nearest_zero**2, eps)
    p_der = jnp.clip(p, data.pdermin, jnp.minimum(data.pdermax, p_disk_max))
    caps = _broadcast_batch(pder_avai.astype(p.dtype), p.shape[0])
    p_excess = p_der - data.pdermin
    unit_excess = unit_phase_sum_batch(
        p_excess,
        data.der_unit_to_phase,
        data.der_unit_to_phase_mask,
    )
    unit_min = unit_phase_sum_batch(
        data.pdermin[None, :], data.der_unit_to_phase, data.der_unit_to_phase_mask
    )
    # Scale only the excess above the lower bounds, so scaling cannot violate
    # them. With zero lower bounds this is the original common phase scaling.
    safe_unit_excess = jnp.where(unit_excess > 0.0, unit_excess, eps)
    unit_scale = jnp.where(
        unit_excess > 0.0,
        jnp.minimum(1.0, jnp.maximum(caps - unit_min, 0.0) / safe_unit_excess),
        1.0,
    )
    p_der = data.pdermin + p_excess * unit_scale[:, data.der_phase_to_unit]
    q_disk_max = _nonnegative_sqrt(data.sdermax**2 - p_der**2, eps)
    q_lower = jnp.maximum(data.qdermin, -q_disk_max)
    q_upper = jnp.minimum(data.qdermax, q_disk_max)
    q_der = jnp.clip(q, q_lower, q_upper)
    return p_der, q_der


def _nonnegative_sqrt(value, eps):
    """Exact sqrt(max(value, 0)) with a finite, zero derivative at value <= 0.

    Mask the input to sqrt as well as its output: differentiating an inactive
    sqrt(0) branch otherwise introduces NaNs at a saturated apparent-power disk.
    """
    positive = value > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, value, eps)), 0.0)


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
