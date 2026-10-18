"""Shared utilities for PDHG feasibility seeking.

Row scaling / spectral-norm estimates for constraint preprocessing, and primal
projection onto the DER feasible set (box limits, per-unit active-power caps,
apparent-power disks). Used by ``core/feasibility/fs.py`` and inference solvers
at each PDHG primal update.
"""

import jax  
import jax.numpy as jnp  


# ---------------------------------------------------------------------------
# Constraint row scaling and step-size preconditioning
# ---------------------------------------------------------------------------

def row_scale(A, eps=1e-12):
    """Row-normalize an unbatched constraint matrix."""
    row_norms = jnp.sqrt(jnp.sum(A**2, axis=1))
    row_norms = jnp.maximum(row_norms, eps)
    row_scale_values = 1.0 / row_norms
    A_scaled = A * row_scale_values[:, None]
    return A_scaled, row_scale_values, row_norms


def row_scale_batch(A, eps):
    """Row-normalize batched constraint matrices."""
    row_norms = jnp.sqrt(jnp.sum(A**2, axis=2))
    row_norms = jnp.maximum(row_norms, eps)
    row_scale_values = 1.0 / row_norms
    A_scaled = A * row_scale_values[:, :, None]
    return A_scaled, row_scale_values


def row_scale_voltage_weighted_batch(A, eps, n_voltage_rows, voltage_weight):
    """Row-scale batched matrices and up-weight the leading voltage rows."""
    A_scaled, row_scale_values = row_scale_batch(A, eps)
    w = jnp.asarray(voltage_weight, dtype=A.dtype)
    if n_voltage_rows > 0:
        row_scale_values = row_scale_values.at[:, :n_voltage_rows].multiply(w)
        A_scaled = A * row_scale_values[:, :, None]
    return A_scaled, row_scale_values


def estimate_spectral_norm_batch(A, eps, iters=6):
    """Estimate batched spectral norms via power iteration."""
    bsz, _, n = A.shape
    del bsz
    v = jnp.ones((A.shape[0], n), dtype=A.dtype) / jnp.sqrt(jnp.asarray(n, dtype=A.dtype))

    def _power_step(_, v_):
        Av = jnp.einsum("bmn,bn->bm", A, v_)
        AtAv = jnp.einsum("bmn,bm->bn", A, Av)
        norm = jnp.linalg.norm(AtAv, axis=1, keepdims=True)
        return AtAv / jnp.maximum(norm, eps)

    v = jax.lax.fori_loop(0, iters, _power_step, v)
    Av = jnp.einsum("bmn,bn->bm", A, v)
    return jnp.sqrt(jnp.sum(Av**2, axis=1) + eps)


# ---------------------------------------------------------------------------
# Primal projection onto the DER feasible set
# ---------------------------------------------------------------------------

def unit_phase_sum_batch(values, der_unit_to_phase, der_unit_to_phase_mask):
    gathered = values[:, der_unit_to_phase]
    mask = der_unit_to_phase_mask[None, :, :].astype(values.dtype)
    return jnp.sum(gathered * mask, axis=2)


def project_der_controls_batch(data, p, q, pder_avai, eps):
    """Project DER (p, q) to box, per-unit availability, and apparent-power disk.

    Same closed-form policy as ``methods.penalty_fs.trainer._project_predictions``:
    enforce ``p >= pdermin``, scale all phases in a DER unit by one factor so
    unit active power respects ``pder_avai``, then clip ``q`` to the disk
    ``|s| <= sdermax`` at the projected ``p``.
    """
    p_der = jnp.maximum(p, data.pdermin)
    caps = _broadcast_batch(pder_avai.astype(p.dtype), p.shape[0])
    unit_p = unit_phase_sum_batch(
        p_der,
        data.der_unit_to_phase,
        data.der_unit_to_phase_mask,
    )
    safe_unit_p = jnp.maximum(unit_p, eps)
    unit_scale = jnp.where(
        unit_p > 0.0,
        jnp.minimum(1.0, jnp.maximum(caps, 0.0) / safe_unit_p),
        1.0,
    )
    p_der = p_der * unit_scale[:, data.der_phase_to_unit]
    q_der_max = jnp.sqrt(jnp.maximum(data.sdermax**2 - p_der**2, eps))
    q_der = jnp.clip(q, -q_der_max, q_der_max)
    return p_der, q_der


def project_der_controls_from_x_batch(data, x, pder_avai, eps):
    """Project concatenated ``[pder; qder]`` controls (batch, 2 * n_der_phase)."""
    n = data.n_der_phase
    p, q = project_der_controls_batch(
        data, x[:, :n], x[:, n : 2 * n], pder_avai, eps
    )
    return jnp.concatenate([p, q], axis=1)


def unit_active_power_violation_batch(data, p, pder_avai):
    caps = _broadcast_batch(pder_avai.astype(p.dtype), p.shape[0])
    unit_p = unit_phase_sum_batch(
        p,
        data.der_unit_to_phase,
        data.der_unit_to_phase_mask,
    )
    return jnp.max(jnp.maximum(unit_p - caps, 0.0), axis=1)


def project_primal_controls_batch(data, x, lx, ux, pder_avai, eps):
    """Box clip then DER-specific projection on ``x = [pder; qder]``.

    Args:
        data: ``NetworkParams`` with DER limits and unit indexing.
        x: Primal controls ``(B, 2 * n_der_phase)``.
        lx, ux: Variable box bounds (broadcast to batch if 1-D).
        pder_avai: Per-unit active-power caps ``(B, n_der_unit)`` or ``(n_der_unit,)``.
        eps: Small constant for safe divisions / square roots.

    Returns:
        Projected controls, same shape as ``x``.
    """
    batch_size = x.shape[0]
    lx = _broadcast_batch(lx.astype(x.dtype), batch_size)
    ux = _broadcast_batch(ux.astype(x.dtype), batch_size)

    x_box = jnp.clip(x, lx, ux)
    num_ders = data.n_der_phase
    p, q = project_der_controls_batch(
        data,
        x_box[:, :num_ders],
        x_box[:, num_ders:],
        pder_avai,
        eps,
    )
    return jnp.concatenate([p, q], axis=1)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _broadcast_batch(values, batch_size: int):
    if values.ndim == 1:
        return jnp.broadcast_to(values[None, :], (batch_size, values.shape[0]))
    return values


# def _project_x_batch(x, lx, ux, s, pder_avai, data, eps):
#     """Box clip, then Lagrangian-style DER projection (batched)."""
#     del s  # sdermax lives on ``data``; kept for call-site symmetry with legacy FS.
#     return project_primal_controls_batch(data, x, lx, ux, pder_avai, eps)
