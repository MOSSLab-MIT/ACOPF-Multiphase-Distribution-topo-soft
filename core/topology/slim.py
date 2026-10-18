"""Slim Woodbury stamping helpers — target blocks of ``Y_scaled_base`` only.

Used by :mod:`core.topology.woodbury` to stamp ``Y_0row`` and ``Y_L0`` without
materializing the full ``(n_total, n_total)`` admittance matrix.
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from core.topology.woodbury import (
    WoodburyStatic,
    _switch_delta_f64,
    _switches_moved,
)


def _stamp_block(
    s: WoodburyStatic,
    full_status: jnp.ndarray,
    row_range: Tuple[int, int],
    col_range: Tuple[int, int],
) -> jnp.ndarray:
    """Apply per-switch stamp deltas onto ``Y_scaled_base[row_range, col_range]``.

    Same per-switch per-term math as ``_stamp_Y_scaled``; only the target block
    differs. Out-of-range updates are zeroed (masked) and local indices are
    clamped so ``.at[].add()`` scatter stays in-bounds for the small target.
    """
    r0, r1 = row_range
    c0, c1 = col_range
    n_rows = r1 - r0
    n_cols = c1 - c0
    Y = s.Y_scaled_base[r0:r1, c0:c1]

    if s.n_switch == 0:
        return Y

    delta = _switch_delta_f64(s, full_status)
    no_change = jnp.logical_not(_switches_moved(s, full_status))

    m = s.max_switch_terms

    def stamp_body(Y_in: jnp.ndarray) -> jnp.ndarray:
        r = jnp.arange(m * m) // m
        c = jnp.arange(m * m) % m

        valid = s.switch_valid[:, r] & s.switch_valid[:, c]
        coeff = delta[:, None] * valid.astype(jnp.float64)

        i_from = s.switch_from_idx[:, r]
        i_to = s.switch_to_idx[:, r]
        j_from = s.switch_from_idx[:, c]
        j_to = s.switch_to_idx[:, c]

        y_series = s.switch_y_series[:, r, c]
        y_shunt = s.switch_y_shunt[:, r, c]
        self_val = y_series + 0.5 * y_shunt
        mutual_val = -y_series

        scale_ff = s.v_scale_full[i_from] * s.v_scale_full[j_from]
        scale_tt = s.v_scale_full[i_to] * s.v_scale_full[j_to]
        scale_ft = s.v_scale_full[i_from] * s.v_scale_full[j_to]
        scale_tf = s.v_scale_full[i_to] * s.v_scale_full[j_from]

        upd_ff = coeff * self_val * scale_ff
        upd_tt = coeff * self_val * scale_tt
        upd_ft = coeff * mutual_val * scale_ft
        upd_tf = coeff * mutual_val * scale_tf

        def write_term(
            Y_cur: jnp.ndarray,
            i_idx: jnp.ndarray,
            j_idx: jnp.ndarray,
            upd: jnp.ndarray,
        ) -> jnp.ndarray:
            row_in = (i_idx >= r0) & (i_idx < r1)
            col_in = (j_idx >= c0) & (j_idx < c1)
            mask = (row_in & col_in).astype(jnp.float64)
            local_i = jnp.clip(i_idx - r0, 0, max(n_rows - 1, 0))
            local_j = jnp.clip(j_idx - c0, 0, max(n_cols - 1, 0))
            upd_masked = upd * mask
            return Y_cur.at[local_i.flatten(), local_j.flatten()].add(
                upd_masked.flatten()
            )

        Y_cur = write_term(Y_in, i_from, j_from, upd_ff)
        Y_cur = write_term(Y_cur, i_to, j_to, upd_tt)
        Y_cur = write_term(Y_cur, i_from, j_to, upd_ft)
        Y_cur = write_term(Y_cur, i_to, j_from, upd_tf)
        return Y_cur

    return jax.lax.cond(no_change, lambda: Y, lambda: stamp_body(Y))


def _stamp_Y_L0_slim(s: WoodburyStatic, full_status: jnp.ndarray) -> jnp.ndarray:
    """(n_net, n_slk) — network-slack cross block of stamped Y_scaled."""
    return _stamp_block(s, full_status, (0, s.n_net), (s.n_net, s.n_total))


def _stamp_Y_0row_slim(s: WoodburyStatic, full_status: jnp.ndarray) -> jnp.ndarray:
    """(n_slk, n_total) — slack rows of stamped Y_scaled."""
    return _stamp_block(s, full_status, (s.n_net, s.n_total), (0, s.n_total))


def _stamp_Y_LL_slim(s: WoodburyStatic, full_status: jnp.ndarray) -> jnp.ndarray:
    """(n_net, n_net) — network block of stamped Y_scaled (slow-path residual only)."""
    return _stamp_block(s, full_status, (0, s.n_net), (0, s.n_net))


# ---------------------------------------------------------------------------
# Branchless (cond-free) variants for vmap-compatible batch paths
# ---------------------------------------------------------------------------


def _stamp_block_branchless(
    s: WoodburyStatic,
    full_status: jnp.ndarray,
    row_range: Tuple[int, int],
    col_range: Tuple[int, int],
) -> jnp.ndarray:
    """``_stamp_block`` without the ``_switches_moved`` cond.

    When no switch moved, ``delta`` is all zero so every scatter-add contributes
    zero and the base block is returned unchanged. Always running the stamp body
    keeps the graph branchless (vmap-friendly) at the cost of harmless zero work.
    """
    r0, r1 = row_range
    c0, c1 = col_range
    n_rows = r1 - r0
    n_cols = c1 - c0
    Y = s.Y_scaled_base[r0:r1, c0:c1]

    if s.n_switch == 0:
        return Y

    delta = _switch_delta_f64(s, full_status)
    m = s.max_switch_terms

    r = jnp.arange(m * m) // m
    c = jnp.arange(m * m) % m
    valid = s.switch_valid[:, r] & s.switch_valid[:, c]
    coeff = delta[:, None] * valid.astype(jnp.float64)

    i_from = s.switch_from_idx[:, r]
    i_to = s.switch_to_idx[:, r]
    j_from = s.switch_from_idx[:, c]
    j_to = s.switch_to_idx[:, c]
    y_series = s.switch_y_series[:, r, c]
    y_shunt = s.switch_y_shunt[:, r, c]
    self_val = y_series + 0.5 * y_shunt
    mutual_val = -y_series

    scale_ff = s.v_scale_full[i_from] * s.v_scale_full[j_from]
    scale_tt = s.v_scale_full[i_to] * s.v_scale_full[j_to]
    scale_ft = s.v_scale_full[i_from] * s.v_scale_full[j_to]
    scale_tf = s.v_scale_full[i_to] * s.v_scale_full[j_from]

    upd_ff = coeff * self_val * scale_ff
    upd_tt = coeff * self_val * scale_tt
    upd_ft = coeff * mutual_val * scale_ft
    upd_tf = coeff * mutual_val * scale_tf

    def _write(Y_cur, i_idx, j_idx, upd):
        row_in = (i_idx >= r0) & (i_idx < r1)
        col_in = (j_idx >= c0) & (j_idx < c1)
        mask = (row_in & col_in).astype(jnp.float64)
        local_i = jnp.clip(i_idx - r0, 0, max(n_rows - 1, 0))
        local_j = jnp.clip(j_idx - c0, 0, max(n_cols - 1, 0))
        return Y_cur.at[local_i.flatten(), local_j.flatten()].add(
            (upd * mask).flatten()
        )

    Y = _write(Y, i_from, j_from, upd_ff)
    Y = _write(Y, i_to, j_to, upd_tt)
    Y = _write(Y, i_from, j_to, upd_ft)
    Y = _write(Y, i_to, j_from, upd_tf)
    return Y


def _stamp_Y_LL_slim_bl(s: WoodburyStatic, full_status: jnp.ndarray) -> jnp.ndarray:
    """Branchless (n_net, n_net) — network block of stamped Y_scaled."""
    return _stamp_block_branchless(s, full_status, (0, s.n_net), (0, s.n_net))


def _stamp_Y_L0_slim_bl(s: WoodburyStatic, full_status: jnp.ndarray) -> jnp.ndarray:
    """Branchless (n_net, n_slk) — network-slack cross block of stamped Y_scaled."""
    return _stamp_block_branchless(s, full_status, (0, s.n_net), (s.n_net, s.n_total))


def _stamp_Y_0row_slim_bl(s: WoodburyStatic, full_status: jnp.ndarray) -> jnp.ndarray:
    """Branchless (n_slk, n_total) — slack rows of stamped Y_scaled."""
    return _stamp_block_branchless(s, full_status, (s.n_net, s.n_total), (0, s.n_total))
