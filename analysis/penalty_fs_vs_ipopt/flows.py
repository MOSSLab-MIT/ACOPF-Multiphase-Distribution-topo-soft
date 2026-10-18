"""Substation and branch power-flow helpers for the penalty_fs vs Ipopt analysis.

Self-contained on purpose: the core library is untouched. The branch maths is
the same as ``core.grids.metrics._branch_current_magnitudes`` (series admittance
only, matching the ``|I| <= branch_imax`` constraint) but also returns the
complex power.
"""

from __future__ import annotations

import jax.numpy as jnp

from core.grids.data import NetworkParams
from core.grids.metrics import calc_substation_power_per_phase, with_slack_voltage


def branch_flows_from_voltage(
    net: NetworkParams, v_full: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """From-end complex power and current of every branch phase-term.

    For branch ``b`` with phase terms ``r, c`` (system-unit voltages
    ``v = v_scale_full * v_full``)::

        I[b, r] = sum_c y_series[b, r, c] * (v[from[b, c]] - v[to[b, c]])
        S[b, r] = v[from[b, r]] * conj(I[b, r])

    ``v_full`` is the full local voltage ``(..., n_total)``. Returns ``(S, I)``,
    each ``(..., n_branch_constraints, max_branch_terms)``; padded / invalid
    terms are exactly zero. Switches are separate elements and not included.
    """
    v_system = net.v_scale_full * v_full
    vf = v_system[..., net.branch_from_idx]
    vt = v_system[..., net.branch_to_idx]
    mask = net.branch_valid.astype(v_full.real.dtype)
    current = jnp.sum(net.branch_y_series * (vf - vt)[..., None, :], axis=-1) * mask
    return vf * jnp.conj(current), current


def flow_terms(net: NetworkParams, v_net: jnp.ndarray, y_0row: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Substation per-phase P/Q and branch |I|, P, Q for a batch of voltages.

    Branch arrays keep the padded ``(B, n_branch, max_terms)`` layout; select the
    valid terms on the host with ``np.asarray(net.branch_valid)``.
    """
    v_full = with_slack_voltage(net, v_net)
    s_sub = calc_substation_power_per_phase(net, v_full, y_0row)
    s_br, i_br = branch_flows_from_voltage(net, v_full)
    return {
        "psub_phase": s_sub.real,
        "qsub_phase": s_sub.imag,
        "branch_i": jnp.abs(i_br),
        "branch_p": s_br.real,
        "branch_q": s_br.imag,
    }
