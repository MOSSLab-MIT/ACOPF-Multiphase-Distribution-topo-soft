"""AC constraint evaluation, violations, and PF-consistent metrics."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from core.grids.data import NetworkParams
from core.grids.power_flow import get_power_flow_batch_fn
from core.topology.utils import y_0row_from_scaled
from core.utils.types import TopologyData


class ACMetrics(NamedTuple):
    """Per-sample AC objective and constraint metrics at a given voltage."""

    cost: jnp.ndarray
    violation: jnp.ndarray
    eq_violation: jnp.ndarray
    eq_violation_sq: jnp.ndarray
    ineq_violation: jnp.ndarray
    ineq_violation_sq: jnp.ndarray
    v_net: jnp.ndarray | None = None


# --- Power ---


def calc_substation_power_per_phase(data: NetworkParams, v_full_local, Y_0row):
    """Per-phase slack complex power from scaled local voltages.

    ``Y_0row`` has shape ``(n_slk, n_total)`` or batched ``(B, n_slk, n_total)``.
    Returns shape ``(n_slk,)`` or ``(B, n_slk)``.
    """
    if v_full_local.ndim == 1:
        i_slack = Y_0row @ v_full_local
        return data.v_slack_local * jnp.conj(i_slack)

    def one(v, Y):
        return calc_substation_power_per_phase(data, v, Y)

    return jax.vmap(one, in_axes=(0, 0))(v_full_local, Y_0row)


def calc_substation_power(data: NetworkParams, v_full_local, Y_0row):
    """Total slack complex power (sum over phases)."""
    return jnp.sum(calc_substation_power_per_phase(data, v_full_local, Y_0row), axis=-1)


def substation_power_from_voltage(
    net: NetworkParams,
    v_full: jnp.ndarray,
    Y_0row=None,
) -> jnp.ndarray:
    """Compute total complex substation power from local full or non-slack voltages."""
    v_full = with_slack_voltage(net, v_full)
    Y_0row = _resolve_y_0row(net, v_full, Y_0row)
    return calc_substation_power(net, v_full, Y_0row)


def _resolve_y_0row(net: NetworkParams, v_full, Y_0row=None, *, branch_status=None):
    """Batched slack admittance rows aligned with ``v_full``."""
    if Y_0row is None:
        if branch_status is None:
            Y_0row = net.Y_scaled_base[net.n_net : net.n_total, :]
        else:
            Y_full = net.topology_matrices_batch(branch_status)[0]
            Y_0row = y_0row_from_scaled(Y_full, net.n_net, net.n_total)
    if Y_0row.ndim == 2 and v_full.ndim > 1:
        Y_0row = jnp.broadcast_to(Y_0row, (v_full.shape[0],) + Y_0row.shape)
    return Y_0row


# --- Voltage ---


def with_slack_voltage(net: NetworkParams, v_net: jnp.ndarray) -> jnp.ndarray:
    """Return full local voltage ``[v_net, v_slack]`` if slack is not already present."""
    if v_net.shape[-1] == net.n_total:
        return v_net
    slack = jnp.broadcast_to(net.v_slack_local, v_net.shape[:-1] + (net.n_slk,))
    return jnp.concatenate([v_net, slack], axis=-1)


# --- Branch ---


def _branch_terminal_voltage(data: NetworkParams, v_full, idx):
    return v_full[idx]


def _angle_constraint_values(data: NetworkParams, v_full):
    vf = _branch_terminal_voltage(data, v_full, data.branch_from_idx)
    vt = _branch_terminal_voltage(data, v_full, data.branch_to_idx)
    mask = data.branch_valid.astype(v_full.real.dtype)
    dot = vf.real * vt.real + vf.imag * vt.imag
    cross = vf.real * vt.imag - vf.imag * vt.real
    upper = (cross - data.dvamax * dot) * mask
    lower = (-cross - data.dvamax * dot) * mask
    return jnp.concatenate([upper.reshape(-1), lower.reshape(-1)])


def _branch_current_magnitudes(data: NetworkParams, v_full):
    v_system = data.v_scale_full * v_full
    vf = v_system[data.branch_from_idx]
    vt = v_system[data.branch_to_idx]
    dv = vf - vt
    mask = data.branch_valid.astype(v_full.real.dtype)
    currents = jnp.sum(data.branch_y_series * dv[:, None, :], axis=2)
    return (jnp.abs(currents) * mask).reshape(-1)


# --- Obj ---


def _group_phase_sum(values, groups, mask):
    gathered = values[groups] * mask.astype(values.dtype)
    return jnp.sum(gathered, axis=1)


def calc_cost(data, psub, pder, pder_avai, obj_weights):
    """Two-weight Julia objective: substation cost + DER cost + curtailment."""
    psub = jnp.reshape(psub, (-1,))
    unit_p = jax.vmap(_group_phase_sum, in_axes=(0, None, None))(
        pder, data.der_unit_to_phase, data.der_unit_to_phase_mask
    )
    cost_sub = data.sub_lin * psub + data.sub_quad * psub**2
    cost_der = jnp.sum(data.der_lin * pder + data.der_quad * pder**2, axis=1)
    obj_cost = cost_sub + cost_der
    obj_curtailment = jnp.sum(pder_avai - unit_p, axis=1)
    return obj_weights[:, 0] * obj_cost + obj_weights[:, 1] * obj_curtailment


def _zbus_residual(net: NetworkParams, s_load_i, s_der_i, v_net_i, w_i, y_inv_i):
    """Z-bus fixed-point residual at non-slack voltages for one sample."""
    s_net = net.compute_net_injection(s_load_i, s_der_i)
    return w_i + y_inv_i @ jnp.conj(s_net / v_net_i) - v_net_i


def _eq_violation_from_v_net(net, s_load, s_der, v_net, w_batch, y_inv_batch):
    """Per-sample max and mean-square Z-bus residual (batched)."""

    def one_sample(s_der_i, s_load_i, v_i, w_i, y_inv_i):
        residual = _zbus_residual(net, s_load_i, s_der_i, v_i, w_i, y_inv_i)
        eq_violation = jnp.max(jnp.abs(residual))
        eq_violation_sq = jnp.mean(jnp.abs(residual) ** 2)
        return eq_violation, eq_violation_sq

    return jax.vmap(one_sample, in_axes=(0, 0, 0, 0, 0))(
        s_der, s_load, v_net, w_batch, y_inv_batch
    )


@partial(jax.jit, static_argnums=(0,))
def compute_ac_metrics_from_voltage(
    net: NetworkParams,
    s_load_batch,
    p_der,
    q_der,
    pder_avai,
    v_full,
    obj_weights,
    topo: TopologyData,
    *,
    branch_status=None,
):
    """AC cost, equality, and inequality metrics at a full or non-slack local voltage.

  ``v_full`` may be non-slack ``(B, n_net)`` or full ``(B, n_total)`` local voltage.
  ``violation`` is ``max(eq_violation, ineq_violation)`` per sample.
    """
    v_full = with_slack_voltage(net, v_full)
    v_net = v_full[..., : net.n_net]
    s_der = p_der + 1j * q_der
    eq_violation, eq_violation_sq = _eq_violation_from_v_net(
        net, s_load_batch, s_der, v_net, topo.w, topo.Y_LL_inv
    )
    psub = substation_power_from_voltage(net, v_full, Y_0row=topo.Y_0row).real
    cost = calc_cost(net, psub, p_der, pder_avai, obj_weights)
    ineq_violation, ineq_violation_sq = calc_ineq_violation(
        net,
        p_der,
        q_der,
        pder_avai,
        v_full,
        Y_0row=topo.Y_0row,
        branch_status=branch_status,
        return_square=True,
    )
    violation = jnp.maximum(eq_violation, ineq_violation)
    return ACMetrics(
        cost=cost,
        violation=violation,
        eq_violation=eq_violation,
        eq_violation_sq=eq_violation_sq,
        ineq_violation=ineq_violation,
        ineq_violation_sq=ineq_violation_sq,
    )


@partial(jax.jit, static_argnums=(0,))
def calc_ac_metrics_pf(
    net: NetworkParams,
    s_load_batch,
    pder,
    qder,
    pder_avai,
    obj_weights,
    topo: TopologyData,
    v_init=None,
):
    """Cost and violations after solving PF at ``(pder, qder)``.

    Matches the inference pipeline (PF, then AC metrics) instead of trusting
    stored OPF voltages from HDF5.

    ``v_init`` warm-starts the PF solve (default ``topo.w``). Target metrics pass
    stored OPF ``batch.v_target`` so the baseline matches dataset primals.
    """
    s_der = pder + 1j * qder
    s_net = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(s_load_batch, s_der)
    pf_b = get_power_flow_batch_fn(getattr(net, "power_flow_solver", "fixed_point"))
    if v_init is None:
        v_init = topo.w
    v_net = pf_b(s_net, v_init, topo.Y_LL_inv, topo.w)
    v_full = with_slack_voltage(net, v_net)
    metrics = compute_ac_metrics_from_voltage(
        net,
        s_load_batch,
        pder,
        qder,
        pder_avai,
        v_full,
        obj_weights,
        topo,
    )
    return metrics._replace(v_net=v_net)


# --- Violation ---


def constraint_values_from_voltage(data: NetworkParams, x, v_full_local, Y_0row):
    """Nonlinear network constraint values at a full local voltage."""
    del x
    vmag = jnp.abs(v_full_local[: data.n_net])
    angle = _angle_constraint_values(data, v_full_local)
    ssub = jnp.abs(calc_substation_power_per_phase(data, v_full_local, Y_0row))
    branch_i = _branch_current_magnitudes(data, v_full_local)
    return jnp.concatenate([vmag, angle, ssub, branch_i])


def constraint_row_layout(net: NetworkParams) -> dict:
    """Slice layout for network inequality rows in ``constraint_values_from_voltage``."""
    n_v = net.n_net
    n_angle = 2 * net.n_branch_constraints * net.max_branch_terms
    n_sub = net.n_slk
    n_branch_i = net.n_branch_constraints * net.max_branch_terms
    sl_v = slice(0, n_v)
    sl_angle = slice(n_v, n_v + n_angle)
    sl_sub = slice(n_v + n_angle, n_v + n_angle + n_sub)
    sl_branch = slice(n_v + n_angle + n_sub, n_v + n_angle + n_sub + n_branch_i)
    return {
        "n_v": n_v,
        "n_angle": n_angle,
        "n_sub": n_sub,
        "n_branch_i": n_branch_i,
        "sl_v": sl_v,
        "sl_angle": sl_angle,
        "sl_sub": sl_sub,
        "sl_branch": sl_branch,
        "substation_row": n_v + n_angle,
    }


def constraint_bounds(data: NetworkParams, dtype=None):
    """Bounds for network inequality constraints (unit cap is primal-only)."""
    if dtype is None:
        dtype = data.vmin.dtype
    layout = constraint_row_layout(data)
    n_angle = layout["n_angle"]
    n_sub = layout["n_sub"]
    n_branch_i = layout["n_branch_i"]

    neg_inf_angle = jnp.full((n_angle,), -jnp.inf, dtype=dtype)
    neg_inf_sub = jnp.full((n_sub,), -jnp.inf, dtype=dtype)
    neg_inf_branch = jnp.full((n_branch_i,), -jnp.inf, dtype=dtype)

    lb = jnp.concatenate(
        [
            data.vmin.astype(dtype),
            neg_inf_angle,
            neg_inf_sub,
            neg_inf_branch,
        ]
    )
    angle_u = jnp.zeros((n_angle,), dtype=dtype)
    branch_u = jnp.repeat(data.branch_imax.astype(dtype), data.max_branch_terms)
    ub = jnp.concatenate(
        [
            data.vmax.astype(dtype),
            angle_u,
            jnp.asarray(data.sub_smax_phase, dtype=dtype),
            branch_u,
        ]
    )
    return lb, ub


def _bounds_violation(values, lower, upper):
    return jnp.maximum(jnp.maximum(lower - values, 0.0), jnp.maximum(values - upper, 0.0))


def _resolve_ac_violation_inputs(data, pder, qder, v_full, Y_0row, branch_status):
    v_full = with_slack_voltage(data, v_full)
    Y_use = _resolve_y_0row(data, v_full, Y_0row, branch_status=branch_status)
    x = jnp.concatenate([pder, qder], axis=1)
    return x, v_full, Y_use


def ac_constraint_violation_vector(data, x_i, v_i, Y_i, av_i):
    """Per-constraint violation vector for one sample (network rows + DER primal)."""
    vals = constraint_values_from_voltage(data, x_i, v_i, Y_i)
    lb, ub = constraint_bounds(data)
    row_vio = _bounds_violation(vals, lb, ub)
    p = x_i[: data.n_der_phase]
    q = x_i[data.n_der_phase :]
    p_vio = _bounds_violation(p, data.pdermin, data.pdermax)
    q_vio = _bounds_violation(q, data.qdermin, data.qdermax)
    s_vio = jnp.maximum(p**2 + q**2 - data.sdermax**2, 0.0)
    unit_p = _group_phase_sum(p, data.der_unit_to_phase, data.der_unit_to_phase_mask)
    unit_vio = jnp.maximum(unit_p - av_i, 0.0)
    return jnp.concatenate([row_vio, p_vio, q_vio, s_vio, unit_vio])


@partial(jax.jit, static_argnums=(0,), static_argnames=("return_square",))
def calc_ineq_violation(
    data,
    pder,
    qder,
    pder_avai,
    v_full,
    Y_0row=None,
    branch_status=None,
    *,
    return_square=False,
):
    """Per-sample max inequality violation, optionally with mean-square penalty."""
    x, v_full, Y_use = _resolve_ac_violation_inputs(
        data, pder, qder, v_full, Y_0row, branch_status
    )

    def one(x_i, v_i, Y_i, av_i):
        vio = ac_constraint_violation_vector(data, x_i, v_i, Y_i, av_i)
        violation = jnp.max(vio)
        if return_square:
            return violation, jnp.mean(vio**2)
        return violation

    return jax.vmap(one, in_axes=(0, 0, 0, 0))(x, v_full, Y_use, pder_avai)


VIOLATION_COMPONENT_KEYS = (
    "voltage_magnitude_violation",
    "voltage_angle_violation",
    "substation_violation",
    "branch_current_violation",
    "unit_available_power_violation",
    "pder_box_violation",
    "qder_box_violation",
    "der_capacity_violation",
)


@partial(jax.jit, static_argnums=(0,))
def calc_violation_components_ac(
    data,
    pder,
    qder,
    pder_avai,
    v_full,
    Y_0row=None,
    branch_status=None,
):
    """Per-sample max violation split by constraint type, for diagnostics.

    Components partition the rows of ``ac_constraint_violation_vector`` (via
    ``constraint_row_layout``) plus the DER primal terms, so the max over all
    components equals ``calc_ineq_violation``'s per-sample value. Returns a dict
    with ``total_violation`` and one ``(B,)`` array per key in
    ``VIOLATION_COMPONENT_KEYS``.
    """
    x, v_full, Y_use = _resolve_ac_violation_inputs(
        data, pder, qder, v_full, Y_0row, branch_status
    )
    layout = constraint_row_layout(data)

    def _max0(values):
        return jnp.max(values, initial=jnp.asarray(0.0, dtype=values.dtype))

    def one(x_i, v_i, Y_i, av_i):
        vals = constraint_values_from_voltage(data, x_i, v_i, Y_i)
        lb, ub = constraint_bounds(data)
        row_vio = _bounds_violation(vals, lb, ub)
        p = x_i[: data.n_der_phase]
        q = x_i[data.n_der_phase :]
        p_vio = _bounds_violation(p, data.pdermin, data.pdermax)
        q_vio = _bounds_violation(q, data.qdermin, data.qdermax)
        s_vio = jnp.maximum(p**2 + q**2 - data.sdermax**2, 0.0)
        unit_p = _group_phase_sum(p, data.der_unit_to_phase, data.der_unit_to_phase_mask)
        unit_vio = jnp.maximum(unit_p - av_i, 0.0)

        components = (
            _max0(row_vio[layout["sl_v"]]),
            _max0(row_vio[layout["sl_angle"]]),
            _max0(row_vio[layout["sl_sub"]]),
            _max0(row_vio[layout["sl_branch"]]),
            _max0(unit_vio),
            _max0(p_vio),
            _max0(q_vio),
            _max0(s_vio),
        )
        total = jnp.max(jnp.asarray(components))
        return (total,) + components

    values = jax.vmap(one, in_axes=(0, 0, 0, 0))(x, v_full, Y_use, pder_avai)
    keys = ("total_violation",) + VIOLATION_COMPONENT_KEYS
    return {key: value for key, value in zip(keys, values)}


@partial(jax.jit, static_argnums=(0,))
def calc_eq_violation(net, s_der, s_load, vr_net, vi_net, w_batch, y_inv_batch):
    """Z-bus fixed-point residual at predicted non-slack voltages (batched)."""
    v_net = vr_net + 1j * vi_net
    return _eq_violation_from_v_net(net, s_load, s_der, v_net, w_batch, y_inv_batch)
