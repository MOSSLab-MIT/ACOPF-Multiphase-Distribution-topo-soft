"""Linearized AC OPF model: power flow, constraint linearization, costs.

Builds ``A x + c ≈ f(x)`` around a frozen voltage point (used by the linear
OPF baseline in ``core/grids/linear_opf_solver.py``) and packages the shifted
bound form ``l − c ≤ A x ≤ u − c`` via ``create_reduced_system``.
"""
from functools import partial

import jax
import jax.numpy as jnp

from core.utils.types import TopologyData
from core.grids.network import NetworkParams
from core.grids.power_flow import PowerFlowSolver, get_power_flow_solve_fn
from core.grids.metrics import constraint_bounds


def _scalar_pf_or_default(solve_pf_fn, power_flow_solver: str):
    """Resolve scalar PF used inside linearization when ``solve_pf_fn`` omitted."""
    if solve_pf_fn is not None:
        return solve_pf_fn
    return get_power_flow_solve_fn(power_flow_solver)


def _full_local_voltage(data: NetworkParams, u_net):
    return jnp.concatenate([u_net, data.v_slack_local])

def _safe_abs(x, eps=1e-12):
    return jnp.maximum(jnp.abs(x), jnp.asarray(eps, dtype=jnp.real(x).dtype))


def _voltage_control_sensitivity(data: NetworkParams, u_hat, topo: TopologyData):
    """Frozen-denominator Z-bus sensitivity from DER controls to non-slack voltage."""
    # With u ~= w + Y_LL_inv @ (conj(s_net) / conj(u_hat)), each DER real
    # control contributes through conj(s_net), and each reactive control through
    # -1j * conj(s_net).  ``der_node`` maps phase-level DER controls to nodes.
    M = topo.Y_LL_inv * (1.0 / jnp.conj(u_hat))[None, :]
    U_p = M[:, data.der_node] 
    U_q = -1j * U_p
    return jnp.concatenate([U_p, U_q], axis=1)


def _full_voltage_sensitivity(data: NetworkParams, U_net):
    slack_rows = jnp.zeros((data.n_slk, U_net.shape[1]), dtype=U_net.dtype)
    return jnp.concatenate([U_net, slack_rows], axis=0)


def _voltage_magnitude_rows(u_hat, U_net):
    scale = jnp.conj(u_hat) / _safe_abs(u_hat)
    A = jnp.real(scale[:, None] * U_net)
    return A


def _angle_rows(data: NetworkParams, v_full, U_full):
    vf = v_full[data.branch_from_idx]
    vt = v_full[data.branch_to_idx]
    Uf = U_full[data.branch_from_idx]
    Ut = U_full[data.branch_to_idx]
    mask = data.branch_valid.astype(v_full.real.dtype)

    z = jnp.conj(vf) * vt
    dz = jnp.conj(Uf) * vt[..., None] + jnp.conj(vf)[..., None] * Ut

    upper = (jnp.imag(z) - data.dvamax * jnp.real(z)) * mask
    lower = (-jnp.imag(z) - data.dvamax * jnp.real(z)) * mask
    A_upper = (jnp.imag(dz) - data.dvamax * jnp.real(dz)) * mask[..., None]
    A_lower = (-jnp.imag(dz) - data.dvamax * jnp.real(dz)) * mask[..., None]

    f0 = jnp.concatenate([upper.reshape(-1), lower.reshape(-1)])
    A = jnp.concatenate(
        [
            A_upper.reshape(-1, U_full.shape[1]),
            A_lower.reshape(-1, U_full.shape[1]),
        ],
        axis=0,
    )
    return f0, A


def _substation_power_rows(data: NetworkParams, v_full, U_full, Y_0row):
    i_slack = Y_0row @ v_full
    s_k = data.v_slack_local * jnp.conj(i_slack)

    di_slack = Y_0row @ U_full
    ds_k = data.v_slack_local[:, None] * jnp.conj(di_slack)
    A = jnp.real((jnp.conj(s_k) / _safe_abs(s_k))[:, None] * ds_k)
    return jnp.abs(s_k), A


def _branch_current_rows(data: NetworkParams, v_full, U_full):
    v_system = data.v_scale_full * v_full
    U_system = data.v_scale_full[:, None] * U_full

    vf = v_system[data.branch_from_idx]
    vt = v_system[data.branch_to_idx]
    dv = vf - vt
    currents = jnp.sum(data.branch_y_series * dv[:, None, :], axis=2)

    Uf = U_system[data.branch_from_idx]
    Ut = U_system[data.branch_to_idx]
    d_dv = Uf - Ut
    d_currents = jnp.sum(data.branch_y_series[..., None] * d_dv[:, None, :, :], axis=2)

    mask = data.branch_valid.astype(v_full.real.dtype)
    scale = jnp.conj(currents) / _safe_abs(currents)
    A = jnp.real(scale[..., None] * d_currents) * mask[..., None]
    f0 = jnp.abs(currents) * mask
    return f0.reshape(-1), A.reshape(-1, U_full.shape[1])


def _linear_model_from_voltage_sensitivity(data: NetworkParams, x0, u_hat, U_net, topo: TopologyData):
    """Build constraint rows from a voltage sensitivity matrix ``du/dx``."""
    v_full = _full_local_voltage(data, u_hat)
    U_full = _full_voltage_sensitivity(data, U_net)

    f0_v = jnp.abs(u_hat)
    A_v = _voltage_magnitude_rows(u_hat, U_net)
    f0_angle, A_angle = _angle_rows(data, v_full, U_full)
    f0_sub, A_sub = _substation_power_rows(data, v_full, U_full, topo.Y_0row)
    f0_branch, A_branch = _branch_current_rows(data, v_full, U_full)

    f0 = jnp.concatenate([f0_v, f0_angle, f0_sub, f0_branch])
    A = jnp.concatenate([A_v, A_angle, A_sub, A_branch], axis=0)
    c = f0 - A @ x0
    return {"A": A, "c": c, "x0": x0, "f0": f0, "u_hat": u_hat}


def create_linear_model(
    data: NetworkParams,
    s_load,
    s_der,
    v_init,
    topo: TopologyData,
    u_hat=None,
    solve_pf_fn=None,
    power_flow_solver: str = PowerFlowSolver.FIXED_POINT.value,
):
    """Create the frozen-denominator Z-bus linear model.

    Solve AC power flow once at the current point, freeze ``1 / conj(u_hat)``
    in the Z-bus fixed-point equation, and analytically linearize every
    constraint row from the resulting voltage affine model.
    """
    x0 = jnp.concatenate([s_der.real, s_der.imag])
    s_hat = data.compute_net_injection(s_load, s_der)
    if u_hat is None:
        solve_pf_fn = _scalar_pf_or_default(solve_pf_fn, power_flow_solver)
        u_hat = solve_pf_fn(s_hat, v_init, topo.Y_LL_inv, topo.w)

    U_net = _voltage_control_sensitivity(data, u_hat, topo)
    return _linear_model_from_voltage_sensitivity(data, x0, u_hat, U_net, topo)


def create_reduced_system(data: NetworkParams, linear_model, pder_avai):
    """Shift linearized bounds: ``l − c ≤ A x ≤ u − c``.

    DER disk and per-unit availability constraints are not rows in ``A``; they
    are handled by the DER projection (``core/grids/der_projection.py``).

    Returns:
        dict with keys ``A_reduced``, ``l_reduced``, ``u_reduced``, ``lx``, ``ux``,
        ``pder_avai``.
    """

    lb, ub = constraint_bounds(data)
    return {
        "l_reduced": lb - linear_model["c"],
        "u_reduced": ub - linear_model["c"],
        "A_reduced": linear_model["A"],
        "lx": data.lx,
        "ux": data.ux,
        "pder_avai": pder_avai,
    }


def create_models(
    data,
    s_load,
    s_der,
    pder_avai,
    v_init,
    topo: TopologyData,
    *,
    u_hat=None,
    solve_pf_fn=None,
    power_flow_solver: str = PowerFlowSolver.FIXED_POINT.value,
):
    """Build linear and reduced constraint models for one operating point.

    If ``u_hat`` is None, solve AC power flow from ``v_init`` and linearize there.
    If ``u_hat`` is set, skip PF and linearize at that fixed voltage.
    """
    solve_pf_fn = _scalar_pf_or_default(solve_pf_fn, power_flow_solver)
    if u_hat is None:
        s_hat = data.compute_net_injection(s_load, s_der)
        v_hat = solve_pf_fn(s_hat, v_init, topo.Y_LL_inv, topo.w)
    else:
        v_hat = u_hat
    linear_model_data = create_linear_model(
        data,
        s_load,
        s_der,
        v_init,
        topo,
        u_hat=v_hat,
        solve_pf_fn=solve_pf_fn,
        power_flow_solver=power_flow_solver,
    )
    reduced_system = create_reduced_system(data, linear_model_data, pder_avai)
    return {
        "v_hat": v_hat,
        "linear_model": linear_model_data,
        "reduced_system": reduced_system,
    }


def _resolve_v_init_batch(v_init_batch, topo: TopologyData, batch_size: int):
    if v_init_batch is None:
        v_init_batch = topo.v_init
    if v_init_batch.ndim == 1:
        v_init_batch = jnp.broadcast_to(
            v_init_batch[None, :], (batch_size, v_init_batch.shape[0])
        )
    return v_init_batch


@partial(
    jax.jit,
    static_argnums=(0,),
    static_argnames=("solve_pf_fn", "power_flow_solver"),
)
def create_models_batch(
    data,
    s_load_batch,
    s_der_batch,
    pder_avai_batch,
    v_init_batch,
    topo: TopologyData,
    *,
    u_hat_batch=None,
    solve_pf_fn=None,
    power_flow_solver: str = PowerFlowSolver.FIXED_POINT.value,
):
    """Batched wrapper around ``create_models`` (``vmap`` over batch dimension).

    When ``u_hat_batch`` is provided, skips PF and linearizes at the fixed voltages.
    Otherwise solves PF from ``v_init_batch`` (or ``topo.v_init``) per sample.

    Returns:
        PyTree of batched dicts with ``v_hat``, ``linear_model``, ``reduced_system``.
    """
    solve_pf_fn = _scalar_pf_or_default(solve_pf_fn, power_flow_solver)
    batch_size = int(s_load_batch.shape[0])
    if u_hat_batch is not None:
        def _one(s_load, s_der, pder_avai, u_hat_i):
            return create_models(
                data,
                s_load,
                s_der,
                pder_avai,
                topo.v_init,
                topo,
                u_hat=u_hat_i,
                solve_pf_fn=solve_pf_fn,
                power_flow_solver=power_flow_solver,
            )

        return jax.vmap(_one, in_axes=(0, 0, 0, 0))(
            s_load_batch, s_der_batch, pder_avai_batch, u_hat_batch
        )

    v_init_batch = _resolve_v_init_batch(v_init_batch, topo, batch_size)
    _fn = partial(create_models, solve_pf_fn=solve_pf_fn, power_flow_solver=power_flow_solver)
    return jax.vmap(_fn, in_axes=(None, 0, 0, 0, 0, 0))(
        data, s_load_batch, s_der_batch, pder_avai_batch, v_init_batch, topo
    )
