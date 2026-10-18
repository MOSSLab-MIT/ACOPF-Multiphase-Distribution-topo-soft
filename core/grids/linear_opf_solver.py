"""Linearized AC OPF via CVXPY (Clarabel default).

Matches Julia ``datagen_topo/src/opf/model.jl`` primal constraints on DER
(``pder``/``qder`` box, apparent-power disk, unit availability) and
linearized network rows from ``create_linear_model``. The model is
re-linearized at each sample's load profile and DER availability (``qder = 0``)
so network and DER constraints are jointly feasible.

Network physics remain a first-order approximation of Julia's nonlinear
voltage / angle / substation / branch limits; nodal Y-bus balance is
encoded implicitly through the frozen-Z voltage sensitivity rather than
as bilinear ``vr, vi`` equalities (those require Ipopt in Julia).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from core.grids.data import NetworkParams
from core.grids.linear_model import create_linear_model
from core.grids.metrics import constraint_bounds
from core.grids.power_flow import (
    PowerFlowSolver,
    get_power_flow_solve_fn,
    solve_power_flow_nr,
)
from core.utils.types import TopologyData

DEFAULT_QP_SOLVER = cp.CLARABEL
QP_SUCCESS_STATUSES = frozenset({cp.OPTIMAL, cp.OPTIMAL_INACCURATE})
DISK_MODE_SOC = "soc"
DISK_MODE_POLY_INNER = "poly_inner"
# Max acceptable Z-bus fixed-point residual for the linearization voltage.
# Above this, the configured solver is treated as diverged (e.g. non-contractive
# fixed point on networks with near-singular Y_LL rows) and Newton is used instead.
PF_RESIDUAL_TOL = 1e-6
COST_COEFF_TOL = 1e-12


class LinearizationPowerFlowError(RuntimeError):
    """Power flow at the linearization point did not converge.

    The linearized model would be invalid; callers should skip the sample
    rather than solve a QP on garbage sensitivities.
    """


@dataclass
class LinearOPFProblemCache:
    """Compiled CVXPY problem reusable across topology samples of one grid."""

    branch_status_key: tuple[int, ...]
    lb: np.ndarray
    ub: np.ndarray
    unit_matrix: np.ndarray
    sdermax: np.ndarray
    der_lin: np.ndarray
    der_quad: np.ndarray
    sub_lin: float
    sub_quad: float
    lx: np.ndarray
    ux: np.ndarray
    problem: cp.Problem
    pder_var: cp.Variable
    qder_var: cp.Variable
    a_param: cp.Parameter
    lb_rhs_param: cp.Parameter
    ub_rhs_param: cp.Parameter
    psub_offset_param: cp.Parameter
    a_psub_param: cp.Parameter
    pder_avai_param: cp.Parameter
    x_linear_coeff_param: cp.Parameter
    der_quad_weight_sqrt_param: cp.Parameter
    sub_quad_weight_sqrt_param: cp.Parameter
    # Updated each sample solve (not part of CVXPY graph constants).
    u_hat: np.ndarray = field(default_factory=lambda: np.array([]))
    U_net: np.ndarray = field(default_factory=lambda: np.array([]))
    x0: np.ndarray = field(default_factory=lambda: np.array([]))
    _solve_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)


def _numpy_complex(z) -> np.ndarray:
    return np.asarray(z, dtype=np.complex128)


def _numpy_real(z) -> np.ndarray:
    return np.asarray(z, dtype=np.float64)


def _voltage_control_sensitivity(net: NetworkParams, u_hat, topo: TopologyData) -> np.ndarray:
    """Frozen-denominator Z-bus sensitivity ``du/dx`` (non-slack nodes)."""
    u_hat_j = jnp.asarray(u_hat)
    M = topo.Y_LL_inv * (1.0 / jnp.conj(u_hat_j))[None, :]
    U_p = M[:, net.der_node]
    U_q = -1j * U_p
    return np.asarray(jnp.concatenate([U_p, U_q], axis=1), dtype=np.complex128)


def _substation_real_power_affine(
    net: NetworkParams,
    topo: TopologyData,
    u_hat: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Affine model ``psub(x) = psub0 + a_psub @ (x - x0)`` at ``u_hat``."""
    y_0row = _numpy_complex(topo.Y_0row)
    v_sl = _numpy_complex(net.v_slack_local)
    v_net = _numpy_complex(u_hat)
    v_full = np.concatenate([v_net, v_sl])
    u_net_sens = _voltage_control_sensitivity(net, u_hat, topo)
    s_slack = v_sl * np.conj(y_0row @ v_full)
    psub0 = float(np.real(np.sum(s_slack)))
    d_i_slack = y_0row[:, : net.n_net] @ u_net_sens
    # Preserve the complex derivative until projecting slack complex power to real P.
    d_s_slack = v_sl[:, None] * np.conj(d_i_slack)
    a_psub = np.real(np.sum(d_s_slack, axis=0)).astype(np.float64)
    return psub0, a_psub


def _unit_active_power_matrix(net: NetworkParams) -> np.ndarray:
    """Matrix ``M`` with ``(M @ pder)_g = sum_{k in unit g} pder_k``."""
    n_unit = net.n_der_unit
    n_phase = net.n_der_phase
    mat = np.zeros((n_unit, n_phase), dtype=np.float64)
    groups = np.asarray(net.der_unit_to_phase)
    mask = np.asarray(net.der_unit_to_phase_mask)
    for g in range(n_unit):
        for j in range(groups.shape[1]):
            if mask[g, j]:
                mat[g, groups[g, j]] = 1.0
    return mat


def _der_linearization_point_from_availability(
    net: NetworkParams,
    pder_avai: np.ndarray,
) -> jnp.ndarray:
    """DER injection for voltage linearization: unit P at availability, Q = 0.

    Phase active powers split equally within each unit so
    ``sum_{k in unit g} p_k = pder_avai[g]`` at the linearization point.
    """
    pder_avai = _numpy_real(pder_avai).reshape(-1)
    if pder_avai.shape[0] != net.n_der_unit:
        raise ValueError(
            f"pder_avai must have length {net.n_der_unit}, got {pder_avai.shape[0]}"
        )
    unit_matrix = _unit_active_power_matrix(net)
    phase_counts = np.maximum(unit_matrix.sum(axis=1), 1.0)
    unit_for_phase = np.asarray(net.der_phase_to_unit, dtype=np.int32)
    pder_phase = pder_avai[unit_for_phase] / phase_counts[unit_for_phase]
    return jnp.asarray(pder_phase, dtype=jnp.float64) + 0j


def _network_bound_rhs(
    lb: np.ndarray,
    ub: np.ndarray,
    c_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """RHS for the nominal linear OPF bounds ``lb <= A x + c <= ub``."""
    return lb - c_offset, ub - c_offset


def _checked_nonnegative_scalar(value: float, name: str) -> float:
    """Return a scalar that is nonnegative up to numerical noise."""
    value = float(value)
    if value < -COST_COEFF_TOL:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return max(value, 0.0)


def _sqrt_checked_nonnegative(value, name: str) -> np.ndarray:
    """Return ``sqrt(value)`` after validating convex quadratic weights."""
    values = _numpy_real(value)
    if np.any(values < -COST_COEFF_TOL):
        min_value = float(np.min(values))
        raise ValueError(f"{name} must be nonnegative, min={min_value}")
    return np.sqrt(np.maximum(values, 0.0))


def _sample_linearization(
    net: NetworkParams,
    topo: TopologyData,
    s_load: np.ndarray,
    *,
    power_flow_solver: str,
    pder_avai: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Linear model at sample load and a DER operating point.

    When ``pder_avai`` is provided, linearize at unit availability with ``qder = 0``.
    Otherwise linearize at zero DER (cache template sizing only).
    """
    s_load_j = jnp.asarray(s_load, dtype=jnp.complex128)
    if pder_avai is None:
        s_der = jnp.zeros(net.n_der_phase, dtype=jnp.complex128)
    else:
        s_der = _der_linearization_point_from_availability(net, pder_avai)
    solve_pf = get_power_flow_solve_fn(power_flow_solver)
    s_net = net.compute_net_injection(s_load_j, s_der)
    v_init = jnp.asarray(topo.v_init)
    Y_LL_inv = jnp.asarray(topo.Y_LL_inv)
    w = jnp.asarray(topo.w)
    u_hat = solve_pf(s_net, v_init, Y_LL_inv, w)
    # Guard: the Z-bus fixed point diverges on networks whose Y_LL_inv has
    # near-singular (huge) rows; a garbage u_hat silently poisons the whole
    # linear model. Verify the fixed-point residual and fall back to Newton.
    residual = jnp.max(jnp.abs(u_hat - (w + Y_LL_inv @ jnp.conj(s_net / u_hat))))
    if float(residual) > PF_RESIDUAL_TOL:
        u_hat = solve_power_flow_nr(s_net, v_init, Y_LL_inv, w, maxiter=20, diffiter=1)
        residual = jnp.max(jnp.abs(u_hat - (w + Y_LL_inv @ jnp.conj(s_net / u_hat))))
        if float(residual) > PF_RESIDUAL_TOL:
            raise LinearizationPowerFlowError(
                "Power flow at the linearization point did not converge "
                f"(residual {float(residual):.3e}); linear model would be invalid."
            )
    lm = create_linear_model(
        net,
        s_load_j,
        s_der,
        jnp.asarray(topo.v_init),
        topo,
        u_hat=u_hat,
        power_flow_solver=power_flow_solver,
    )
    u_hat_np = _numpy_complex(u_hat)
    psub0, a_psub = _substation_real_power_affine(net, topo, u_hat_np)
    return {
        "u_hat": u_hat_np,
        "U_net": _voltage_control_sensitivity(net, u_hat_np, topo),
        "A": _numpy_real(lm["A"]),
        "c": _numpy_real(lm["c"]),
        "x0": _numpy_real(lm["x0"]),
        "psub0": np.asarray(psub0, dtype=np.float64),
        "a_psub": np.asarray(a_psub, dtype=np.float64),
    }


def _build_cvxpy_problem(
    cache_fields: dict[str, Any],
    *,
    solver: str,
    solver_kwargs: dict[str, Any] | None,
) -> tuple[
    cp.Problem,
    cp.Variable,
    cp.Variable,
    cp.Parameter,
    cp.Parameter,
    cp.Parameter,
    cp.Parameter,
    cp.Parameter,
    cp.Parameter,
    cp.Parameter,
    cp.Parameter,
    cp.Parameter,
    dict[str, Any],
]:
    n_der = int(cache_fields["n_der"])
    n_con = int(cache_fields["A"].shape[0])
    n_x = 2 * n_der
    n_unit = int(cache_fields["unit_matrix"].shape[0])

    pder = cp.Variable(n_der, name="pder")
    qder = cp.Variable(n_der, name="qder")
    x = cp.hstack([pder, qder])

    a_param = cp.Parameter((n_con, n_x), name="A")
    lb_rhs_param = cp.Parameter(n_con, name="lb_rhs")
    ub_rhs_param = cp.Parameter(n_con, name="ub_rhs")
    psub_offset_param = cp.Parameter(name="psub_offset")
    a_psub_param = cp.Parameter(n_x, name="a_psub")
    pder_avai_param = cp.Parameter(n_unit, nonneg=True, name="pder_avai")
    x_linear_coeff_param = cp.Parameter(n_x, name="x_linear_coeff")
    der_quad_weight_sqrt_param = cp.Parameter(
        n_der, nonneg=True, name="der_quad_weight_sqrt"
    )
    sub_quad_weight_sqrt_param = cp.Parameter(
        nonneg=True, name="sub_quad_weight_sqrt"
    )

    lx = cache_fields["lx"]
    ux = cache_fields["ux"]
    unit_matrix = cache_fields["unit_matrix"]
    sdermax = cache_fields["sdermax"]
    disk_mode = cache_fields["disk_mode"]
    disk_polygon_sides = int(cache_fields["disk_polygon_sides"])

    ax = a_param @ x
    constraints = [
        ax <= ub_rhs_param,
        ax >= lb_rhs_param,
        # Julia: pder box, qder box, apparent-power disk, unit availability.
        pder >= lx[:n_der],
        pder <= ux[:n_der],
        qder >= lx[n_der:],
        qder <= ux[n_der:],
        unit_matrix @ pder <= pder_avai_param,
    ]
    if disk_mode == DISK_MODE_SOC:
        for k in range(n_der):
            constraints.append(
                cp.norm(cp.hstack([pder[k], qder[k]]), 2) <= float(sdermax[k])
            )
    elif disk_mode == DISK_MODE_POLY_INNER:
        radius_scale = float(np.cos(np.pi / disk_polygon_sides))
        for theta in np.linspace(0.0, 2.0 * np.pi, disk_polygon_sides, endpoint=False):
            c_theta = float(np.cos(theta))
            s_theta = float(np.sin(theta))
            constraints.append(c_theta * pder + s_theta * qder <= radius_scale * sdermax)
    else:
        raise ValueError(f"Unknown disk constraint mode: {disk_mode!r}")

    # Previous non-DPP formulation kept for reference:
    # psub = psub0_param + a_psub_param @ (x - x0_param)
    # cost_der = cp.sum(
    #     cp.multiply(der_lin, pder) + cp.multiply(der_quad, cp.square(pder))
    # )
    # cost_sub = sub_lin * psub + sub_quad * cp.square(psub)
    # unit_p = unit_matrix @ pder
    # curtail = cp.sum(pder_avai_param - unit_p)
    # objective = cp.Minimize(
    #     w_cost_param * (cost_der + cost_sub) + w_curtail_param * curtail
    # )
    psub = cp.Variable(name="psub")
    psub_quad_scaled = cp.Variable(name="psub_quad_scaled")
    constraints.append(psub == psub_offset_param + a_psub_param @ x)
    constraints.append(psub_quad_scaled == sub_quad_weight_sqrt_param * psub)
    cost_der_quad = cp.sum_squares(cp.multiply(der_quad_weight_sqrt_param, pder))
    cost_sub_quad = cp.square(psub_quad_scaled)
    cost_linear = x_linear_coeff_param @ x
    objective = cp.Minimize(
        cost_der_quad + cost_sub_quad + cost_linear
    )
    problem = cp.Problem(objective, constraints)
    # Problem is DPP, but we solve with ignore_dpp=True: DPP canonicalization materializes
    # a parameter tensor of size ~(cone_rows × (n_vars+1)) × (n_params+1). With dense
    # a_param (n_con × n_x) at 8500-bus scale (~7e4 × 240 ≈ 1.7e7 entries), cvxpy
    # format_constraints can allocate ~4.8e9-row intermediates (~36 GiB) and OOM.
    solve_kwargs = {
        "solver": solver,
        "ignore_dpp": True,
        **(solver_kwargs or {}),
    }
    return (
        problem,
        pder,
        qder,
        a_param,
        lb_rhs_param,
        ub_rhs_param,
        psub_offset_param,
        a_psub_param,
        pder_avai_param,
        x_linear_coeff_param,
        der_quad_weight_sqrt_param,
        sub_quad_weight_sqrt_param,
        solve_kwargs,
    )


def _sync_linearization_params(
    cache: LinearOPFProblemCache,
    lin: dict[str, np.ndarray],
) -> None:
    """Push a sample linearization into CVXPY parameters."""
    cache.u_hat = lin["u_hat"]
    cache.U_net = lin["U_net"]
    cache.x0 = lin["x0"]
    cache.a_param.value = lin["A"]
    lb_rhs, ub_rhs = _network_bound_rhs(
        cache.lb, cache.ub, lin["c"]
    )
    cache.lb_rhs_param.value = lb_rhs
    cache.ub_rhs_param.value = ub_rhs
    cache.a_psub_param.value = lin["a_psub"]
    cache.psub_offset_param.value = float(lin["psub0"] - lin["a_psub"] @ lin["x0"])


def _sync_objective_params(
    cache: LinearOPFProblemCache,
    pder_avai: np.ndarray,
    obj_weights: np.ndarray,
) -> None:
    """Push DPP-safe objective coefficients for one sample."""
    w_cost = _checked_nonnegative_scalar(obj_weights[0], "obj_weights[0]")
    w_curtail = _checked_nonnegative_scalar(obj_weights[1], "obj_weights[1]")
    cache.pder_avai_param.value = pder_avai
    n_der = cache.der_lin.shape[0]
    unit_phase_counts = np.sum(cache.unit_matrix, axis=0)
    # Constants from psub_offset and sum(pder_avai) are restored in reported objectives.
    x_linear_coeff = w_cost * cache.sub_lin * np.asarray(cache.a_psub_param.value)
    x_linear_coeff[:n_der] += w_cost * cache.der_lin - w_curtail * unit_phase_counts
    cache.x_linear_coeff_param.value = x_linear_coeff
    cache.der_quad_weight_sqrt_param.value = _sqrt_checked_nonnegative(
        w_cost * cache.der_quad, "weighted DER quadratic costs"
    )
    cache.sub_quad_weight_sqrt_param.value = _sqrt_checked_nonnegative(
        w_cost * cache.sub_quad, "weighted substation quadratic cost"
    )


def release_linear_opf_cache_state(
    cache: LinearOPFProblemCache,
    *,
    clear_linearization: bool = True,
    clear_parameter_values: bool = True,
    clear_solver_cache: bool = True,
) -> None:
    """Release large per-sample CVXPY/linearization state after a solve.

    The reusable problem graph is kept intact; only values and solver workspaces
    that are overwritten before the next solve are dropped. This prevents long
    8500-bus sweeps from retaining one dense ``A`` value / solver state per
    iteration through CVXPY or Clarabel internals.
    """
    if clear_linearization:
        cache.u_hat = np.array([])
        cache.U_net = np.array([])
        cache.x0 = np.array([])

    if clear_parameter_values:
        for param in (
            cache.a_param,
            cache.lb_rhs_param,
            cache.ub_rhs_param,
            cache.psub_offset_param,
            cache.a_psub_param,
            cache.pder_avai_param,
            cache.x_linear_coeff_param,
            cache.der_quad_weight_sqrt_param,
            cache.sub_quad_weight_sqrt_param,
        ):
            param.value = None

    if clear_solver_cache:
        solver_cache = getattr(cache.problem, "_solver_cache", None)
        if isinstance(solver_cache, dict):
            solver_cache.clear()


def _original_objective_value(
    cache: LinearOPFProblemCache,
    pder: np.ndarray,
    qder: np.ndarray,
    pder_avai: np.ndarray,
    obj_weights: np.ndarray,
) -> float:
    """Evaluate the original objective, including dropped constants."""
    x = np.concatenate([pder, qder])
    psub = float(cache.psub_offset_param.value + cache.a_psub_param.value @ x)
    cost_der = float(np.sum(cache.der_lin * pder + cache.der_quad * pder**2))
    cost_sub = float(cache.sub_lin * psub + cache.sub_quad * psub**2)
    unit_p = cache.unit_matrix @ pder
    curtail = float(np.sum(pder_avai - unit_p))
    return float(obj_weights[0] * (cost_der + cost_sub) + obj_weights[1] * curtail)


def build_linear_opf_cache(
    net: NetworkParams,
    topo: TopologyData,
    config: dict,
    *,
    branch_status_key: np.ndarray | None = None,
    power_flow_solver: str = PowerFlowSolver.FIXED_POINT.value,
    solver: str = DEFAULT_QP_SOLVER,
    solver_kwargs: dict[str, Any] | None = None,
    disk_mode: str = DISK_MODE_SOC,
    disk_polygon_sides: int = 32,
) -> LinearOPFProblemCache:
    """Assemble a topology-keyed CVXPY shell; linearization is synced per sample."""
    del config  # sample loads enter via ``solve_linear_opf`` parameters.
    if branch_status_key is None:
        key = tuple(int(v) for v in np.asarray(topo.branch_status).ravel().tolist())
    else:
        key = tuple(int(v) for v in np.asarray(branch_status_key).ravel().tolist())

    lb, ub = constraint_bounds(net)
    lb = _numpy_real(lb)
    ub = _numpy_real(ub)
    unit_matrix = _unit_active_power_matrix(net)
    if disk_mode == DISK_MODE_POLY_INNER and disk_polygon_sides < 4:
        raise ValueError("disk_polygon_sides must be at least 4 for poly_inner mode")

    # Template linearization (zero load) only sizes the CVXPY graph.
    template = _sample_linearization(
        net,
        topo,
        np.zeros(net.n_net, dtype=np.complex128),
        power_flow_solver=power_flow_solver,
    )
    fields = {
        "A": template["A"],
        "lb": lb,
        "ub": ub,
        "lx": _numpy_real(net.lx),
        "ux": _numpy_real(net.ux),
        "unit_matrix": unit_matrix,
        "sdermax": _numpy_real(net.sdermax),
        "der_lin": _numpy_real(net.der_lin),
        "der_quad": _numpy_real(net.der_quad),
        "sub_lin": float(np.asarray(net.sub_lin)),
        "sub_quad": float(np.asarray(net.sub_quad)),
        "n_der": net.n_der_phase,
        "disk_mode": disk_mode,
        "disk_polygon_sides": disk_polygon_sides,
    }
    (
        problem,
        pder_var,
        qder_var,
        a_param,
        lb_rhs_param,
        ub_rhs_param,
        psub_offset_param,
        a_psub_param,
        pder_avai_param,
        x_linear_coeff_param,
        der_quad_weight_sqrt_param,
        sub_quad_weight_sqrt_param,
        solve_kwargs,
    ) = _build_cvxpy_problem(fields, solver=solver, solver_kwargs=solver_kwargs)

    cache = LinearOPFProblemCache(
        branch_status_key=key,
        lb=lb,
        ub=ub,
        unit_matrix=unit_matrix,
        sdermax=fields["sdermax"],
        der_lin=fields["der_lin"],
        der_quad=fields["der_quad"],
        sub_lin=fields["sub_lin"],
        sub_quad=fields["sub_quad"],
        lx=fields["lx"],
        ux=fields["ux"],
        problem=problem,
        pder_var=pder_var,
        qder_var=qder_var,
        a_param=a_param,
        lb_rhs_param=lb_rhs_param,
        ub_rhs_param=ub_rhs_param,
        psub_offset_param=psub_offset_param,
        a_psub_param=a_psub_param,
        pder_avai_param=pder_avai_param,
        x_linear_coeff_param=x_linear_coeff_param,
        der_quad_weight_sqrt_param=der_quad_weight_sqrt_param,
        sub_quad_weight_sqrt_param=sub_quad_weight_sqrt_param,
        _solve_kwargs=solve_kwargs,
    )
    _sync_linearization_params(cache, template)
    return cache


def affine_voltage(
    cache: LinearOPFProblemCache,
    x: np.ndarray,
) -> np.ndarray:
    """Non-slack voltage ``u_hat + U_net (x - x0)`` at the current linearization."""
    x = _numpy_real(x).reshape(-1)
    return cache.u_hat + cache.U_net @ (x - cache.x0)


def solve_linear_opf(
    cache: LinearOPFProblemCache,
    s_load: np.ndarray,
    pder_avai: np.ndarray,
    obj_weights: np.ndarray,
    *,
    net: NetworkParams,
    topo: TopologyData,
    power_flow_solver: str = PowerFlowSolver.FIXED_POINT.value,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve linearized OPF for one sample; returns ``(pder, qder, info)``.

    When the QP is not optimal, emits a warning, sets ``info["solved"]`` to False,
    and returns the linearization-point DER setpoint (``x0``) so callers can continue.

    When power flow at the linearization point fails to converge, emits a warning,
    sets ``info["skipped"]=True``, and returns zero DER so the batch loop can continue.
    """
    s_load = _numpy_complex(s_load).reshape(-1)
    pder_avai = _numpy_real(pder_avai).reshape(-1)
    obj_weights = _numpy_real(obj_weights).reshape(-1)
    if obj_weights.shape[0] != 2:
        raise ValueError(f"obj_weights must have length 2, got {obj_weights.shape[0]}")

    n_der = net.n_der_phase
    lin_start = time.perf_counter()
    try:
        lin = _sample_linearization(
            net,
            topo,
            s_load,
            power_flow_solver=power_flow_solver,
            pder_avai=pder_avai,
        )
    except LinearizationPowerFlowError as exc:
        linearization_time = time.perf_counter() - lin_start
        warnings.warn(f"Skipping linear OPF sample: {exc}", stacklevel=2)
        return (
            np.zeros(n_der, dtype=np.float64),
            np.zeros(n_der, dtype=np.float64),
            {
                "status": "linearization_pf_failed",
                "solved": False,
                "skipped": True,
                "objective": float("nan"),
                "u_linear": np.asarray(topo.v_init, dtype=np.complex128).reshape(-1),
                "solve_time": 0.0,
                "linearization_time": linearization_time,
                "qp_wall_time": 0.0,
                "qp_overhead_time": 0.0,
                "skip_reason": str(exc),
            },
        )
    linearization_time = time.perf_counter() - lin_start

    qp_start = time.perf_counter()
    _sync_linearization_params(cache, lin)
    _sync_objective_params(cache, pder_avai, obj_weights)

    cache.problem.solve(**cache._solve_kwargs)
    qp_wall_time = time.perf_counter() - qp_start
    status = cache.problem.status
    solved = status in QP_SUCCESS_STATUSES

    qp_solver_time = getattr(cache.problem.solver_stats, "solve_time", None)
    if qp_solver_time is None:
        qp_solver_time = qp_wall_time

    if solved:
        pder = np.asarray(cache.pder_var.value, dtype=np.float64).reshape(-1)
        qder = np.asarray(cache.qder_var.value, dtype=np.float64).reshape(-1)
        objective = _original_objective_value(cache, pder, qder, pder_avai, obj_weights)
    else:
        # Fall back to the linearization point so downstream PF/metrics can continue.
        x0 = np.asarray(cache.x0, dtype=np.float64).reshape(-1)
        pder = x0[:n_der].copy()
        qder = x0[n_der:].copy()
        objective = float("nan")
        warnings.warn(
            f"Linear OPF QP not solved (status={status}); using linearization-point fallback.",
            stacklevel=2,
        )

    x = np.concatenate([pder, qder])
    info = {
        "status": status,
        "solved": solved,
        "skipped": False,
        "objective": objective,
        "u_linear": affine_voltage(cache, x),
        # Clarabel-reported optimizer time (comparable to Ipopt ``meta/solve_time``).
        "solve_time": float(qp_solver_time),
        "linearization_time": linearization_time,
        "qp_wall_time": qp_wall_time,
        "qp_overhead_time": max(qp_wall_time - float(qp_solver_time), 0.0),
    }
    return pder, qder, info
