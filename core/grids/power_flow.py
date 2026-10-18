"""Multi-solver AC power flow in FeederFlow-style **scaled local** coordinates.

Problem setup
-------------
Partition ``Y_scaled = [[Y_LL, Y_L0], [Y_sn, Y_ss]]`` with fixed slack voltage
``v_slack_local``.  For non-slack complex voltages ``u`` and net injections ``s``,

    s = u * conj(Y_LL @ u + Y_L0 @ v_slack_local).

Equivalently, with ``w = -Y_LL_inv @ (Y_L0 @ v_slack_local)``,

    u = w + Y_LL_inv @ conj(s / u)                                    (Z-bus map)

Forward solvers (this module)
-----------------------------
* ``solve_power_flow_fixed_point`` — Z-bus iteration (cheap; matches FeederFlow-style
  fixed-point iteration for well-conditioned feeders).
* ``solve_power_flow_nr`` — Newton on the fixed-point residual with a **dense** Jacobian
  (``jacfwd`` + linear solve).  Stronger when the Z-bus map is stiff or slow.
* ``solve_power_flow_gmres`` — Same Newton step, but the linear system is solved with
  **GMRES** + JVP (matrix-free). Pair with explicit truncated AD or see below.

When fixed-point struggles (e.g. very large or ill-conditioned networks), prefer NR or
GMRES; see ``estimate_fixed_point_contractivity`` for a cheap heuristic on the Z-bus map.

Autodiff
--------
* **Registry key** ``newton_gmres`` (default for large stiff cases, e.g. 240-bus heuristic):
  uses ``solve_power_flow_gmres_implicit`` — **implicit adjoint** at the converged root
  (custom VJP; efficient vs backprop through Newton scans).
* **Truncated differentiation**: for fixed-point, dense NR, and **explicit**
  ``newton_gmres_explicit``, the last ``diffiter`` iterations are differentiated; earlier
  iterations use ``stop_gradient``.
* Custom factory: ``make_solve_power_flow_gmres_implicit`` — dense NR has no implicit VJP here.
"""

import enum
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres


class PowerFlowSolver(str, enum.Enum):
    """Which forward solve to use for Z-bus / Newton residuals."""

    FIXED_POINT = "fixed_point"
    NEWTON_DENSE = "newton_dense"
    # Implicit adjoint at converged u (custom VJP); forward = same GMRES Newton iteration.
    NEWTON_GMRES = "newton_gmres"
    # Explicit truncated differentiation through Newton scans (debug / parity).
    NEWTON_GMRES_EXPLICIT = "newton_gmres_explicit"


def _prepare_inputs(s, v_init, Y_LL_inv, w):
    s = s.astype(Y_LL_inv.dtype)
    v_init = v_init.astype(Y_LL_inv.dtype)
    w = w.astype(Y_LL_inv.dtype)
    return s, v_init, w


def _fixed_point_map(s, Y_LL_inv, w):
    """Return F(v) = w + Y_LL_inv @ conj(s / v)."""

    def fp(v):
        return w + Y_LL_inv @ jnp.conj(s / v)

    return fp


def _residual_real_from_fp(fp):
    """Real residual r(z) = z_complex - F(z_complex), stacked [Re; Im]."""

    def residual_real(z):
        n = z.shape[0] // 2
        v = z[:n] + 1j * z[n:]
        r = v - fp(v)
        return jnp.concatenate([r.real, r.imag])

    return residual_real


def _newton_scan(v_init, step_fn, maxiter, diffiter):
    """Newton outer iterations with diffiter differentiable trailing steps."""

    def scan_body(carry, _):
        v_curr, _ = carry
        v_next = step_fn(v_curr)
        return (v_next, False), None

    init_val = (v_init, False)
    bulk_iters = max(maxiter - diffiter, 0)
    (v_final, _), _ = jax.lax.stop_gradient(
        jax.lax.scan(scan_body, init_val, None, length=bulk_iters, unroll=10)
    )
    (v_final, _), _ = jax.lax.scan(scan_body, (v_final, False), None, length=diffiter)
    return v_final


@partial(jax.jit, static_argnums=(4, 5))
def solve_power_flow_nr(s, v_init, Y_LL_inv, w, maxiter=4, diffiter=3):
    """Newton–Raphson on the Z-bus fixed-point residual (dense Jacobian)."""

    s, v_init, w = _prepare_inputs(s, v_init, Y_LL_inv, w)
    fp = _fixed_point_map(s, Y_LL_inv, w)
    residual_real = _residual_real_from_fp(fp)

    def step_fn(v):
        z = jnp.concatenate([v.real, v.imag])
        r = residual_real(z)
        J = jax.jacfwd(residual_real)(z)
        dz = jnp.linalg.solve(J, r)
        z_next = z - dz
        n = v.shape[0]
        return z_next[:n] + 1j * z_next[n:]

    return _newton_scan(v_init, step_fn, maxiter, diffiter)


solve_power_flow_nr_batch = jax.jit(
    jax.vmap(solve_power_flow_nr, in_axes=(0, 0, 0, 0))
)


@partial(jax.jit, static_argnums=(4, 5))
def solve_power_flow_gmres(s, v_init, Y_LL_inv, w, maxiter=4, diffiter=3):
    """Newton on the Z-bus residual; each linear solve uses GMRES + JVP."""

    s, v_init, w = _prepare_inputs(s, v_init, Y_LL_inv, w)
    fp = _fixed_point_map(s, Y_LL_inv, w)
    residual_real = _residual_real_from_fp(fp)

    def step_fn(v):
        z = jnp.concatenate([v.real, v.imag])
        r = residual_real(z)

        def jvp_fn(z_, dz):
            return jax.jvp(residual_real, (z_,), (dz,))[1]

        dz, _ = gmres(lambda dz_: jvp_fn(z, dz_), -r, tol=1e-7)
        z_next = z + dz
        n = v.shape[0]
        return z_next[:n] + 1j * z_next[n:]

    return _newton_scan(v_init, step_fn, maxiter, diffiter)


solve_power_flow_gmres_batch = jax.jit(
    jax.vmap(solve_power_flow_gmres, in_axes=(0, 0, 0, 0))
)


def _residual_for_vjp(z, s_flat, Y_LL_inv, w):
    """Residual F(z, s) = z - fixed_point(z, s) in real-valued form."""
    n = z.shape[0] // 2
    v = z[:n] + 1j * z[n:]
    s_c = s_flat[:n] + 1j * s_flat[n:]
    r = v - (w + Y_LL_inv @ jnp.conj(s_c / v))
    return jnp.concatenate([r.real, r.imag])


def make_solve_power_flow_gmres_implicit(maxiter=4, diffiter=3):
    """Build a JITted solver with implicit adjoint (custom VJP) for converged `u` vs `s`."""

    def solve_fn(s, v_init, Y_LL_inv, w):
        return solve_power_flow_gmres(s, v_init, Y_LL_inv, w, maxiter, diffiter)

    def fwd(s, v_init, Y_LL_inv, w):
        v = solve_fn(s, v_init, Y_LL_inv, w)
        return v, (v, s, Y_LL_inv, w)

    def bwd(residuals, grad_output):
        v, s, Y_LL_inv, w = residuals
        n = v.shape[0]

        z = jnp.concatenate([v.real, v.imag])
        s_flat = jnp.concatenate([s.real, s.imag])
        grad_z = jnp.concatenate([grad_output.real, -grad_output.imag])

        def transpose_jvp_fn(lambda_vec):
            _, vjp_fn = jax.vjp(
                lambda z_in: _residual_for_vjp(z_in, s_flat, Y_LL_inv, w),
                z,
            )
            return vjp_fn(lambda_vec)[0]

        lambda_sol, _ = gmres(transpose_jvp_fn, grad_z, tol=1e-7)

        _, vjp_s_fn = jax.vjp(
            lambda s_in: _residual_for_vjp(z, s_in, Y_LL_inv, w),
            s_flat,
        )
        grad_s_flat = -vjp_s_fn(lambda_sol)[0]
        grad_s = grad_s_flat[:n] - 1j * grad_s_flat[n:]

        return (grad_s, None, None, None)

    solve_fn = jax.custom_vjp(solve_fn)
    solve_fn.defvjp(fwd, bwd)
    return jax.jit(solve_fn)


solve_power_flow_gmres_implicit = make_solve_power_flow_gmres_implicit()

solve_power_flow_gmres_implicit_batch = jax.jit(
    jax.vmap(solve_power_flow_gmres_implicit, in_axes=(0, 0, 0, 0))
)


@partial(jax.jit, static_argnums=(4, 5))
def solve_power_flow_fixed_point(s, v_init, Y_LL_inv, w, maxiter=7, diffiter=3):
    """Z-bus fixed-point iteration: v <- w + Y_LL_inv @ conj(s / v)."""

    s, v_init, w = _prepare_inputs(s, v_init, Y_LL_inv, w)
    fp = _fixed_point_map(s, Y_LL_inv, w)

    def body_fn(v, _):
        return fp(v), None

    v_final, _ = jax.lax.stop_gradient(
        jax.lax.scan(body_fn, v_init, None, length=maxiter - diffiter, unroll=10)
    )
    v_final, _ = jax.lax.scan(body_fn, v_final, None, length=diffiter)

    return v_final


solve_power_flow_fixed_point_batch = jax.jit(
    jax.vmap(solve_power_flow_fixed_point, in_axes=(0, 0, 0, 0))
)


@partial(jax.jit, static_argnums=(4,))
def solve_power_flow_fixed_point_full(s, v_init, Y_LL_inv, w, maxiter=7):
    """Z-bus fixed-point with gradients through all ``maxiter`` iterations (no stop_gradient split)."""

    s, v_init, w = _prepare_inputs(s, v_init, Y_LL_inv, w)
    fp = _fixed_point_map(s, Y_LL_inv, w)

    def body_fn(v, _):
        return fp(v), None

    v_final, _ = jax.lax.scan(body_fn, v_init, None, length=maxiter, unroll=10)
    return v_final


# --- Solver registry (requires solver definitions above) ---
_SOLVER_SCALAR_FUNCS = {
    PowerFlowSolver.FIXED_POINT.value: solve_power_flow_fixed_point,
    PowerFlowSolver.NEWTON_DENSE.value: solve_power_flow_nr,
    PowerFlowSolver.NEWTON_GMRES.value: solve_power_flow_gmres_implicit,
    PowerFlowSolver.NEWTON_GMRES_EXPLICIT.value: solve_power_flow_gmres,
}

_SOLVER_BATCH_FUNCS = {
    PowerFlowSolver.FIXED_POINT.value: solve_power_flow_fixed_point_batch,
    PowerFlowSolver.NEWTON_DENSE.value: solve_power_flow_nr_batch,
    PowerFlowSolver.NEWTON_GMRES.value: solve_power_flow_gmres_implicit_batch,
    PowerFlowSolver.NEWTON_GMRES_EXPLICIT.value: solve_power_flow_gmres_batch,
}


def get_power_flow_solve_fn(solver: str | PowerFlowSolver):
    """Scalar JITted solver for `(s, v_init, Y_LL_inv, w, *iter_kw)`.

    For ``newton_gmres``, this is the **implicit** GMRES wrapper (custom VJP from the
    converged residual). Use ``newton_gmres_explicit`` for truncated scan autodiff.
    """

    key = solver.value if isinstance(solver, PowerFlowSolver) else solver
    return _SOLVER_SCALAR_FUNCS[key]


def get_power_flow_batch_fn(solver: str | PowerFlowSolver):
    """Batched JITted solver ``vmap`` over leading batch axis.

    ``newton_gmres`` maps to ``solve_power_flow_gmres_implicit_batch``.
    """

    key = solver.value if isinstance(solver, PowerFlowSolver) else solver
    return _SOLVER_BATCH_FUNCS[key]


# def estimate_fixed_point_contractivity(Y_LL_inv, s, v_trial):
#     """Rough spectral-radius estimate of the Z-bus map Jacobian at ``v_trial`` (power iter)."""

#     diag_term = -jnp.conj(s / (v_trial**2))

#     u = jnp.ones_like(v_trial, dtype=Y_LL_inv.dtype)
#     for _ in range(3):
#         u = Y_LL_inv @ (diag_term * u)
#         u_norm = jnp.linalg.norm(u)
#         u = u / (u_norm + 1e-16)

#     Au = Y_LL_inv @ (diag_term * u)
#     rho = jnp.linalg.norm(Au) / (jnp.linalg.norm(u) + 1e-16)
#     return rho


def power_flow_residual(u, s, Y_LL, Y_L0, v_slack_local):
    """Return ``u * conj(Y_LL @ u + Y_L0 @ v_slack) - s``."""
    current = Y_LL @ u + Y_L0 @ v_slack_local
    return u * jnp.conj(current) - s


power_flow_residual_batch = jax.jit(
    jax.vmap(power_flow_residual, in_axes=(0, 0, 0, 0, None))
)
