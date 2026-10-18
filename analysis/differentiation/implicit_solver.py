"""Fixed-point power flow with implicit GMRES adjoint — benchmark-only factory.

Forward uses ``solve_power_flow_fixed_point`` with ``diffiter=0`` (no AD through
the scan). Backward mirrors ``make_solve_power_flow_gmres_implicit`` in
``core.grids.power_flow`` but for the Z-bus fixed-point residual at convergence.
"""

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from core.grids.power_flow import solve_power_flow_fixed_point

jax.config.update("jax_enable_x64", True)

_GMRES_TOL = 1e-7


def _residual_for_vjp(z, s_flat, Y_LL_inv, w):
    """Residual F(z, s) = z - G(z, s) in real-valued form.

    Duplicated from ``core.grids.power_flow._residual_for_vjp`` so this module
    stays self-contained under ``analysis/differentiation/``.
    """
    n = z.shape[0] // 2
    v = z[:n] + 1j * z[n:]
    s_c = s_flat[:n] + 1j * s_flat[n:]
    r = v - (w + Y_LL_inv @ jnp.conj(s_c / v))
    return jnp.concatenate([r.real, r.imag])


def make_solve_power_flow_fixed_point_implicit(maxiter: int = 7):
    """Build a JITted solver with implicit adjoint (custom VJP) w.r.t. ``s``."""

    def solve_fn(s, v_init, Y_LL_inv, w):
        return solve_power_flow_fixed_point(
            s, v_init, Y_LL_inv, w, maxiter=maxiter, diffiter=0
        )

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

        lambda_sol, _ = gmres(transpose_jvp_fn, grad_z, tol=_GMRES_TOL)

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
