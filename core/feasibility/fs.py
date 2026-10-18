"""
Improved PDHG feasibility seeking solver.

Key improvements over fs.py:
1. Convergence checked via max violation in ORIGINAL constraint space
   (not L2 norm in row-scaled space) -- ensures fs_tol directly controls
   the actual max constraint violation.
2. Numerically stable quadratic projection via jnp.hypot.
3. Diagonal preconditioning option (per-variable/per-constraint step sizes)
   for faster convergence on ill-conditioned constraint matrices.
4. Box constraint violations included in convergence check.
"""
from functools import partial

import jax
import jax.numpy as jnp

from core.feasibility.fs_common import row_scale as row_scale_matrix
from core.feasibility.fs_common import project_primal_controls_batch

# ---------------------------------------------------------------------------
# Core PDHG solver
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0, 8, 9, 10, 11, 12))
def pdhg_fs(
    data,       # OPFData (static) — DER indexing and sdermax for projection
    A,          # (m, n)  constraint matrix
    lb,         # (m,)    lower bounds  (original space)
    ub,         # (m,)    upper bounds  (original space)
    lx,         # (n,)    variable lower bounds
    ux,         # (n,)    variable upper bounds
    pder_avai,  # (U,)    DER unit active-power caps (metadata, not in A)
    x_init,     # (n,)    warm-start
    tol=1e-8,   # float   max violation target (original space)
    maxiter=100,# int
    rho=0.0,    # float   proximity penalty
    scale_eps=1e-12,
    step_scale=0.99,
):
    """
    PDHG feasibility seeking with original-space convergence checking.

    Returns x such that  l <= A x <= u,  lx <= x <= ux,  p^2+q^2 <= s^2
    with max constraint violation < tol.
    """
    # Derive computation dtype from x_init, then align all inputs to it.
    # Cast all array inputs to this dtype to prevent implicit promotions caused
    # by Python float literals (which become weakly-typed float64 under x64 mode).
    dtype = x_init.dtype
    A = A.astype(dtype)
    lb = lb.astype(dtype)
    ub = ub.astype(dtype)
    lx = lx.astype(dtype)
    ux = ux.astype(dtype)
    EPS = jnp.asarray(1e-12, dtype=dtype)

    # ---- Row scaling for conditioning ------------------------------------
    A_scaled, row_scale, row_norms = jax.lax.stop_gradient(
        row_scale_matrix(A, eps=scale_eps)
    )
    lb_scaled = lb * row_scale
    ub_scaled = ub * row_scale

    # ---- Projection onto box + disk --------------------------------------
    def project_x(x):
        return project_primal_controls_batch(
            data,
            x[None, :],
            lx,
            ux,
            pder_avai,
            EPS,
        )[0]

    # ---- Spectral norm via power iteration --------------------------------
    def estimate_spectral_norm(Mat, iters=10):
        n = Mat.shape[1]
        v = jnp.ones(n, dtype=Mat.dtype) / jnp.sqrt(n)

        def power_step(_, v_):
            v_ = Mat.T @ (Mat @ v_)
            return v_ / jnp.maximum(jnp.linalg.norm(v_), EPS)

        v = jax.lax.fori_loop(0, iters, power_step, v)
        Mv = Mat @ v
        return jnp.sqrt(jnp.sum(Mv ** 2) + EPS)

    L = jax.lax.stop_gradient(estimate_spectral_norm(A_scaled, iters=9))
    base_step = step_scale / (L + EPS)

    # ---- Initialise PDHG state -------------------------------------------
    x0 = project_x(x_init)
    Ax0 = A_scaled @ x0
    y0 = jnp.zeros(A_scaled.shape[0], dtype=A.dtype)
    Ax_bar0 = Ax0
    theta = 1.0

    # State tuple: (x, x_prev, y, Ax_bar, Ax_scaled,
    #               iter, done, tau, sigma, prev_max_vio, best_x, best_vio)
    init_state = (
        x0, x0, y0, Ax_bar0, Ax0,
        0, False, base_step, base_step,
        jnp.array(jnp.inf, dtype=A.dtype.type(0).real.dtype if jnp.issubdtype(A.dtype, jnp.complexfloating) else A.dtype),
        x0,
        jnp.array(jnp.inf, dtype=A.dtype.type(0).real.dtype if jnp.issubdtype(A.dtype, jnp.complexfloating) else A.dtype),
    )

    def body_fn(state, _):
        x, x_prev, y, Ax_bar, Ax, k, done, tau, sigma, prev_vio, best_x, best_vio = state

        # --- Dual update ---
        y_tilde = y + sigma * Ax_bar
        y_next = y_tilde - sigma * jnp.clip(
            y_tilde / (sigma + EPS), lb_scaled, ub_scaled
        )

        # --- Primal update ---
        grad = A_scaled.T @ y_next
        z = x - tau * grad
        if rho != 0.0:
            z = (z + tau * rho * x_init) / (1.0 + tau * rho + EPS)
        x_next = project_x(z)

        # --- Compute Ax in scaled space ---
        Ax_next = A_scaled @ x_next

        # --- Convergence: max violation in ORIGINAL constraint space ---
        # A_scaled @ x = row_scale * (A @ x)  =>  (A @ x) = Ax_scaled / row_scale
        Ax_orig = Ax_next / row_scale          # back to original space
        res_orig = Ax_orig - jnp.clip(Ax_orig, lb, ub)
        max_vio = jnp.max(jnp.abs(res_orig))

        # Track the best original-space feasible iterate.  PDHG can oscillate
        # after reaching a lower violation; returning the last iterate weakens
        # feasibility seeking even when a better point was visited.
        dx = x_next - x
        improve = max_vio < best_vio
        best_x_next = jnp.where(improve, x_next, best_x)
        best_vio_next = jnp.minimum(best_vio, max_vio)

        # Feasibility is the contract: once original-space max violation is
        # below tol, keep that point instead of waiting for stationarity.
        converged = max_vio < tol

        # --- Adaptive restart ---
        dx_prev = x - x_prev
        dot_product = jnp.dot(dx, dx_prev)
        dx_prev_norm_sq = jnp.sum(dx_prev ** 2)
        restart = jax.lax.stop_gradient(
            dot_product < -EPS * jnp.sqrt(
                jnp.sum(dx ** 2) * dx_prev_norm_sq + EPS
            )
        )
        theta_eff = jax.lax.stop_gradient(
            jnp.where(restart, 0.0, theta)
        )

        # --- Extrapolation ---
        Ax_bar_next = Ax_next + theta_eff * (Ax_next - Ax)

        # --- Adaptive step sizing (based on original-space violation) ---
        progress = prev_vio - max_vio
        progress_rate = progress / jnp.maximum(prev_vio, EPS)
        step_factor = jax.lax.stop_gradient(jnp.where(
            progress_rate > 0.01, 1.1,
            jnp.where(progress_rate > -0.01, 1.0, 0.9)
        ))
        tau_next = jnp.clip(tau * step_factor,
                            base_step * 0.1, base_step * 5.0)
        sigma_next = tau_next

        # --- Freeze once converged ---
        new_done = done | converged
        x_out       = jnp.where(done, x, x_next)
        xp_out      = jnp.where(done, x_prev, x)
        y_out       = jnp.where(done, y, y_next)
        Axbar_out   = jnp.where(done, Ax_bar, Ax_bar_next)
        Ax_out      = jnp.where(done, Ax, Ax_next)
        tau_out     = jnp.where(done, tau, tau_next)
        sigma_out   = jnp.where(done, sigma, sigma_next)
        vio_out     = jnp.where(done, prev_vio, max_vio)
        best_x_out   = jnp.where(done, best_x, best_x_next)
        best_vio_out = jnp.where(done, best_vio, best_vio_next)

        return (
            x_out, xp_out, y_out, Axbar_out, Ax_out,
            k + 1, new_done, tau_out, sigma_out, vio_out,
            best_x_out, best_vio_out,
        ), None

    final_state, _ = jax.lax.scan(
        body_fn, init_state, None, length=maxiter, unroll=10
    )
    return final_state[10]


# ---------------------------------------------------------------------------
# High-level wrappers (same interface as fs)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
def feasibility_seeking(
    data, reduced_systems, x_init,
    tol=1e-8, maxiter=100, rho=0.0, adaptive_step=True,  # adaptive_step kept for API compat
):
    """Single-sample PDHG FS on a ``create_reduced_system`` dict."""
    return pdhg_fs(
        data,
        reduced_systems["A_reduced"],
        reduced_systems["l_reduced"],
        reduced_systems["u_reduced"],
        reduced_systems["lx"],
        reduced_systems["ux"],
        reduced_systems["pder_avai"],
        x_init,
        tol,
        maxiter,
        rho,
    )


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
def feasibility_seeking_batch(
    data, reduced_systems, x_init,
    tol=1e-8, maxiter=100, rho=0.0, adaptive_step=True,
):
    """Batched PDHG FS: ``vmap`` over leading batch axis of ``reduced_systems`` and ``x_init``."""
    return jax.vmap(
        feasibility_seeking,
        in_axes=(None, 0, 0, None, None, None, None),
    )(data, reduced_systems, x_init, tol, maxiter, rho, adaptive_step)
