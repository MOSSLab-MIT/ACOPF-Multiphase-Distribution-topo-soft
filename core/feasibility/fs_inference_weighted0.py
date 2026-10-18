"""
Inference-focused PDHG feasibility solver with voltage row emphasis.

The first ``n_voltage_rows`` constraints (voltage magnitude rows by convention) use

    row_scale_i = voltage_weight / ||A_i||

instead of ``1 / ||A_i||``. This increases dual pressure on voltage constraints
without duplicating rows. The spectral norm of ``A_scaled`` grows with
``voltage_weight``, so ``base_step = step_fraction / L`` shrinks accordingly.

Defaults: ``voltage_weight=8``, ``step_fraction=2.5``, ``n_voltage_rows=0``
(interpreted as ``data.n_net``). With ``w=8``, spectral norm grows and the nominal
``0.99/L`` step is conservative; ``step_fraction=1.5`` was best on IEEE123 stressed
split (see ``experiments/feasibility_restoration/weighted_row_scale_stepsize/``).
Set ``voltage_weight=1`` to recover unweighted row normalization.
"""
from functools import partial

import jax
import jax.numpy as jnp

from core.feasibility.fs_common import (
    estimate_spectral_norm_batch,
    project_primal_controls_batch,
    row_scale_voltage_weighted_batch,
)


@partial(
    jax.jit,
    static_argnums=(0, 3, 4, 5, 6, 7, 8, 9),
)
def feasibility_seeking_batch_inference(
    data,
    reduced_systems,
    x_init,
    tol=1e-8,
    maxiter=100,
    rho=0.0,
    n_voltage_rows: int = 0,
    voltage_weight: float = 8.0,
    step_fraction: float = 2.5,
    voltage_margin: float = 0.0,
):
    # del tol  # tol unused: full maxiter scan (no per-sample freeze)

    dtype = x_init.dtype
    eps = jnp.asarray(1e-12, dtype=dtype)
    n_v = n_voltage_rows if n_voltage_rows > 0 else data.n_net

    A = reduced_systems["A_reduced"].astype(dtype)
    lb = reduced_systems["l_reduced"].astype(dtype)
    ub = reduced_systems["u_reduced"].astype(dtype)
    if voltage_margin > 0.0:
        margin = jnp.asarray(voltage_margin, dtype=dtype)
        lb = lb.at[:, :n_v].add(margin)
        ub = ub.at[:, :n_v].add(-margin)
    lx = reduced_systems["lx"].astype(dtype)
    ux = reduced_systems["ux"].astype(dtype)
    pder_avai = reduced_systems["pder_avai"].astype(dtype)

    if lx.ndim == 1:
        lx = jnp.broadcast_to(lx[None, :], x_init.shape)
    if ux.ndim == 1:
        ux = jnp.broadcast_to(ux[None, :], x_init.shape)

    A_scaled, row_scale = jax.lax.stop_gradient(
        row_scale_voltage_weighted_batch(A, eps, n_v, voltage_weight)
    )
    lb_scaled = lb * row_scale
    ub_scaled = ub * row_scale

    L = jax.lax.stop_gradient(estimate_spectral_norm_batch(A_scaled, eps, iters=6))
    base_step = jnp.asarray(step_fraction, dtype=dtype) / (L + eps)

    x0 = project_primal_controls_batch(data, x_init, lx, ux, pder_avai, eps)
    Ax0 = jnp.einsum("bmn,bn->bm", A_scaled, x0)
    y0 = jnp.zeros_like(Ax0)
    prev_vio0 = jnp.full((x0.shape[0],), jnp.inf, dtype=dtype)
    best_vio0 = jnp.full((x0.shape[0],), jnp.inf, dtype=dtype)

    # (x, x_prev, y, Ax_bar, Ax, tau, sigma, prev_vio, best_x, best_vio)
    state = (
        x0,
        x0,
        y0,
        Ax0,
        Ax0,
        base_step,
        base_step,
        prev_vio0,
        x0,
        best_vio0,
    )

    def _scan_body(st, _):
        x, x_prev, y, Ax_bar, Ax, tau, sigma, prev_vio, best_x, best_vio = st

        sigma_col = sigma[:, None]
        y_tilde = y + sigma_col * Ax_bar
        y_next = y_tilde - sigma_col * jnp.clip(
            y_tilde / (sigma_col + eps), lb_scaled, ub_scaled
        )

        grad = jnp.einsum("bmn,bm->bn", A_scaled, y_next)
        tau_col = tau[:, None]
        z = x - tau_col * grad
        if rho != 0.0:
            trho = tau_col * jnp.asarray(rho, dtype=dtype)
            z = (z + trho * x_init) / (1.0 + trho + eps)

        x_next = project_primal_controls_batch(data, z, lx, ux, pder_avai, eps)
        Ax_next = jnp.einsum("bmn,bn->bm", A_scaled, x_next)

        Ax_orig = Ax_next / row_scale
        res_orig = Ax_orig - jnp.clip(Ax_orig, lb, ub)
        max_vio = jnp.max(jnp.abs(res_orig), axis=1)

        dx = x_next - x
        improve = max_vio < best_vio
        best_x_next = jnp.where(improve[:, None], x_next, best_x)
        best_vio_next = jnp.minimum(best_vio, max_vio)

        dx_prev = x - x_prev
        dot_product = jnp.sum(dx * dx_prev, axis=1)
        dx_norm_sq = jnp.sum(dx**2, axis=1)
        dx_prev_norm_sq = jnp.sum(dx_prev**2, axis=1)
        restart = dot_product < -eps * jnp.sqrt(dx_norm_sq * dx_prev_norm_sq + eps)
        theta_eff = jnp.where(restart, 0.0, 1.0).astype(dtype)
        Ax_bar_next = Ax_next + theta_eff[:, None] * (Ax_next - Ax)

        progress = prev_vio - max_vio
        progress_rate = progress / jnp.maximum(prev_vio, eps)
        step_factor = jnp.where(
            progress_rate > 0.01,
            1.1,
            jnp.where(progress_rate > -0.01, 1.0, 0.9),
        ).astype(dtype)
        tau_next = jnp.clip(tau * step_factor, base_step * 0.1, base_step * 5.0)
        sigma_next = tau_next

        # Per-sample early-stop freeze (disabled): scan always runs maxiter steps.
        # Previously, once max_vio < tol we kept (x, y, tau, ...) fixed via jnp.where(done, ...).

        return (
            x_next,
            x,
            y_next,
            Ax_bar_next,
            Ax_next,
            tau_next,
            sigma_next,
            max_vio,
            best_x_next,
            best_vio_next,
        ), max_vio

    final_state, _ = jax.lax.scan(_scan_body, state, None, length=maxiter, unroll=10)
    return final_state[8]
