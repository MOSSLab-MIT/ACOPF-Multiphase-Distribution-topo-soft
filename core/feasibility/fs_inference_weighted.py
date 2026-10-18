"""
Chunked early-stop PDHG feasibility solver with voltage row emphasis.

This module keeps the same PDHG update used by ``fs_inference_weighted`` but
changes inference control flow:

1. A small inner ``lax.scan`` chunk preserves efficient unrolled iterations.
2. An outer ``lax.while_loop`` stops once every sample in the batch has reached
   the original-space tolerance or ``maxiter`` has been exhausted.

Early stopping is batch-level, while freezing is per-sample. Once a sample
reaches ``tol``, its state is held fixed inside subsequent chunks so the
returned point is faithful to the first converged iterate.
"""
from functools import partial

import jax
import jax.numpy as jnp

from core.feasibility.fs_common import (
    estimate_spectral_norm_batch,
    project_primal_controls_batch,
    row_scale_voltage_weighted_batch,
)

_EPS = 1e-12
_SPECTRAL_NORM_ITERS = 6

_FAST_PROGRESS_RATE = 0.01
_SLOW_PROGRESS_RATE = -0.01
_STEP_GROWTH = 1.1
_STEP_HOLD = 1.0
_STEP_SHRINK = 0.9
_MIN_STEP_FACTOR = 0.1
_MAX_STEP_FACTOR = 5.0

_STATE_DONE_INDEX = 8
_STATE_BEST_X_INDEX = 9
_STATE_BEST_VIO_INDEX = 10
_STATE_ITER_INDEX = 11


def feasibility_seeking_batch_inference(
    data,
    reduced_systems,
    x_init,
    tol=1e-6,
    maxiter=100,
    rho=0.0,
    n_voltage_rows: int = 0,
    voltage_weight: float = 8.0,
    step_fraction: float = 2.5,
    voltage_margin: float = 0.0,
    chunk_size: int = 20,
):
    """Return the best feasible iterate from chunked early-stop inference PDHG."""
    _validate_chunk_size(chunk_size)
    best_x, _, _, _ = feasibility_seeking_batch_inference_with_stats(
        data,
        reduced_systems,
        x_init,
        tol,
        maxiter,
        rho,
        n_voltage_rows,
        voltage_weight,
        step_fraction,
        voltage_margin,
        chunk_size,
    )
    return best_x


def feasibility_seeking_batch_inference_with_stats(
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
    chunk_size: int = 20,
):
    """Return ``(best_x, iterations, done, best_vio)`` for benchmarking."""
    _validate_chunk_size(chunk_size)
    return _feasibility_seeking_batch_inference_impl(
        data,
        reduced_systems,
        x_init,
        tol,
        maxiter,
        rho,
        n_voltage_rows,
        voltage_weight,
        step_fraction,
        voltage_margin,
        chunk_size,
    )


@partial(
    jax.jit,
    static_argnums=(0, 3, 4, 5, 6, 7, 8, 9, 10),
)
def _feasibility_seeking_batch_inference_impl(
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
    chunk_size: int = 20,
):
    dtype = x_init.dtype
    eps = jnp.asarray(_EPS, dtype=dtype)
    tol_value = jnp.asarray(tol, dtype=dtype)
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

    L = jax.lax.stop_gradient(
        estimate_spectral_norm_batch(A_scaled, eps, iters=_SPECTRAL_NORM_ITERS)
    )
    base_step = jnp.asarray(step_fraction, dtype=dtype) / (L + eps)

    x0 = project_primal_controls_batch(data, x_init, lx, ux, pder_avai, eps)
    Ax0 = jnp.einsum("bmn,bn->bm", A_scaled, x0)
    y0 = jnp.zeros_like(Ax0)
    prev_vio0 = jnp.full((x0.shape[0],), jnp.inf, dtype=dtype)
    best_vio0 = jnp.full((x0.shape[0],), jnp.inf, dtype=dtype)
    done0 = jnp.zeros((x0.shape[0],), dtype=bool)
    iter0 = jnp.asarray(0, dtype=jnp.int32)

    # (x, x_prev, y, Ax_bar, Ax, tau, sigma, prev_vio,
    #  done, best_x, best_vio, iter_count)
    state = (
        x0,
        x0,
        y0,
        Ax0,
        Ax0,
        base_step,
        base_step,
        prev_vio0,
        done0,
        x0,
        best_vio0,
        iter0,
    )

    def _pdhg_step(st, _):
        x, x_prev, y, Ax_bar, Ax, tau, sigma, prev_vio, done, best_x, best_vio, k = st
        active_iter = k < maxiter
        sample_active = (~done) & active_iter

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

        improve = max_vio < best_vio
        best_x_next = jnp.where(improve[:, None], x_next, best_x)
        best_vio_next = jnp.minimum(best_vio, max_vio)

        dx = x_next - x
        dx_prev = x - x_prev
        dot_product = jnp.sum(dx * dx_prev, axis=1)
        dx_norm_sq = jnp.sum(dx**2, axis=1)
        dx_prev_norm_sq = jnp.sum(dx_prev**2, axis=1)
        restart = dot_product < -eps * jnp.sqrt(dx_norm_sq * dx_prev_norm_sq + eps)
        theta_eff = jnp.where(restart, 0.0, 1.0).astype(dtype)
        Ax_bar_next = Ax_next + theta_eff[:, None] * (Ax_next - Ax)

        progress = prev_vio - max_vio
        progress_rate = progress / jnp.maximum(prev_vio, eps)
        step_factor = jnp.full_like(progress_rate, _STEP_SHRINK)
        step_factor = jnp.where(
            progress_rate > _SLOW_PROGRESS_RATE,
            _STEP_HOLD,
            step_factor,
        )
        step_factor = jnp.where(
            progress_rate > _FAST_PROGRESS_RATE,
            _STEP_GROWTH,
            step_factor,
        ).astype(dtype)
        tau_next = jnp.clip(
            tau * step_factor,
            base_step * _MIN_STEP_FACTOR,
            base_step * _MAX_STEP_FACTOR,
        )
        sigma_next = tau_next

        done_next = done | (active_iter & (max_vio < tol_value))

        return (
            jnp.where(sample_active[:, None], x_next, x),
            jnp.where(sample_active[:, None], x, x_prev),
            jnp.where(sample_active[:, None], y_next, y),
            jnp.where(sample_active[:, None], Ax_bar_next, Ax_bar),
            jnp.where(sample_active[:, None], Ax_next, Ax),
            jnp.where(sample_active, tau_next, tau),
            jnp.where(sample_active, sigma_next, sigma),
            jnp.where(sample_active, max_vio, prev_vio),
            done_next,
            jnp.where(sample_active[:, None], best_x_next, best_x),
            jnp.where(sample_active, best_vio_next, best_vio),
            k + jnp.asarray(1, dtype=jnp.int32),
        ), None

    def _needs_more(st):
        done = st[_STATE_DONE_INDEX]
        k = st[_STATE_ITER_INDEX]
        return (k < maxiter) & (~jnp.all(done))

    def _run_chunk(st):
        st, _ = jax.lax.scan(
            _pdhg_step,
            st,
            None,
            length=chunk_size,
            unroll=chunk_size,
        )
        return st

    final_state = jax.lax.while_loop(_needs_more, _run_chunk, state)
    best_x = final_state[_STATE_BEST_X_INDEX]
    iter_count = jnp.minimum(
        final_state[_STATE_ITER_INDEX],
        jnp.asarray(maxiter, dtype=jnp.int32),
    )
    done = final_state[_STATE_DONE_INDEX]
    best_vio = final_state[_STATE_BEST_VIO_INDEX]
    return best_x, iter_count, done, best_vio


def _validate_chunk_size(chunk_size: int) -> None:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
