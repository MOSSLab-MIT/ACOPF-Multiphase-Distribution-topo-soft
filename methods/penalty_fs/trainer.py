"""
Penalty + FS training utilities for multiphase distribution OPF.

Training uses fixed penalty ``rho`` on AC inequality violations:
``loss = cost + rho * ineq_violation_sq``. Deploy/test runs ``post_process_slfs``
(inference SLFS) when NN predictions violate limits.

Shared pieces: NN → project to DER feasible set → PF → ``calc_ineq_violation`` /
``calc_cost``. Topology via ``precompute_topology`` + ``TopologyData``.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from core.grids.data import NetworkParams
from core.nn.input import nn_features_for_apply
from core.feasibility.fs_common import project_der_controls_from_x_batch
from core.utils.loss import normalize_cost_for_loss
from core.utils.loss import compute_optimality_metrics
from core.grids.metrics import (
    calc_ac_metrics_pf,
    calc_ineq_violation,
    compute_ac_metrics_from_voltage,
)
from core.grids.metrics import with_slack_voltage
from core.grids.power_flow import get_power_flow_batch_fn
from core.nn.model import update_model
from core.feasibility.slfs import slfs_inference_core
from core.topology.batch import precompute_topology
from core.utils.workflow import prepare_batch_data
from core.utils.types import TopologyData
from core.utils.runtime import get_gpu_memory_usage

_PROJ_EPS = 1e-12
_TRAIN_METRIC_SUM_KEYS = (
    "loss",
    "cost",
    "cost_norm",
    "violation",
    "eq_violation",
    "ineq_violation",
    "ineq_violation_sq",
    "grad_norm",
)

@partial(jax.jit, static_argnums=(2,))
def _project_predictions(x_pred_raw, pder_avai, net: NetworkParams):
    eps = jnp.asarray(_PROJ_EPS, dtype=x_pred_raw.dtype)
    return project_der_controls_from_x_batch(net, x_pred_raw, pder_avai, eps)


def calc_batch_quantities(
    net: NetworkParams,
    s_load_batch,
    x_pred,
    pder_avai,
    obj_weights,
    num_der,
    topo: TopologyData,
    v_init=None,
):
    """
    Calculate quantities needed for Lagrangian loss.
    
    Args:
        net: NetworkParams object
        s_load_batch: Load power (batch, n_bus-1)
        x_pred: Current DER setpoints [p_der; q_der] (batch, 2*n_der)
        pder_avai: DER availability (batch, n_der)
        obj_weights: Objective weights (batch, 2)
        num_der: Number of DERs
        topo: Precomputed topology matrices (required when topology varies)
        v_init: Optional pre-computed voltage to skip PF solve
    
    Returns:
        v_final: Voltages (batch, n_bus-1)
        s_der: DER setpoints (batch, n_der)
        cost: Objective cost (batch,)
        violation: Total max constraint violation (batch,)
        ineq_violation_sq: Mean-square inequality penalty (batch,)
    """
    pder = x_pred[:, :num_der]
    qder = x_pred[:, num_der:]
    s_der = pder + 1j * qder

    s_net = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(s_load_batch, s_der)
    Y_LL_inv_batch = topo.Y_LL_inv
    w_batch = topo.w

    if v_init is not None:
        v_final = v_init
    else:
        v_init_batch = w_batch
        pf_b = get_power_flow_batch_fn(getattr(net, "power_flow_solver", "fixed_point"))
        v_final = pf_b(s_net, v_init_batch, Y_LL_inv_batch, w_batch)

    v_full = with_slack_voltage(net, v_final)
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

    return (
        v_final,
        s_der,
        metrics.cost,
        metrics.violation,
        metrics.eq_violation,
        metrics.ineq_violation,
        metrics.ineq_violation_sq,
    )


@partial(jax.jit, static_argnums=(0,))
def _metrics_pred_and_fs(net, batch, x_pred, x_pred_fs, v_pred, v_fs, topo: TopologyData):
    """Evaluate cost/violation for NN-only and FS-corrected (x, v) pairs in one JIT region."""
    xs = jnp.stack([x_pred, x_pred_fs])
    vs = jnp.stack([v_pred, v_fs])

    def one(x, v_net):
        pder = x[:, : net.n_der_phase]
        qder = x[:, net.n_der_phase :]
        v_full = with_slack_voltage(net, v_net)
        metrics = compute_ac_metrics_from_voltage(
            net,
            batch.s_load,
            pder,
            qder,
            batch.pder_avai,
            v_full,
            batch.obj_weights,
            topo,
        )
        return metrics.cost, metrics.violation

    costs, violations = jax.vmap(one)(xs, vs)
    return costs[0], violations[0], costs[1], violations[1]


def _scalar_stats(arr):
    """Per-batch mean/std/min/max for a 1D JAX array."""
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def _branch_metric_fields(
    cost, violation, cost_error, abs_cost_error, optimality_gap, pder_norms, qder_norms
):
    """Scalar summary fields for one pred or fs branch within a batch."""
    cost_s = _scalar_stats(cost)
    vio_s = _scalar_stats(violation)
    err_s = _scalar_stats(cost_error)
    abs_err_s = _scalar_stats(abs_cost_error)
    gap_s = _scalar_stats(optimality_gap)
    return {
        'cost_mean': cost_s['mean'],
        'cost_std': cost_s['std'],
        'cost_min': cost_s['min'],
        'cost_max': cost_s['max'],
        'violation_mean': vio_s['mean'],
        'violation_std': vio_s['std'],
        'violation_min': vio_s['min'],
        'violation_max': vio_s['max'],
        'cost_error_mean': err_s['mean'],
        'cost_error_std': err_s['std'],
        'cost_error_min': err_s['min'],
        'cost_error_max': err_s['max'],
        'abs_cost_error_mean': abs_err_s['mean'],
        'abs_cost_error_std': abs_err_s['std'],
        'abs_cost_error_min': abs_err_s['min'],
        'abs_cost_error_max': abs_err_s['max'],
        'optimality_gap_mean': gap_s['mean'],
        'optimality_gap_std': gap_s['std'],
        'optimality_gap_min': gap_s['min'],
        'optimality_gap_max': gap_s['max'],
        'pder_l1_norm': pder_norms['l1_norm'],
        'pder_l2_norm': pder_norms['l2_norm'],
        'pder_l_inf_norm': pder_norms['l_inf_norm'],
        'qder_l1_norm': qder_norms['l1_norm'],
        'qder_l2_norm': qder_norms['l2_norm'],
        'qder_l_inf_norm': qder_norms['l_inf_norm'],
    }


def _accumulate_branch_metrics(
    metrics_sum, cost, violation, cost_error, abs_cost_error, optimality_gap
):
    metrics_sum['cost'] += float(cost.sum())
    metrics_sum['violation'] += float(violation.sum())
    metrics_sum['max_violation'] = max(metrics_sum['max_violation'], float(violation.max()))
    metrics_sum['cost_error'] += float(cost_error.sum())
    metrics_sum['abs_cost_error'] += float(abs_cost_error.sum())
    metrics_sum['max_abs_cost_error'] = max(
        metrics_sum['max_abs_cost_error'], float(abs_cost_error.max())
    )
    metrics_sum['optimality_gap'] += float(optimality_gap.sum())


@partial(jax.jit, static_argnums=(4,))
def compute_penalty_fs_loss(
    params, state, inputs_batch, targets_batch, session, rho, topo: TopologyData,
    cost_scale=None,
):
    """Compute Lagrangian loss: cost + rho * ineq_violation_sq (``session`` = OPFData)."""
    net = session.net
    batch = prepare_batch_data(inputs_batch, targets_batch, session, topo=topo)

    nn_inputs = nn_features_for_apply(
        inputs_batch, session, use_compact=session.use_compact_nn
    )
    x_pred_raw = state.apply_fn(params, nn_inputs)
    x_pred = _project_predictions(x_pred_raw, batch.pder_avai, net)

    v_final, s_der, cost, violation, eq_violation, ineq_violation, ineq_violation_sq = (
        calc_batch_quantities(
        net,
        batch.s_load,
        x_pred,
        batch.pder_avai,
        batch.obj_weights,
        net.n_der_phase,
        topo,
    )
    )

    # rho_scaled = rho / float(net.n_total + net.n_branch)
    cost_for_loss = normalize_cost_for_loss(cost, cost_scale)
    loss = cost_for_loss.mean() + rho * ineq_violation_sq.mean()

    metrics = {
        'loss': loss,
        'cost': cost.mean(),
        'cost_norm': cost_for_loss.mean(),
        'violation': violation.mean(),
        'max_violation': violation.max(),
        'eq_violation': eq_violation.max(),
        'ineq_violation': ineq_violation.max(),
        'ineq_violation_sq': ineq_violation_sq.mean(),
    }
    
    internals = {
        'batch': batch,
        'cost': cost,
        'pder_final': s_der.real,
        'v_final': v_final,
        'violation': violation,
    }
    
    return loss, metrics, internals
    


@partial(jax.jit, static_argnums=(3,))
def train_step_penalty_fs(state, inputs_batch, targets_batch, session, rho, topo: TopologyData,
                          cost_scale=None):
    """One Optax step: ``value_and_grad`` on ``compute_penalty_fs_loss``.

    Skips the update when the global gradient norm is non-finite.
    """
    def loss_fn(params):
        loss, metrics, _ = compute_penalty_fs_loss(
            params, state, inputs_batch, targets_batch, session, rho, topo,
            cost_scale=cost_scale,
        )
        return loss, metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    

    # Compute gradient norm
    grad_norm = optax.global_norm(grads)
    metrics['grad_norm'] = grad_norm

    is_finite = jnp.isfinite(grad_norm)

    # jnp.where will broadcast the scalar 'is_finite' to the whole tree
    grads = jax.tree_util.tree_map(
        lambda g: jnp.where(is_finite, g, jnp.zeros_like(g)), 
        grads
    )
    # Update model
    state = update_model(state, grads)    
    
    return state, loss, metrics


def train_epoch_penalty_fs(state, train_loader, data, rho, cost_scale=None):
    """Run one training epoch with fixed ``rho``; sync metrics once at epoch end."""
    metrics_sum = {k: jnp.asarray(0.0) for k in _TRAIN_METRIC_SUM_KEYS}
    metrics_sum["max_violation"] = jnp.asarray(0.0)
    num_batches = 0

    for inputs_batch in train_loader:
        topo = precompute_topology(data, inputs_batch)
        inputs_batch = jnp.array(inputs_batch)
        targets_batch = jnp.zeros((inputs_batch.shape[0], 1))

        state, _, metrics = train_step_penalty_fs(
            state, inputs_batch, targets_batch, data, rho, topo, cost_scale=cost_scale
        )

        for k in _TRAIN_METRIC_SUM_KEYS:
            metrics_sum[k] = metrics_sum[k] + metrics[k]
        metrics_sum["max_violation"] = jnp.maximum(
            metrics_sum["max_violation"], metrics["max_violation"]
        )
        num_batches += 1

    inv = 1.0 / float(num_batches)
    avg_metrics = {k: float(metrics_sum[k] * inv) for k in _TRAIN_METRIC_SUM_KEYS}
    avg_metrics["max_violation"] = float(metrics_sum["max_violation"])
    return state, avg_metrics


def eval_epoch_penalty_fs(state, eval_loader, data, rho, cost_scale=None):
    """Run evaluation epoch."""
    # Use full data object, not just data.net
    metrics_sum = {
        'loss': 0.0,
        'cost': 0.0,
        'cost_norm': 0.0,
        'violation': 0.0,
        'max_violation': 0.0,
        'eq_violation': 0.0,
        'ineq_violation': 0.0,
        'ineq_violation_sq': 0.0,
        'violation_target': 0.0,
        'max_violation_target': 0.0,
        'eq_violation_target': 0.0,
        'max_eq_violation_target': 0.0,
        'ineq_violation_target': 0.0,
        'max_ineq_violation_target': 0.0,
        'cost_error': 0.0,
        'abs_cost_error': 0.0,
        'max_abs_cost_error': 0.0,
        'optimality_gap': 0.0,
    }
    num_batches = 0
    eval_start_time = time.time()
    
    for inputs_batch, targets_batch in eval_loader:
        topo = precompute_topology(data, inputs_batch)
        loss, metrics, internals = compute_penalty_fs_loss(
            state.params, state, inputs_batch, targets_batch, data, rho, topo,
            cost_scale=cost_scale,
        )

        batch = internals['batch']
        net = data.net
        target_metrics = calc_ac_metrics_pf(
            net,
            batch.s_load,
            batch.pder_target,
            batch.qder_target,
            batch.pder_avai,
            batch.obj_weights,
            topo,
            v_init=batch.v_target,
        )
        
        cost_error, abs_cost_error, optimality_gap = compute_optimality_metrics(
            internals['cost'], target_metrics.cost
        )
        
        metrics['violation_target'] = target_metrics.violation.mean()
        metrics['max_violation_target'] = target_metrics.violation.max()
        metrics['eq_violation_target'] = target_metrics.eq_violation.mean()
        metrics['max_eq_violation_target'] = target_metrics.eq_violation.max()
        metrics['ineq_violation_target'] = target_metrics.ineq_violation.mean()
        metrics['max_ineq_violation_target'] = target_metrics.ineq_violation.max()
        metrics['cost_error'] = cost_error.mean()
        metrics['abs_cost_error'] = abs_cost_error.mean()
        metrics['optimality_gap'] = optimality_gap.mean()
        metrics_sum['max_abs_cost_error'] = max(
            metrics_sum['max_abs_cost_error'], float(abs_cost_error.max())
        )
        
        for k, v in metrics.items():
            if k in (
                'max_violation',
                'max_violation_target',
                'max_eq_violation_target',
                'max_ineq_violation_target',
            ):
                metrics_sum[k] = max(metrics_sum[k], float(v))
            else:
                metrics_sum[k] += float(v)
        num_batches += 1
    
    total_time = time.time() - eval_start_time
    _max_keys = [
        'max_violation',
        'max_violation_target',
        'max_eq_violation_target',
        'max_ineq_violation_target',
        'max_abs_cost_error',
    ]
    avg_metrics = {
        k: v / num_batches for k, v in metrics_sum.items()
        if k not in _max_keys
    }
    avg_metrics['max_violation'] = metrics_sum['max_violation']
    avg_metrics['max_violation_target'] = metrics_sum['max_violation_target']
    avg_metrics['max_eq_violation_target'] = metrics_sum['max_eq_violation_target']
    avg_metrics['max_ineq_violation_target'] = metrics_sum['max_ineq_violation_target']
    avg_metrics['max_abs_cost_error'] = metrics_sum['max_abs_cost_error']
    avg_metrics['total_time'] = total_time
    avg_metrics['avg_batch_time'] = total_time / num_batches if num_batches > 0 else 0.0
    
    return avg_metrics


def compute_norms(predicted, target):
    """
    Compute L1, L2, and L_inf norms between predicted and target vectors.
    
    Args:
        predicted: Predicted array (shape: [batch_size, n_der])
        target: Target array (same shape as predicted)
    
    Returns:
        Dictionary with 'l1_norm', 'l2_norm', 'l_inf_norm' keys
    """
    diff = predicted - target
    l1_norm = jnp.mean(jnp.abs(diff))
    l2_norm = jnp.mean(jnp.sqrt(jnp.sum(diff ** 2, axis=1)))
    l_inf_norm = jnp.max(jnp.abs(diff))
    
    return {
        'l1_norm': float(l1_norm),
        'l2_norm': float(l2_norm),
        'l_inf_norm': float(l_inf_norm),
    }


@partial(jax.jit, static_argnums=(0,))
def compute_voltage(net: NetworkParams, batch, x_pred, topo: TopologyData):
    p_der = x_pred[:, :net.n_der_phase]
    q_der = x_pred[:, net.n_der_phase:]
    s_der = p_der + 1j * q_der
    s_injection = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)
    pf_b = get_power_flow_batch_fn(getattr(net, "power_flow_solver", "fixed_point"))
    v_final = pf_b(s_injection, topo.w, topo.Y_LL_inv, topo.w)
    return v_final

@partial(jax.jit, static_argnums=(0,))
def compute_violation(net: NetworkParams, x_pred, batch, topo: TopologyData):
    p_der = x_pred[:, :net.n_der_phase]
    q_der = x_pred[:, net.n_der_phase:]
    s_der = p_der + 1j * q_der
    s_injection = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)
    pf_b = get_power_flow_batch_fn(getattr(net, "power_flow_solver", "fixed_point"))
    v_final = pf_b(s_injection, topo.w, topo.Y_LL_inv, topo.w) # so equality is satisfied
    v_full = with_slack_voltage(net, v_final)
    violation = calc_ineq_violation(
        net, p_der, q_der, batch.pder_avai, v_full, Y_0row=topo.Y_0row
    )
    return violation, s_der


@partial(jax.jit, static_argnums=(0, 4))
def post_process_slfs(net: NetworkParams, batch, s_der, x_pred, fs_config, topo: TopologyData):
    """Deploy-time SLFS refinement when the NN prediction violates AC limits.

    Solves PF at the projected NN controls; if ``max(violation) > 1e-6``, runs
    ``slfs_inference_core`` and returns FS-refined ``(pder, qder)`` and voltages.
    Otherwise returns the NN prediction unchanged (cheap path).

    Uses batch-level ``lax.cond`` so compile traces both branches; see inline
    comments on ``while_loop`` alternative for mixed per-sample flags under vmap.
    """
    p_der = x_pred[:, :net.n_der_phase]
    q_der = x_pred[:, net.n_der_phase:]
    s_injection = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)
    pf_b = get_power_flow_batch_fn(getattr(net, "power_flow_solver", "fixed_point"))
    v_final = pf_b(s_injection, topo.w, topo.Y_LL_inv, topo.w)
    v_full = with_slack_voltage(net, v_final)
    violation = calc_ineq_violation(
        net, p_der, q_der, batch.pder_avai, v_full, Y_0row=topo.Y_0row
    )
    max_violation = jnp.max(violation)

    def run_slfs_branch(dynamic_args):
        batch_val, s_der_val, v_final_val = dynamic_args
        topo_slfs = topo._replace(v_init=v_final_val)
        s_der_new, p_der_new, q_der_new, _ = slfs_inference_core(
            net,
            batch_val.s_load,
            s_der_val,
            batch_val.pder_avai,
            topo_slfs,
            fs_config,
        )
        s_injection_new = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(
            batch_val.s_load, s_der_new
        )
        v_final_new = pf_b(s_injection_new, topo.w, topo.Y_LL_inv, topo.w)
        return jnp.concatenate([p_der_new, q_der_new], axis=1), v_final_new

    v_pred = v_final
    need_slfs = max_violation > 1e-6

    # jax.lax.cond with a *batched* predicate under vmap becomes lax.select: both branches
    # are evaluated for every batch lane and results are masked (see JAX docs + issue #8409).
    # That wastes SLFS work when only some samples violate. Scalar batch-level cond below
    # avoids vmap(cond), but both branches are still traced at compile time.
    x_pred_final, v_fs = jax.lax.cond(
        need_slfs,
        run_slfs_branch,
        lambda _: (x_pred, v_final),
        operand=(batch, s_der, v_final),
    )

    # while_loop trick (jax-ml/jax#8409): 0 iterations if need_slfs is False (skip SLFS entirely
    # at runtime); 1 iteration if True. Helps memory/AD vs cond when the expensive branch must
    # not run; under vmap with mixed per-sample flags the expensive branch still runs for all.
    # (x_pred_final, v_fs), _ = jax.lax.while_loop(
    #     lambda carry: carry[1],
    #     lambda _: (run_slfs_branch((batch, s_der, v_final)), jnp.array(False)),
    #     ((x_pred, v_final), need_slfs),
    # )
    return x_pred_final, v_fs, v_pred


def _empty_branch_metrics_sum():
    return {
        'cost': 0.0,
        'violation': 0.0,
        'max_violation': 0.0,
        'cost_error': 0.0,
        'abs_cost_error': 0.0,
        'max_abs_cost_error': 0.0,
        'optimality_gap': 0.0,
        'raw_speedup': 0.0,
        'speedup': 0.0,
    }


def test_penalty_fs_fn(state, data, batch_size, test_samples, fs_config):
    """
    Lagrangian+FS test with dual physics metrics and split timing.

    Branches:
      - pred: NN inference + projection only (no FS); metrics at (x_pred, v_pred).
      - fs: full deploy pipeline (NN + post_process_slfs); metrics at (x_pred_fs, v_fs).

    Timing:
      - shared.total_inverse_time: precompute_topology (host dedup + Y_LL_inv).
      - pred.total_raw_prediction_time: inference_step + _project_predictions.
      - fs.total_fs_time: post_process_slfs only.
      - fs.total_prediction_time: raw + fs (end-to-end deploy latency).

    Note: pred.raw_* is pure NN deploy timing (no FS overhead).
    """
    test_batched = (
        data.make_test_dataset(samples=test_samples)
        .batch(batch_size, drop_remainder=True)
        .prefetch(1)
    )

    @jax.jit
    def inference_step(inputs):
        return state.apply_fn(state.params, inputs)

    net = data.net
    n_der = net.n_der_phase

    wb_inputs, wb_targets = next(test_batched.as_numpy_iterator())
    wb_inputs_np = np.asarray(wb_inputs)
    wb_targets_np = np.asarray(wb_targets)
    topo_wb = precompute_topology(data, wb_inputs_np)
    wb_inputs_j = jnp.asarray(wb_inputs_np)
    wb_targets_j = jnp.asarray(wb_targets_np)
    wb_batch = prepare_batch_data(wb_inputs_j, wb_targets_j, data, topo=topo_wb)
    wb_nn = nn_features_for_apply(wb_inputs_j, data, use_compact=data.use_compact_nn)
    wb_x = _project_predictions(inference_step(wb_nn), wb_batch.pder_avai, net)
    wb_x_fs, v_fs_wb, v_pred_wb = post_process_slfs(
        net,
        wb_batch,
        wb_x[:, :n_der] + 1j * wb_x[:, n_der:],
        wb_x,
        fs_config,
        topo_wb,
    )
    wb_x_fs.block_until_ready()
    wb_m = _metrics_pred_and_fs(
        net, wb_batch, wb_x, wb_x_fs, v_pred_wb, v_fs_wb, topo_wb
    )
    wb_m[0].block_until_ready()
    del topo_wb, wb_batch, wb_x, wb_x_fs

    batch_inverse_times = []
    batch_raw_prediction_times = []
    batch_fs_times = []
    batch_prediction_times = []
    metrics_sum_pred = _empty_branch_metrics_sum()
    metrics_sum_fs = _empty_branch_metrics_sum()
    metrics_sum_shared = {
        'cost_target': 0.0,
        'distance': 0.0,
        'violation_target': 0.0,
        'max_violation_target': 0.0,
        'solve_time': 0.0,
    }
    batch_metrics_list = []
    num_batches = 0
    num_samples = 0

    for batch_idx, (inputs_batch, targets_batch) in enumerate(
        test_batched.as_numpy_iterator()
    ):
        inputs_np = np.asarray(inputs_batch)
        targets_np = np.asarray(targets_batch)
        inv_start = time.perf_counter()
        topo = precompute_topology(data, inputs_np)
        topo.Y_LL_inv.block_until_ready()
        batch_inverse_time = time.perf_counter() - inv_start
        batch_inverse_times.append(batch_inverse_time)
        inputs_batch = jnp.asarray(inputs_np)
        targets_batch = jnp.asarray(targets_np)
        batch = prepare_batch_data(inputs_batch, targets_batch, data, topo=topo)

        batch_size_actual = int(inputs_batch.shape[0])
        num_batches += 1
        num_samples += batch_size_actual

        solve_time_batch = targets_batch[:, data.slices['target']['solve_time']].flatten()
        total_solve_time_batch = float(jnp.sum(solve_time_batch))

        nn_inputs = nn_features_for_apply(
            inputs_batch, data, use_compact=data.use_compact_nn
        )
        raw_start = time.time()
        x_pred_raw = inference_step(nn_inputs)
        x_pred = _project_predictions(x_pred_raw, batch.pder_avai, net)
        v_pred_raw = compute_voltage(net, batch, x_pred, topo) # v_pred_raw is the voltage from the PF at x_pred -- make the comparison fairer
        v_pred_raw.block_until_ready()
        raw_prediction_time = time.time() - raw_start
        batch_raw_prediction_times.append(raw_prediction_time)

        fs_start = time.time()
        x_pred_fs, v_fs, v_pred = post_process_slfs(
            net,
            batch,
            x_pred[:, :n_der] + 1j * x_pred[:, n_der:],
            x_pred,
            fs_config,
            topo,
        )
        x_pred_fs.block_until_ready()
        fs_time = time.time() - fs_start
        batch_fs_times.append(fs_time)
        prediction_time = raw_prediction_time + fs_time
        batch_prediction_times.append(prediction_time)

        raw_speedup_batch = solve_time_batch / (raw_prediction_time / batch_size_actual)
        speedup_batch = solve_time_batch / (prediction_time / batch_size_actual)

        cost_pred, vio_pred, cost_fs, vio_fs = _metrics_pred_and_fs(
            net, batch, x_pred, x_pred_fs, v_pred, v_fs, topo
        )

        target_metrics = calc_ac_metrics_pf(
            net,
            batch.s_load,
            batch.pder_target,
            batch.qder_target,
            batch.pder_avai,
            batch.obj_weights,
            topo,
            v_init=batch.v_target,
        )
        cost_target = target_metrics.cost
        violation_target = target_metrics.violation

        cost_error_pred, abs_cost_error_pred, optimality_gap_pred = compute_optimality_metrics(
            cost_pred, cost_target
        )
        cost_error_fs, abs_cost_error_fs, optimality_gap_fs = compute_optimality_metrics(
            cost_fs, cost_target
        )
        distance = jnp.linalg.norm(x_pred_fs - x_pred, axis=1) ** 2

        pder_norms_pred = compute_norms(x_pred[:, :n_der], batch.pder_target)
        qder_norms_pred = compute_norms(x_pred[:, n_der:], batch.qder_target)
        pder_norms_fs = compute_norms(x_pred_fs[:, :n_der], batch.pder_target)
        qder_norms_fs = compute_norms(x_pred_fs[:, n_der:], batch.qder_target)

        pred_fields = _branch_metric_fields(
            cost_pred,
            vio_pred,
            cost_error_pred,
            abs_cost_error_pred,
            optimality_gap_pred,
            pder_norms_pred,
            qder_norms_pred,
        )
        fs_fields = _branch_metric_fields(
            cost_fs,
            vio_fs,
            cost_error_fs,
            abs_cost_error_fs,
            optimality_gap_fs,
            pder_norms_fs,
            qder_norms_fs,
        )
        distance_s = _scalar_stats(distance)
        cost_target_s = _scalar_stats(cost_target)
        violation_target_s = _scalar_stats(violation_target)
        raw_speedup_s = _scalar_stats(raw_speedup_batch)
        speedup_s = _scalar_stats(speedup_batch)

        batch_metrics_list.append({
            'batch_idx': batch_idx,
            'batch_size': batch_size_actual,
            'shared': {
                'inverse_time': batch_inverse_time,
                'cost_target_mean': cost_target_s['mean'],
                'cost_target_std': cost_target_s['std'],
                'cost_target_min': cost_target_s['min'],
                'cost_target_max': cost_target_s['max'],
                'violation_target_mean': violation_target_s['mean'],
                'violation_target_std': violation_target_s['std'],
                'violation_target_min': violation_target_s['min'],
                'violation_target_max': violation_target_s['max'],
                'distance_mean': distance_s['mean'],
                'distance_std': distance_s['std'],
                'distance_min': distance_s['min'],
                'distance_max': distance_s['max'],
                'solve_time_mean': float(solve_time_batch.mean()),
                'solve_time_total': total_solve_time_batch,
            },
            'pred': {
                **pred_fields,
                'raw_prediction_time': raw_prediction_time,
                'raw_speedup_mean': raw_speedup_s['mean'],
                'raw_speedup_std': raw_speedup_s['std'],
                'raw_speedup_min': raw_speedup_s['min'],
                'raw_speedup_max': raw_speedup_s['max'],
            },
            'fs': {
                **fs_fields,
                'prediction_time': prediction_time,
                'fs_time': fs_time,
                'speedup_mean': speedup_s['mean'],
                'speedup_std': speedup_s['std'],
                'speedup_min': speedup_s['min'],
                'speedup_max': speedup_s['max'],
            },
        })

        _accumulate_branch_metrics(
            metrics_sum_pred,
            cost_pred,
            vio_pred,
            cost_error_pred,
            abs_cost_error_pred,
            optimality_gap_pred,
        )
        metrics_sum_pred['raw_speedup'] += float(raw_speedup_batch.sum())
        _accumulate_branch_metrics(
            metrics_sum_fs,
            cost_fs,
            vio_fs,
            cost_error_fs,
            abs_cost_error_fs,
            optimality_gap_fs,
        )
        metrics_sum_fs['speedup'] += float(speedup_batch.sum())

        metrics_sum_shared['cost_target'] += float(cost_target.sum())
        metrics_sum_shared['distance'] += float(distance.sum())
        metrics_sum_shared['violation_target'] += float(violation_target.sum())
        metrics_sum_shared['max_violation_target'] = max(
            metrics_sum_shared['max_violation_target'], float(violation_target.max())
        )
        metrics_sum_shared['solve_time'] += total_solve_time_batch

        del topo, batch, x_pred, x_pred_fs, v_pred, v_fs

    total_inverse_time = sum(batch_inverse_times)
    total_raw_prediction_time = sum(batch_raw_prediction_times)
    total_fs_time = sum(batch_fs_times)
    total_prediction_time = sum(batch_prediction_times)
    test_memory = get_gpu_memory_usage()
    n = num_samples if num_samples > 0 else 1
    nb = num_batches if num_batches > 0 else 1

    pred_summary = {
        'cost': metrics_sum_pred['cost'] / n,
        'violation': metrics_sum_pred['violation'] / n,
        'max_violation': metrics_sum_pred['max_violation'],
        'cost_error': metrics_sum_pred['cost_error'] / n,
        'abs_cost_error': metrics_sum_pred['abs_cost_error'] / n,
        'max_abs_cost_error': metrics_sum_pred['max_abs_cost_error'],
        'optimality_gap': metrics_sum_pred['optimality_gap'] / n,
        'raw_speedup': metrics_sum_pred['raw_speedup'] / n,
        'total_raw_prediction_time': total_raw_prediction_time,
        'avg_raw_prediction_time_per_sample': total_raw_prediction_time / n,
        'avg_raw_prediction_time_per_batch': total_raw_prediction_time / nb,
        'pder_l1_norm': float(np.mean([b['pred']['pder_l1_norm'] for b in batch_metrics_list])),
        'pder_l2_norm': float(np.mean([b['pred']['pder_l2_norm'] for b in batch_metrics_list])),
        'pder_l_inf_norm': float(np.max([b['pred']['pder_l_inf_norm'] for b in batch_metrics_list])),
        'qder_l1_norm': float(np.mean([b['pred']['qder_l1_norm'] for b in batch_metrics_list])),
        'qder_l2_norm': float(np.mean([b['pred']['qder_l2_norm'] for b in batch_metrics_list])),
        'qder_l_inf_norm': float(np.max([b['pred']['qder_l_inf_norm'] for b in batch_metrics_list])),
    }

    fs_summary = {
        'cost': metrics_sum_fs['cost'] / n,
        'violation': metrics_sum_fs['violation'] / n,
        'max_violation': metrics_sum_fs['max_violation'],
        'cost_error': metrics_sum_fs['cost_error'] / n,
        'abs_cost_error': metrics_sum_fs['abs_cost_error'] / n,
        'max_abs_cost_error': metrics_sum_fs['max_abs_cost_error'],
        'optimality_gap': metrics_sum_fs['optimality_gap'] / n,
        'speedup': metrics_sum_fs['speedup'] / n,
        'total_prediction_time': total_prediction_time,
        'total_fs_time': total_fs_time,
        'avg_prediction_time_per_sample': total_prediction_time / n,
        'avg_prediction_time_per_batch': total_prediction_time / nb,
        'avg_fs_time_per_sample': total_fs_time / n,
        'avg_fs_time_per_batch': total_fs_time / nb,
        'pder_l1_norm': float(np.mean([b['fs']['pder_l1_norm'] for b in batch_metrics_list])),
        'pder_l2_norm': float(np.mean([b['fs']['pder_l2_norm'] for b in batch_metrics_list])),
        'pder_l_inf_norm': float(np.max([b['fs']['pder_l_inf_norm'] for b in batch_metrics_list])),
        'qder_l1_norm': float(np.mean([b['fs']['qder_l1_norm'] for b in batch_metrics_list])),
        'qder_l2_norm': float(np.mean([b['fs']['qder_l2_norm'] for b in batch_metrics_list])),
        'qder_l_inf_norm': float(np.max([b['fs']['qder_l_inf_norm'] for b in batch_metrics_list])),
    }

    shared_summary = {
        'total_inverse_time': total_inverse_time,
        'avg_inverse_time_per_batch': total_inverse_time / nb,
        'avg_inverse_time_per_sample': total_inverse_time / n,
        'total_solve_time': metrics_sum_shared['solve_time'],
        'avg_solve_time_per_sample': metrics_sum_shared['solve_time'] / n,
        'cost_target': metrics_sum_shared['cost_target'] / n,
        'violation_target': metrics_sum_shared['violation_target'] / n,
        'max_violation_target': metrics_sum_shared['max_violation_target'],
        'distance': metrics_sum_shared['distance'] / n,
        'gpu_memory': test_memory,
    }

    return {
        'batch_size': batch_size,
        'num_batches': num_batches,
        'num_samples': num_samples,
        'shared': shared_summary,
        'pred': pred_summary,
        'fs': fs_summary,
        'batch_metrics': batch_metrics_list,
    }
