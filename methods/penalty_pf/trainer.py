"""
Penalty + power-flow (penalty_pf) training utilities for multiphase distribution OPF.

Pipeline (identical at train, eval, and test time):
NN → project DER controls to their device-feasible set → AC power flow →
``compute_ac_metrics_from_voltage`` (cost, violations).

Training loss: ``loss = cost + rho * ineq_violation_sq`` where ``cost`` is the soft-OPF objective
(economic cost + curtailment + per-sample ``ρ_V, ρ_I, ρ_θ`` L1 penalties on voltage / branch-current /
angle limits, matching the Julia labels) and ``ineq_violation_sq`` covers only the constraints that
stay hard (substation; DER limits are enforced by projection). Equality (power-flow) constraints are
satisfied by construction because voltages come from the PF solve. Topology via
``precompute_topology`` + ``TopologyData``.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from core.grids.data import NetworkParams
from core.nn.input import nn_features_for_apply
from core.grids.der_projection import project_der_controls_from_x_batch
from core.utils.loss import normalize_cost_for_loss
from core.utils.loss import compute_optimality_metrics
from core.grids.metrics import (
    accumulate_soft_penalty_sums,
    calc_ac_metrics_pf,
    calc_soft_penalties,
    compute_ac_metrics_from_voltage,
)
from core.grids.metrics import with_slack_voltage
from core.grids.power_flow import get_power_flow_batch_fn
from core.nn.model import update_model
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
    Calculate quantities needed for the penalty loss.
    
    Args:
        net: NetworkParams object
        s_load_batch: Load power (batch, n_bus-1)
        x_pred: Current DER setpoints [p_der; q_der] (batch, 2*n_der)
        pder_avai: DER availability (batch, n_der)
        obj_weights: Objective weights [w_cost, w_curtail, rho_v, rho_i, rho_theta] (batch, 5)
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


@partial(jax.jit, static_argnums=(4,))
def compute_penalty_pf_loss(
    params, state, inputs_batch, targets_batch, session, rho, topo: TopologyData,
    cost_scale=None,
):
    """Penalty loss: soft-OPF cost + rho * hard ineq_violation_sq (``session`` = OPFData)."""
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
def train_step_penalty_pf(state, inputs_batch, targets_batch, session, rho, topo: TopologyData,
                          cost_scale=None):
    """One Optax step: ``value_and_grad`` on ``compute_penalty_pf_loss``.

    Skips the update when the global gradient norm is non-finite.
    """
    def loss_fn(params):
        loss, metrics, _ = compute_penalty_pf_loss(
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


def train_epoch_penalty_pf(state, train_loader, data, rho, cost_scale=None):
    """Run one training epoch with fixed ``rho``; sync metrics once at epoch end."""
    metrics_sum = {k: jnp.asarray(0.0) for k in _TRAIN_METRIC_SUM_KEYS}
    metrics_sum["max_violation"] = jnp.asarray(0.0)
    num_batches = 0

    for inputs_batch in train_loader:
        topo = precompute_topology(data, inputs_batch)
        inputs_batch = jnp.array(inputs_batch)
        targets_batch = jnp.zeros((inputs_batch.shape[0], 1))

        state, _, metrics = train_step_penalty_pf(
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


def eval_epoch_penalty_pf(state, eval_loader, data, rho, cost_scale=None):
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
        loss, metrics, internals = compute_penalty_pf_loss(
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
    """Non-slack voltages from the PF solve at ``x_pred`` (initial guess ``topo.w``, as in training)."""
    p_der = x_pred[:, :net.n_der_phase]
    q_der = x_pred[:, net.n_der_phase:]
    s_der = p_der + 1j * q_der
    s_injection = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)
    pf_b = get_power_flow_batch_fn(getattr(net, "power_flow_solver", "fixed_point"))
    v_final = pf_b(s_injection, topo.w, topo.Y_LL_inv, topo.w)
    return v_final


@partial(jax.jit, static_argnums=(0,))
def _prediction_metrics(net: NetworkParams, batch, x_pred, v_pred, topo: TopologyData):
    """Per-sample cost and max violation at ``(x_pred, v_pred)`` via the training metric path."""
    _, _, cost, violation, _, _, _ = calc_batch_quantities(
        net,
        batch.s_load,
        x_pred,
        batch.pder_avai,
        batch.obj_weights,
        net.n_der_phase,
        topo,
        v_init=v_pred,
    )
    return cost, violation


def test_penalty_pf_fn(state, data, batch_size, test_samples):
    """Test the penalty_pf pipeline (NN → projection → PF) against solver targets.

    Timing:
      - ``total_inverse_time``: ``precompute_topology`` (host dedup + ``Y_LL_inv``).
      - ``total_prediction_time``: NN forward + DER projection + PF solve.

    Speedup is per-sample solver time divided by per-sample prediction time.
    """
    test_batched = (
        data.make_test_dataset(samples=test_samples)
        .batch(batch_size, drop_remainder=True)
        .prefetch(1)
    )
    net = data.net
    n_der = net.n_der_phase

    @jax.jit
    def inference_step(inputs):
        return state.apply_fn(state.params, inputs)

    def predict_and_solve_pf(nn_inputs, batch, topo):
        """Deploy pipeline: NN → DER projection → AC power flow (same as training)."""
        x_pred = _project_predictions(inference_step(nn_inputs), batch.pder_avai, net)
        v_pred = compute_voltage(net, batch, x_pred, topo)
        return x_pred, v_pred

    # Warm-up (compile) on the first batch so timings exclude JIT compilation.
    wb_inputs, wb_targets = next(test_batched.as_numpy_iterator())
    wb_inputs_np = np.asarray(wb_inputs)
    topo_wb = precompute_topology(data, wb_inputs_np)
    wb_inputs_j = jnp.asarray(wb_inputs_np)
    wb_batch = prepare_batch_data(
        wb_inputs_j, jnp.asarray(np.asarray(wb_targets)), data, topo=topo_wb
    )
    wb_nn = nn_features_for_apply(wb_inputs_j, data, use_compact=data.use_compact_nn)
    wb_x, wb_v = predict_and_solve_pf(wb_nn, wb_batch, topo_wb)
    wb_cost, _ = _prediction_metrics(net, wb_batch, wb_x, wb_v, topo_wb)
    wb_cost.block_until_ready()
    del topo_wb, wb_batch, wb_x, wb_v, wb_cost

    batch_inverse_times = []
    batch_prediction_times = []
    metrics_sum = {
        "cost": 0.0,
        "cost_target": 0.0,
        "violation": 0.0,
        "max_violation": 0.0,
        "violation_target": 0.0,
        "max_violation_target": 0.0,
        "cost_error": 0.0,
        "abs_cost_error": 0.0,
        "max_abs_cost_error": 0.0,
        "optimality_gap": 0.0,
        "speedup": 0.0,
        "solve_time": 0.0,
    }
    batch_metrics_list = []
    penalty_sums: dict[str, float] = {}
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

        solve_time_batch = targets_np[:, data.slices["target"]["solve_time"]].flatten()
        total_solve_time_batch = float(np.sum(solve_time_batch))

        nn_inputs = nn_features_for_apply(
            inputs_batch, data, use_compact=data.use_compact_nn
        )
        pred_start = time.perf_counter()
        x_pred, v_pred = predict_and_solve_pf(nn_inputs, batch, topo)
        v_pred.block_until_ready()
        batch_pred_time = time.perf_counter() - pred_start
        batch_prediction_times.append(batch_pred_time)
        speedup_batch = solve_time_batch / (batch_pred_time / batch_size_actual)

        cost, violation = _prediction_metrics(net, batch, x_pred, v_pred, topo)
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
        accumulate_soft_penalty_sums(
            penalty_sums, batch.obj_weights, calc_soft_penalties(net, v_pred), target_metrics.soft_penalty
        )
        cost_error, abs_cost_error, optimality_gap = compute_optimality_metrics(
            cost, cost_target
        )
        pder_norms = compute_norms(x_pred[:, :n_der], batch.pder_target)
        qder_norms = compute_norms(x_pred[:, n_der:], batch.qder_target)

        batch_metrics_list.append(
            {
                "batch_idx": batch_idx,
                "batch_size": batch_size_actual,
                "inverse_time": batch_inverse_time,
                "prediction_time": batch_pred_time,
                "cost_mean": float(cost.mean()),
                "violation_mean": float(violation.mean()),
                "violation_max": float(violation.max()),
                "cost_error_mean": float(cost_error.mean()),
                "abs_cost_error_mean": float(abs_cost_error.mean()),
                "abs_cost_error_max": float(abs_cost_error.max()),
                "optimality_gap_mean": float(optimality_gap.mean()),
                "pder_l1_norm": pder_norms["l1_norm"],
                "pder_l2_norm": pder_norms["l2_norm"],
                "pder_l_inf_norm": pder_norms["l_inf_norm"],
                "qder_l1_norm": qder_norms["l1_norm"],
                "qder_l2_norm": qder_norms["l2_norm"],
                "qder_l_inf_norm": qder_norms["l_inf_norm"],
            }
        )

        metrics_sum["cost"] += float(cost.sum())
        metrics_sum["cost_target"] += float(cost_target.sum())
        metrics_sum["violation"] += float(violation.sum())
        metrics_sum["max_violation"] = max(metrics_sum["max_violation"], float(violation.max()))
        metrics_sum["violation_target"] += float(violation_target.sum())
        metrics_sum["max_violation_target"] = max(
            metrics_sum["max_violation_target"], float(violation_target.max())
        )
        metrics_sum["cost_error"] += float(cost_error.sum())
        metrics_sum["abs_cost_error"] += float(abs_cost_error.sum())
        metrics_sum["max_abs_cost_error"] = max(
            metrics_sum["max_abs_cost_error"], float(abs_cost_error.max())
        )
        metrics_sum["optimality_gap"] += float(optimality_gap.sum())
        metrics_sum["speedup"] += float(np.sum(speedup_batch))
        metrics_sum["solve_time"] += total_solve_time_batch

        del topo, batch, x_pred, v_pred

    total_inverse_time = sum(batch_inverse_times)
    total_prediction_time = sum(batch_prediction_times)
    test_memory = get_gpu_memory_usage()
    nb = num_batches if num_batches > 0 else 1
    n = num_samples if num_samples > 0 else 1

    return {
        "batch_size": batch_size,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "total_inverse_time": total_inverse_time,
        "avg_inverse_time_per_batch": total_inverse_time / nb,
        "avg_inverse_time_per_sample": total_inverse_time / n,
        "total_prediction_time": total_prediction_time,
        "avg_prediction_time_per_batch": total_prediction_time / nb,
        "avg_prediction_time_per_sample": total_prediction_time / n,
        "gpu_memory": test_memory,
        "cost": metrics_sum["cost"] / n,
        "cost_target": metrics_sum["cost_target"] / n,
        "violation": metrics_sum["violation"] / n,
        "max_violation": metrics_sum["max_violation"],
        "violation_target": metrics_sum["violation_target"] / n,
        "max_violation_target": metrics_sum["max_violation_target"],
        "cost_error": metrics_sum["cost_error"] / n,
        "abs_cost_error": metrics_sum["abs_cost_error"] / n,
        "max_abs_cost_error": metrics_sum["max_abs_cost_error"],
        "optimality_gap": metrics_sum["optimality_gap"] / n,
        "speedup": metrics_sum["speedup"] / n,
        "total_solve_time": metrics_sum["solve_time"],
        "avg_solve_time_per_sample": metrics_sum["solve_time"] / n,
        "pder_l1_norm": float(np.mean([b["pder_l1_norm"] for b in batch_metrics_list])),
        "pder_l2_norm": float(np.mean([b["pder_l2_norm"] for b in batch_metrics_list])),
        "pder_l_inf_norm": float(np.max([b["pder_l_inf_norm"] for b in batch_metrics_list])),
        "qder_l1_norm": float(np.mean([b["qder_l1_norm"] for b in batch_metrics_list])),
        "qder_l2_norm": float(np.mean([b["qder_l2_norm"] for b in batch_metrics_list])),
        "qder_l_inf_norm": float(np.max([b["qder_l_inf_norm"] for b in batch_metrics_list])),
        **{k: v / n for k, v in penalty_sums.items()},
        "batch_metrics": batch_metrics_list,
    }
