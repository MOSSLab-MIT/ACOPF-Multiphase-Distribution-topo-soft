"""
SLFS / FSNet training utilities for multiphase distribution OPF.

Training loss: ``cost + weight_dist * ||x_pred - x_fs||^2`` after successive
linearization and PDHG feasibility seeking. Deploy/test uses
``forward_pass_inference`` when available.

Shared pieces: NN → ``forward_pass`` / ``forward_pass_inference`` → PF →
``calc_ineq_violation`` / ``calc_cost``. Topology via ``precompute_topology`` +
``TopologyData``.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from core.utils.workflow import prepare_batch_data
from core.utils.types import TopologyData
from core.grids.data import OPFData
from core.utils.loss import compute_optimality_metrics, normalize_cost_for_loss
from core.grids.metrics import (
    VIOLATION_COMPONENT_KEYS,
    calc_ac_metrics_pf,
    calc_cost,
    calc_violation_components_ac,
    compute_ac_metrics_from_voltage,
)
from core.grids.power_flow import get_power_flow_batch_fn
from core.nn.input import nn_features_for_apply
from core.nn.model import update_model
from core.feasibility.slfs import forward_pass, forward_pass_inference
from core.feasibility.slfs_config import FeasibilityConfig
from core.utils.runtime import get_gpu_memory_usage
from core.topology.batch import precompute_topology
from core.topology.utils import make_topology_data
from core.grids.metrics import substation_power_from_voltage, with_slack_voltage

_TRAIN_METRIC_SUM_KEYS = (
    "loss",
    "cost",
    "cost_norm",
    "distance",
    "violation",
    "eq_violation",
    "ineq_violation",
    "grad_norm",
)

def _batch_pf_or_config(solve_pf_batch_fn, fs_config: FeasibilityConfig):
    """Use injected batch PF when provided; otherwise registry fn from ``fs_config``."""
    if solve_pf_batch_fn is not None:
        return solve_pf_batch_fn
    return get_power_flow_batch_fn(fs_config.power_flow_solver)


@partial(jax.jit, static_argnums=(4, 6), static_argnames=("solve_pf_batch_fn",))
def compute_slfs_fsnet_loss(params, state, inputs_batch, targets_batch, session: OPFData, weight_dist, fs_config, topo: TopologyData, solve_pf_batch_fn=None, cost_scale=None):
    """SLFS/FSNet loss with precomputed topology matrices (outside JIT)."""
    net = session.net
    batch = prepare_batch_data(inputs_batch, targets_batch, session, topo=topo)

    # Use provided matrices (stop_gradient since topology matrices are constants)
    topo = make_topology_data(
        Y_0row=jax.lax.stop_gradient(topo.Y_0row),
        Y_LL_inv=jax.lax.stop_gradient(topo.Y_LL_inv),
        w=jax.lax.stop_gradient(topo.w),
        branch_status=jax.lax.stop_gradient(topo.branch_status),
        v_init=jax.lax.stop_gradient(topo.v_init),
    )

    nn_inputs = nn_features_for_apply(
        inputs_batch, session, use_compact=session.use_compact_nn
    )
    outs = forward_pass(
        params, state, nn_inputs, batch, net, topo, fs_config
    )

    _pf_batch = _batch_pf_or_config(solve_pf_batch_fn, fs_config)
    s_injection = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(batch.s_load, outs.s_der_final)
    v_final = _pf_batch(
        s_injection,
        outs.v_hat,
        topo.Y_LL_inv,
        topo.w
    )

    v_fs_full = with_slack_voltage(net, v_final)
    psub = substation_power_from_voltage(net, v_fs_full, Y_0row=topo.Y_0row).real

    pder_final = outs.pder_final
    qder_final = outs.qder_final
    cost = calc_cost(net, psub, pder_final, batch.pder_avai, batch.obj_weights)

    # Compute distance loss (L2 between NN prediction and FS solution)
    distance = jnp.sum((outs.pder_pred - pder_final) ** 2 + (outs.qder_pred - qder_final) ** 2, axis=1)

    # Report total and component violations; loss remains cost + distance only.
    ac_metrics = compute_ac_metrics_from_voltage(
        net,
        batch.s_load,
        pder_final,
        qder_final,
        batch.pder_avai,
        v_fs_full,
        batch.obj_weights,
        topo,
        branch_status=batch.branch_status,
    )
    violation = ac_metrics.violation
    eq_violation = ac_metrics.eq_violation
    ineq_violation = ac_metrics.ineq_violation

    cost_for_loss = normalize_cost_for_loss(cost, cost_scale)
    total_loss = cost_for_loss.mean() + distance.mean() * weight_dist
    # total_loss = cost.mean() + distance.mean() * weight_dist
    metrics = {
        'loss': total_loss,
        'cost': cost.mean(),
        'cost_norm': cost_for_loss.mean(),
        'distance': distance.mean(),
        'violation': violation.mean(),
        'max_violation': violation.max(),
        'eq_violation': eq_violation.max(),
        'ineq_violation': ineq_violation.max(),
    }

    internals = {
        'batch': batch,
        'outs': outs,
        'v_final': v_final,
        'cost': cost,
        'pder_final': pder_final,
        'qder_final': qder_final,
        'violation': violation,
        'v_fs_full': v_fs_full,
        'Y_0row': topo.Y_0row,
    }

    return total_loss, metrics, internals


@partial(jax.jit, static_argnums=(3, 5), static_argnames=("solve_pf_batch_fn",))
def train_step_slfs_fsnet(state, inputs_batch, targets_batch, session: OPFData, weight_dist, fs_config,
                  topo: TopologyData, solve_pf_batch_fn=None, cost_scale=None):
    """Single training step with precomputed topology matrices."""
    def loss_fn(params):
        loss, metrics, _ = compute_slfs_fsnet_loss(
            params, state, inputs_batch, targets_batch, session,
            weight_dist, fs_config, topo, solve_pf_batch_fn=solve_pf_batch_fn,
            cost_scale=cost_scale)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Replace NaN gradient components with zeros to prevent corrupting optimizer state.
    # NaN gradients arise from numerical instability in the SLFS forward/backward pass
    # (e.g., ill-conditioned Jacobians in FS, singular topology matrices).
    grads = jax.tree.map(
        lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g), g),
        grads
    )

    # Compute gradient norm (after NaN → 0 replacement, so it reflects real signal)
    grad_norm = optax.global_norm(grads)
    metrics['grad_norm'] = grad_norm

    state = update_model(state, grads)

    return state, loss, metrics


def train_epoch_slfs_fsnet(state, train_loader, data, target_dim, weight_dist, fs_config,
                solve_pf_batch_fn=None, cost_scale=None):
    """Run one training epoch and return state, average loss, and aggregated metrics.

    Topology matrices are always precomputed outside JIT via precompute_topology.
    Sync metrics once at epoch end.
    """
    metrics_sum = {k: jnp.asarray(0.0) for k in _TRAIN_METRIC_SUM_KEYS}
    metrics_sum["max_violation"] = jnp.asarray(0.0)
    num_batches = 0
    targets_batch_template = None

    for inputs_batch in train_loader:
        batch_size = inputs_batch.shape[0]

        # Precompute topology 
        topo = precompute_topology(data, inputs_batch)

        if targets_batch_template is None or targets_batch_template.shape[0] != batch_size:
            targets_batch_template = jnp.zeros((batch_size, target_dim))
        targets_batch = targets_batch_template

        state, _, metrics = train_step_slfs_fsnet(
            state, inputs_batch, targets_batch, data,
            weight_dist, fs_config, topo, solve_pf_batch_fn=solve_pf_batch_fn,
            cost_scale=cost_scale,
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


def eval_epoch_slfs_fsnet(state, test_loader, data, weight_dist, fs_config,
               solve_pf_batch_fn=None, cost_scale=None):
    """Run full evaluation and return aggregated metrics with timing information.

    Uses cached intermediate results from compute_loss to compute target
    metrics without redundant forward passes.  Topology matrices are always
    precomputed outside JIT.
    """
    metrics_sum = {
        'loss': 0.0,
        'cost': 0.0,
        'cost_norm': 0.0,
        'distance': 0.0,
        'violation': 0.0,
        'max_violation': 0.0,
        'eq_violation': 0.0,
        'ineq_violation': 0.0,
        'violation_target': 0.0,
        'max_violation_target': 0.0,
        'eq_violation_target': 0.0,
        'max_eq_violation_target': 0.0,
        'ineq_violation_target': 0.0,
        'max_ineq_violation_target': 0.0,
        'cost_error': 0.0,
        'abs_cost_error': 0.0,
        'max_abs_cost_error': 0.0,
        'optimality_gap': 0.0
    }
    num_batches = 0

    # Use compile-time eval context to get accurate timing
    eval_start_time = time.time()

    for inputs_batch, targets_batch in test_loader:
        # Precompute topology BEFORE converting inputs to JAX
        topo = precompute_topology(data, inputs_batch)

        loss, metrics, internals = compute_slfs_fsnet_loss(
            state.params, state, inputs_batch, targets_batch, data,
            weight_dist, fs_config,
            topo, solve_pf_batch_fn=solve_pf_batch_fn,
            cost_scale=cost_scale,
        )

        # Unpack cached internals to compute target metrics
        batch = internals['batch']
        cost = internals['cost']

        target_metrics = calc_ac_metrics_pf(
            data.net,
            batch.s_load,
            batch.pder_target,
            batch.qder_target,
            batch.pder_avai,
            batch.obj_weights,
            topo,
            v_init=batch.v_target,
        )
        cost_error, abs_cost_error, optimality_gap = compute_optimality_metrics(
            cost, target_metrics.cost
        )
        
        # Update metrics dict with target metrics
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
        
        # Aggregate metrics
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
    
    # Average out metrics
    _max_keys = [
        'max_violation',
        'max_violation_target',
        'max_eq_violation_target',
        'max_ineq_violation_target',
        'max_abs_cost_error',
    ]
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()
                   if k not in _max_keys}
    avg_metrics['max_violation'] = metrics_sum['max_violation']
    avg_metrics['max_violation_target'] = metrics_sum['max_violation_target']
    avg_metrics['max_eq_violation_target'] = metrics_sum['max_eq_violation_target']
    avg_metrics['max_ineq_violation_target'] = metrics_sum['max_ineq_violation_target']
    avg_metrics['max_abs_cost_error'] = metrics_sum['max_abs_cost_error']
    avg_metrics['total_time'] = total_time
    avg_metrics['avg_batch_time'] = total_time / num_batches if num_batches > 0 else 0.0
    
    return avg_metrics




@jax.jit
def _compute_norms_jax(predicted, target):
    """JIT-ed norm kernel for repeated test-time norm evaluation."""
    diff = predicted - target
    l1_norm = jnp.mean(jnp.abs(diff))
    l2_norm = jnp.mean(jnp.sqrt(jnp.sum(diff ** 2, axis=1)))
    l_inf_norm = jnp.max(jnp.abs(diff))
    return l1_norm, l2_norm, l_inf_norm




def test_slfs_fsnet_fn(
    state,
    data,
    batch_size,
    test_samples,
    fs_config,
    use_fast_inference: bool = True,
    solve_pf_batch_fn=None,
):
    """Test function that measures prediction time and quality metrics.

    Precomputes topology matrices per batch, calls forward_pass,
    then evaluates quality metrics in a second pass over the accumulated results.

    Args:
        state: Training state containing model parameters and apply_fn.
        data: OPFData object.
        batch_size: Batch size for testing.
        test_samples: Number of testing samples (None → use all).
        fs_config: Feasibility-seeking configuration.
        use_fast_inference: Use inference-optimized forward function when True.

    Returns:
        Dictionary containing timing metrics (inverse, forward, prediction, metrics)
        and quality metrics (cost, violation, optimality_gap, etc.).
    """
    # Stream batches (prefetch=1) and free GPU tensors each step — same pattern as
    # penalty_fs test_fn.  Do not accumulate topo / model_output across batches.
    test_batched = (
        data.make_test_dataset(samples=test_samples)
        .batch(batch_size, drop_remainder=True)
        .prefetch(1)
    )
    forward_fn = forward_pass_inference if use_fast_inference else forward_pass

    # ------------------------------------------------------------------
    # Warm-up pass (ensures JIT compilation is done before timing)
    # ------------------------------------------------------------------
    wb_inputs, wb_targets = next(test_batched.as_numpy_iterator())
    wb_inputs_np = np.asarray(wb_inputs)
    wb_targets_np = np.asarray(wb_targets)
    topo_wb = precompute_topology(data, wb_inputs_np)
    topo_wb.Y_LL_inv.block_until_ready()
    wb_inputs_j = jnp.asarray(wb_inputs_np)
    wb_targets_j = jnp.asarray(wb_targets_np)
    wb_batch = prepare_batch_data(wb_inputs_j, wb_targets_j, data, topo=topo_wb)
    wb_nn = nn_features_for_apply(wb_inputs_j, data, use_compact=data.use_compact_nn)
    wb_out = forward_fn(
        state.params, state, wb_nn, wb_batch, data.net, topo_wb, fs_config
    )
    wb_out.pder_final.block_until_ready()
    del topo_wb, wb_batch, wb_nn, wb_out

    # ------------------------------------------------------------------
    # Timed prediction + metrics (single pass; release GPU memory per batch)
    # ------------------------------------------------------------------
    batch_inverse_times = []
    batch_forward_times = []
    batch_prediction_times = []
    metrics_sum = {
        'cost': 0.0,
        'cost_target': 0.0,
        'distance': 0.0,
        'violation': 0.0,
        'max_violation': 0.0,
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
        'speedup': 0.0,
        'solve_time': 0.0,
    }
    for component_key in VIOLATION_COMPONENT_KEYS:
        metrics_sum[component_key] = 0.0
        metrics_sum[f'max_{component_key}'] = 0.0
        metrics_sum[f'{component_key}_target'] = 0.0
        metrics_sum[f'max_{component_key}_target'] = 0.0
    batch_metrics_list = []
    num_batches = 0
    num_samples = 0

    for batch_idx, (inputs_batch, targets_batch) in enumerate(
        test_batched.as_numpy_iterator()
    ):
        inputs_np = np.asarray(inputs_batch)
        targets_np = np.asarray(targets_batch)
        inv_start_time = time.perf_counter()
        topo = precompute_topology(data, inputs_np)
        topo.Y_LL_inv.block_until_ready()
        batch_inverse_time = time.perf_counter() - inv_start_time
        batch_inverse_times.append(batch_inverse_time)

        inputs_batch = jnp.asarray(inputs_np)
        targets_batch = jnp.asarray(targets_np)
        batch = prepare_batch_data(inputs_batch, targets_batch, data, topo=topo)
        solve_time_batch = targets_batch[
            :, data.slices['target']['solve_time']
        ].flatten()

        batch_size_actual = int(inputs_batch.shape[0])
        num_batches += 1
        num_samples += batch_size_actual

        nn_inputs = nn_features_for_apply(
            inputs_batch, data, use_compact=data.use_compact_nn
        )
        fwd_start_time = time.perf_counter()
        model_output = forward_fn(
            state.params, state, nn_inputs, batch, data.net, topo, fs_config
        )
        model_output.pder_final.block_until_ready()
        batch_forward_time = time.perf_counter() - fwd_start_time
        batch_prediction_times.append(batch_forward_time)
        batch_forward_times.append(batch_forward_time)

        total_solve_time_batch = float(jnp.sum(solve_time_batch))
        avg_prediction_time_per_sample = batch_forward_time / batch_size_actual
        speedup_batch = solve_time_batch / avg_prediction_time_per_sample

        _pf_batch = _batch_pf_or_config(solve_pf_batch_fn, fs_config)
        s_injection = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(
            batch.s_load, model_output.s_der_final
        )
        v_final = _pf_batch(
            s_injection,
            model_output.v_hat,
            topo.Y_LL_inv,
            topo.w,
        )
        v_fs_full = with_slack_voltage(data.net, v_final)
        psub = substation_power_from_voltage(data.net, v_fs_full, Y_0row=topo.Y_0row).real

        pder_final = model_output.pder_final
        qder_final = model_output.qder_final
        cost = calc_cost(data.net, psub, pder_final, batch.pder_avai, batch.obj_weights)

        ac_metrics = compute_ac_metrics_from_voltage(
            data.net,
            batch.s_load,
            pder_final,
            qder_final,
            batch.pder_avai,
            v_fs_full,
            batch.obj_weights,
            topo,
            branch_status=batch.branch_status,
        )
        violation = ac_metrics.violation

        target_metrics = calc_ac_metrics_pf(
            data.net,
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
        cost_error, abs_cost_error, optimality_gap = compute_optimality_metrics(
            cost, cost_target
        )

        violation_components = calc_violation_components_ac(
            data.net,
            pder_final,
            qder_final,
            batch.pder_avai,
            v_fs_full,
            Y_0row=topo.Y_0row,
            branch_status=batch.branch_status,
        )
        # Target components at the PF-consistent voltage from calc_ac_metrics_pf,
        # so their max matches the ineq part of violation_target.
        violation_components_target = calc_violation_components_ac(
            data.net,
            batch.pder_target,
            batch.qder_target,
            batch.pder_avai,
            target_metrics.v_net,
            Y_0row=topo.Y_0row,
            branch_status=batch.branch_status,
        )

        distance = jnp.sum(
            (model_output.pder_pred - pder_final) ** 2
            + (model_output.qder_pred - qder_final) ** 2,
            axis=1,
        )

        pder_l1, pder_l2, pder_linf = _compute_norms_jax(
            model_output.pder_pred, pder_final
        )
        qder_l1, qder_l2, qder_linf = _compute_norms_jax(
            model_output.qder_pred, qder_final
        )

        batch_metrics = {
            'batch_idx': batch_idx,
            'batch_size': batch_size_actual,
            'inverse_time': batch_inverse_time,
            'forward_time': batch_forward_time,
            'prediction_time': batch_forward_time,
            'cost_mean': float(cost.mean()),
            'cost_std': float(cost.std()),
            'cost_min': float(cost.min()),
            'cost_max': float(cost.max()),
            'cost_target_mean': float(cost_target.mean()),
            'cost_target_std': float(cost_target.std()),
            'cost_target_min': float(cost_target.min()),
            'cost_target_max': float(cost_target.max()),
            'distance_mean': float(distance.mean()),
            'distance_std': float(distance.std()),
            'distance_min': float(distance.min()),
            'distance_max': float(distance.max()),
            'violation_mean': float(violation.mean()),
            'violation_std': float(violation.std()),
            'violation_min': float(violation.min()),
            'violation_max': float(violation.max()),
            'violation_target_mean': float(violation_target.mean()),
            'violation_target_std': float(violation_target.std()),
            'violation_target_min': float(violation_target.min()),
            'violation_target_max': float(violation_target.max()),
            'cost_error_mean': float(cost_error.mean()),
            'cost_error_std': float(cost_error.std()),
            'cost_error_min': float(cost_error.min()),
            'cost_error_max': float(cost_error.max()),
            'abs_cost_error_mean': float(abs_cost_error.mean()),
            'abs_cost_error_std': float(abs_cost_error.std()),
            'abs_cost_error_min': float(abs_cost_error.min()),
            'abs_cost_error_max': float(abs_cost_error.max()),
            'optimality_gap_mean': float(optimality_gap.mean()),
            'optimality_gap_std': float(optimality_gap.std()),
            'optimality_gap_min': float(optimality_gap.min()),
            'optimality_gap_max': float(optimality_gap.max()),
            'speedup_mean': float(speedup_batch.mean()),
            'speedup_std': float(speedup_batch.std()),
            'speedup_min': float(speedup_batch.min()),
            'speedup_max': float(speedup_batch.max()),
            'solve_time_mean': float(solve_time_batch.mean()),
            'solve_time_total': total_solve_time_batch,
            'pder_l1_norm': float(pder_l1),
            'pder_l2_norm': float(pder_l2),
            'pder_l_inf_norm': float(pder_linf),
            'qder_l1_norm': float(qder_l1),
            'qder_l2_norm': float(qder_l2),
            'qder_l_inf_norm': float(qder_linf),
        }
        for component_key in VIOLATION_COMPONENT_KEYS:
            component = violation_components[component_key]
            component_target = violation_components_target[component_key]
            batch_metrics[f'{component_key}_mean'] = float(component.mean())
            batch_metrics[f'{component_key}_max'] = float(component.max())
            batch_metrics[f'{component_key}_target_mean'] = float(component_target.mean())
            batch_metrics[f'{component_key}_target_max'] = float(component_target.max())
        batch_metrics_list.append(batch_metrics)

        metrics_sum['cost'] += float(cost.sum())
        metrics_sum['cost_target'] += float(cost_target.sum())
        metrics_sum['distance'] += float(distance.sum())
        metrics_sum['violation'] += float(violation.sum())
        metrics_sum['max_violation'] = max(
            metrics_sum['max_violation'], float(violation.max())
        )
        metrics_sum['violation_target'] += float(violation_target.sum())
        metrics_sum['max_violation_target'] = max(
            metrics_sum['max_violation_target'], float(violation_target.max())
        )
        metrics_sum['cost_error'] += float(cost_error.sum())
        metrics_sum['abs_cost_error'] += float(abs_cost_error.sum())
        metrics_sum['max_abs_cost_error'] = max(
            metrics_sum['max_abs_cost_error'], float(abs_cost_error.max())
        )
        metrics_sum['optimality_gap'] += float(optimality_gap.sum())
        metrics_sum['speedup'] += float(speedup_batch.sum())
        metrics_sum['solve_time'] += total_solve_time_batch
        for component_key in VIOLATION_COMPONENT_KEYS:
            component = violation_components[component_key]
            component_target = violation_components_target[component_key]
            metrics_sum[component_key] += float(component.sum())
            metrics_sum[f'max_{component_key}'] = max(
                metrics_sum[f'max_{component_key}'], float(component.max())
            )
            metrics_sum[f'{component_key}_target'] += float(component_target.sum())
            metrics_sum[f'max_{component_key}_target'] = max(
                metrics_sum[f'max_{component_key}_target'], float(component_target.max())
            )

        del topo, batch, model_output, v_final, v_fs_full

    total_inverse_time = sum(batch_inverse_times)
    total_forward_time = sum(batch_forward_times)
    total_prediction_time = sum(batch_prediction_times)
    test_memory = get_gpu_memory_usage()

    avg_metrics = {
        'batch_size': batch_size,
        'num_batches': num_batches,
        'num_samples': num_samples,
        # --- topology / inverse step ---
        'total_inverse_time': total_inverse_time,
        'avg_inverse_time_per_batch': (
            total_inverse_time / num_batches if num_batches > 0 else 0.0
        ),
        'avg_inverse_time_per_sample': (
            total_inverse_time / num_samples if num_samples > 0 else 0.0
        ),
        # --- forward pass or inference function ---
        'total_forward_time': total_forward_time,
        'avg_forward_time_per_batch': (
            total_forward_time / num_batches if num_batches > 0 else 0.0
        ),
        'avg_forward_time_per_sample': (
            total_forward_time / num_samples if num_samples > 0 else 0.0
        ),
        # --- combined prediction time ---
        'total_prediction_time': total_prediction_time,
        'avg_prediction_time_per_batch': (
            total_prediction_time / num_batches if num_batches > 0 else 0.0
        ),
        'avg_prediction_time_per_sample': (
            total_prediction_time / num_samples if num_samples > 0 else 0.0
        ),
        'gpu_memory': test_memory,
        # --- quality metrics ---
        'cost': metrics_sum['cost'] / num_samples,
        'cost_target': metrics_sum['cost_target'] / num_samples,
        'distance': metrics_sum['distance'] / num_samples,
        'violation': metrics_sum['violation'] / num_samples,
        'max_violation': metrics_sum['max_violation'],
        'violation_target': metrics_sum['violation_target'] / num_samples,
        'max_violation_target': metrics_sum['max_violation_target'],
        'cost_error': metrics_sum['cost_error'] / num_samples,
        'abs_cost_error': metrics_sum['abs_cost_error'] / num_samples,
        'max_abs_cost_error': metrics_sum['max_abs_cost_error'],
        'optimality_gap': metrics_sum['optimality_gap'] / num_samples,
        'speedup': metrics_sum['speedup'] / num_samples,
        'total_solve_time': metrics_sum['solve_time'],
        'avg_solve_time_per_sample': metrics_sum['solve_time'] / num_samples,
        'pder_l1_norm': float(
            np.mean([b['pder_l1_norm'] for b in batch_metrics_list])
        ),
        'pder_l2_norm': float(
            np.mean([b['pder_l2_norm'] for b in batch_metrics_list])
        ),
        'pder_l_inf_norm': float(
            np.max([b['pder_l_inf_norm'] for b in batch_metrics_list])
        ),
        'qder_l1_norm': float(
            np.mean([b['qder_l1_norm'] for b in batch_metrics_list])
        ),
        'qder_l2_norm': float(
            np.mean([b['qder_l2_norm'] for b in batch_metrics_list])
        ),
        'qder_l_inf_norm': float(
            np.max([b['qder_l_inf_norm'] for b in batch_metrics_list])
        ),
        'batch_metrics': batch_metrics_list,
    }
    for component_key in VIOLATION_COMPONENT_KEYS:
        avg_metrics[component_key] = metrics_sum[component_key] / num_samples
        avg_metrics[f'max_{component_key}'] = metrics_sum[f'max_{component_key}']
        avg_metrics[f'{component_key}_target'] = (
            metrics_sum[f'{component_key}_target'] / num_samples
        )
        avg_metrics[f'max_{component_key}_target'] = metrics_sum[
            f'max_{component_key}_target'
        ]

    return avg_metrics
