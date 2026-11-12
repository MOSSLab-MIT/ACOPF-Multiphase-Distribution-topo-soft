"""Direct Prediction: NN predicts DER setpoints and voltages jointly.

Loss at the predicted ``(pder, qder, vr, vi)``:
``mean(normalized_cost) + rho_eq * mean(eq_violation_sq)`` where ``cost`` is the soft-OPF
objective (economic + per-sample ``ρ_V, ρ_I, ρ_θ, ρ_sub`` L1 penalties on voltage / branch-current /
angle / substation limits, matching the Julia labels). DER limits are enforced by projection;
their violations are reported as diagnostics only.

Testing reports two evaluations of the same projected prediction:

- raw (``cost``, ``violation``, ``optimality_gap``, ...): metrics at the NN-predicted voltages, which
  need not satisfy power flow;
- PF-repaired (``*_pf`` keys): AC power flow solved at the projected ``(pder, qder)`` from the
  no-load voltage ``topo.w`` (the Proposed method's deploy pipeline), then metrics at the PF voltages. The
  predicted voltages are discarded; ``speedup_pf`` counts the PF solve in the prediction time.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from core.grids.data import OPFData
from core.grids.network import NetworkParams
from core.nn.input import nn_features_for_apply
from core.grids.der_projection import project_der_controls_batch
from core.utils.loss import normalize_cost_for_loss
from core.grids.metrics import (
    accumulate_soft_penalty_sums,
    calc_ac_metrics_pf,
    calc_soft_penalties,
    compute_ac_metrics_from_voltage,
    compute_optimality_metrics,
    weighted_soft_penalty_sums,
)
from core.grids.power_flow import get_power_flow_batch_fn
from core.nn.state import update_model
from core.topology.batch import precompute_topology
from core.utils.batch import prepare_batch_data
from core.utils.types import TopologyData
from core.utils.runtime import get_gpu_memory_usage

_TRAIN_METRIC_SUM_KEYS = (
    "loss",
    "cost",
    "cost_norm",
    "violation",
    "eq_violation",
    "ineq_violation",
    "eq_violation_sq",
    "ineq_violation_sq",
    "grad_norm",
)


def split_vars(net: NetworkParams, x_pred):
    """Direct method: NN predicts [pder; qder; vr; vi] for ``n_net`` non-slack buses."""
    p_der = x_pred[:, : net.n_der_phase]
    q_der = x_pred[:, net.n_der_phase : 2 * net.n_der_phase]
    vr = x_pred[:, 2 * net.n_der_phase : 2 * net.n_der_phase + net.n_net]
    vi = x_pred[:, 2 * net.n_der_phase + net.n_net : 2 * net.n_der_phase + 2 * net.n_net]
    return p_der, q_der, vr, vi


@partial(jax.jit, static_argnums=(0,))
def _project_predictions(net: NetworkParams, x_pred, pder_avai):
    p_der_raw, q_der_raw, vr_raw, vi_raw = split_vars(net, x_pred)
    eps = jnp.asarray(1e-12, dtype=x_pred.dtype)
    p_der, q_der = project_der_controls_batch(
        net, p_der_raw, q_der_raw, pder_avai, eps
    )
    return jnp.concatenate([p_der, q_der, vr_raw, vi_raw], axis=1)


@partial(jax.jit, static_argnums=(0,))
def solve_pf_at_predicted_der(net: NetworkParams, s_load_batch, x_pred, topo: TopologyData):
    """Non-slack PF voltages at the projected DER setpoints in ``x_pred``; predicted voltages unused.

    Same solve as the Proposed method's deploy pipeline: initial guess ``topo.w``, network PF solver.
    """
    pder, qder, _, _ = split_vars(net, x_pred)
    s_net = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(s_load_batch, pder + 1j * qder)
    pf_b = get_power_flow_batch_fn(getattr(net, "power_flow_solver", "fixed_point"))
    return pf_b(s_net, topo.w, topo.Y_LL_inv, topo.w)


@partial(jax.jit, static_argnums=(0,))
def pf_repaired_metrics(net: NetworkParams, batch, x_pred, v_pf, topo: TopologyData):
    """AC metrics (cost, violations, soft penalties) at the projected DERs and their PF voltages."""
    pder, qder, _, _ = split_vars(net, x_pred)
    return compute_ac_metrics_from_voltage(
        net, batch.s_load, pder, qder, batch.pder_avai, v_pf, batch.obj_weights, topo
    )


def calc_batch_quantities(
    net: NetworkParams,
    s_load_batch,
    x_pred,
    pder_avai,
    obj_weights,
    topo: TopologyData,
):
    """Direct voltages from NN; topology-aware eq/ineq violations."""
    pder, qder, vr, vi = split_vars(net, x_pred)
    v_net = vr + 1j * vi
    s_der = pder + 1j * qder
    metrics = compute_ac_metrics_from_voltage(
        net,
        s_load_batch,
        pder,
        qder,
        pder_avai,
        v_net,
        obj_weights,
        topo,
    )
    return (
        v_net,
        s_der,
        metrics.cost,
        metrics.eq_violation,
        metrics.eq_violation_sq,
        metrics.ineq_violation,
        metrics.ineq_violation_sq,
        metrics.violation,
    )


@partial(jax.jit, static_argnums=(4,))
def compute_direct_prediction_loss(
    params, state, inputs_batch, targets_batch, session: OPFData, rho_eq, topo: TopologyData,
    cost_scale=None,
):
    net = session.net
    batch = prepare_batch_data(inputs_batch, targets_batch, session, topo=topo)

    nn_inputs = nn_features_for_apply(
        inputs_batch, session, use_compact=session.use_compact_nn
    )
    x_pred_raw = state.apply_fn(params, nn_inputs)
    x_pred = _project_predictions(net, x_pred_raw, batch.pder_avai)

    (
        v_final,
        s_der,
        cost,
        eq_violation,
        eq_violation_sq,
        ineq_violation,
        ineq_violation_sq,
        violation,
    ) = calc_batch_quantities(
        net, batch.s_load, x_pred, batch.pder_avai, batch.obj_weights, topo
    )

    cost_for_loss = normalize_cost_for_loss(cost, cost_scale)
    loss = cost_for_loss.mean() + rho_eq * eq_violation_sq.mean()

    metrics = {
        "loss": loss,
        "cost": cost.mean(),
        "cost_norm": cost_for_loss.mean(),
        "violation": violation.mean(),
        "max_violation": violation.max(),
        "eq_violation": eq_violation.max(),
        "ineq_violation": ineq_violation.max(),
        "eq_violation_sq": eq_violation_sq.mean(),
        "ineq_violation_sq": ineq_violation_sq.mean(),
    }

    internals = {
        "batch": batch,
        "cost": cost,
        "pder_final": s_der.real,
        "v_final": v_final,
        "violation": violation,
        "topo": topo,
    }
    return loss, metrics, internals


@partial(jax.jit, static_argnums=(3,))
def train_step_direct_prediction(state, inputs_batch, targets_batch, session: OPFData, rho_eq, topo: TopologyData,
                          cost_scale=None):
    def loss_fn(params):
        loss, metrics, _ = compute_direct_prediction_loss(
            params, state, inputs_batch, targets_batch, session, rho_eq, topo,
            cost_scale=cost_scale,
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grad_norm = optax.global_norm(grads)
    metrics["grad_norm"] = grad_norm
    is_finite = jnp.isfinite(grad_norm)
    grads = jax.tree_util.tree_map(
        lambda g: jnp.where(is_finite, g, jnp.zeros_like(g)), grads
    )
    state = update_model(state, grads)
    return state, loss, metrics


def train_epoch_direct_prediction(
    state, train_loader, data, rho_eq, rho_rate, violation_threshold, decay_rate, rho_max,
    cost_scale=None,
):
    """Run one training epoch; sync metrics once at epoch end."""
    metrics_sum = {k: jnp.asarray(0.0) for k in _TRAIN_METRIC_SUM_KEYS}
    metrics_sum["max_violation"] = jnp.asarray(0.0)
    num_batches = 0

    for inputs_batch in train_loader:
        topo = precompute_topology(data, inputs_batch)
        inputs_batch = jnp.array(inputs_batch)
        targets_batch = jnp.zeros((inputs_batch.shape[0], 1))

        state, _, metrics = train_step_direct_prediction(
            state, inputs_batch, targets_batch, data, rho_eq, topo,
            cost_scale=cost_scale,
        )

        eq_violation_val = float(metrics["eq_violation"])
        eq_violation_sq_val = float(metrics["eq_violation_sq"])
        if eq_violation_val > violation_threshold:
            rho_eq = rho_eq + rho_rate * eq_violation_sq_val
            rho_eq = min(rho_eq, rho_max)
        else:
            rho_eq *= decay_rate

        for k in _TRAIN_METRIC_SUM_KEYS:
            metrics_sum[k] = metrics_sum[k] + metrics[k]
        metrics_sum["max_violation"] = jnp.maximum(
            metrics_sum["max_violation"], metrics["max_violation"]
        )
        num_batches += 1

    inv = 1.0 / float(num_batches)
    avg_metrics = {k: float(metrics_sum[k] * inv) for k in _TRAIN_METRIC_SUM_KEYS}
    avg_metrics["max_violation"] = float(metrics_sum["max_violation"])
    return state, avg_metrics, rho_eq


def eval_epoch_direct_prediction(state, eval_loader, data, rho_eq, cost_scale=None):
    net = data.net
    metrics_sum = {
        "loss": 0.0,
        "cost": 0.0,
        "cost_norm": 0.0,
        "violation": 0.0,
        "max_violation": 0.0,
        "eq_violation": 0.0,
        "ineq_violation": 0.0,
        "eq_violation_sq": 0.0,
        "ineq_violation_sq": 0.0,
        "violation_target": 0.0,
        "max_violation_target": 0.0,
        "eq_violation_target": 0.0,
        "max_eq_violation_target": 0.0,
        "ineq_violation_target": 0.0,
        "max_ineq_violation_target": 0.0,
        "cost_error": 0.0,
        "abs_cost_error": 0.0,
        "max_abs_cost_error": 0.0,
        "optimality_gap": 0.0,
    }
    num_batches = 0
    eval_start_time = time.time()

    for inputs_batch, targets_batch in eval_loader:
        inputs_np = np.asarray(inputs_batch)
        targets_np = np.asarray(targets_batch)
        topo = precompute_topology(data, inputs_np)
        inputs_batch = jnp.asarray(inputs_np)
        targets_batch = jnp.asarray(targets_np)

        loss, metrics, internals = compute_direct_prediction_loss(
            state.params, state, inputs_batch, targets_batch, data, rho_eq, topo,
            cost_scale=cost_scale,
        )

        batch = internals["batch"]
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
        cost_error, abs_cost_error, optimality_gap = compute_optimality_metrics(
            internals["cost"], cost_target
        )

        metrics["violation_target"] = target_metrics.violation.mean()
        metrics["max_violation_target"] = target_metrics.violation.max()
        metrics["eq_violation_target"] = target_metrics.eq_violation.mean()
        metrics["max_eq_violation_target"] = target_metrics.eq_violation.max()
        metrics["ineq_violation_target"] = target_metrics.ineq_violation.mean()
        metrics["max_ineq_violation_target"] = target_metrics.ineq_violation.max()
        metrics["cost_error"] = cost_error.mean()
        metrics["abs_cost_error"] = abs_cost_error.mean()
        metrics["optimality_gap"] = optimality_gap.mean()
        metrics_sum["max_abs_cost_error"] = max(
            metrics_sum["max_abs_cost_error"], float(abs_cost_error.max())
        )

        for k, v in metrics.items():
            if k in (
                "max_violation",
                "max_violation_target",
                "max_eq_violation_target",
                "max_ineq_violation_target",
            ):
                metrics_sum[k] = max(metrics_sum[k], float(v))
            else:
                metrics_sum[k] += float(v)
        num_batches += 1

    total_time = time.time() - eval_start_time
    _max_keys = [
        "max_violation",
        "max_violation_target",
        "max_eq_violation_target",
        "max_ineq_violation_target",
        "max_abs_cost_error",
    ]
    avg_metrics = {
        k: v / num_batches
        for k, v in metrics_sum.items()
        if k not in _max_keys
    }
    avg_metrics["max_violation"] = metrics_sum["max_violation"]
    avg_metrics["max_violation_target"] = metrics_sum["max_violation_target"]
    avg_metrics["max_eq_violation_target"] = metrics_sum["max_eq_violation_target"]
    avg_metrics["max_ineq_violation_target"] = metrics_sum["max_ineq_violation_target"]
    avg_metrics["max_abs_cost_error"] = metrics_sum["max_abs_cost_error"]
    avg_metrics["total_time"] = total_time
    avg_metrics["avg_batch_time"] = total_time / num_batches if num_batches > 0 else 0.0
    return avg_metrics


def compute_norms(predicted, target):
    diff = predicted - target
    return {
        "l1_norm": float(jnp.mean(jnp.abs(diff))),
        "l2_norm": float(jnp.mean(jnp.sqrt(jnp.sum(diff**2, axis=1)))),
        "l_inf_norm": float(jnp.max(jnp.abs(diff))),
    }


def test_direct_prediction_fn(state, data, batch_size, test_samples):
    test_batched = (
        data.make_test_dataset(samples=test_samples)
        .batch(batch_size, drop_remainder=True)
        .prefetch(1)
    )
    net = data.net

    @jax.jit
    def inference_step(inputs):
        return state.apply_fn(state.params, inputs)

    wb_inputs, wb_targets = next(test_batched.as_numpy_iterator())
    wb_inputs_np = np.asarray(wb_inputs)
    wb_targets_np = np.asarray(wb_targets)
    topo_wb = precompute_topology(data, wb_inputs_np)
    wb_inputs_j = jnp.asarray(wb_inputs_np)
    wb_targets_j = jnp.asarray(wb_targets_np)
    wb_batch = prepare_batch_data(wb_inputs_j, wb_targets_j, data, topo=topo_wb)
    wb_nn = nn_features_for_apply(wb_inputs_j, data, use_compact=data.use_compact_nn)
    wb_x = _project_predictions(net, inference_step(wb_nn), wb_batch.pder_avai)
    wb_x.block_until_ready()
    wb_v_pf = solve_pf_at_predicted_der(net, wb_batch.s_load, wb_x, topo_wb)
    pf_repaired_metrics(net, wb_batch, wb_x, wb_v_pf, topo_wb).cost.block_until_ready()
    del topo_wb, wb_batch, wb_x, wb_v_pf

    batch_inverse_times = []
    batch_prediction_times = []
    batch_pf_times = []
    metrics_sum = {
        "cost_pf": 0.0,
        "violation_pf": 0.0,
        "max_violation_pf": 0.0,
        "cost_error_pf": 0.0,
        "abs_cost_error_pf": 0.0,
        "max_abs_cost_error_pf": 0.0,
        "optimality_gap_pf": 0.0,
        "speedup_pf": 0.0,
        "cost": 0.0,
        "cost_target": 0.0,
        "distance": 0.0,
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

        nn_inputs = nn_features_for_apply(
            inputs_batch, data, use_compact=data.use_compact_nn
        )
        pred_start = time.perf_counter()
        x_pred_raw = inference_step(nn_inputs)
        x_pred = _project_predictions(net, x_pred_raw, batch.pder_avai)
        x_pred.block_until_ready()
        batch_pred_time = time.perf_counter() - pred_start
        batch_prediction_times.append(batch_pred_time)

        pf_start = time.perf_counter()
        v_pf = solve_pf_at_predicted_der(net, batch.s_load, x_pred, topo)
        v_pf.block_until_ready()
        batch_pf_time = time.perf_counter() - pf_start
        batch_pf_times.append(batch_pf_time)

        total_solve_time_batch = float(jnp.sum(solve_time_batch))
        avg_prediction_time_per_sample = batch_pred_time / batch_size_actual
        speedup_batch = solve_time_batch / avg_prediction_time_per_sample
        speedup_pf_batch = solve_time_batch / ((batch_pred_time + batch_pf_time) / batch_size_actual)

        (
            v_final,
            s_der,
            cost,
            _eq_violation,
            _eq_violation_sq,
            _ineq_violation,
            _ineq_violation_sq,
            violation,
        ) = calc_batch_quantities(
            net, batch.s_load, x_pred, batch.pder_avai, batch.obj_weights, topo
        )
        pen_pred = calc_soft_penalties(net, v_final, topo.Y_0row)
        del v_final, s_der
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
        accumulate_soft_penalty_sums(penalty_sums, batch.obj_weights, pen_pred, target_metrics.soft_penalty)
        cost_error, abs_cost_error, optimality_gap = compute_optimality_metrics(
            cost, cost_target
        )
        pf_metrics = pf_repaired_metrics(net, batch, x_pred, v_pf, topo)
        cost_error_pf, abs_cost_error_pf, optimality_gap_pf = compute_optimality_metrics(
            pf_metrics.cost, cost_target
        )
        for key, value in weighted_soft_penalty_sums(
            batch.obj_weights, pf_metrics.soft_penalty, "_pf"
        ).items():
            penalty_sums[key] = penalty_sums.get(key, 0.0) + value
        pder_pred = x_pred[:, : net.n_der_phase]
        qder_pred = x_pred[:, net.n_der_phase : 2 * net.n_der_phase]
        pder_norms = compute_norms(pder_pred, batch.pder_target)
        qder_norms = compute_norms(qder_pred, batch.qder_target)

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
                "pf_time": batch_pf_time,
                "cost_pf_mean": float(pf_metrics.cost.mean()),
                "violation_pf_mean": float(pf_metrics.violation.mean()),
                "violation_pf_max": float(pf_metrics.violation.max()),
                "optimality_gap_pf_mean": float(optimality_gap_pf.mean()),
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
        metrics_sum["speedup"] += float(speedup_batch.sum())
        metrics_sum["solve_time"] += total_solve_time_batch
        metrics_sum["cost_pf"] += float(pf_metrics.cost.sum())
        metrics_sum["violation_pf"] += float(pf_metrics.violation.sum())
        metrics_sum["max_violation_pf"] = max(
            metrics_sum["max_violation_pf"], float(pf_metrics.violation.max())
        )
        metrics_sum["cost_error_pf"] += float(cost_error_pf.sum())
        metrics_sum["abs_cost_error_pf"] += float(abs_cost_error_pf.sum())
        metrics_sum["max_abs_cost_error_pf"] = max(
            metrics_sum["max_abs_cost_error_pf"], float(abs_cost_error_pf.max())
        )
        metrics_sum["optimality_gap_pf"] += float(optimality_gap_pf.sum())
        metrics_sum["speedup_pf"] += float(speedup_pf_batch.sum())

        del topo, batch, x_pred, v_pf, pf_metrics

    total_inverse_time = sum(batch_inverse_times)
    total_prediction_time = sum(batch_prediction_times)
    total_pf_time = sum(batch_pf_times)
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
        "gpu_memory": test_memory,
        "cost": metrics_sum["cost"] / num_samples,
        "cost_target": metrics_sum["cost_target"] / num_samples,
        "violation": metrics_sum["violation"] / num_samples,
        "max_violation": metrics_sum["max_violation"],
        "violation_target": metrics_sum["violation_target"] / num_samples,
        "max_violation_target": metrics_sum["max_violation_target"],
        "cost_error": metrics_sum["cost_error"] / num_samples,
        "abs_cost_error": metrics_sum["abs_cost_error"] / num_samples,
        "max_abs_cost_error": metrics_sum["max_abs_cost_error"],
        "optimality_gap": metrics_sum["optimality_gap"] / num_samples,
        "speedup": metrics_sum["speedup"] / num_samples,
        "total_pf_time": total_pf_time,
        "cost_pf": metrics_sum["cost_pf"] / num_samples,
        "violation_pf": metrics_sum["violation_pf"] / num_samples,
        "max_violation_pf": metrics_sum["max_violation_pf"],
        "cost_error_pf": metrics_sum["cost_error_pf"] / num_samples,
        "abs_cost_error_pf": metrics_sum["abs_cost_error_pf"] / num_samples,
        "max_abs_cost_error_pf": metrics_sum["max_abs_cost_error_pf"],
        "optimality_gap_pf": metrics_sum["optimality_gap_pf"] / num_samples,
        "speedup_pf": metrics_sum["speedup_pf"] / num_samples,
        "total_solve_time": metrics_sum["solve_time"],
        "avg_solve_time_per_sample": metrics_sum["solve_time"] / num_samples,
        "pder_l1_norm": float(
            np.mean([b["pder_l1_norm"] for b in batch_metrics_list])
        ),
        "pder_l2_norm": float(
            np.mean([b["pder_l2_norm"] for b in batch_metrics_list])
        ),
        "pder_l_inf_norm": float(
            np.max([b["pder_l_inf_norm"] for b in batch_metrics_list])
        ),
        "qder_l1_norm": float(
            np.mean([b["qder_l1_norm"] for b in batch_metrics_list])
        ),
        "qder_l2_norm": float(
            np.mean([b["qder_l2_norm"] for b in batch_metrics_list])
        ),
        "qder_l_inf_norm": float(
            np.max([b["qder_l_inf_norm"] for b in batch_metrics_list])
        ),
        **{k: v / n for k, v in penalty_sums.items()},
        "batch_metrics": batch_metrics_list,
    }
