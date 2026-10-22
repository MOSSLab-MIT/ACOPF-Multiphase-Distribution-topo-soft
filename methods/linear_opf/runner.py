"""Benchmark loop for fixed-reference linear OPF baseline (test-only, no NN)."""

import gc
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from core.grids.data import OPFData
from core.grids.metrics import (
    accumulate_soft_penalty_sums,
    calc_ac_metrics_pf,
    compute_ac_metrics_from_voltage,
    substation_power_from_voltage,
    with_slack_voltage,
)
from core.grids.power_flow import get_power_flow_batch_fn
from core.topology.batch import precompute_topology
from core.utils.loss import compute_optimality_metrics
from core.utils.workflow import prepare_batch_data
from methods.linear_opf.solver import solve_linear_opf_batch

# Env-gated per-stage RSS logging for OOM diagnosis (e.g. 8500-bus SLURM runs).
# Disabled by default; enable with LINEAR_OPF_RSS_LOG=1. Adds a /proc read +
# print after each stage (topo/QP/PF/metrics/gc) and during warmup. Zero
# overhead when disabled (a single bool check at each call site).
_RSS_LOG_ENABLED = os.environ.get("LINEAR_OPF_RSS_LOG", "") == "1"


def _rss_gb() -> float:
    """Current process resident set size (GB) from /proc/self/status (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024**2)
    except OSError:
        pass
    return float("nan")


def _rss_log(stage: str, batch_idx: int | None = None) -> None:
    """Log RSS after a processing stage; no-op unless LINEAR_OPF_RSS_LOG=1."""
    if not _RSS_LOG_ENABLED:
        return
    label = (
        f"[rss] batch={batch_idx} stage={stage}"
        if batch_idx is not None
        else f"[rss] warmup stage={stage}"
    )
    print(f"{label} rss={_rss_gb():.2f} GB", flush=True)


def _compute_norms_jax(pred, target):
    diff = pred - target
    l1 = jnp.sum(jnp.abs(diff))
    l2 = jnp.sqrt(jnp.sum(diff**2))
    linf = jnp.max(jnp.abs(diff))
    return l1, l2, linf


def benchmark_linear_opf(
    _unused_state,
    data: OPFData,
    batch_size: int,
    test_samples: int | None,
    *,
    sample_offset: int = 0,
    solve_pf_batch_fn=None,
) -> dict:
    """Run linear OPF on test HDF5 batches and return aggregate benchmark metrics."""
    test_batched = (
        data.make_test_dataset(samples=test_samples, sample_offset=sample_offset)
        .batch(batch_size, drop_remainder=True)
        .prefetch(1)
    )
    solve_pf_batch_fn = solve_pf_batch_fn or get_power_flow_batch_fn(data.power_flow_solver)

    # Warm up CVXPY cache, topology inverse, and JAX PF kernels on one batch.
    _rss_log("start")
    wb_inputs, wb_targets = next(test_batched.as_numpy_iterator())
    wb_topo = precompute_topology(data, np.asarray(wb_inputs))
    wb_topo.Y_LL_inv.block_until_ready()
    _rss_log("topo")
    wb_batch = prepare_batch_data(
        jnp.asarray(wb_inputs),
        jnp.asarray(wb_targets),
        data,
        topo=wb_topo,
    )
    wb_pder, wb_qder, wb_qp_infos = solve_linear_opf_batch(
        data,
        wb_topo,
        np.asarray(wb_batch.s_load),
        np.asarray(wb_batch.pder_avai),
        np.asarray(wb_batch.obj_weights),
    )
    _rss_log("qp")
    if not any(info.get("skipped", False) for info in wb_qp_infos):
        wb_u_linear = jnp.asarray(
            np.stack([info["u_linear"] for info in wb_qp_infos], axis=0),
            dtype=jnp.complex128,
        )
        wb_s_injection = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(
            wb_batch.s_load,
            jnp.asarray(wb_pder + 1j * wb_qder),
        )
        wb_v = solve_pf_batch_fn(
            wb_s_injection,
            wb_u_linear,
            wb_topo.Y_LL_inv,
            wb_topo.w,
        )
        wb_v.block_until_ready()
        del wb_u_linear, wb_s_injection, wb_v
    _rss_log("pf")
    del wb_topo, wb_batch, wb_pder, wb_qder, wb_qp_infos
    gc.collect()
    _rss_log("gc")

    batch_inverse_times: list[float] = []
    batch_qp_times: list[float] = []
    batch_lin_times: list[float] = []
    batch_qp_solver_times: list[float] = []
    batch_qp_wall_times: list[float] = []
    batch_qp_overhead_times: list[float] = []
    batch_pf_times: list[float] = []
    batch_e2e_times: list[float] = []
    metrics_sum = {
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
        "solver_speedup": 0.0,
        "solver_pf_speedup": 0.0,
        "solve_time": 0.0,
        "qp_solver_time": 0.0,
        "linearization_time": 0.0,
        "qp_overhead_time": 0.0,
    }
    batch_metrics_list: list[dict] = []
    penalty_sums: dict[str, float] = {}
    num_batches = 0
    num_samples = 0
    num_qp_failed = 0
    num_skipped = 0

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
        _rss_log("topo", batch_idx)

        inputs_batch = jnp.asarray(inputs_np)
        targets_batch = jnp.asarray(targets_np)
        batch = prepare_batch_data(inputs_batch, targets_batch, data, topo=topo)
        solve_time_batch = targets_batch[:, data.slices["target"]["solve_time"]].flatten()
        batch_size_actual = int(inputs_batch.shape[0])
        num_batches += 1
        num_samples += batch_size_actual

        qp_start = time.perf_counter()
        pder_qp, qder_qp, qp_infos = solve_linear_opf_batch(
            data,
            topo,
            np.asarray(batch.s_load),
            np.asarray(batch.pder_avai),
            np.asarray(batch.obj_weights),
        )
        qp_time = time.perf_counter() - qp_start
        batch_qp_times.append(qp_time)
        batch_e2e_times.append(qp_time)
        _rss_log("qp", batch_idx)

        # Linearization-PF failures are skipped (not QP failures). batch_size=1 is
        # the usual path; mixed skip/eval in one batch is unsupported.
        num_skipped_batch = sum(1 for info in qp_infos if info.get("skipped", False))
        num_skipped += num_skipped_batch
        num_qp_failed_batch = sum(
            1
            for info in qp_infos
            if (not info.get("solved", True)) and (not info.get("skipped", False))
        )
        num_qp_failed += num_qp_failed_batch
        for i, info in enumerate(qp_infos):
            if info.get("skipped", False):
                print(
                    f"    [skip] batch={batch_idx} sample={i}: "
                    f"{info.get('skip_reason', info.get('status', 'unknown'))}",
                    flush=True,
                )

        lin_time_batch = float(sum(info["linearization_time"] for info in qp_infos))
        qp_solver_time_batch = float(sum(info["solve_time"] for info in qp_infos))
        qp_wall_time_batch = float(sum(info["qp_wall_time"] for info in qp_infos))
        qp_overhead_time_batch = float(sum(info["qp_overhead_time"] for info in qp_infos))
        batch_lin_times.append(lin_time_batch)
        batch_qp_solver_times.append(qp_solver_time_batch)
        batch_qp_wall_times.append(qp_wall_time_batch)
        batch_qp_overhead_times.append(qp_overhead_time_batch)
        metrics_sum["linearization_time"] += lin_time_batch
        metrics_sum["qp_overhead_time"] += qp_overhead_time_batch
        metrics_sum["qp_solver_time"] += qp_solver_time_batch

        if num_skipped_batch == batch_size_actual:
            batch_pf_times.append(0.0)
            batch_metrics_list.append(
                {
                    "batch_idx": batch_idx,
                    "batch_size": batch_size_actual,
                    "num_qp_failed": 0,
                    "num_skipped": num_skipped_batch,
                    "inverse_time": batch_inverse_time,
                    "qp_time": qp_time,
                    "linearization_time": lin_time_batch,
                    "qp_solver_time": qp_solver_time_batch,
                    "qp_wall_time": qp_wall_time_batch,
                    "qp_overhead_time": qp_overhead_time_batch,
                    "pf_time": 0.0,
                    "e2e_time": qp_time,
                }
            )
            del topo, batch, inputs_batch, targets_batch, pder_qp, qder_qp, qp_infos
            gc.collect()
            _rss_log("gc", batch_idx)
            continue
        if num_skipped_batch:
            raise RuntimeError(
                f"Batch {batch_idx} has {num_skipped_batch}/{batch_size_actual} "
                "linearization-PF skips; use batch_size=1 so skipped samples can "
                "be dropped without rewriting the metrics path."
            )

        u_linear = jnp.asarray(
            np.stack([info["u_linear"] for info in qp_infos], axis=0),
            dtype=jnp.complex128,
        )
        pf_start = time.perf_counter()
        s_injection = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(
            batch.s_load,
            jnp.asarray(pder_qp + 1j * qder_qp),
        )
        v_final = solve_pf_batch_fn(
            s_injection,
            u_linear,
            topo.Y_LL_inv,
            topo.w,
        )
        v_final.block_until_ready()
        pf_time = time.perf_counter() - pf_start
        batch_pf_times.append(pf_time)
        batch_e2e_times[-1] += pf_time
        _rss_log("pf", batch_idx)

        total_solve_time_batch = float(jnp.sum(solve_time_batch))
        qp_solver_time_per_sample = jnp.asarray(
            [info["solve_time"] for info in qp_infos], dtype=jnp.float64
        )
        e2e_time_batch = qp_time + pf_time
        avg_e2e_per_sample = e2e_time_batch / batch_size_actual
        speedup_batch = solve_time_batch / avg_e2e_per_sample
        solver_speedup_batch = solve_time_batch / qp_solver_time_per_sample
        pf_time_per_sample = pf_time / batch_size_actual
        solver_pf_time_per_sample = qp_solver_time_per_sample + pf_time_per_sample
        solver_pf_speedup_batch = solve_time_batch / solver_pf_time_per_sample

        v_fs_full = with_slack_voltage(data.net, v_final)
        psub = substation_power_from_voltage(data.net, v_fs_full, Y_0row=topo.Y_0row).real
        pder = jnp.asarray(pder_qp)
        qder = jnp.asarray(qder_qp)

        ac_metrics = compute_ac_metrics_from_voltage(
            data.net,
            batch.s_load,
            pder,
            qder,
            batch.pder_avai,
            v_fs_full,
            batch.obj_weights,
            topo,
            branch_status=batch.branch_status,
        )
        cost = ac_metrics.cost
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
        accumulate_soft_penalty_sums(
            penalty_sums, batch.obj_weights, ac_metrics.soft_penalty, target_metrics.soft_penalty
        )
        _rss_log("metrics", batch_idx)
        cost_error, abs_cost_error, optimality_gap = compute_optimality_metrics(
            cost, cost_target
        )
        distance = jnp.sum(
            (pder - batch.pder_target) ** 2 + (qder - batch.qder_target) ** 2,
            axis=1,
        )
        pder_l1, pder_l2, pder_linf = _compute_norms_jax(pder, batch.pder_target)
        qder_l1, qder_l2, qder_linf = _compute_norms_jax(qder, batch.qder_target)

        batch_metrics = {
            "batch_idx": batch_idx,
            "batch_size": batch_size_actual,
            "num_qp_failed": num_qp_failed_batch,
            "inverse_time": batch_inverse_time,
            "qp_time": qp_time,
            "linearization_time": lin_time_batch,
            "qp_solver_time": qp_solver_time_batch,
            "qp_wall_time": qp_wall_time_batch,
            "qp_overhead_time": qp_overhead_time_batch,
            "pf_time": pf_time,
            "e2e_time": e2e_time_batch,
            "cost_mean": float(cost.mean()),
            "cost_std": float(cost.std()),
            "cost_min": float(cost.min()),
            "cost_max": float(cost.max()),
            "cost_target_mean": float(cost_target.mean()),
            "cost_target_std": float(cost_target.std()),
            "cost_target_min": float(cost_target.min()),
            "cost_target_max": float(cost_target.max()),
            "distance_mean": float(distance.mean()),
            "distance_std": float(distance.std()),
            "distance_min": float(distance.min()),
            "distance_max": float(distance.max()),
            "violation_mean": float(violation.mean()),
            "violation_std": float(violation.std()),
            "violation_min": float(violation.min()),
            "violation_max": float(violation.max()),
            "violation_target_mean": float(violation_target.mean()),
            "violation_target_std": float(violation_target.std()),
            "violation_target_min": float(violation_target.min()),
            "violation_target_max": float(violation_target.max()),
            "cost_error_mean": float(cost_error.mean()),
            "cost_error_std": float(cost_error.std()),
            "cost_error_min": float(cost_error.min()),
            "cost_error_max": float(cost_error.max()),
            "abs_cost_error_mean": float(abs_cost_error.mean()),
            "abs_cost_error_std": float(abs_cost_error.std()),
            "abs_cost_error_min": float(abs_cost_error.min()),
            "abs_cost_error_max": float(abs_cost_error.max()),
            "optimality_gap_mean": float(optimality_gap.mean()),
            "optimality_gap_std": float(optimality_gap.std()),
            "optimality_gap_min": float(optimality_gap.min()),
            "optimality_gap_max": float(optimality_gap.max()),
            "speedup_mean": float(speedup_batch.mean()),
            "speedup_std": float(speedup_batch.std()),
            "speedup_min": float(speedup_batch.min()),
            "speedup_max": float(speedup_batch.max()),
            "solver_speedup_mean": float(solver_speedup_batch.mean()),
            "solver_speedup_std": float(solver_speedup_batch.std()),
            "solver_speedup_min": float(solver_speedup_batch.min()),
            "solver_speedup_max": float(solver_speedup_batch.max()),
            "solver_pf_speedup_mean": float(solver_pf_speedup_batch.mean()),
            "solver_pf_speedup_std": float(solver_pf_speedup_batch.std()),
            "solver_pf_speedup_min": float(solver_pf_speedup_batch.min()),
            "solver_pf_speedup_max": float(solver_pf_speedup_batch.max()),
            "solve_time_mean": float(solve_time_batch.mean()),
            "solve_time_total": total_solve_time_batch,
            "qp_solver_time_mean": float(qp_solver_time_per_sample.mean()),
            "qp_solver_time_total": qp_solver_time_batch,
            "pder_l1_norm": float(pder_l1),
            "pder_l2_norm": float(pder_l2),
            "pder_l_inf_norm": float(pder_linf),
            "qder_l1_norm": float(qder_l1),
            "qder_l2_norm": float(qder_l2),
            "qder_l_inf_norm": float(qder_linf),
        }
        batch_metrics_list.append(batch_metrics)

        metrics_sum["cost"] += float(cost.sum())
        metrics_sum["cost_target"] += float(cost_target.sum())
        metrics_sum["distance"] += float(distance.sum())
        metrics_sum["violation"] += float(violation.sum())
        metrics_sum["max_violation"] = max(
            metrics_sum["max_violation"], float(violation.max())
        )
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
        metrics_sum["solver_speedup"] += float(solver_speedup_batch.sum())
        metrics_sum["solver_pf_speedup"] += float(solver_pf_speedup_batch.sum())
        metrics_sum["solve_time"] += total_solve_time_batch

        del (
            topo,
            batch,
            inputs_batch,
            targets_batch,
            pder_qp,
            qder_qp,
            qp_infos,
            u_linear,
            s_injection,
            v_final,
            v_fs_full,
            psub,
            pder,
            qder,
            cost,
            ac_metrics,
            violation,
            target_metrics,
            cost_target,
            violation_target,
            cost_error,
            abs_cost_error,
            optimality_gap,
            distance,
            qp_solver_time_per_sample,
            speedup_batch,
            solver_speedup_batch,
            solver_pf_speedup_batch,
        )
        gc.collect()
        _rss_log("gc", batch_idx)

    total_inverse_time = sum(batch_inverse_times)
    total_qp_time = sum(batch_qp_times)
    total_lin_time = sum(batch_lin_times)
    total_qp_solver_time = sum(batch_qp_solver_times)
    total_qp_wall_time = sum(batch_qp_wall_times)
    total_qp_overhead_time = sum(batch_qp_overhead_times)
    total_pf_time = sum(batch_pf_times)
    total_e2e_time = sum(batch_e2e_times)
    total_solver_pf_time = total_qp_solver_time + total_pf_time
    num_evaluated = num_samples - num_skipped
    evaluated = [b for b in batch_metrics_list if "pder_l1_norm" in b]
    return {
        "batch_size": batch_size,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "num_skipped": num_skipped,
        "num_evaluated": num_evaluated,
        "num_qp_failed": num_qp_failed,
        "num_qp_solved": num_evaluated - num_qp_failed,
        "total_inverse_time": total_inverse_time,
        "avg_inverse_time_per_batch": total_inverse_time / num_batches if num_batches else 0.0,
        "avg_inverse_time_per_sample": total_inverse_time / num_samples if num_samples else 0.0,
        "total_qp_time": total_qp_time,
        "avg_qp_time_per_batch": total_qp_time / num_batches if num_batches else 0.0,
        "avg_qp_time_per_sample": total_qp_time / num_samples if num_samples else 0.0,
        "total_linearization_time": total_lin_time,
        "avg_linearization_time_per_sample": total_lin_time / num_samples if num_samples else 0.0,
        "total_qp_solver_time": total_qp_solver_time,
        "avg_qp_solver_time_per_sample": total_qp_solver_time / num_samples if num_samples else 0.0,
        "total_qp_wall_time": total_qp_wall_time,
        "avg_qp_wall_time_per_sample": total_qp_wall_time / num_samples if num_samples else 0.0,
        "total_qp_overhead_time": total_qp_overhead_time,
        "avg_qp_overhead_time_per_sample": total_qp_overhead_time / num_samples if num_samples else 0.0,
        "total_pf_time": total_pf_time,
        "avg_pf_time_per_batch": total_pf_time / num_batches if num_batches else 0.0,
        "avg_pf_time_per_sample": total_pf_time / num_evaluated if num_evaluated else 0.0,
        "total_e2e_time": total_e2e_time,
        "avg_e2e_time_per_batch": total_e2e_time / num_batches if num_batches else 0.0,
        "avg_e2e_time_per_sample": total_e2e_time / num_samples if num_samples else 0.0,
        "mean_qp_time": total_qp_time / num_samples if num_samples else 0.0,
        "mean_pf_time": total_pf_time / num_evaluated if num_evaluated else 0.0,
        "cost": metrics_sum["cost"] / num_evaluated if num_evaluated else 0.0,
        "cost_target": metrics_sum["cost_target"] / num_evaluated if num_evaluated else 0.0,
        "distance": metrics_sum["distance"] / num_evaluated if num_evaluated else 0.0,
        "violation": metrics_sum["violation"] / num_evaluated if num_evaluated else 0.0,
        "max_violation": metrics_sum["max_violation"],
        "violation_target": metrics_sum["violation_target"] / num_evaluated if num_evaluated else 0.0,
        "max_violation_target": metrics_sum["max_violation_target"],
        "cost_error": metrics_sum["cost_error"] / num_evaluated if num_evaluated else 0.0,
        "abs_cost_error": metrics_sum["abs_cost_error"] / num_evaluated if num_evaluated else 0.0,
        "max_abs_cost_error": metrics_sum["max_abs_cost_error"],
        "optimality_gap": metrics_sum["optimality_gap"] / num_evaluated if num_evaluated else 0.0,
        "speedup": metrics_sum["speedup"] / num_evaluated if num_evaluated else 0.0,
        "solver_speedup": metrics_sum["solver_speedup"] / num_evaluated if num_evaluated else 0.0,
        "solver_pf_speedup": metrics_sum["solver_pf_speedup"] / num_evaluated if num_evaluated else 0.0,
        "total_solve_time": metrics_sum["solve_time"],
        "total_solver_pf_time": total_solver_pf_time,
        "avg_solver_pf_time_per_sample": total_solver_pf_time / num_evaluated if num_evaluated else 0.0,
        "avg_solve_time_per_sample": metrics_sum["solve_time"] / num_evaluated if num_evaluated else 0.0,
        "avg_ipopt_solve_time_per_sample": metrics_sum["solve_time"] / num_evaluated if num_evaluated else 0.0,
        "avg_clarabel_solve_time_per_sample": total_qp_solver_time / num_samples if num_samples else 0.0,
        "pder_l1_norm": float(np.mean([b["pder_l1_norm"] for b in evaluated])) if evaluated else 0.0,
        "pder_l2_norm": float(np.mean([b["pder_l2_norm"] for b in evaluated])) if evaluated else 0.0,
        "pder_l_inf_norm": float(np.max([b["pder_l_inf_norm"] for b in evaluated])) if evaluated else 0.0,
        "qder_l1_norm": float(np.mean([b["qder_l1_norm"] for b in evaluated])) if evaluated else 0.0,
        "qder_l2_norm": float(np.mean([b["qder_l2_norm"] for b in evaluated])) if evaluated else 0.0,
        "qder_l_inf_norm": float(np.max([b["qder_l_inf_norm"] for b in evaluated])) if evaluated else 0.0,
        **{k: (v / num_evaluated if num_evaluated else 0.0) for k, v in penalty_sums.items()},
        "batch_metrics": batch_metrics_list,
    }
