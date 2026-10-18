#!/usr/bin/env python3
"""Benchmark SLFS inference vs JAXopt voltage least-squares on perturbed OPF targets.

Both solvers start from the same projected perturbation of HDF5 target DER controls.
Headline timing is solver/refinement only; topology, perturbation, and PF init are
reported separately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

# Repo root on path when run as a script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.feasibility.fs_common import project_der_controls_from_x_batch
from core.feasibility.slfs import slfs_inference_core
from core.feasibility.slfs_config import FeasibilityConfig
from core.grids.data import NetworkParams, OPFData
from core.grids.metrics import (
    ac_constraint_violation_vector,
    compute_ac_metrics_from_voltage,
    with_slack_voltage,
)
from core.grids.power_flow import PowerFlowSolver, get_power_flow_batch_fn
from core.topology.batch import precompute_topology
from core.utils.datasets import load_opf_data
from core.utils.runtime import configure_tensorflow_for_data_loading, set_random_seeds
from core.utils.types import OPFDataBuildOptions, TopologyData
from core.utils.workflow import prepare_batch_data

from jaxopt import LevenbergMarquardt

_PROJ_EPS = 1e-12
_DEFAULT_PERTURB_SCALES = (0.01, 0.05, 0.10)

# 240-bus (906 network nodes) is stiff: fixed-point PF (7 iter) does not converge
# reliably for v0 init. Other preset grids keep fixed_point.
_GRIDS_USING_NEWTON_GMRES = frozenset({"240"})
_240_CASE_MARKER = "240bus_Master"
_DEFAULT_PF_SOLVER = PowerFlowSolver.FIXED_POINT.value

GRID_PRESETS: dict[str, dict[str, str]] = {
    "13": {
        "train_h5": "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5",
        "test_h5": "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5",
        "data_json": "datagen_topo/datasets/IEEE13Nodeckt/network_config.json",
    },
    "123": {
        "train_h5": "datagen_topo/datasets/IEEE123Master/IEEE123Master_training.h5",
        "test_h5": "datagen_topo/datasets/IEEE123Master/IEEE123Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/IEEE123Master/network_config.json",
    },
    "240": {
        "train_h5": "datagen_topo/datasets/240bus_Master/240bus_Master_training.h5",
        "test_h5": "datagen_topo/datasets/240bus_Master/240bus_Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/240bus_Master/network_config.json",
    },
    "906": {
        "train_h5": "datagen_topo/datasets/906bus_Master/906bus_Master_training.h5",
        "test_h5": "datagen_topo/datasets/906bus_Master/906bus_Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/906bus_Master/network_config.json",
    },
    "8500": {
        "train_h5": "datagen_topo/datasets/8500_Master/8500_Master_training.h5",
        "test_h5": "datagen_topo/datasets/8500_Master/8500_Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/8500_Master/network_config.json",
    },
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Runtime knobs for one benchmark run."""

    perturb_scales: tuple[float, ...]
    perturb_seed: int
    batch_size: int
    test_samples: int | None
    fs_config: FeasibilityConfig
    lm_maxiter: int
    lm_tol: float
    lm_damping: float
    eq_weight: float
    ineq_weight: float


def assert_der_recovery_unique(net: NetworkParams) -> None:
    """Voltage recovery needs one DER phase per network node."""
    nodes = np.asarray(net.der_node, dtype=np.int32)
    if nodes.size == 0:
        return
    uniq, counts = np.unique(nodes, return_counts=True)
    if np.any(counts > 1):
        dup_nodes = uniq[counts > 1]
        raise ValueError(
            "Voltage least-squares recovery requires a unique der_node per DER phase; "
            f"found duplicate network nodes {dup_nodes.tolist()}."
        )


def batch_Y_LL(session: OPFData, branch_status: jnp.ndarray) -> jnp.ndarray:
    """Topology-aware ``Y_LL`` for LS residuals (benchmark-only; not on ``TopologyData``)."""
    net = session.net
    status = jnp.asarray(branch_status, dtype=jnp.int32)
    if status.ndim == 1:
        status = status[None, :]
    if status.shape[-1] == net.n_branch:
        status = status[:, net.changing_switches]

    backend = session.topology_backend
    if hasattr(backend, "static"):
        from core.topology.slim import _stamp_Y_LL_slim
        from core.topology.woodbury import _expand_compact

        static = backend.static

        def one_row(compact_row: jnp.ndarray) -> jnp.ndarray:
            full = _expand_compact(static, compact_row)
            return _stamp_Y_LL_slim(static, full)

        return jax.vmap(one_row)(status)

    mats = net.topology_matrices_batch(status)
    y_scaled = mats[0]
    if y_scaled.ndim == 2:
        return y_scaled[: net.n_net, : net.n_net][None, ...]
    return y_scaled[:, : net.n_net, : net.n_net]


def _voltage_to_flat(v_net: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([v_net.real, v_net.imag], axis=-1)


def _flat_to_voltage(v_flat: jnp.ndarray, n_net: int) -> jnp.ndarray:
    return v_flat[..., :n_net] + 1j * v_flat[..., n_net:]


def recover_der_from_voltage(
    net: NetworkParams,
    v_net: jnp.ndarray,
    Y_LL: jnp.ndarray,
    w: jnp.ndarray,
    s_load: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Recover DER controls from non-slack voltage via AC injection physics."""
    dv = v_net - w
    if Y_LL.ndim == 3:
        yv = jnp.einsum("bmn,bn->bm", Y_LL, dv)
    else:
        yv = Y_LL @ dv
    s_calc = v_net * jnp.conj(yv)
    s_node_der = s_calc + s_load
    if v_net.ndim == 1:
        pder = s_node_der[net.der_node].real
        qder = s_node_der[net.der_node].imag
    else:
        pder = s_node_der[:, net.der_node].real
        qder = s_node_der[:, net.der_node].imag
    return pder, qder, s_calc


def make_voltage_residual_fun(
    net: NetworkParams,
    eq_weight: float,
    ineq_weight: float,
) -> Callable[..., jnp.ndarray]:
    """Constraint-projection residual for ``jaxopt.LevenbergMarquardt``."""
    n_net = net.n_net

    def residual(
        v_flat: jnp.ndarray,
        Y_LL: jnp.ndarray,
        w: jnp.ndarray,
        s_load: jnp.ndarray,
        pder_avai: jnp.ndarray,
        y_0row: jnp.ndarray,
    ) -> jnp.ndarray:
        v_net = _flat_to_voltage(v_flat, n_net)
        pder, qder, s_calc = recover_der_from_voltage(net, v_net, Y_LL, w, s_load)
        s_der = jax.lax.complex(pder, qder)
        s_net = net.compute_net_injection(s_load, s_der)
        eq_c = s_calc - s_net
        eq = jnp.concatenate([eq_c.real, eq_c.imag]) * eq_weight
        x = jnp.concatenate([pder, qder])
        v_full = with_slack_voltage(net, v_net)
        ineq = ac_constraint_violation_vector(net, x, v_full, y_0row, pder_avai)
        hinge = jnp.sqrt(jnp.maximum(ineq, 0.0) + _PROJ_EPS) * ineq_weight
        return jnp.concatenate([eq, hinge])

    return residual


def perturb_der_targets(
    key: jnp.ndarray,
    net: NetworkParams,
    pder_target: jnp.ndarray,
    qder_target: jnp.ndarray,
    pder_avai: jnp.ndarray,
    scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic relative perturbation of target DER controls with projection."""
    key_p, key_q = jax.random.split(key)
    n_der = net.n_der_phase
    p_scale = scale * jnp.maximum(jnp.abs(pder_target), 1e-3)
    q_scale = scale * jnp.maximum(jnp.abs(qder_target), 1e-3)
    p_pert = pder_target + jax.random.normal(key_p, pder_target.shape) * p_scale
    q_pert = qder_target + jax.random.normal(key_q, qder_target.shape) * q_scale
    x_proj = project_der_controls_from_x_batch(
        net,
        jnp.concatenate([p_pert, q_pert], axis=1),
        pder_avai,
        _PROJ_EPS,
    )
    return x_proj[:, :n_der], x_proj[:, n_der:]


@partial(jax.jit, static_argnums=(0, 5))
def _prepare_inputs_jit(
    net: NetworkParams,
    pder_target: jnp.ndarray,
    qder_target: jnp.ndarray,
    pder_avai: jnp.ndarray,
    key: jnp.ndarray,
    scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    p_pert, q_pert = perturb_der_targets(key, net, pder_target, qder_target, pder_avai, scale)
    s_der = jax.lax.complex(p_pert, q_pert)
    return p_pert, q_pert, s_der


@partial(jax.jit, static_argnums=(0, 5))
def _run_slfs_jit(
    net: NetworkParams,
    s_load: jnp.ndarray,
    s_der_init: jnp.ndarray,
    pder_avai: jnp.ndarray,
    topo_v_init: jnp.ndarray,
    fs_config: FeasibilityConfig,
    Y_LL_inv: jnp.ndarray,
    w: jnp.ndarray,
    y_0row: jnp.ndarray,
    branch_status: jnp.ndarray,
):
    from core.utils.types import TopologyData

    topo = TopologyData(
        v_init=topo_v_init,
        Y_0row=y_0row,
        Y_LL_inv=Y_LL_inv,
        w=w,
        branch_status=branch_status,
    )
    return slfs_inference_core(net, s_load, s_der_init, pder_avai, topo, fs_config)


def _create_lm_solver(
    net: NetworkParams,
    lm_maxiter: int,
    lm_tol: float,
    lm_damping: float,
    eq_weight: float,
    ineq_weight: float,
):

    return LevenbergMarquardt(
        residual_fun=make_voltage_residual_fun(net, eq_weight, ineq_weight),
        maxiter=lm_maxiter,
        tol=lm_tol,
        damping_parameter=lm_damping,
        implicit_diff=False,
        materialize_jac=False,
        jit=True,
    )


class _LmBatchRunner:
    """Cache ``jax.vmap`` over per-sample LM solves so tracing happens once."""

    def __init__(self, solver) -> None:
        self.solver = solver

        def one(
            v_flat: jnp.ndarray,
            Y_LL_i: jnp.ndarray,
            w_i: jnp.ndarray,
            s_load_i: jnp.ndarray,
            pder_avai_i: jnp.ndarray,
            y_0row_i: jnp.ndarray,
        ):
            out = solver.run(v_flat, Y_LL_i, w_i, s_load_i, pder_avai_i, y_0row_i)
            resid = jnp.linalg.norm(out.state.residual)
            return out.params, resid, jnp.asarray(out.state.iter_num, dtype=jnp.float64)

        self._vmapped = jax.vmap(one)

    def __call__(
        self,
        v0: jnp.ndarray,
        Y_LL: jnp.ndarray,
        w: jnp.ndarray,
        s_load: jnp.ndarray,
        pder_avai: jnp.ndarray,
        y_0row: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._vmapped(_voltage_to_flat(v0), Y_LL, w, s_load, pder_avai, y_0row)


def _warmup_benchmark_solvers(
    session: OPFData,
    cfg: BenchmarkConfig,
    warm_batch: _PreparedBatch,
    solve_pf_batch_fn,
    lm_runner: _LmBatchRunner,
) -> None:
    """JIT-compile solver kernels before timed batches (all perturbation scales)."""
    net = session.net
    topo = warm_batch.topo
    topo.Y_LL_inv.block_until_ready()
    branch_status = jnp.asarray(warm_batch.branch_status)

    for scale in cfg.perturb_scales:
        key = jax.random.PRNGKey(cfg.perturb_seed)
        _, _, s_der_pert = _prepare_inputs_jit(
            net,
            warm_batch.pder_target,
            warm_batch.qder_target,
            warm_batch.pder_avai,
            key,
            scale,
        )
        Y_LL = batch_Y_LL(session, branch_status)
        Y_LL.block_until_ready()
        s_injection = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(
            warm_batch.s_load,
            s_der_pert,
        )
        v0 = solve_pf_batch_fn(
            s_injection,
            warm_batch.v_init,
            topo.Y_LL_inv,
            topo.w,
        )
        v0.block_until_ready()

        slfs_out = _run_slfs_jit(
            net,
            warm_batch.s_load,
            s_der_pert,
            warm_batch.pder_avai,
            v0,
            cfg.fs_config,
            topo.Y_LL_inv,
            topo.w,
            topo.Y_0row,
            topo.branch_status,
        )
        _block_tree(slfs_out)

        lm_out = lm_runner(
            v0,
            Y_LL,
            topo.w,
            warm_batch.s_load,
            warm_batch.pder_avai,
            topo.Y_0row,
        )
        _block_tree(lm_out)


def _quality_metrics(
    net: NetworkParams,
    pder: jnp.ndarray,
    qder: jnp.ndarray,
    v_net: jnp.ndarray,
    s_load: jnp.ndarray,
    Y_LL: jnp.ndarray,
    topo: TopologyData,
    pder_avai: jnp.ndarray,
    pder_target: jnp.ndarray,
    qder_target: jnp.ndarray,
    eq_weight: float,
    ineq_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Post-solve quality metrics (solver residual + core AC eq/ineq violations)."""
    residual_fun = make_voltage_residual_fun(net, eq_weight, ineq_weight)
    v_flat = _voltage_to_flat(v_net)

    def one_residual(
        v_flat_i,
        Y_LL_i,
        w_i,
        s_load_i,
        pder_avai_i,
        y_0row_i,
    ):
        resid = residual_fun(v_flat_i, Y_LL_i, w_i, s_load_i, pder_avai_i, y_0row_i)
        resid_norm = jnp.linalg.norm(resid)
        eq_part = resid[: 4 * net.n_net]
        eq_norm = jnp.linalg.norm(eq_part)
        return resid_norm, eq_norm

    resid_norm, eq_norm = jax.vmap(one_residual)(
        v_flat,
        Y_LL,
        topo.w,
        s_load,
        pder_avai,
        topo.Y_0row,
    )
    batch_size = pder.shape[0]
    obj_weights = jnp.zeros((batch_size, 2), dtype=pder.dtype)
    ac_metrics = compute_ac_metrics_from_voltage(
        net,
        s_load,
        pder,
        qder,
        pder_avai,
        v_net,
        obj_weights,
        topo,
        branch_status=topo.branch_status,
    )
    dist = jnp.sum((pder - pder_target) ** 2 + (qder - qder_target) ** 2, axis=1)
    return (
        resid_norm,
        eq_norm,
        ac_metrics.eq_violation,
        ac_metrics.ineq_violation,
        dist,
    )


def _block_tree(tree) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def benchmark_batch(
    session: OPFData,
    batch,
    cfg: BenchmarkConfig,
    perturb_scale: float,
    solve_pf_batch_fn,
    lm_runner: _LmBatchRunner,
) -> dict[str, Any]:
    """Run one timed batch for SLFS and voltage LM with shared perturbed inputs."""
    net = session.net
    key = jax.random.PRNGKey(cfg.perturb_seed)

    topo = batch.topo
    topo.Y_LL_inv.block_until_ready()

    prep_start = time.perf_counter()
    p_pert, q_pert, s_der_pert = _prepare_inputs_jit(
        net,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        key,
        perturb_scale,
    )
    Y_LL = batch_Y_LL(session, batch.branch_status)
    s_injection = jax.vmap(net.compute_net_injection, in_axes=(0, 0))(
        batch.s_load, s_der_pert
    )
    v0 = solve_pf_batch_fn(s_injection, batch.v_init, topo.Y_LL_inv, topo.w)
    v0.block_until_ready()
    prep_time = time.perf_counter() - prep_start

    topo_slfs = topo._replace(v_init=v0)

    slfs_start = time.perf_counter()
    slfs_out = _run_slfs_jit(
        net,
        batch.s_load,
        s_der_pert,
        batch.pder_avai,
        topo_slfs.v_init,
        cfg.fs_config,
        topo.Y_LL_inv,
        topo.w,
        topo.Y_0row,
        topo.branch_status,
    )
    _block_tree(slfs_out)
    slfs_time = time.perf_counter() - slfs_start

    lm_start = time.perf_counter()
    v_lm_flat, lm_resid, lm_iters = lm_runner(
        v0,
        Y_LL,
        topo.w,
        batch.s_load,
        batch.pder_avai,
        topo.Y_0row,
    )
    _block_tree((v_lm_flat, lm_resid, lm_iters))
    lm_time = time.perf_counter() - lm_start

    v_lm = _flat_to_voltage(v_lm_flat, net.n_net)
    p_lm, q_lm, _ = recover_der_from_voltage(
        net, v_lm, Y_LL, topo.w, batch.s_load
    )

    metrics_start = time.perf_counter()
    (
        slfs_resid,
        slfs_eq,
        slfs_eq_vio,
        slfs_ineq,
        slfs_dist,
    ) = _quality_metrics(
        net,
        slfs_out[1],
        slfs_out[2],
        slfs_out[3],
        batch.s_load,
        Y_LL,
        topo,
        batch.pder_avai,
        batch.pder_target,
        batch.qder_target,
        cfg.eq_weight,
        cfg.ineq_weight,
    )
    (
        lm_resid_q,
        lm_eq,
        lm_eq_vio,
        lm_ineq,
        lm_dist,
    ) = _quality_metrics(
        net,
        p_lm,
        q_lm,
        v_lm,
        batch.s_load,
        Y_LL,
        topo,
        batch.pder_avai,
        batch.pder_target,
        batch.qder_target,
        cfg.eq_weight,
        cfg.ineq_weight,
    )
    metrics_time = time.perf_counter() - metrics_start

    batch_size = int(batch.s_load.shape[0])
    return {
        "batch_size": batch_size,
        "perturb_scale": perturb_scale,
        "prep_time": prep_time,
        "slfs_solver_time": slfs_time,
        "lm_solver_time": lm_time,
        "metrics_time": metrics_time,
        "slfs_solver_time_per_sample": slfs_time / batch_size,
        "lm_solver_time_per_sample": lm_time / batch_size,
        "slfs_residual_norm_mean": float(slfs_resid.mean()),
        "lm_residual_norm_mean": float(lm_resid_q.mean()),
        "slfs_eq_norm_mean": float(slfs_eq.mean()),
        "lm_eq_norm_mean": float(lm_eq.mean()),
        "slfs_max_eq_mean": float(slfs_eq_vio.mean()),
        "lm_max_eq_mean": float(lm_eq_vio.mean()),
        "slfs_max_ineq_mean": float(slfs_ineq.mean()),
        "lm_max_ineq_mean": float(lm_ineq.mean()),
        "slfs_target_distance_mean": float(slfs_dist.mean()),
        "lm_target_distance_mean": float(lm_dist.mean()),
        "lm_iter_mean": float(lm_iters.mean()),
        "speedup_slfs_over_lm": lm_time / slfs_time if slfs_time > 0 else float("inf"),
    }


@dataclass
class _PreparedBatch:
    s_load: jnp.ndarray
    pder_avai: jnp.ndarray
    pder_target: jnp.ndarray
    qder_target: jnp.ndarray
    v_init: jnp.ndarray
    branch_status: jnp.ndarray
    topo: TopologyData


def _load_batches(session: OPFData, batch_size: int, test_samples: int | None):
    test_batched = (
        session.make_test_dataset(samples=test_samples)
        .batch(batch_size, drop_remainder=True)
        .prefetch(1)
    )
    for inputs_np, targets_np in test_batched.as_numpy_iterator():
        inputs_np = np.asarray(inputs_np)
        targets_np = np.asarray(targets_np)
        topo = precompute_topology(session, inputs_np)
        inputs_j = jnp.asarray(inputs_np)
        targets_j = jnp.asarray(targets_np)
        batch = prepare_batch_data(inputs_j, targets_j, session, topo=topo)
        yield _PreparedBatch(
            s_load=batch.s_load,
            pder_avai=batch.pder_avai,
            pder_target=batch.pder_target,
            qder_target=batch.qder_target,
            v_init=batch.v_init,
            branch_status=batch.branch_status,
            topo=topo,
        )


def run_benchmark(
    session: OPFData,
    cfg: BenchmarkConfig,
    *,
    power_flow_solver: str = PowerFlowSolver.FIXED_POINT.value,
    grid: str | None = None,
) -> dict[str, Any]:
    """Full benchmark over test HDF5 batches and perturbation scales."""
    assert_der_recovery_unique(session.net)
    solve_pf_batch_fn = get_power_flow_batch_fn(power_flow_solver)
    lm_solver = _create_lm_solver(
        session.net,
        cfg.lm_maxiter,
        cfg.lm_tol,
        cfg.lm_damping,
        cfg.eq_weight,
        cfg.ineq_weight,
    )
    lm_runner = _LmBatchRunner(lm_solver)

    warm = next(iter(_load_batches(session, cfg.batch_size, cfg.test_samples)))
    _warmup_benchmark_solvers(session, cfg, warm, solve_pf_batch_fn, lm_runner)

    all_results: dict[str, Any] = {
        "grid": grid,
        "power_flow_solver": power_flow_solver,
        "n_net": int(session.net.n_net),
        "n_der_phase": int(session.net.n_der_phase),
        "batch_size": cfg.batch_size,
        "perturb_scales": list(cfg.perturb_scales),
        "scales": {},
    }

    for scale in cfg.perturb_scales:
        batch_metrics = []
        totals = {
            "prep_time": 0.0,
            "slfs_solver_time": 0.0,
            "lm_solver_time": 0.0,
            "metrics_time": 0.0,
            "num_samples": 0,
        }
        for batch in _load_batches(session, cfg.batch_size, cfg.test_samples):
            metrics = benchmark_batch(
                session,
                batch,
                cfg,
                scale,
                solve_pf_batch_fn,
                lm_runner,
            )
            batch_metrics.append(metrics)
            totals["prep_time"] += metrics["prep_time"]
            totals["slfs_solver_time"] += metrics["slfs_solver_time"]
            totals["lm_solver_time"] += metrics["lm_solver_time"]
            totals["metrics_time"] += metrics["metrics_time"]
            totals["num_samples"] += metrics["batch_size"]

        n = max(totals["num_samples"], 1)
        all_results["scales"][str(scale)] = {
            "num_samples": totals["num_samples"],
            "num_batches": len(batch_metrics),
            "avg_prep_time_per_sample": totals["prep_time"] / n,
            "avg_slfs_solver_time_per_sample": totals["slfs_solver_time"] / n,
            "avg_lm_solver_time_per_sample": totals["lm_solver_time"] / n,
            "avg_metrics_time_per_sample": totals["metrics_time"] / n,
            "speedup_slfs_over_lm": totals["lm_solver_time"] / max(totals["slfs_solver_time"], 1e-12),
            "batch_metrics": batch_metrics,
        }

    return all_results


def print_benchmark_report(results: dict[str, Any]) -> None:
    """Print concise timing/scalability summary."""
    print(f"Network: n_net={results['n_net']}  n_der_phase={results['n_der_phase']}")
    print(f"Batch size: {results['batch_size']}")
    print()
    header = (
        f"{'scale':>8}  {'slfs_ms':>10}  {'lm_ms':>10}  {'lm/slfs':>8}  "
        f"{'slfs_res':>10}  {'lm_res':>10}  "
        f"{'slfs_eq':>10}  {'lm_eq':>10}  {'slfs_ineq':>10}  {'lm_ineq':>10}"
    )
    print(header)
    print("-" * len(header))
    for scale, block in results["scales"].items():
        slfs_ms = 1e3 * block["avg_slfs_solver_time_per_sample"]
        lm_ms = 1e3 * block["avg_lm_solver_time_per_sample"]
        ratio = block["speedup_slfs_over_lm"]
        b0 = block["batch_metrics"][0] if block["batch_metrics"] else {}
        print(
            f"{float(scale):8.3f}  {slfs_ms:10.3f}  {lm_ms:10.3f}  {ratio:8.3f}  "
            f"{b0.get('slfs_residual_norm_mean', float('nan')):10.3e}  "
            f"{b0.get('lm_residual_norm_mean', float('nan')):10.3e}  "
            f"{b0.get('slfs_max_eq_mean', float('nan')):10.3e}  "
            f"{b0.get('lm_max_eq_mean', float('nan')):10.3e}  "
            f"{b0.get('slfs_max_ineq_mean', float('nan')):10.3e}  "
            f"{b0.get('lm_max_ineq_mean', float('nan')):10.3e}"
        )
    print()
    print("Headline columns are solver-only ms/sample (excludes topo inverse and input prep).")


def _is_240_bus_case(grid: str | None, data_json: str) -> bool:
    if grid in _GRIDS_USING_NEWTON_GMRES:
        return True
    return _240_CASE_MARKER in Path(data_json).as_posix()


def resolve_power_flow_solver(
    *,
    grid: str | None,
    data_json: str,
    explicit: str | None,
) -> str:
    """Pick PF solver for shared v0 init; grid 240 defaults to newton_gmres."""
    if explicit is not None:
        return explicit
    if _is_240_bus_case(grid, data_json):
        return PowerFlowSolver.NEWTON_GMRES.value
    return _DEFAULT_PF_SOLVER


def _resolve_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.grid:
        preset = GRID_PRESETS.get(args.grid)
        if preset is None:
            valid = ", ".join(sorted(GRID_PRESETS))
            raise ValueError(f"Unknown --grid {args.grid!r}; expected one of: {valid}")
        root = _ROOT
        return (
            str(root / preset["train_h5"]),
            str(root / preset["test_h5"]),
            str(root / preset["data_json"]),
        )
    if not (args.train_h5 and args.test_h5 and args.data_json):
        raise ValueError("Provide --grid PRESET or all of --train_h5, --test_h5, --data_json")
    return args.train_h5, args.test_h5, args.data_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SLFS vs JAXopt voltage least-squares on perturbed OPF targets"
    )
    parser.add_argument("--grid", type=str, default=None, choices=sorted(GRID_PRESETS))
    parser.add_argument("--train_h5", type=str, default=None)
    parser.add_argument("--test_h5", type=str, default=None)
    parser.add_argument("--data_json", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_samples", type=int, default=64)
    parser.add_argument(
        "--perturb_scales",
        type=float,
        nargs="+",
        default=list(_DEFAULT_PERTURB_SCALES),
    )
    parser.add_argument("--perturb_seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lin_steps", type=int, default=2)
    parser.add_argument("--fs_tol", type=float, default=1e-6)
    parser.add_argument("--fs_maxiter", type=int, default=100)
    parser.add_argument("--fs_rho", type=float, default=0.0)
    parser.add_argument("--fs_damp", type=float, default=1.0)
    parser.add_argument("--lm_maxiter", type=int, default=30)
    parser.add_argument("--lm_tol", type=float, default=1e-6)
    parser.add_argument("--lm_damping", type=float, default=1e-6)
    parser.add_argument("--eq_weight", type=float, default=1.0)
    parser.add_argument("--ineq_weight", type=float, default=1.0)
    parser.add_argument(
        "--power_flow_solver",
        type=str,
        default=None,
        choices=[m.value for m in PowerFlowSolver],
        help=(
            "PF solver for shared v0 init (default: fixed_point; "
            "grid 240 / 240bus_Master uses newton_gmres)."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON output path under analysis/ or experiments/",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    configure_tensorflow_for_data_loading()
    jax.config.update("jax_enable_x64", True)

    args = parse_args(argv)
    set_random_seeds(args.seed)

    train_h5, test_h5, data_json = _resolve_paths(args)
    for path in (train_h5, test_h5, data_json):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    power_flow_solver = resolve_power_flow_solver(
        grid=args.grid,
        data_json=data_json,
        explicit=args.power_flow_solver,
    )
    if _is_240_bus_case(args.grid, data_json) and args.power_flow_solver is None:
        print(
            "Note: 240-bus case uses newton_gmres for PF init "
            "(fixed_point does not converge reliably on this feeder)."
        )

    build = OPFDataBuildOptions(power_flow_solver=power_flow_solver)
    session = load_opf_data(
        train_h5=train_h5,
        data_json=data_json,
        test_h5_paths={"feasible": test_h5},
        build=build,
    )

    fs_config = FeasibilityConfig(
        fs_tol=args.fs_tol,
        fs_maxiter=args.fs_maxiter,
        lin_steps=args.lin_steps,
        diff_steps=0,
        fs_rho=args.fs_rho,
        fs_damp=args.fs_damp,
        power_flow_solver=power_flow_solver,
    ).validate()

    cfg = BenchmarkConfig(
        perturb_scales=tuple(args.perturb_scales),
        perturb_seed=args.perturb_seed,
        batch_size=args.batch_size,
        test_samples=args.test_samples,
        fs_config=fs_config,
        lm_maxiter=args.lm_maxiter,
        lm_tol=args.lm_tol,
        lm_damping=args.lm_damping,
        eq_weight=args.eq_weight,
        ineq_weight=args.ineq_weight,
    )

    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")
    print(f"Case: {data_json}")
    print(f"Test HDF5: {test_h5}")
    print(f"Power-flow solver: {power_flow_solver}")
    print()

    results = run_benchmark(
        session,
        cfg,
        power_flow_solver=power_flow_solver,
        grid=args.grid,
    )
    print_benchmark_report(results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote {out_path}")

    return results


if __name__ == "__main__":
    main()
