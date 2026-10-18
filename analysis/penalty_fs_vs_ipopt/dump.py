#!/usr/bin/env python3
"""Dump penalty_fs pred / after-SLFS DER and |V| vs Ipopt on feasible tests.

Loads ``results/penalty_fs/{grid}_penalty_fs_rho1e5/seed_1/models/last``, runs
NN + project and ``post_process_slfs``, and writes NPZ + summary JSON under
``analysis/penalty_fs_vs_ipopt/results/{grid}/``.

Example (repo root, conda ``new_opt``)::

    python analysis/penalty_fs_vs_ipopt/dump.py --grids 13 123 240 906 8500
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.nn.input import nn_features_for_apply
from core.nn.model import create_train_state
from core.topology.batch import precompute_topology
from core.utils.checkpoint import (
    load_checkpoint_config,
    resolve_checkpoint_model_spec,
    restore_checkpoint_state,
    to_json_serializable,
)
from core.utils.datasets import prepare_checkpoint_test_data
from core.utils.runtime import (
    clear_gpu_memory,
    configure_tensorflow_for_data_loading,
    set_protobuf_implementation,
    set_random_seeds,
)
from core.utils.types import OPFDataBuildOptions
from core.utils.workflow import prepare_batch_data
from methods.penalty_fs.trainer import (
    _project_predictions,
    compute_norms,
    post_process_slfs,
)

from analysis.batch_size_inference_time.benchmark import (
    DEFAULT_CASE_SUFFIX,
    DEFAULT_FS_RHO,
    DEFAULT_FS_STEP_FRACTION,
    DEFAULT_N_VOLTAGE_ROWS,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_VOLTAGE_WEIGHT,
    GRID_PRESETS,
    build_fs_config,
    load_fixed_sample_pool,
    resolve_grid,
    resolve_power_flow_solver,
    resolve_run_checkpoint,
)
from analysis.penalty_fs_vs_ipopt.flows import flow_terms

set_protobuf_implementation()
configure_tensorflow_for_data_loading()

DEFAULT_GRIDS = ("13", "123", "240", "906", "8500")
DEFAULT_SEED = 1
DEFAULT_BATCH_SIZE = 32
DEFAULT_OUTPUT_ROOT = Path("analysis/penalty_fs_vs_ipopt/results")
DEFAULT_TEST_VARIANT = "feasible"


def _error_stats(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / max-abs over all elements."""
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _split_batches(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slice into batches; keep a final ragged batch if needed."""
    n = int(inputs.shape[0])
    if n < 1:
        raise ValueError("Need at least one sample")
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append((inputs[start:end], targets[start:end]))
    return batches


def dump_grid(
    *,
    grid: str,
    results_root: str,
    case_suffix: str,
    seed: int,
    batch_size: int,
    test_samples: int | None,
    output_root: Path,
    woodbury_ir_steps: str = "auto",
    woodbury_batch_mode: str = "vmap",
    fs_rho: float = DEFAULT_FS_RHO,
    voltage_weight: float = DEFAULT_VOLTAGE_WEIGHT,
    fs_step_fraction: float = DEFAULT_FS_STEP_FRACTION,
    n_voltage_rows: int = DEFAULT_N_VOLTAGE_ROWS,
    train_samples: int | None = None,
) -> dict[str, Any]:
    """Run penalty_fs inference and write NPZ + summary for one grid."""
    preset = resolve_grid(grid)
    case_dir = preset["case_dir"]
    set_random_seeds(seed)
    clear_gpu_memory()

    checkpoint_path = resolve_run_checkpoint(
        checkpoint_path=None,
        results_root=results_root,
        case_dir=case_dir,
        case_suffix=case_suffix,
        seed=seed,
    )
    checkpoint_path, model_spec = resolve_checkpoint_model_spec(checkpoint_path)
    checkpoint_cfg = load_checkpoint_config(checkpoint_path)
    power_flow_solver = resolve_power_flow_solver(grid, checkpoint_cfg)

    build = OPFDataBuildOptions(
        power_flow_solver=power_flow_solver,
        woodbury_ir_steps=woodbury_ir_steps,
        woodbury_batch_mode=woodbury_batch_mode,
    )
    test_h5 = str((_ROOT / preset["test_h5"]).resolve())
    data, train_dataset = prepare_checkpoint_test_data(
        train_h5=str((_ROOT / preset["train_h5"]).resolve()),
        data_json=str((_ROOT / preset["data_json"]).resolve()),
        test_h5_paths={DEFAULT_TEST_VARIANT: test_h5},
        build=build,
        train_samples=train_samples,
    )

    model_type = checkpoint_cfg.get("model_type", "mlp")
    state = restore_checkpoint_state(
        data=data,
        train_dataset=train_dataset,
        checkpoint_path=checkpoint_path,
        seed=seed,
        create_state_fn=partial(create_train_state, model_type=model_type),
        model_spec=model_spec,
        dout=2 * data.net.n_der_phase,
    )

    fs_config = build_fs_config(
        checkpoint_cfg,
        power_flow_solver=power_flow_solver,
        fs_rho=fs_rho,
        voltage_weight=voltage_weight,
        fs_step_fraction=fs_step_fraction,
        n_voltage_rows=n_voltage_rows,
        lin_steps=None,
        fs_maxiter=None,
        fs_tol=None,
    )

    # Cap by available test size when --test-samples omitted.
    with h5py.File(data.test_h5_path, "r") as f:
        n_available = int(f["input/pd"].shape[0])
    n_use = n_available if test_samples is None else min(int(test_samples), n_available)
    inputs_pool, targets_pool = load_fixed_sample_pool(data, n_use)
    host_batches = _split_batches(inputs_pool, targets_pool, batch_size)

    net = data.net
    n_der = int(net.n_der_phase)

    @jax.jit
    def inference_step(inputs):
        return state.apply_fn(state.params, inputs)

    pder_pred_chunks: list[np.ndarray] = []
    qder_pred_chunks: list[np.ndarray] = []
    pder_fs_chunks: list[np.ndarray] = []
    qder_fs_chunks: list[np.ndarray] = []
    pder_ipopt_chunks: list[np.ndarray] = []
    qder_ipopt_chunks: list[np.ndarray] = []
    vmag_pred_chunks: list[np.ndarray] = []
    vmag_fs_chunks: list[np.ndarray] = []
    vmag_ipopt_chunks: list[np.ndarray] = []
    # Flow dumps (Penalty+SLFS and Ipopt only): key -> list of per-batch arrays.
    flow_chunks: dict[str, list[np.ndarray]] = {}
    branch_mask = np.asarray(net.branch_valid)

    @jax.jit
    def flow_step(v_fs, v_ipopt, y_0row):
        return flow_terms(net, v_fs, y_0row), flow_terms(net, v_ipopt, y_0row)

    print(
        f"[{grid}] checkpoint={checkpoint_path} "
        f"pf={power_flow_solver} samples={n_use} batch_size={batch_size}"
    )

    for bi, (inputs_np, targets_np) in enumerate(host_batches):
        topo = precompute_topology(data, inputs_np)
        inputs_j = jnp.asarray(inputs_np)
        targets_j = jnp.asarray(targets_np)
        batch = prepare_batch_data(inputs_j, targets_j, data, topo=topo)
        nn_inputs = nn_features_for_apply(
            inputs_j, data, use_compact=data.use_compact_nn
        )

        x_pred_raw = inference_step(nn_inputs)
        x_pred = _project_predictions(x_pred_raw, batch.pder_avai, net)
        x_fs, v_fs, v_pred = post_process_slfs(
            net,
            batch,
            x_pred[:, :n_der] + 1j * x_pred[:, n_der:],
            x_pred,
            fs_config,
            topo,
        )
        x_fs.block_until_ready()
        v_fs.block_until_ready()
        v_pred.block_until_ready()

        flows_fs, flows_ipopt = flow_step(v_fs, batch.v_target, topo.Y_0row)
        for stage, flows in (("fs", flows_fs), ("ipopt", flows_ipopt)):
            for name, val in flows.items():
                val = np.asarray(val)
                if name.startswith("branch_"):
                    # Valid phase-terms only; float32 bounds the 8500-node file size.
                    val = val[:, branch_mask].astype(np.float32)
                flow_chunks.setdefault(f"{name}_{stage}", []).append(val)

        pder_pred_chunks.append(np.asarray(x_pred[:, :n_der], dtype=np.float64))
        qder_pred_chunks.append(np.asarray(x_pred[:, n_der:], dtype=np.float64))
        pder_fs_chunks.append(np.asarray(x_fs[:, :n_der], dtype=np.float64))
        qder_fs_chunks.append(np.asarray(x_fs[:, n_der:], dtype=np.float64))
        pder_ipopt_chunks.append(np.asarray(batch.pder_target, dtype=np.float64))
        qder_ipopt_chunks.append(np.asarray(batch.qder_target, dtype=np.float64))
        vmag_pred_chunks.append(np.asarray(np.abs(v_pred), dtype=np.float64))
        vmag_fs_chunks.append(np.asarray(np.abs(v_fs), dtype=np.float64))
        vmag_ipopt_chunks.append(np.asarray(np.abs(batch.v_target), dtype=np.float64))

        if (bi + 1) % 10 == 0 or bi + 1 == len(host_batches):
            print(f"  batch {bi + 1}/{len(host_batches)}")

    arrays = {
        "pder_pred": np.concatenate(pder_pred_chunks, axis=0),
        "qder_pred": np.concatenate(qder_pred_chunks, axis=0),
        "pder_fs": np.concatenate(pder_fs_chunks, axis=0),
        "qder_fs": np.concatenate(qder_fs_chunks, axis=0),
        "pder_ipopt": np.concatenate(pder_ipopt_chunks, axis=0),
        "qder_ipopt": np.concatenate(qder_ipopt_chunks, axis=0),
        "vmag_pred": np.concatenate(vmag_pred_chunks, axis=0),
        "vmag_fs": np.concatenate(vmag_fs_chunks, axis=0),
        "vmag_ipopt": np.concatenate(vmag_ipopt_chunks, axis=0),
    }

    # Match trainer.compute_norms aggregation used in testing_results JSON.
    pder_norms_pred = compute_norms(
        jnp.asarray(arrays["pder_pred"]), jnp.asarray(arrays["pder_ipopt"])
    )
    qder_norms_pred = compute_norms(
        jnp.asarray(arrays["qder_pred"]), jnp.asarray(arrays["qder_ipopt"])
    )
    pder_norms_fs = compute_norms(
        jnp.asarray(arrays["pder_fs"]), jnp.asarray(arrays["pder_ipopt"])
    )
    qder_norms_fs = compute_norms(
        jnp.asarray(arrays["qder_fs"]), jnp.asarray(arrays["qder_ipopt"])
    )

    error_summary = {
        "pred": {
            "pder": _error_stats(arrays["pder_pred"], arrays["pder_ipopt"]),
            "qder": _error_stats(arrays["qder_pred"], arrays["qder_ipopt"]),
            "vmag": _error_stats(arrays["vmag_pred"], arrays["vmag_ipopt"]),
        },
        "fs": {
            "pder": _error_stats(arrays["pder_fs"], arrays["pder_ipopt"]),
            "qder": _error_stats(arrays["qder_fs"], arrays["qder_ipopt"]),
            "vmag": _error_stats(arrays["vmag_fs"], arrays["vmag_ipopt"]),
        },
    }

    out_dir = output_root / case_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"seed_{seed}_{DEFAULT_TEST_VARIANT}.npz"
    summary_path = out_dir / f"seed_{seed}_{DEFAULT_TEST_VARIANT}_summary.json"

    np.savez_compressed(npz_path, **arrays)

    flows_path = out_dir / f"seed_{seed}_{DEFAULT_TEST_VARIANT}_flows.npz"
    flow_arrays = {k: np.concatenate(v, axis=0) for k, v in flow_chunks.items()}
    np.savez_compressed(flows_path, **flow_arrays)

    summary: dict[str, Any] = {
        "grid": grid,
        "case_dir": case_dir,
        "case_suffix": case_suffix,
        "seed": seed,
        "test_variant": DEFAULT_TEST_VARIANT,
        "test_h5": test_h5,
        "checkpoint_path": str(checkpoint_path),
        "power_flow_solver": power_flow_solver,
        "num_samples": int(arrays["pder_pred"].shape[0]),
        "n_der_phase": n_der,
        "n_net": int(arrays["vmag_ipopt"].shape[1]),
        "batch_size": batch_size,
        "fs_config": {
            "lin_steps": fs_config.lin_steps,
            "diff_steps": fs_config.diff_steps,
            "fs_tol": fs_config.fs_tol,
            "fs_maxiter": fs_config.fs_maxiter,
            "fs_rho": fs_config.fs_rho,
            "voltage_weight": fs_config.voltage_weight,
            "fs_step_fraction": fs_config.fs_step_fraction,
            "n_voltage_rows": fs_config.n_voltage_rows,
            "power_flow_solver": fs_config.power_flow_solver,
        },
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "error_summary": error_summary,
        "compute_norms": {
            "pred": {"pder": pder_norms_pred, "qder": qder_norms_pred},
            "fs": {"pder": pder_norms_fs, "qder": qder_norms_fs},
        },
        "npz_path": str(npz_path),
        "flows_npz_path": str(flows_path),
    }
    summary_path.write_text(
        json.dumps(to_json_serializable(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[{grid}] wrote {npz_path}")
    print(f"[{grid}] wrote {flows_path}")
    print(f"[{grid}] wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump penalty_fs pred/FS DER and |V| vs Ipopt (feasible)"
    )
    parser.add_argument(
        "--grids",
        nargs="+",
        default=list(DEFAULT_GRIDS),
        choices=sorted(GRID_PRESETS.keys()),
        help="Grid keys (default: 13 123 240 906 8500)",
    )
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--case-suffix", type=str, default=DEFAULT_CASE_SUFFIX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Optional cap on feasible samples (default: all)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for per-grid NPZ/summary outputs",
    )
    parser.add_argument("--woodbury-ir-steps", type=str, default="auto",
                        choices=["auto", "0", "1", "2"])
    parser.add_argument("--woodbury-batch-mode", type=str, default="vmap",
                        choices=["vmap", "map"])
    parser.add_argument("--fs-rho", type=float, default=DEFAULT_FS_RHO)
    parser.add_argument("--voltage-weight", type=float, default=DEFAULT_VOLTAGE_WEIGHT)
    parser.add_argument(
        "--fs-step-fraction", type=float, default=DEFAULT_FS_STEP_FRACTION
    )
    parser.add_argument("--n-voltage-rows", type=int, default=DEFAULT_N_VOLTAGE_ROWS)
    parser.add_argument("--train-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (_ROOT / output_root).resolve()

    print(f"JAX devices: {jax.devices()}")
    for grid in args.grids:
        dump_grid(
            grid=grid,
            results_root=args.results_root,
            case_suffix=args.case_suffix,
            seed=args.seed,
            batch_size=args.batch_size,
            test_samples=args.test_samples,
            output_root=output_root,
            woodbury_ir_steps=args.woodbury_ir_steps,
            woodbury_batch_mode=args.woodbury_batch_mode,
            fs_rho=args.fs_rho,
            voltage_weight=args.voltage_weight,
            fs_step_fraction=args.fs_step_fraction,
            n_voltage_rows=args.n_voltage_rows,
            train_samples=args.train_samples,
        )


if __name__ == "__main__":
    main()
