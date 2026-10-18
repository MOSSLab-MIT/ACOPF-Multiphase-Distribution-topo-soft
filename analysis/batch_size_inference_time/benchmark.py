#!/usr/bin/env python3
"""Benchmark penalty_fs deploy inference time vs batch size.

Primary metric: mean wall time per batch for NN + project + PF +
``post_process_slfs`` (same scope as ``avg_prediction_time_per_batch`` in
``methods/penalty_fs/trainer.py``). Topology inverse is recorded separately
and excluded from the primary pipeline time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.feasibility.slfs_config import FeasibilityConfig
from core.grids.power_flow import PowerFlowSolver
from core.nn.input import nn_features_for_apply
from core.nn.model import create_train_state
from core.topology.batch import precompute_topology
from core.utils.checkpoint import (
    load_checkpoint_config,
    resolve_checkpoint_model_spec,
    resolve_checkpoint_path,
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
    compute_voltage,
    post_process_slfs,
)

set_protobuf_implementation()
configure_tensorflow_for_data_loading()

BENCHMARK_NAME = "batch_size_inference_time"
DEFAULT_CASE_SUFFIX = "penalty_fs_rho1e5"
DEFAULT_RESULTS_ROOT = "results/penalty_fs"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_TEST_SAMPLES = 256
DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10
DEFAULT_SEED = 1
DEFAULT_FS_RHO = 0.0
DEFAULT_VOLTAGE_WEIGHT = 8.0
DEFAULT_FS_STEP_FRACTION = 2.5
DEFAULT_N_VOLTAGE_ROWS = 0

TIMED_SCOPE = "raw_nn_project_pf_plus_post_process_slfs"
EXCLUDES = ("topology_inverse", "quality_metrics_pf")

# 240-bus PF does not converge with fixed-point; match train_jobs_penalty_fs.yaml.
GRID_POWER_FLOW_SOLVER_OVERRIDES: dict[str, str] = {
    "240": PowerFlowSolver.NEWTON_GMRES.value,
}

GRID_PRESETS: dict[str, dict[str, str]] = {
    "13": {
        "case_dir": "IEEE13Nodeckt",
        "train_h5": "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_training.h5",
        "test_h5": "datagen_topo/datasets/IEEE13Nodeckt/IEEE13Nodeckt_test_feasible.h5",
        "data_json": "datagen_topo/datasets/IEEE13Nodeckt/network_config.json",
    },
    "123": {
        "case_dir": "IEEE123Master",
        "train_h5": "datagen_topo/datasets/IEEE123Master/IEEE123Master_training.h5",
        "test_h5": "datagen_topo/datasets/IEEE123Master/IEEE123Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/IEEE123Master/network_config.json",
    },
    "240": {
        "case_dir": "240bus_Master",
        "train_h5": "datagen_topo/datasets/240bus_Master/240bus_Master_training.h5",
        "test_h5": "datagen_topo/datasets/240bus_Master/240bus_Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/240bus_Master/network_config.json",
    },
    "906": {
        "case_dir": "906bus_Master",
        "train_h5": "datagen_topo/datasets/906bus_Master/906bus_Master_training.h5",
        "test_h5": "datagen_topo/datasets/906bus_Master/906bus_Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/906bus_Master/network_config.json",
    },
    "8500": {
        "case_dir": "8500_Master",
        "train_h5": "datagen_topo/datasets/8500_Master/8500_Master_training.h5",
        "test_h5": "datagen_topo/datasets/8500_Master/8500_Master_test_feasible.h5",
        "data_json": "datagen_topo/datasets/8500_Master/network_config.json",
    },
}


def resolve_grid(grid: str) -> dict[str, str]:
    if grid not in GRID_PRESETS:
        raise ValueError(f"Unknown grid {grid!r}; choose from {sorted(GRID_PRESETS)}")
    return GRID_PRESETS[grid]


def resolve_power_flow_solver(
    grid: str,
    checkpoint_cfg: dict[str, Any],
) -> str:
    """Return PF solver for benchmark inference (grid overrides beat checkpoint)."""
    if grid in GRID_POWER_FLOW_SOLVER_OVERRIDES:
        return GRID_POWER_FLOW_SOLVER_OVERRIDES[grid]
    return str(
        checkpoint_cfg.get("power_flow_solver", PowerFlowSolver.FIXED_POINT.value)
    )


def _jax_device_string() -> str:
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if gpus:
        return str(gpus[0])
    return str(jax.devices()[0])


def _latest_checkpoint_epoch(models_dir: Path) -> Path:
    """Pick ``checkpoint_epoch_*`` with the largest epoch number (not lexicographic)."""
    candidates = [
        p for p in models_dir.glob("checkpoint_epoch_*") if p.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_epoch_* under {models_dir}")

    def _epoch_num(path: Path) -> int:
        suffix = path.name.removeprefix("checkpoint_epoch_")
        return int(suffix)

    return max(candidates, key=_epoch_num)


def resolve_models_checkpoint(models_dir: Path) -> Path:
    """Prefer ``models/last/`` (final train save); else highest ``checkpoint_epoch_*``."""
    last_dir = models_dir / "last"
    if last_dir.is_dir():
        # ``last/`` usually holds ``checkpoint_epoch_<N>/``; resolve that child.
        if (last_dir / "config.json").exists():
            return last_dir
        if any(last_dir.glob("checkpoint_epoch_*")):
            return _latest_checkpoint_epoch(last_dir)
        return resolve_checkpoint_path(last_dir)
    return _latest_checkpoint_epoch(models_dir)


def resolve_run_checkpoint(
    *,
    checkpoint_path: str | None,
    results_root: str,
    case_dir: str,
    case_suffix: str,
    seed: int,
) -> Path:
    """Resolve an explicit path or ``{results_root}/{case}_{suffix}/seed_{N}/models``."""
    if checkpoint_path:
        path = Path(checkpoint_path).resolve()
        # If the user pointed at models/, prefer last/ then highest epoch.
        if path.is_dir() and path.name == "models":
            return resolve_models_checkpoint(path)
        if path.is_dir() and path.name == "last":
            return resolve_models_checkpoint(path.parent)
        if path.is_dir() and any(path.glob("checkpoint_epoch_*")):
            return _latest_checkpoint_epoch(path)
        return resolve_checkpoint_path(path)

    models_dir = (
        Path(results_root).resolve()
        / f"{case_dir}_{case_suffix}"
        / f"seed_{seed}"
        / "models"
    )
    if not models_dir.exists():
        raise FileNotFoundError(
            f"No models directory at {models_dir}. "
            "Pass --checkpoint-path or check --results-root / --case-suffix / --seed."
        )
    return resolve_models_checkpoint(models_dir)


def align_test_samples(requested: int, batch_sizes: Sequence[int]) -> int:
    """Round down so every batch size sees a complete sample pool (no ragged last batch)."""
    if requested < 1:
        raise ValueError(f"test_samples must be >= 1, got {requested}")
    max_b = max(batch_sizes)
    if max_b < 1:
        raise ValueError("batch_sizes must be positive")
    n = (requested // max_b) * max_b
    if n < max_b:
        raise ValueError(
            f"test_samples={requested} is smaller than max batch size {max_b}"
        )
    return n


def load_fixed_sample_pool(
    data,
    test_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the first ``test_samples`` host arrays from the test HDF5."""
    inputs = np.asarray(data._load_inputs(data.test_h5_path, samples=test_samples))
    targets = np.asarray(data._load_targets(data.test_h5_path, samples=test_samples))
    if inputs.shape[0] < test_samples:
        raise ValueError(
            f"Requested {test_samples} test samples but only {inputs.shape[0]} available"
        )
    return inputs[:test_samples], targets[:test_samples]


def regroup_batches(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slice a fixed pool into full batches of size ``batch_size`` (drop remainder)."""
    n = inputs.shape[0]
    n_full = (n // batch_size) * batch_size
    if n_full < batch_size:
        raise ValueError(
            f"Need at least {batch_size} samples for batch_size={batch_size}, got {n}"
        )
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, n_full, batch_size):
        end = start + batch_size
        batches.append((inputs[start:end], targets[start:end]))
    return batches


def build_fs_config(
    checkpoint_cfg: dict[str, Any],
    *,
    power_flow_solver: str,
    fs_rho: float,
    voltage_weight: float,
    fs_step_fraction: float,
    n_voltage_rows: int,
    lin_steps: int | None,
    fs_maxiter: int | None,
    fs_tol: float | None,
) -> FeasibilityConfig:
    return FeasibilityConfig(
        fs_tol=float(fs_tol if fs_tol is not None else checkpoint_cfg.get("fs_tol", 1e-6)),
        fs_maxiter=int(
            fs_maxiter
            if fs_maxiter is not None
            else checkpoint_cfg.get("fs_maxiter", 100)
        ),
        lin_steps=int(
            lin_steps if lin_steps is not None else checkpoint_cfg.get("lin_steps", 2)
        ),
        diff_steps=int(checkpoint_cfg.get("diff_steps", 0)),
        fs_rho=float(fs_rho),
        power_flow_solver=str(power_flow_solver),
        voltage_weight=float(voltage_weight),
        fs_step_fraction=float(fs_step_fraction),
        n_voltage_rows=int(n_voltage_rows),
    ).validate()


def _time_one_batch(
    *,
    data,
    net,
    n_der: int,
    inference_step: Callable,
    fs_config: FeasibilityConfig,
    inputs_np: np.ndarray,
    targets_np: np.ndarray,
) -> dict[str, float]:
    """Time topology inverse (recorded) and full pipeline (primary) for one batch."""
    inv_start = time.perf_counter()
    topo = precompute_topology(data, inputs_np)
    topo.Y_LL_inv.block_until_ready()
    inverse_time = time.perf_counter() - inv_start

    inputs_j = jnp.asarray(inputs_np)
    targets_j = jnp.asarray(targets_np)
    batch = prepare_batch_data(inputs_j, targets_j, data, topo=topo)
    nn_inputs = nn_features_for_apply(
        inputs_j, data, use_compact=data.use_compact_nn
    )

    raw_start = time.perf_counter()
    x_pred_raw = inference_step(nn_inputs)
    x_pred = _project_predictions(x_pred_raw, batch.pder_avai, net)
    v_pred_raw = compute_voltage(net, batch, x_pred, topo)
    v_pred_raw.block_until_ready()
    raw_time = time.perf_counter() - raw_start

    fs_start = time.perf_counter()
    x_pred_fs, _, _ = post_process_slfs(
        net,
        batch,
        x_pred[:, :n_der] + 1j * x_pred[:, n_der:],
        x_pred,
        fs_config,
        topo,
    )
    x_pred_fs.block_until_ready()
    fs_time = time.perf_counter() - fs_start

    return {
        "inverse_time": inverse_time,
        "raw_time": raw_time,
        "fs_time": fs_time,
        "pipeline_time": raw_time + fs_time,
    }


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=0)) if arr.size > 1 else 0.0
    return mean, std


def measure_batch_size(
    *,
    data,
    state,
    fs_config: FeasibilityConfig,
    host_batches: list[tuple[np.ndarray, np.ndarray]],
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    """Warmup then timed repeats; return mean±std pipeline time per batch."""
    net = data.net
    n_der = net.n_der_phase

    @jax.jit
    def inference_step(inputs):
        return state.apply_fn(state.params, inputs)

    time_kwargs = dict(
        data=data,
        net=net,
        n_der=n_der,
        inference_step=inference_step,
        fs_config=fs_config,
    )

    # Warmup on the first batch only (covers JIT for this B).
    first_in, first_tg = host_batches[0]
    for _ in range(warmup):
        _time_one_batch(inputs_np=first_in, targets_np=first_tg, **time_kwargs)

    repeat_pipeline: list[float] = []
    repeat_raw: list[float] = []
    repeat_fs: list[float] = []
    repeat_inverse: list[float] = []
    repeat_pipeline_per_sample: list[float] = []

    batch_size = int(host_batches[0][0].shape[0])
    for _ in range(repeats):
        pipe_times: list[float] = []
        raw_times: list[float] = []
        fs_times: list[float] = []
        inv_times: list[float] = []
        for inputs_np, targets_np in host_batches:
            times = _time_one_batch(
                inputs_np=inputs_np, targets_np=targets_np, **time_kwargs
            )
            pipe_times.append(times["pipeline_time"])
            raw_times.append(times["raw_time"])
            fs_times.append(times["fs_time"])
            inv_times.append(times["inverse_time"])

        mean_pipe = float(np.mean(pipe_times))
        repeat_pipeline.append(mean_pipe)
        repeat_raw.append(float(np.mean(raw_times)))
        repeat_fs.append(float(np.mean(fs_times)))
        repeat_inverse.append(float(np.mean(inv_times)))
        repeat_pipeline_per_sample.append(mean_pipe / batch_size)

    pipe_mean, pipe_std = _mean_std(repeat_pipeline)
    raw_mean, raw_std = _mean_std(repeat_raw)
    fs_mean, fs_std = _mean_std(repeat_fs)
    inv_mean, inv_std = _mean_std(repeat_inverse)
    pps_mean, pps_std = _mean_std(repeat_pipeline_per_sample)

    return {
        "num_batches": len(host_batches),
        "batch_size": batch_size,
        "pipeline_time_per_batch_mean_s": pipe_mean,
        "pipeline_time_per_batch_std_s": pipe_std,
        "raw_time_per_batch_mean_s": raw_mean,
        "raw_time_per_batch_std_s": raw_std,
        "fs_time_per_batch_mean_s": fs_mean,
        "fs_time_per_batch_std_s": fs_std,
        "inverse_time_per_batch_mean_s": inv_mean,
        "inverse_time_per_batch_std_s": inv_std,
        "pipeline_time_per_sample_mean_s": pps_mean,
        "pipeline_time_per_sample_std_s": pps_std,
    }


def run_benchmark(
    *,
    grid: str,
    checkpoint_path: Path,
    batch_sizes: Sequence[int],
    test_samples: int,
    warmup: int,
    repeats: int,
    seed: int,
    case_suffix: str,
    out_path: Path | None = None,
    resume: bool = True,
    woodbury_ir_steps: str = "auto",
    woodbury_batch_mode: str = "vmap",
    fs_rho: float = DEFAULT_FS_RHO,
    voltage_weight: float = DEFAULT_VOLTAGE_WEIGHT,
    fs_step_fraction: float = DEFAULT_FS_STEP_FRACTION,
    n_voltage_rows: int = DEFAULT_N_VOLTAGE_ROWS,
    lin_steps: int | None = None,
    fs_maxiter: int | None = None,
    fs_tol: float | None = None,
    train_samples: int | None = None,
) -> dict[str, Any]:
    """Load penalty_fs checkpoint and sweep batch sizes."""
    preset = resolve_grid(grid)
    aligned_n = align_test_samples(test_samples, batch_sizes)

    set_random_seeds(seed)
    clear_gpu_memory()

    checkpoint_path, model_spec = resolve_checkpoint_model_spec(checkpoint_path)
    checkpoint_cfg = load_checkpoint_config(checkpoint_path)
    power_flow_solver = resolve_power_flow_solver(grid, checkpoint_cfg)

    build = OPFDataBuildOptions(
        power_flow_solver=power_flow_solver,
        woodbury_ir_steps=woodbury_ir_steps,
        woodbury_batch_mode=woodbury_batch_mode,
    )
    test_h5_paths = {"feasible": str((_ROOT / preset["test_h5"]).resolve())}
    data, train_dataset = prepare_checkpoint_test_data(
        train_h5=str((_ROOT / preset["train_h5"]).resolve()),
        data_json=str((_ROOT / preset["data_json"]).resolve()),
        test_h5_paths=test_h5_paths,
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
        lin_steps=lin_steps,
        fs_maxiter=fs_maxiter,
        fs_tol=fs_tol,
    )

    inputs_pool, targets_pool = load_fixed_sample_pool(data, aligned_n)
    by_batch_size: dict[str, dict[str, float]] = {}
    if out_path is not None and resume:
        existing = load_partial_results(out_path)
        if existing is not None:
            _validate_resume_compatible(
                existing,
                out_path=out_path,
                grid=grid,
                case_suffix=case_suffix,
                batch_sizes=batch_sizes,
                warmup=warmup,
                repeats=repeats,
                test_samples_requested=test_samples,
                aligned_test_samples=aligned_n,
            )
            by_batch_size = dict(existing.get("by_batch_size", {}))

    to_run = pending_batch_sizes(batch_sizes, by_batch_size)
    if not to_run:
        print(f"All {len(batch_sizes)} batch sizes already in {out_path}; skipping.")
    elif len(to_run) < len(batch_sizes):
        print(
            f"Resuming: {len(by_batch_size)} done, "
            f"{len(to_run)} remaining batch sizes {to_run}"
        )

    failed_batch_size: int | None = None
    for batch_size in to_run:
        print(f"\n=== batch_size={batch_size} ===")
        try:
            clear_gpu_memory()
            host_batches = regroup_batches(inputs_pool, targets_pool, batch_size)
            metrics = measure_batch_size(
                data=data,
                state=state,
                fs_config=fs_config,
                host_batches=host_batches,
                warmup=warmup,
                repeats=repeats,
            )
            by_batch_size[str(batch_size)] = metrics
            print(
                f"  pipeline/batch: {metrics['pipeline_time_per_batch_mean_s']:.6f}s "
                f"± {metrics['pipeline_time_per_batch_std_s']:.6f}s "
                f"(n_batches={metrics['num_batches']})"
            )
        except Exception:
            failed_batch_size = batch_size
            if out_path is not None:
                payload = build_result_payload(
                    grid=grid,
                    preset=preset,
                    checkpoint_path=checkpoint_path,
                    case_suffix=case_suffix,
                    warmup=warmup,
                    repeats=repeats,
                    aligned_n=aligned_n,
                    test_samples=test_samples,
                    batch_sizes=batch_sizes,
                    fs_config=fs_config,
                    by_batch_size=by_batch_size,
                    failed_batch_size=failed_batch_size,
                )
                write_results(payload, out_path)
            raise

        if out_path is not None:
            payload = build_result_payload(
                grid=grid,
                preset=preset,
                checkpoint_path=checkpoint_path,
                case_suffix=case_suffix,
                warmup=warmup,
                repeats=repeats,
                aligned_n=aligned_n,
                test_samples=test_samples,
                batch_sizes=batch_sizes,
                fs_config=fs_config,
                by_batch_size=by_batch_size,
            )
            write_results(payload, out_path)

    return build_result_payload(
        grid=grid,
        preset=preset,
        checkpoint_path=checkpoint_path,
        case_suffix=case_suffix,
        warmup=warmup,
        repeats=repeats,
        aligned_n=aligned_n,
        test_samples=test_samples,
        batch_sizes=batch_sizes,
        fs_config=fs_config,
        by_batch_size=by_batch_size,
        failed_batch_size=failed_batch_size,
    )


def write_results(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(to_json_serializable(payload), f, indent=2)
    print(f"\nWrote {out_path}")


def load_partial_results(out_path: Path) -> dict[str, Any] | None:
    """Load an existing results JSON for resume, or ``None`` if missing."""
    if not out_path.exists():
        return None
    with open(out_path) as f:
        return json.load(f)


def _validate_resume_compatible(
    existing: dict[str, Any],
    *,
    out_path: Path,
    grid: str,
    case_suffix: str,
    batch_sizes: Sequence[int],
    warmup: int,
    repeats: int,
    test_samples_requested: int,
    aligned_test_samples: int,
) -> None:
    """Reject resume when the on-disk file does not match this run's settings."""
    checks: tuple[tuple[str, Any], ...] = (
        ("grid", grid),
        ("case_suffix", case_suffix),
        ("warmup", warmup),
        ("repeats", repeats),
        ("test_samples_requested", test_samples_requested),
        ("test_samples", aligned_test_samples),
        ("batch_sizes", list(batch_sizes)),
    )
    for key, expected in checks:
        if key in existing and existing[key] != expected:
            raise ValueError(
                f"Cannot resume {out_path}: existing {key}={existing[key]!r} "
                f"!= requested {expected!r}. Delete the file or pass --no-resume."
            )


def pending_batch_sizes(
    batch_sizes: Sequence[int],
    by_batch_size: dict[str, dict[str, float]],
) -> list[int]:
    """Return batch sizes not yet present in ``by_batch_size``."""
    return [b for b in batch_sizes if str(b) not in by_batch_size]


def _result_status(
    batch_sizes: Sequence[int],
    by_batch_size: dict[str, dict[str, float]],
    *,
    failed_batch_size: int | None = None,
) -> str:
    if failed_batch_size is not None:
        return "failed"
    if len(by_batch_size) >= len(batch_sizes):
        return "complete"
    return "partial"


def build_result_payload(
    *,
    grid: str,
    preset: dict[str, str],
    checkpoint_path: Path,
    case_suffix: str,
    warmup: int,
    repeats: int,
    aligned_n: int,
    test_samples: int,
    batch_sizes: Sequence[int],
    fs_config: FeasibilityConfig,
    by_batch_size: dict[str, dict[str, float]],
    failed_batch_size: int | None = None,
) -> dict[str, Any]:
    status = _result_status(
        batch_sizes, by_batch_size, failed_batch_size=failed_batch_size
    )
    payload: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "method": "penalty_fs",
        "case_suffix": case_suffix,
        "grid": grid,
        "case_dir": preset["case_dir"],
        "variant": "feasible",
        "checkpoint_path": str(checkpoint_path),
        "warmup": warmup,
        "repeats": repeats,
        "test_samples": aligned_n,
        "test_samples_requested": test_samples,
        "batch_sizes": list(batch_sizes),
        "device": _jax_device_string(),
        "dtype": "float64",
        "timed_scope": TIMED_SCOPE,
        "excludes": list(EXCLUDES),
        "status": status,
        "fs_config": {
            "lin_steps": fs_config.lin_steps,
            "diff_steps": fs_config.diff_steps,
            "fs_maxiter": fs_config.fs_maxiter,
            "fs_tol": fs_config.fs_tol,
            "fs_rho": fs_config.fs_rho,
            "voltage_weight": fs_config.voltage_weight,
            "fs_step_fraction": fs_config.fs_step_fraction,
            "n_voltage_rows": fs_config.n_voltage_rows,
            "power_flow_solver": fs_config.power_flow_solver,
        },
        "by_batch_size": by_batch_size,
    }
    if failed_batch_size is not None:
        payload["failed_batch_size"] = failed_batch_size
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="penalty_fs batch-size inference-time benchmark"
    )
    parser.add_argument(
        "--grid",
        type=str,
        required=True,
        choices=sorted(GRID_PRESETS),
        help="Grid preset key",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Explicit checkpoint_epoch_* directory (or parent with epochs)",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=DEFAULT_RESULTS_ROOT,
        help="Root used when --checkpoint-path is omitted",
    )
    parser.add_argument(
        "--case-suffix",
        type=str,
        default=DEFAULT_CASE_SUFFIX,
        help="Method case suffix for checkpoint discovery",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to sweep",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
        help="Sample pool size (rounded down to a multiple of max batch size)",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: analysis/.../results/results_<grid>.json)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from --out if present; skip completed batch sizes (default: on)",
    )
    parser.add_argument("--train-samples", type=int, default=None)
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
    parser.add_argument("--lin-steps", type=int, default=None)
    parser.add_argument("--fs-maxiter", type=int, default=None)
    parser.add_argument("--fs-tol", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preset = resolve_grid(args.grid)
    ckpt = resolve_run_checkpoint(
        checkpoint_path=args.checkpoint_path,
        results_root=args.results_root,
        case_dir=preset["case_dir"],
        case_suffix=args.case_suffix,
        seed=args.seed,
    )
    out = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent / "results" / f"results_{args.grid}.json"
    )

    print(f"Grid: {args.grid}")
    print(f"Checkpoint: {ckpt}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Warmup={args.warmup} repeats={args.repeats}")
    print(f"Device: {_jax_device_string()}")

    payload = run_benchmark(
        grid=args.grid,
        checkpoint_path=ckpt,
        batch_sizes=args.batch_sizes,
        test_samples=args.test_samples,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        case_suffix=args.case_suffix,
        out_path=out,
        resume=args.resume,
        woodbury_ir_steps=args.woodbury_ir_steps,
        woodbury_batch_mode=args.woodbury_batch_mode,
        fs_rho=args.fs_rho,
        voltage_weight=args.voltage_weight,
        fs_step_fraction=args.fs_step_fraction,
        n_voltage_rows=args.n_voltage_rows,
        lin_steps=args.lin_steps,
        fs_maxiter=args.fs_maxiter,
        fs_tol=args.fs_tol,
        train_samples=args.train_samples,
    )
    write_results(payload, out)


if __name__ == "__main__":
    main()
