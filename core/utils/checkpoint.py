"""Checkpoint save/restore and JSON result helpers."""

import json
from pathlib import Path
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.training import checkpoints
from core.grids.data import OPFData
from core.nn.model import compute_input_stats
from core.grids.power_flow import PowerFlowSolver
from core.nn.input import log_nn_input_layout, resolve_nn_input_for_checkpoint, save_nn_input_artifacts
from core.utils.types import CheckpointModelSpec


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    """Resolve either a checkpoint directory or its latest ``checkpoint_epoch_*`` child."""
    if (checkpoint_path / "config.json").exists():
        return checkpoint_path
    candidates = sorted(checkpoint_path.glob("checkpoint_epoch_*"))
    if not candidates:
        return checkpoint_path
    return candidates[-1]


def load_checkpoint_config(checkpoint_path: Path):
    """Load optional checkpoint ``config.json``."""
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def resolve_checkpoint_param(value, config, key, label):
    """Resolve a CLI value with fallback to checkpoint config."""
    resolved = value if value is not None else config.get(key)
    if resolved is None:
        raise ValueError(
            f"Missing {label}. Provide --{key} or ensure it exists in config.json."
        )
    return resolved


def save_checkpoint(state, epoch, checkpoint_dir, data, args):
    """Save Flax train state plus repository-specific config artifacts."""
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoints.save_checkpoint(
        ckpt_dir=str(checkpoint_path),
        target=state,
        step=epoch,
        overwrite=True,
    )

    config = {
        "epoch": epoch,
        "model_type": getattr(args, "model_type", "mlp"),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "min_lr_ratio": getattr(args, "min_lr_ratio", 0.5),
        "batch_size": args.batch_size,
        "weight_dist": args.weight_dist,
        "lin_steps": args.lin_steps,
        "diff_steps": args.diff_steps,
        "fs_tol": args.fs_tol,
        "fs_maxiter": args.fs_maxiter,
        "power_flow_solver": getattr(
            args,
            "power_flow_solver",
            getattr(data, "power_flow_solver", PowerFlowSolver.FIXED_POINT.value),
        ),
        "n_total": data.net.n_total,
        "n_der_phase": data.net.n_der_phase,
    }
    cost_norm_stats = getattr(args, "cost_norm_stats", None)
    if cost_norm_stats is not None:
        config["cost_norm"] = cost_norm_stats.to_dict()

    nn_layout = getattr(data, "nn_layout", None)
    input_stat = getattr(args, "input_stat", None)
    if nn_layout is not None:
        config["nn_input"] = nn_layout.summary_dict()
    if nn_layout is not None and input_stat is not None:
        save_nn_input_artifacts(checkpoint_path, nn_layout, input_stat)

    with open(checkpoint_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Checkpoint saved to {checkpoint_path}")


def to_json_serializable(obj):
    """Convert JAX/NumPy scalars and arrays into JSON-compatible Python values."""
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_training_summary(training_results: list[dict], **extra: Any) -> dict[str, Any]:
    """Aggregate run-level training metadata from per-epoch metrics."""
    summary: dict[str, Any] = {
        "num_epochs": len(training_results),
        "total_training_time": sum(
            float(metric.get("epoch_time", 0.0)) for metric in training_results
        ),
    }
    summary.update(extra)
    return summary


def save_results(
    training_results,
    testing_results,
    results_dir,
    *,
    training_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Save training and testing metrics as JSON files.

    Per-epoch metrics go to ``training_metrics.json``. Run-level aggregates
    (total wall time, epoch count, optional peak memory) go to
    ``training_summary.json``. Last in-epoch validation metrics go to
    ``testing_metrics_last.json``.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    saved_summary: dict[str, Any] | None = None

    if training_results:
        train_path = results_dir / "training_metrics.json"
        with open(train_path, "w") as f:
            json.dump(to_json_serializable(training_results), f, indent=2)
        print(f"Training results saved to {train_path}")

        saved_summary = training_summary or build_training_summary(training_results)
        summary_path = results_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(to_json_serializable(saved_summary), f, indent=2)
        print(f"Training summary saved to {summary_path}")

    if testing_results:
        test_path = results_dir / "testing_metrics_last.json"
        with open(test_path, "w") as f:
            json.dump(to_json_serializable(testing_results), f, indent=2)
        print(f"Testing results saved to {test_path}")

    return saved_summary


def resolve_checkpoint_model_spec(
    checkpoint_path: str | Path,
    *,
    hidden_size: Optional[int] = None,
    num_layers: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> tuple[Path, CheckpointModelSpec]:
    path = resolve_checkpoint_path(Path(checkpoint_path).resolve())
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    checkpoint_cfg = load_checkpoint_config(path)
    spec = CheckpointModelSpec(
        hidden_size=resolve_checkpoint_param(
            hidden_size, checkpoint_cfg, "hidden_size", "hidden_size"
        ),
        num_layers=resolve_checkpoint_param(
            num_layers, checkpoint_cfg, "num_layers", "num_layers"
        ),
        learning_rate=resolve_checkpoint_param(
            learning_rate, checkpoint_cfg, "learning_rate", "learning_rate"
        ),
    )
    return path, spec


def restore_checkpoint_state(
    *,
    data: OPFData,
    train_dataset: Any,
    checkpoint_path: Path,
    seed: int,
    create_state_fn: Callable[..., Any],
    model_spec: CheckpointModelSpec,
    dout: int,
    scheduler: Any = None,
) -> Any:
    def _recompute_stats(use_compact: bool = True):
        print("Computing input statistics...")
        return compute_input_stats(train_dataset, data, use_compact=use_compact)

    layout, input_stat, use_compact = resolve_nn_input_for_checkpoint(
        data, checkpoint_path, recompute_stats_fn=_recompute_stats
    )
    data.use_compact_nn = use_compact
    data.nn_layout = layout
    data.include_topo_nn_input = getattr(layout, "include_topo_nn_input", True)
    log_nn_input_layout(layout)

    din = layout.din_compact if use_compact else layout.din_full
    print(
        f"  Input dimension: din={din} (compact={use_compact}) | Output dimension: {dout}\n"
    )

    print("Initializing model state...")
    rng = nnx.Rngs(seed)
    state = create_state_fn(
        din=din,
        dhid=model_spec.hidden_size,
        dout=dout,
        num_layers=model_spec.num_layers,
        scheduler=scheduler if scheduler is not None else model_spec.learning_rate,
        input_stat=input_stat,
        data=data,
        rng=rng,
    )

    print(f"Restoring checkpoint from {checkpoint_path}...")
    state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_path), target=state)
    print("Checkpoint restored.\n")

    return state
