"""Train/test orchestration and batch extraction for method entry scripts."""

import json
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp

from core.grids.data import OPFData
from core.nn.model import compute_input_stats
from core.utils.loss import compute_cost_norm_stats_from_eval
from core.utils.datasets import primary_test_dataset, resolve_cost_norm_h5_path
from core.utils.types import (
    BatchData,
    CostNormStats,
    TestH5Paths,
    TopologyData,
    VariantTestSpec,
)


@partial(jax.jit, static_argnums=(2,))
def prepare_batch_data(
    inputs_batch,
    targets_batch,
    session: OPFData,
    topo: TopologyData | None = None,
) -> BatchData:
    """Extract physics tensors and OPF labels from flat HDF5 batches."""
    net = session.net
    pd = inputs_batch[:, session.slices["input"]["pd"]]
    qd = inputs_batch[:, session.slices["input"]["qd"]]
    pder_avai = inputs_batch[:, session.slices["input"]["pder_avai"]]
    obj_weights = inputs_batch[:, session.slices["input"]["obj_weights"]]
    s_load = jax.lax.complex(pd, qd)
    if topo is not None:
        branch_status = topo.branch_status
        switchable_status = branch_status[:, net.changing_switches]
    else:
        topo_key = session.topology_input_key
        branch_status_compact = jnp.asarray(
            inputs_batch[:, session.slices["input"][topo_key]], dtype=jnp.int32
        )
        branch_status = session.expand_branch_status(branch_status_compact)
        switchable_status = branch_status[:, net.changing_switches]

    pder_target = targets_batch[:, session.slices["target"]["pder"]]
    qder_target = targets_batch[:, session.slices["target"]["qder"]]
    vr_target = targets_batch[:, session.slices["target"]["vr"]]
    vi_target = targets_batch[:, session.slices["target"]["vi"]]
    v_target = jax.lax.complex(vr_target, vi_target)

    if topo is not None:
        v_init = (
            topo.w
            if topo.w.ndim > 1
            else jnp.broadcast_to(topo.w[None, :], (pd.shape[0], net.n_net))
        )
    else:
        v_init = jnp.broadcast_to(net.w[None, :], (pd.shape[0], net.n_net))

    return BatchData(
        s_load=s_load,
        pder_avai=pder_avai,
        obj_weights=obj_weights,
        branch_status=branch_status,
        switchable_status=switchable_status,
        v_init=v_init,
        v_target=v_target,
        pder_target=pder_target,
        qder_target=qder_target,
    )


def make_results_root(results_dir: str, case_name: str, seed: int) -> Path:
    """Return <results_dir>/<case_name>/seed_<seed> as an absolute path."""
    return (Path(results_dir) / case_name / f"seed_{seed}").resolve()


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create every directory in paths (parents included)."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def filter_existing_test_datasets(
    test_h5_paths: TestH5Paths,
) -> List[Tuple[str, str]]:
    """Return existing datasets; warn and skip missing optional extras."""

    primary_name, _ = primary_test_dataset(test_h5_paths)
    result: List[Tuple[str, str]] = []
    for name, path in test_h5_paths.items():
        is_primary = name == primary_name
        if not path:
            if is_primary:
                raise ValueError(f"Primary test dataset {name!r} has an empty path")
            warnings.warn(
                f"Skipping test dataset {name!r}: empty path",
                UserWarning,
                stacklevel=2,
            )
            continue
        if not Path(path).exists():
            if is_primary:
                raise ValueError(
                    f"Primary test dataset {name!r} not found at {path}"
                )
            warnings.warn(
                f"Skipping test dataset {name!r}: not found at {path}",
                UserWarning,
                stacklevel=2,
            )
            continue
        result.append((name, path))
    return result


def compute_training_normalization(
    train_dataset: Any,
    eval_dataset: Any,
    data: OPFData,
    *,
    batch_size: int,
    mode: str = "robust",
    val_h5_path: Optional[str] = None,
    eval_samples: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], CostNormStats]:
    """Compute NN input statistics from the training set and the cost-normalization scale from the validation (or reference test) HDF5."""
    if verbose:
        print("Computing input statistics...")
    input_stat = compute_input_stats(
        train_dataset, data, use_compact=getattr(data, "use_compact_nn", True)
    )

    cost_norm_h5_path = resolve_cost_norm_h5_path(data, val_h5_path=val_h5_path)
    if verbose:
        if val_h5_path is None:
            print(
                "No --val_h5; computing cost normalization from reference test HDF5: "
                f"{cost_norm_h5_path}"
            )
        else:
            print("Computing cost normalization stats from validation set...")
    cost_norm_stats = compute_cost_norm_stats_from_eval(
        data,
        mode=mode,
        samples=eval_samples,
        h5_path=cost_norm_h5_path,
    )
    if verbose:
        print(
            f"  cost_norm_mode={cost_norm_stats.mode} | "
            f"cost_max={cost_norm_stats.cost_max:.6g} | "
            f"cost_p95={cost_norm_stats.cost_p95:.6g} | "
            f"cost_scale={cost_norm_stats.cost_scale:.6g} | "
            f"n_samples={cost_norm_stats.n_samples}"
        )

    return input_stat, cost_norm_stats


def create_train_results_layout(
    results_dir: str | Path,
    case_name: str,
    seed: int,
) -> Path:
    """Create models/, training/ and testing/ under the results root and return it."""
    root = make_results_root(results_dir, case_name, seed)
    ensure_dirs([root / "models", root / "training", root / "testing"])
    print(f"Results directory: {root.absolute()}")
    return root


def create_test_results_layout(
    results_dir: str | Path,
    case_name: str,
    seed: int,
) -> Path:
    """Create testing/ under the results root and return it."""
    root = make_results_root(results_dir, case_name, seed)
    ensure_dirs([root / "testing"])
    print(f"Results directory: {root}")
    return root


def save_variant_test_metrics(
    variant_results_dir: Path,
    batch_size: int,
    test_metrics: dict[str, Any],
    *,
    pop_batch_metrics: bool = True,
) -> tuple[Path, Optional[Path]]:
    """Write testing_results_batch_<B>.json (and per-batch details when kept) for one test variant; returns the paths."""
    from core.utils.checkpoint import to_json_serializable

    variant_results_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(test_metrics)
    batch_metrics = metrics.pop("batch_metrics", None) if pop_batch_metrics else None

    summary_path = variant_results_dir / f"testing_results_batch_{batch_size}.json"
    with open(summary_path, "w") as f:
        json.dump(to_json_serializable(metrics), f, indent=2)

    details_path = None
    if batch_metrics is not None:
        details_path = variant_results_dir / f"testing_batch_details_batch_{batch_size}.json"
        with open(details_path, "w") as f:
            json.dump(batch_metrics, f, indent=2)

    return summary_path, details_path


def run_variant_batch_tests(
    state: Any,
    base_data: OPFData,
    spec: VariantTestSpec,
    test_fn: Callable[..., dict[str, Any]],
    *,
    test_fn_kwargs: Optional[dict[str, Any]] = None,
    print_metrics: Optional[Callable[[int, dict[str, Any]], None]] = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Run test_fn over existing dataset variants and batch sizes; save JSON summaries."""
    from core.utils.runtime import clear_gpu_memory

    test_fn_kwargs = test_fn_kwargs or {}
    all_results: dict[str, dict[int, dict[str, Any]]] = {}

    print("\n" + "=" * 80)
    print("Running performance tests with different batch sizes and datasets...")
    print("=" * 80)

    for variant_name, variant_h5_path in filter_existing_test_datasets(spec.test_h5_paths):
        print("\n" + "-" * 80)
        print(f"Testing on dataset variant: {variant_name}")
        print(f"Dataset path: {variant_h5_path}")
        print("-" * 80)

        data_variant = base_data.with_test_h5(variant_h5_path)
        variant_results_dir = spec.testing_results_dir / variant_name
        all_results[variant_name] = {}

        for batch_size in spec.batch_sizes:
            if spec.clear_device_memory:
                clear_gpu_memory()
                if spec.gpu_clear_delay_s > 0:
                    time.sleep(spec.gpu_clear_delay_s)

            print(f"\n  Testing with batch_size={batch_size}...")
            test_metrics = test_fn(
                state,
                data_variant,
                batch_size,
                spec.test_samples,
                **test_fn_kwargs,
            )

            if print_metrics is not None:
                print_metrics(batch_size, test_metrics)

            summary_path, details_path = save_variant_test_metrics(
                variant_results_dir,
                batch_size,
                test_metrics,
                pop_batch_metrics=spec.pop_batch_metrics,
            )
            print(f"    Summary results saved to {summary_path}")
            if details_path is not None:
                print(f"    Per-batch details saved to {details_path}")

            all_results[variant_name][batch_size] = test_metrics

    print("\n" + "=" * 80)
    print("Performance testing complete!")
    print("=" * 80)
    return all_results
