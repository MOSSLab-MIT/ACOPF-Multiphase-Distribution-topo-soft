#!/usr/bin/env python3
"""Test-only entrypoint for fixed-reference linear OPF baseline."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# CVXPY + Clarabel stay CPU-only. JAX backend for power flow is controlled by
# JAX_PLATFORMS (default cpu; set cuda on NVIDIA GPU nodes).
# Must be set before any ``import jax``.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_jax_platform = os.environ.get("JAX_PLATFORMS", "cpu").strip().lower()
# JAX treats "gpu" as ROCm on some builds; cluster jaxlib exposes cuda, not rocm.
if _jax_platform == "gpu":
    _jax_platform = "cuda"
    os.environ["JAX_PLATFORMS"] = _jax_platform
if _jax_platform == "cpu":
    # Force (not setdefault): Slurm's GPU plugin may pre-set
    # CUDA_VISIBLE_DEVICES to a device whose driver mismatches the installed
    # TF/jaxlib (cuInit error 303), which makes `import tensorflow` segfault.
    # Clearing it keeps both JAX and TF strictly on the CPU for this baseline.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from core.utils.runtime import (
    configure_tensorflow_for_data_loading,
    set_protobuf_implementation,
    set_random_seeds,
)

set_protobuf_implementation()

import argparse

import jax

# Only hide TF GPUs when running on an accelerator (so JAX owns it). On the
# CPU-only path CUDA_VISIBLE_DEVICES is forced empty above; calling TF device
# enumeration here probes a mismatched driver (cuInit error 303) and can
# segfault, so skip it.
if _jax_platform != "cpu":
    configure_tensorflow_for_data_loading()
jax.config.update("jax_enable_x64", True)

from core.utils.cli import add_test_dataset_cli_args, print_banner
from core.grids.power_flow import PowerFlowSolver, get_power_flow_batch_fn
from core.utils.datasets import resolve_and_validate_test_paths
from core.utils.types import OPFDataBuildOptions
from core.utils.workflow import create_test_results_layout, save_variant_test_metrics
from core.utils.datasets import load_opf_data
from methods.linear_opf.reporting import print_linear_opf_benchmark_metrics
from methods.linear_opf.runner import benchmark_linear_opf
from methods.linear_opf.sharding import resolve_shard_test_range, shard_results_dir


def parse_args():
    """Parse command-line arguments (sharding flags select a contiguous slice of the test samples)."""
    parser = argparse.ArgumentParser(
        description="Test fixed-reference linear OPF baseline (CVXPY + Clarabel)"
    )
    add_test_dataset_cli_args(parser)
    parser.add_argument("--data_json", type=str, required=True, help="Network config JSON")
    parser.add_argument("--test_samples", type=int, default=None, help="Number of testing samples")
    parser.add_argument(
        "--test_sample_offset",
        type=int,
        default=None,
        help="Absolute sample start index (mutually exclusive with num_shards > 1)",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard index in [0, num_shards) for parallel sample-range jobs",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of sample shards; >1 writes under testing/<variant>/shards/shard_XX/",
    )
    parser.add_argument(
        "--test_batch_sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes for testing",
    )
    parser.add_argument("--case_name", type=str, default="linear_opf", help="Case name")
    parser.add_argument("--results_dir", type=str, default="results/linear_opf", help="Results directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--power_flow_solver",
        type=str,
        default=PowerFlowSolver.FIXED_POINT.value,
        choices=[m.value for m in PowerFlowSolver],
        help="Power flow solver for post-QP metrics",
    )

    # Topology (Woodbury IR steps + batch mode)
    parser.add_argument("--woodbury_ir_steps", type=str, default="auto",
        choices=["auto", "0", "1", "2"],
        help="Woodbury SMW-IR correction steps: auto (probe at init), 0, 1, or 2. "
             "Experiment defaults: 906/8500=0, 13/123=1, 240=2.")
    parser.add_argument("--woodbury_batch_mode", type=str, default="vmap",
        choices=["vmap", "map"],
        help="Batch computation mode for topology matrices: vmap (fast, higher VRAM) "
             "or map (lax.map, lower VRAM). Use 'map' if vmap causes GPU OOM.")

    return parser.parse_args()


def _run_single_variant_benchmark(
    args,
    data,
    test_h5_paths,
    results_root,
    solve_pf_batch_fn,
) -> None:
    variant_name, test_h5_path = next(iter(test_h5_paths.items()))
    sample_offset, sample_count, shard_id, num_shards = resolve_shard_test_range(
        test_h5_path,
        test_samples=args.test_samples,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        test_sample_offset=args.test_sample_offset,
    )

    print("\n" + "=" * 80)
    print("Running linear OPF benchmark...")
    print("=" * 80)
    print(f"  Variant:       {variant_name}")
    print(f"  Dataset path:  {test_h5_path}")
    print(f"  Sample offset: {sample_offset}")
    print(f"  Sample count:  {sample_count}")
    if num_shards > 1:
        print(f"  Shard:         {shard_id + 1}/{num_shards}")
    print()

    for batch_size in args.test_batch_sizes:
        print(f"\n  Testing with batch_size={batch_size}...")
        test_metrics = benchmark_linear_opf(
            None,
            data,
            batch_size,
            sample_count,
            sample_offset=sample_offset,
            solve_pf_batch_fn=solve_pf_batch_fn,
        )
        test_metrics["shard_id"] = shard_id
        test_metrics["num_shards"] = num_shards
        test_metrics["sample_offset"] = sample_offset
        test_metrics["shard_sample_count"] = sample_count

        print_linear_opf_benchmark_metrics(batch_size, test_metrics)
        variant_results_dir = shard_results_dir(
            results_root / "testing",
            variant_name,
            shard_id,
            num_shards,
        )
        summary_path, details_path = save_variant_test_metrics(
            variant_results_dir,
            batch_size,
            test_metrics,
            pop_batch_metrics=True,
        )
        print(f"    Summary results saved to {summary_path}")
        if details_path is not None:
            print(f"    Per-batch details saved to {details_path}")

    print("\n" + "=" * 80)
    print("Performance testing complete!")
    print("=" * 80)


def main():
    """Run the linear OPF baseline on every configured test dataset and save per-variant JSON metrics."""
    args = parse_args()
    print_banner("Linear OPF Baseline - Test", args)

    set_random_seeds(args.seed)
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")
    print()

    test_h5_paths = resolve_and_validate_test_paths(
        test_h5=args.test_h5,
        test_h5_variants=args.test_h5_variants,
        test_datasets=args.test_dataset,
    )
    _, reference_h5 = next(iter(test_h5_paths.items()))

    build = OPFDataBuildOptions(
        power_flow_solver=args.power_flow_solver,
        woodbury_ir_steps=args.woodbury_ir_steps,
        woodbury_batch_mode=args.woodbury_batch_mode,
    )
    data = load_opf_data(
        train_h5=reference_h5,
        data_json=args.data_json,
        test_h5_paths=test_h5_paths,
        build=build,
    )
    solve_pf_batch_fn = get_power_flow_batch_fn(data.power_flow_solver)

    results_root = create_test_results_layout(args.results_dir, args.case_name, args.seed)
    print()

    _run_single_variant_benchmark(
        args,
        data,
        test_h5_paths,
        results_root,
        solve_pf_batch_fn,
    )


if __name__ == "__main__":
    main()


#python methods/linear_opf/test.py   --data_json datagen_topo/datasets/906bus_Master/network_config.json   --test_h5 datagen_topo/datasets/906bus_Master/906bus_Master_test_feasible.h5   --test_samples 100   --case_name 906bus_Master_linear_opf