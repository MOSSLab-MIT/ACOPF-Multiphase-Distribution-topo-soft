#!/usr/bin/env python3
"""Test-only script for OPF Neural Network using penalty_fs checkpoint + FS post-processing."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.utils.runtime import (
    clear_gpu_memory,
    configure_tensorflow_for_data_loading,
    set_protobuf_implementation,
    set_random_seeds,
)

set_protobuf_implementation()

import argparse
from functools import partial

import jax

configure_tensorflow_for_data_loading()
jax.config.update("jax_enable_x64", True)

from core.utils.cli import add_test_dataset_cli_args, print_banner
from core.grids.power_flow import PowerFlowSolver
from core.nn.model import create_train_state
from core.feasibility.slfs_config import FeasibilityConfig
from core.utils.checkpoint import (
    load_checkpoint_config,
    resolve_checkpoint_model_spec,
    restore_checkpoint_state,
)
from core.utils.datasets import prepare_checkpoint_test_data, resolve_and_validate_test_paths
from core.utils.types import OPFDataBuildOptions, VariantTestSpec
from core.utils.workflow import create_test_results_layout, run_variant_batch_tests
from methods.penalty_fs.reporting import print_penalty_fs_test_metrics
from methods.penalty_fs.trainer import test_penalty_fs_fn


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test OPF neural network with penalty_fs checkpoint + FS post-processing"
    )

    # Data
    parser.add_argument("--train_h5", type=str, required=True, help="Training HDF5 file (for input stats)")
    add_test_dataset_cli_args(parser)
    parser.add_argument("--data_json", type=str, required=True, help="Network config JSON")
    parser.add_argument("--train_samples", type=int, default=None, help="Number of training samples (for input stats)")
    parser.add_argument("--test_samples", type=int, default=None, help="Number of testing samples")

    # Checkpoint
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to a checkpoint_epoch_* directory")

    # Model (optional; falls back to checkpoint config.json if omitted)
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["mlp", "gcn", "modified_gcn"],
                        help="Neural initializer architecture (default: from checkpoint config.json, else mlp)")
    parser.add_argument("--hidden_size", type=int, default=None, help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (only for state init)")

    # Testing
    parser.add_argument("--test_batch_sizes", type=int, nargs="+", default=[1], help="Batch sizes for testing")

    # Feasibility-seeking (inference SLFS)
    parser.add_argument("--lin_steps", type=int, default=2)
    parser.add_argument("--diff_steps", type=int, default=0)
    parser.add_argument("--fs_tol", type=float, default=1e-6)
    parser.add_argument("--fs_maxiter", type=int, default=50)
    parser.add_argument("--fs_rho", type=float, default=1e-3)
    parser.add_argument("--voltage_weight", type=float, default=8.0)
    parser.add_argument("--fs_step_fraction", type=float, default=2.5)
    parser.add_argument("--n_voltage_rows", type=int, default=0)

    # Logging / results
    parser.add_argument("--case_name", type=str, default="penalty_fs", help="Case name")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--power_flow_solver", type=str, default=PowerFlowSolver.FIXED_POINT.value, choices=[m.value for m in PowerFlowSolver], help="Power flow: fixed_point | newton_dense | newton_gmres | newton_gmres_explicit. Default fixed_point.")

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


def main():
    args = parse_args()
    print_banner("OPF Neural Network - penalty_fs Test", args)

    clear_gpu_memory()
    set_random_seeds(args.seed)
    print(f"JAX devices: {jax.devices()}")
    print()

    checkpoint_path, model_spec = resolve_checkpoint_model_spec(
        args.checkpoint_path,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
    )
    test_h5_paths = resolve_and_validate_test_paths(
        test_h5=args.test_h5,
        test_h5_variants=args.test_h5_variants,
        test_datasets=args.test_dataset,
    )
    build = OPFDataBuildOptions(
        power_flow_solver=args.power_flow_solver,
        woodbury_ir_steps=args.woodbury_ir_steps,
        woodbury_batch_mode=args.woodbury_batch_mode,
    )
    data, train_dataset = prepare_checkpoint_test_data(
        train_h5=args.train_h5,
        data_json=args.data_json,
        test_h5_paths=test_h5_paths,
        build=build,
        train_samples=args.train_samples,
    )

    model_type = args.model_type or load_checkpoint_config(checkpoint_path).get(
        "model_type", "mlp"
    )
    print(f"  Model type: {model_type}")

    state = restore_checkpoint_state(
        data=data,
        train_dataset=train_dataset,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
        create_state_fn=partial(create_train_state, model_type=model_type),
        model_spec=model_spec,
        dout=2 * data.net.n_der_phase,
    )

    fs_config = FeasibilityConfig(
        fs_tol=args.fs_tol,
        fs_maxiter=args.fs_maxiter,
        lin_steps=args.lin_steps,
        diff_steps=args.diff_steps,
        fs_rho=args.fs_rho,
        power_flow_solver=data.power_flow_solver,
        voltage_weight=args.voltage_weight,
        fs_step_fraction=args.fs_step_fraction,
        n_voltage_rows=args.n_voltage_rows,
    ).validate()

    results_root = create_test_results_layout(args.results_dir, args.case_name, args.seed)
    print()

    run_variant_batch_tests(
        state,
        data,
        VariantTestSpec(
            test_h5_paths=test_h5_paths,
            testing_results_dir=results_root / "testing",
            batch_sizes=args.test_batch_sizes,
            test_samples=args.test_samples,
        ),
        test_penalty_fs_fn,
        test_fn_kwargs={"fs_config": fs_config},
        print_metrics=print_penalty_fs_test_metrics,
    )


if __name__ == "__main__":
    main()
