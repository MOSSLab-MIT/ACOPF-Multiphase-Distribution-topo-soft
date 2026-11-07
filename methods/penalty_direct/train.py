#!/usr/bin/env python3
"""
Training script for OPF Neural Network with penalty_direct method.

NN predicts DER setpoints and voltages jointly. DER controls are projected;
training minimizes the soft-OPF objective divided by ``--cost_scale`` plus a
power-flow equality penalty.
"""
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
import time
import warnings

import jax
import tensorflow as tf
from flax import nnx

configure_tensorflow_for_data_loading()

from core.utils.cli import add_cost_scale_cli_arg, add_test_dataset_cli_args, print_banner
from core.grids.power_flow import PowerFlowSolver
from core.nn.state import create_train_state_full
from core.utils.checkpoint import build_training_summary, save_checkpoint, save_results
from core.utils.datasets import (
    load_opf_data,
    prepare_training_datasets,
    resolve_and_validate_test_paths,
    resolve_and_validate_val_h5,
)
from core.utils.runtime import get_gpu_memory_usage, merge_peak_gpu_memory, print_gpu_memory_usage
from core.utils.schedule import create_lr_schedule
from core.utils.types import OPFDataBuildOptions, VariantTestSpec
from core.utils.workflow import (
    compute_training_normalization,
    create_train_results_layout,
    run_variant_batch_tests,
)
from methods.penalty_direct.reporting import print_penalty_direct_test_metrics
from methods.penalty_direct.trainer import (
    eval_epoch_penalty_direct,
    test_penalty_direct_fn,
    train_epoch_penalty_direct,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train OPF neural network with penalty_direct method"
    )

    # Data
    parser.add_argument("--train_h5", type=str, required=True, help="Training HDF5 file")
    add_test_dataset_cli_args(parser)
    parser.add_argument("--data_json", type=str, required=True, help="Network config JSON")
    parser.add_argument("--train_samples", type=int, default=None, help="Number of training samples")
    parser.add_argument("--eval_samples", type=int, default=None, help="Number of eval samples")

    # Model
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Warmup epochs")
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.5,
        help="Minimum LR as a fraction of peak LR (cosine decay end value)",
    )
    add_cost_scale_cli_arg(parser)
    parser.add_argument("--grad_norm_stop_threshold", type=float, default=1e-4, help="Early stop threshold")

    # Testing
    parser.add_argument("--test_batch_sizes", type=int, nargs="+", default=[1], help="Batch sizes for post-training testing")
    parser.add_argument("--test_samples", type=int, default=None, help="Number of testing samples")

    # Penalty (adaptive)
    parser.add_argument("--rho_eq_init", type=float, default=1e-3, help="Initial power-flow equality penalty weight")
    parser.add_argument("--rho_rate", type=float, default=1.0, help="Penalty rate multiplier")
    parser.add_argument("--violation_threshold", type=float, default=1e-3, help="Violation threshold for decay")
    parser.add_argument("--rho_decay_rate", type=float, default=0.9, help="Penalty decay rate")
    parser.add_argument("--rho_max", type=float, default=1e10, help="Maximum penalty weight")

    # Logging / checkpointing
    parser.add_argument("--case_name", type=str, default="penalty_direct", help="Case name")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--save_every", type=int, default=100, help="Save every N epochs")
    parser.add_argument("--log_every", type=int, default=1, help="Log every N epochs")
    parser.add_argument("--save_best", action="store_true", default=False, help="Save best model")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # Solver / topology
    parser.add_argument("--power_flow_solver", type=str, default=PowerFlowSolver.FIXED_POINT.value, choices=[m.value for m in PowerFlowSolver], help="Power flow: fixed_point | newton_dense | newton_gmres | newton_gmres_explicit. Default fixed_point.")
    parser.add_argument("--topology_backend", type=str, default="woodbury", choices=["woodbury", "stamp"], help="Topology backend: woodbury (default) or stamp (stamping+inv reference).")
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
    """Main training function."""
    args = parse_args()
    print_banner("OPF Neural Network - penalty_direct Training", args)

    clear_gpu_memory()
    set_random_seeds(args.seed)
    print(f"JAX devices: {jax.devices()}")
    print()

    test_h5_paths = resolve_and_validate_test_paths(
        test_h5=args.test_h5,
        test_h5_variants=args.test_h5_variants,
        test_datasets=args.test_dataset,
    )
    val_h5_path = resolve_and_validate_val_h5(args.val_h5)
    if args.save_best and val_h5_path is None:
        warnings.warn(
            "--save_best has no effect without --val_h5 (in-epoch validation disabled).",
            UserWarning,
            stacklevel=2,
        )
    build = OPFDataBuildOptions(
        topology_backend=args.topology_backend,
        power_flow_solver=args.power_flow_solver,
        woodbury_ir_steps=args.woodbury_ir_steps,
        woodbury_batch_mode=args.woodbury_batch_mode,
    )
    data = load_opf_data(
        train_h5=args.train_h5,
        data_json=args.data_json,
        test_h5_paths=test_h5_paths,
        build=build,
    )
    data, train_dataset, eval_dataset, steps_per_epoch = prepare_training_datasets(
        data,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        val_h5_path=val_h5_path,
    )

    input_stat = compute_training_normalization(train_dataset, data)
    args.input_stat = input_stat
    cost_scale = args.cost_scale
    print(f"  Loss cost scale (--cost_scale): {cost_scale:.6g}")

    din = data.nn_layout.din_compact
    dout = 2 * data.net.n_der_phase + 2 * data.net.n_net
    print(
        f"  Input dimension: din_compact={din} (din_full={data.nn_layout.din_full}) "
        f"| Output dimension: {dout}\n"
    )

    print("Initializing model...")
    lr_schedule = create_lr_schedule(
        args.learning_rate,
        args.warmup_epochs,
        args.epochs,
        steps_per_epoch,
        min_lr_ratio=args.min_lr_ratio,
    )
    rng = nnx.Rngs(args.seed)
    state = create_train_state_full(
        din=din,
        dhid=args.hidden_size,
        dout=dout,
        num_layers=args.num_layers,
        scheduler=lr_schedule,
        input_stat=input_stat,
        data=data,
        rng=rng,
    )
    print(f"  Model: {args.num_layers} layers, {args.hidden_size} hidden units\n")

    results_root = create_train_results_layout(args.results_dir, args.case_name, args.seed)
    checkpoint_dir = results_root / "models"
    training_results_dir = results_root / "training"
    testing_results_dir = results_root / "testing"

    print()
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)

    best_eval_loss = float("inf")
    training_results = []
    last_eval_results = None
    peak_memory_across_training = {}

    rho_eq = args.rho_eq_init

    if val_h5_path is None:
        print("No --val_h5 provided; skipping in-epoch validation.")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loader = (
            train_dataset.batch(args.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = (
                eval_dataset.batch(args.batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE)
                .as_numpy_iterator()
            )

        state, train_metrics, rho_eq = train_epoch_penalty_direct(
            state,
            train_loader,
            data,
            rho_eq,
            args.rho_rate,
            args.violation_threshold,
            args.rho_decay_rate,
            args.rho_max,
            cost_scale=cost_scale,
        )
        epoch_time = time.time() - epoch_start

        train_metrics["rho_eq"] = rho_eq

        if epoch % args.log_every == 0:
            print(
                f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.2f}s) | "
                f"Rho_eq: {rho_eq:.2e}"
            )
            print(
                f"  Train - Loss: {train_metrics['loss']:.6f} | Cost: {train_metrics['cost']:.6f} | "
                f"Vio: {train_metrics['violation']:.6e} | Max Vio: {train_metrics['max_violation']:.6e} | "
                f"Grad: {train_metrics['grad_norm']:.6e}"
            )

            if eval_loader is not None:
                eval_metrics = eval_epoch_penalty_direct(
                    state, eval_loader, data, rho_eq, cost_scale=cost_scale
                )
                last_eval_results = eval_metrics.copy()

                print(
                    f"  Eval  - Loss: {eval_metrics['loss']:.6f} | Cost: {eval_metrics['cost']:.6f} | "
                    f"Vio: {eval_metrics['violation']:.6e} | Max Vio: {eval_metrics['max_violation']:.6e}"
                )
                print(
                    f"  Eval  - Target Vio: {eval_metrics['violation_target']:.6e} | Max Target Vio: "
                    f"{eval_metrics['max_violation_target']:.6e} | |CostErr| {eval_metrics['abs_cost_error']:.6f} "
                    f"(signed {eval_metrics['cost_error']:+.6f}) | OptGap: {eval_metrics['optimality_gap']:.6%}"
                )
                print(
                    f"  Timing - Total: {eval_metrics['total_time']:.2f}s | "
                    f"Avg/batch: {eval_metrics['avg_batch_time']:.4f}s"
                )

                mem = get_gpu_memory_usage()
                merge_peak_gpu_memory(peak_memory_across_training, mem)
                print_gpu_memory_usage(mem)

                if eval_metrics["loss"] < best_eval_loss and args.save_best:
                    best_eval_loss = eval_metrics["loss"]
                    save_checkpoint(state, epoch, checkpoint_dir / "best", data, args)

        train_metrics["epoch"] = epoch
        train_metrics["epoch_time"] = epoch_time
        training_results.append(train_metrics)

        if epoch % args.save_every == 0:
            save_checkpoint(state, epoch, checkpoint_dir, data, args)

    save_checkpoint(state, args.epochs, checkpoint_dir / "last", data, args)

    merge_peak_gpu_memory(peak_memory_across_training, get_gpu_memory_usage())

    summary_kwargs = {}
    if peak_memory_across_training:
        summary_kwargs["peak_memory_across_training"] = peak_memory_across_training
    training_summary = build_training_summary(training_results, **summary_kwargs)
    save_results(
        training_results,
        last_eval_results,
        training_results_dir,
        training_summary=training_summary,
    )

    total_training_time = training_summary["total_training_time"]
    print(
        f"\nTotal training time: {total_training_time:.2f}s "
        f"({total_training_time / 60.0:.1f} min)"
    )
    if peak_memory_across_training:
        print("Peak GPU memory during training:")
        for device_name, peaks in peak_memory_across_training.items():
            line = (
                f"  {device_name}: peak in use "
                f"{peaks['peak_bytes_in_use_gb']:.3f} GB"
            )
            if "peak_bytes_reserved_gb" in peaks:
                line += f", peak reserved {peaks['peak_bytes_reserved_gb']:.3f} GB"
            print(line)

    print("\n" + "=" * 80)
    print("Training complete!")
    if args.save_best and val_h5_path is not None:
        print(f"Best eval loss: {best_eval_loss:.6f}")
    print("=" * 80)

    run_variant_batch_tests(
        state,
        data,
        VariantTestSpec(
            test_h5_paths=test_h5_paths,
            testing_results_dir=testing_results_dir,
            batch_sizes=args.test_batch_sizes,
            test_samples=args.test_samples,
        ),
        test_penalty_direct_fn,
        print_metrics=print_penalty_direct_test_metrics,
    )


if __name__ == "__main__":
    main()
