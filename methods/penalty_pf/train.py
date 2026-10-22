#!/usr/bin/env python3
"""
Training script for OPF Neural Network with the penalty_pf method.

NN predicts DER setpoints, which are projected to the DER feasible set and fed
to an AC power-flow solve; training uses a fixed penalty rho on AC inequality
violations. Testing runs exactly the same pipeline (no post-processing).
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

import jax
import tensorflow as tf
from flax import nnx

configure_tensorflow_for_data_loading()
jax.config.update("jax_enable_x64", True)

import warnings

from core.utils.cli import (
    add_test_dataset_cli_args,
    print_banner,
    print_training_schedule_summary,
)
from core.grids.power_flow import PowerFlowSolver
from core.nn.model import create_train_state
from core.utils.checkpoint import build_training_summary, save_checkpoint, save_results
from core.utils.datasets import (
    load_opf_data,
    prepare_training_datasets,
    resolve_and_validate_test_paths,
    resolve_and_validate_val_h5,
)
from core.utils.runtime import create_lr_schedule, get_gpu_memory_usage, merge_peak_gpu_memory, print_gpu_memory_usage
from core.utils.types import OPFDataBuildOptions, VariantTestSpec
from core.utils.workflow import (
    compute_training_normalization,
    create_train_results_layout,
    run_variant_batch_tests,
)
from methods.penalty_pf.reporting import print_penalty_pf_test_metrics
from methods.penalty_pf.trainer import (
    eval_epoch_penalty_pf,
    test_penalty_pf_fn,
    train_epoch_penalty_pf,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train OPF neural network with penalty_pf method"
    )

    # Data
    parser.add_argument("--train_h5", type=str, required=True, help="Training HDF5 file")
    add_test_dataset_cli_args(parser)
    parser.add_argument("--data_json", type=str, required=True, help="Network config JSON")
    parser.add_argument("--train_samples", type=int, default=None, help="Number of training samples")
    parser.add_argument("--eval_samples", type=int, default=None, help="Number of eval samples")

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["mlp", "gcn", "modified_gcn"],
        default="mlp",
        help="NN architecture predicting DER setpoints (default: mlp). "
        "gcn/modified_gcn use full (non-compact) NN inputs.",
    )
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--include_topo_nn_input", action=argparse.BooleanOptionalAction, default=True, help="Include changing_switch_status in MLP input (default: True). Use --no-include_topo_nn_input for ablation.")

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
    parser.add_argument("--grad_norm_stop_threshold", type=float, default=1e-4, help="Early stop threshold")

    # Testing
    parser.add_argument("--test_batch_sizes", type=int, nargs="+", default=[1], help="Batch sizes for post-training testing")
    parser.add_argument("--test_samples", type=int, default=None, help="Number of testing samples")

    # Penalty
    parser.add_argument("--rho", type=float, default=1e-3, help="Fixed penalty weight rho")

    # Logging / checkpointing
    parser.add_argument("--case_name", type=str, default="penalty_pf", help="Case name")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--save_every", type=int, default=100, help="Save every N epochs")
    parser.add_argument("--log_every", type=int, default=1, help="Log every N epochs")
    parser.add_argument("--eval_every", type=int, default=None, help="Run full eval every N epochs (default: same as --log_every)")
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
    print_banner("OPF Neural Network - penalty_pf Training", args)

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
        include_topo_nn_input=args.include_topo_nn_input,
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

    # GCN initializers read per-node topology/loads from the full input row.
    if args.model_type != "mlp":
        data.use_compact_nn = False

    input_stat, cost_norm_stats = compute_training_normalization(
        train_dataset,
        eval_dataset,
        data,
        batch_size=args.batch_size,
        val_h5_path=val_h5_path,
        eval_samples=args.eval_samples,
    )
    args.input_stat = input_stat
    args.cost_norm_stats = cost_norm_stats
    cost_scale = cost_norm_stats.cost_scale_jax()

    din = (
        data.nn_layout.din_compact
        if data.use_compact_nn
        else data.nn_layout.din_full
    )
    dout = 2 * data.net.n_der_phase
    print(
        f"  Input dimension: din={din} (compact={data.use_compact_nn}, "
        f"din_full={data.nn_layout.din_full}) "
        f"| include_topo_nn_input={data.include_topo_nn_input} "
        f"| Output dimension: {dout}\n"
    )

    print("Initializing model...")
    print_training_schedule_summary(
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        min_lr_ratio=args.min_lr_ratio,
        rho=args.rho,
    )
    lr_schedule = create_lr_schedule(
        args.learning_rate,
        args.warmup_epochs,
        args.epochs,
        steps_per_epoch,
        min_lr_ratio=args.min_lr_ratio,
    )
    rng = nnx.Rngs(args.seed)
    state = create_train_state(
        din=din,
        dhid=args.hidden_size,
        dout=dout,
        num_layers=args.num_layers,
        scheduler=lr_schedule,
        input_stat=input_stat,
        data=data,
        rng=rng,
        model_type=args.model_type,
    )
    print(f"  Model: {args.model_type}, {args.num_layers} layers, {args.hidden_size} hidden units\n")

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

    rho = args.rho
    eval_every = args.eval_every if args.eval_every is not None else args.log_every

    if val_h5_path is None:
        print("No --val_h5 provided; skipping in-epoch validation.")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loader = (
            train_dataset.batch(args.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )

        state, train_metrics = train_epoch_penalty_pf(
            state, train_loader, data, rho, cost_scale=cost_scale
        )
        epoch_time = time.time() - epoch_start

        train_metrics["rho"] = rho

        if epoch % args.log_every == 0:
            print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.2f}s) | Rho: {rho:.2e}")
            print(
                f"  Train - Loss: {train_metrics['loss']:.6f} | Cost: {train_metrics['cost']:.6f} | "
                f"Vio: {train_metrics['violation']:.6e} | Max Vio: {train_metrics['max_violation']:.6e} | "
                f"Grad: {train_metrics['grad_norm']:.6e}"
            )

        if eval_dataset is not None and epoch % eval_every == 0:
            eval_loader = (
                eval_dataset.batch(args.batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE)
                .as_numpy_iterator()
            )
            eval_metrics = eval_epoch_penalty_pf(
                state, eval_loader, data, rho, cost_scale=cost_scale
            )
            last_eval_results = eval_metrics.copy()

            if epoch % args.log_every == 0:
                print(
                    f"  Eval  - Loss: {eval_metrics['loss']:.6f} | Cost: {eval_metrics['cost']:.6f} | "
                    f"Vio: {eval_metrics['violation']:.6e} | Max Vio: {eval_metrics['max_violation']:.6e}"
                )
                print(
                    f"  Eval  - Target Vio: {eval_metrics['violation_target']:.6e} | Max Target Vio: "
                    f"{eval_metrics['max_violation_target']:.6e} | |CostErr|: {eval_metrics['abs_cost_error']:.6f} "
                    f"(signed {eval_metrics['cost_error']:+.6f}, max {eval_metrics['max_abs_cost_error']:.6f}) | "
                    f"OptGap: {eval_metrics['optimality_gap']:.6%}"
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
        test_penalty_pf_fn,
        print_metrics=print_penalty_pf_test_metrics,
    )


if __name__ == "__main__":
    main()
