"""Argparse helpers for method train/test entry scripts."""

import argparse
import re
from typing import Optional, Tuple

from core.utils.runtime import lr_schedule_params

_TEST_DATASET_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

_BANNER_SKIP_KEYS = frozenset({"input_stat", "cost_norm_stats"})


def print_banner(title: str, args) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        if arg in _BANNER_SKIP_KEYS:
            continue
        print(f"  {arg}: {value}")
    print()


def print_training_schedule_summary(
    *,
    learning_rate: float,
    warmup_epochs: int,
    epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.5,
    rho: Optional[float] = None,
) -> None:
    """Print resolved LR warmup/cosine schedule (requires ``steps_per_epoch``)."""
    params = lr_schedule_params(
        learning_rate, warmup_epochs, epochs, steps_per_epoch, min_lr_ratio
    )
    print("Training schedule:")
    print(f"  learning_rate (peak): {params['peak_lr']:g}")
    print(
        f"  warmup_epochs: {params['warmup_epochs']} "
        f"({params['warmup_steps']} optimizer steps)"
    )
    print(
        f"  steps_per_epoch: {params['steps_per_epoch']}  |  "
        f"total_epochs: {params['total_epochs']}  |  "
        f"total_steps: {params['total_steps']}"
    )
    print(
        f"  initial_lr: {params['init_lr']:g}  ->  peak_lr: {params['peak_lr']:g}  ->  "
        f"min_lr: {params['min_lr']:g}  (min_lr_ratio={params['min_lr_ratio']})"
    )
    print(f"  cosine_decay_steps: {params['decay_steps']}")
    if rho is not None:
        print(f"  penalty rho: {rho:g}  (loss = cost + rho * ineq_violation_sq)")
    print()


def parse_test_dataset_arg(value: str) -> Tuple[str, str]:
    """Parse ``NAME=PATH`` for ``--test_dataset``."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--test_dataset must be NAME=PATH, got {value!r}"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(
            f"--test_dataset NAME and PATH must be non-empty, got {value!r}"
        )
    if not _TEST_DATASET_NAME_RE.match(name):
        raise argparse.ArgumentTypeError(
            f"Invalid test dataset name {name!r}; use letters, digits, and underscores "
            f"(must start with a letter)"
        )
    return name, path


def add_test_dataset_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register shared test-dataset CLI flags on ``parser``."""
    parser.add_argument(
        "--val_h5",
        type=str,
        default=None,
        help=(
            "Validation HDF5 for in-epoch eval. Cost normalization uses this file "
            "when set; otherwise the reference test HDF5 (first --test_h5_variants "
            "entry). If omitted, in-epoch validation is skipped."
        ),
    )
    parser.add_argument(
        "--test_h5",
        type=str,
        default=None,
        help=(
            "Reference test HDF5 path for OPFData layout init; dataset name is "
            "inferred from *_test_<name>.h5 stem (else file stem)"
        ),
    )
    parser.add_argument(
        "--test_h5_variants",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Post-training test datasets as a list; first entry is the reference "
            "test HDF5 for OPFData layout init. Bare paths infer names from "
            "*_test_<name>.h5 stems (else file stem) or use NAME=PATH "
            "(e.g. infeasible=path.h5). Empty strings are skipped."
        ),
    )
    parser.add_argument(
        "--test_dataset",
        action="append",
        type=parse_test_dataset_arg,
        default=None,
        metavar="NAME=PATH",
        help="Optional named test dataset (repeatable), e.g. infeasible=path.h5",
    )
