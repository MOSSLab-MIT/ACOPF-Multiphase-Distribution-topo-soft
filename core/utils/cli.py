"""Argparse helpers for method train/test entry scripts."""

import argparse
import math
import re
from typing import Tuple

_TEST_DATASET_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

_BANNER_SKIP_KEYS = frozenset({"input_stat"})


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


def positive_float(value: str) -> float:
    """Argparse type for a finite float > 0."""
    try:
        x = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a float, got {value!r}") from None
    if not math.isfinite(x) or x <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a finite float > 0, got {value!r}")
    return x


def add_cost_scale_cli_arg(parser: argparse.ArgumentParser) -> None:
    """Register ``--cost_scale``: training loss uses ``cost / cost_scale``."""
    parser.add_argument(
        "--cost_scale",
        type=positive_float,
        default=1.0,
        help=(
            "Fixed divisor of the OPF objective in the training loss "
            "(loss cost term = cost / cost_scale). Default 1.0 (unscaled)."
        ),
    )


def add_test_dataset_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register shared test-dataset CLI flags on ``parser``."""
    parser.add_argument(
        "--val_h5",
        type=str,
        default=None,
        help=(
            "Validation HDF5 for in-epoch eval. If omitted, in-epoch validation "
            "is skipped."
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
