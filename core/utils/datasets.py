"""Test dataset path resolution and OPFData loading for train/test scripts."""

import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import jax

from core.grids.data import OPFData
from core.grids.power_flow import PowerFlowSolver
from core.nn.input import log_nn_input_layout, validate_load_support
from core.utils.cli import parse_test_dataset_arg
from core.utils.types import OPFDataBuildOptions, TestH5Paths

_TEST_H5_STEM_RE = re.compile(r"^.+_test_(?P<dataset>[a-zA-Z][a-zA-Z0-9_]*)$")


def apply_power_flow_solver_override(
    data, solver: Union[str, PowerFlowSolver]
) -> None:
    """Attach a validated power-flow registry key to ``data`` for ``get_power_flow_*_fn``."""
    if isinstance(solver, PowerFlowSolver):
        key = solver.value
    else:
        try:
            key = PowerFlowSolver(solver).value
        except ValueError as err:
            valid = ", ".join(m.value for m in PowerFlowSolver)
            raise ValueError(
                f"Invalid power_flow_solver {solver!r}; expected one of: {valid}"
            ) from err
    data.power_flow_solver = key
    net = getattr(data, "net", None)
    if net is not None:
        net.power_flow_solver = key


def primary_test_dataset(test_h5_paths: TestH5Paths) -> Tuple[str, str]:
    """Return the first configured test dataset (reference HDF5 for OPFData init)."""
    if not test_h5_paths:
        raise ValueError("At least one test dataset must be configured")
    return next(iter(test_h5_paths.items()))


def resolve_val_h5(val_h5: Optional[str]) -> Optional[str]:
    """Return stripped validation path, or ``None`` when unset."""
    if val_h5 is None:
        return None
    path = val_h5.strip()
    return path or None


def validate_val_h5(val_h5: str) -> None:
    """Fail fast if the validation HDF5 file is missing."""
    if not val_h5:
        raise ValueError("Validation dataset has an empty path")
    if not Path(val_h5).exists():
        parent = Path(val_h5).parent
        hint = ""
        if parent.is_dir():
            h5_files = sorted(p.name for p in parent.glob("*.h5"))
            if h5_files:
                hint = f"\n  HDF5 files in {parent}: {', '.join(h5_files)}"
        raise ValueError(f"Validation dataset not found at {val_h5}.{hint}")


def resolve_and_validate_val_h5(
    val_h5: Optional[str],
    *,
    print_path: bool = True,
) -> Optional[str]:
    """Resolve ``--val_h5`` and validate the file exists when provided."""
    path = resolve_val_h5(val_h5)
    if path is None:
        return None
    validate_val_h5(path)
    if print_path:
        print(f"Validation dataset: {path}")
        print()
    return path


def resolve_cost_norm_h5_path(
    data: OPFData,
    *,
    val_h5_path: Optional[str] = None,
) -> str:
    """Return the HDF5 used for cost normalization.

    Uses ``--val_h5`` when set; otherwise falls back to the reference test HDF5
    bound on ``data`` (first ``--test_h5_variants`` entry).
    """
    if val_h5_path is not None:
        return val_h5_path
    return data.test_h5_path


def _resolve_dataset_name(path: str) -> str:
    """Infer a dataset key from ``*_test_<name>.h5`` stems, else use the file stem."""
    match = _TEST_H5_STEM_RE.match(Path(path).stem)
    if match is not None:
        return match.group("dataset")
    return Path(path).stem


def _parse_test_h5_variants_list(entries: Sequence[str]) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    for entry in entries:
        if not entry.strip():
            continue

        if "=" in entry:
            name, path = parse_test_dataset_arg(entry)
        else:
            name, path = _resolve_dataset_name(entry), entry

        if name in paths:
            raise ValueError(f"Duplicate test dataset name: {name}")
        paths[name] = path

    return paths


def resolve_test_dataset_paths(
    test_h5: Optional[str] = None,
    test_h5_variants: Optional[Sequence[str]] = None,
    test_datasets: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, str]:
    """Resolve ordered test dataset paths; first entry is reference for OPFData init."""
    if test_h5_variants is not None:
        paths = _parse_test_h5_variants_list(test_h5_variants)
    elif test_h5 is not None:
        paths = {_resolve_dataset_name(test_h5): test_h5}
    else:
        raise ValueError(
            "Must provide test dataset via --test_h5 or --test_h5_variants"
        )

    if test_datasets:
        for name, path in test_datasets:
            if name in paths:
                raise ValueError(f"Duplicate test dataset name: {name}")
            paths[name] = path

    if not paths:
        raise ValueError("At least one test dataset must be configured")
    return paths


def validate_primary_test_dataset(test_h5_paths: Dict[str, str]) -> None:
    """Fail fast if the reference (first) test dataset file is missing."""
    name, path = primary_test_dataset(test_h5_paths)
    if not path:
        raise ValueError(f"Primary test dataset {name!r} has an empty path")
    if not Path(path).exists():
        parent = Path(path).parent
        hint = ""
        if parent.is_dir():
            h5_files = sorted(p.name for p in parent.glob("*.h5"))
            if h5_files:
                hint = f"\n  HDF5 files in {parent}: {', '.join(h5_files)}"
        raise ValueError(
            f"Primary test dataset {name!r} not found at {path}.{hint}"
        )


def print_test_dataset_paths(test_h5_paths: Dict[str, str]) -> None:
    print("Test dataset paths:")
    for variant, path in test_h5_paths.items():
        print(f"  {variant}: {path}")
    print()


def resolve_and_validate_test_paths(
    *,
    test_h5: Optional[str] = None,
    test_h5_variants: Optional[Sequence[str]] = None,
    test_datasets: Optional[Sequence[Tuple[str, str]]] = None,
    print_paths: bool = True,
) -> TestH5Paths:
    test_h5_paths = resolve_test_dataset_paths(
        test_h5=test_h5,
        test_h5_variants=test_h5_variants,
        test_datasets=test_datasets,
    )
    validate_primary_test_dataset(test_h5_paths)
    if print_paths:
        print_test_dataset_paths(test_h5_paths)
    return test_h5_paths


def load_opf_data(
    *,
    train_h5: str,
    data_json: str,
    test_h5_paths: TestH5Paths,
    build: OPFDataBuildOptions,
) -> OPFData:
    _, test_h5_path = primary_test_dataset(test_h5_paths)
    print("Loading data...")
    data = OPFData(
        train_h5,
        test_h5_path,
        data_json,
        topology_backend=build.topology_backend,
        woodbury_ir_steps=build.woodbury_ir_steps,
        woodbury_batch_mode=build.woodbury_batch_mode,
    )
    apply_power_flow_solver_override(data, build.power_flow_solver)
    print(
        f"  Network: {data.net.n_net} non-slack nodes, {data.net.n_slk} slack nodes, "
        f"{data.net.n_branch} branches, {data.net.n_der_phase} DER phases"
    )
    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    print(f"  Precision: x64 {'enabled' if x64_enabled else 'disabled'}")
    print(f"  Power flow solver: {data.power_flow_solver}")
    return data


def prepare_training_datasets(
    data: OPFData,
    *,
    train_samples: Optional[int],
    eval_samples: Optional[int],
    batch_size: int,
    val_h5_path: Optional[str] = None,
    log_layout: bool = True,
    validate_loads: bool = True,
) -> Tuple[OPFData, object, Optional[object], int]:
    train_dataset = data.make_train_dataset(samples=train_samples)
    eval_dataset = None
    if val_h5_path is not None:
        eval_data = (
            data.with_test_h5(val_h5_path)
            if val_h5_path != data.test_h5_path
            else data
        )
        eval_dataset = eval_data.make_test_dataset(samples=eval_samples)
    num_train_samples = len(train_dataset)
    steps_per_epoch = num_train_samples // batch_size
    print(f"  Training samples: {num_train_samples} | Steps per epoch: {steps_per_epoch}")
    if eval_dataset is not None:
        print(f"  Validation samples: {len(eval_dataset)}")

    if log_layout:
        log_nn_input_layout(data.nn_layout, n_train_samples=num_train_samples)
    if validate_loads:
        validate_load_support(data)

    return data, train_dataset, eval_dataset, steps_per_epoch


def prepare_checkpoint_test_data(
    *,
    train_h5: str,
    data_json: str,
    test_h5_paths: TestH5Paths,
    build: OPFDataBuildOptions,
    train_samples: Optional[int] = None,
) -> tuple[OPFData, object]:
    data = load_opf_data(
        train_h5=train_h5,
        data_json=data_json,
        test_h5_paths=test_h5_paths,
        build=build,
    )
    print()
    train_dataset = data.make_train_dataset(samples=train_samples)
    validate_load_support(data)
    return data, train_dataset
