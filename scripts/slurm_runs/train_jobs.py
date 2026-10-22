#!/usr/bin/env python3
"""YAML-driven training job generator and runner for Slurm arrays."""

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

SLURM_RUNS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SLURM_RUNS_DIR.parents[1]
DEFAULT_CONFIG = SLURM_RUNS_DIR / "train_jobs_penalty_direct.yaml"
DEFAULT_OUTPUT_ROOT = SLURM_RUNS_DIR
LAUNCHER_RELPATH = "scripts/slurm_runs/train_jobs.py"
DATASET_ROOT = "datagen_topo/datasets"

# Slurm resource profiles ordered by increasing footprint (for max-profile selection).
_RESOURCE_ORDER = ("small", "medium", "large")

# CLI flags that take no value when True in YAML.
_BOOLEAN_FLAGS = frozenset(
    {
        "save_best",
    }
)

# argparse.BooleanOptionalAction flags: emit --flag or --no-flag.
_BOOLEAN_OPTIONAL_FLAGS = frozenset(
    {
        "include_topo_nn_input",
    }
)


def make_results_root(results_dir: str, case_name: str, seed: int) -> str:
    """Relative training output directory for one job.

    Train entrypoints pass ``--results_dir`` and ``--seed``; checkpoints land at
    ``{results_dir}/{case_name}/seed_{seed}/`` relative to the repository root.
    """
    base = results_dir.rstrip("/")
    return f"{base}/{case_name}/seed_{seed}"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a training job YAML config."""
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    _validate_config(config)
    return config


def _normalize_train_variants(matrix: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return train-variant entries; default is a single unsuffixed training HDF5."""
    raw = matrix.get("train_variants")
    if raw is None:
        return [{"suffix": ""}]
    if not raw:
        raise ValueError("matrix.train_variants must be a non-empty list when present")
    variants: list[dict[str, str]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"matrix.train_variants[{index}] must be a mapping with key 'suffix'"
            )
        suffix = entry.get("suffix")
        if suffix is None or str(suffix).strip() == "":
            raise ValueError(
                f"matrix.train_variants[{index}].suffix must be a non-empty string"
            )
        variants.append({"suffix": str(suffix)})
    return variants


def _validate_config(config: dict[str, Any]) -> None:
    required_top = ("name", "matrix", "grid_profiles", "method_profiles", "resources")
    for key in required_top:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    matrix = config["matrix"]
    for key in ("methods", "grids", "seeds"):
        if key not in matrix or not matrix[key]:
            raise ValueError(f"matrix.{key} must be a non-empty list")

    train_variants = _normalize_train_variants(matrix)
    uses_train_variants = any(variant["suffix"] for variant in train_variants)

    for grid in matrix["grids"]:
        grid_key = str(grid)
        if grid_key not in config["grid_profiles"]:
            raise ValueError(f"grid_profiles missing entry for grid {grid_key!r}")
        profile = config["grid_profiles"][grid_key]
        if "case_dir" not in profile:
            raise ValueError(f"grid_profiles.{grid_key} missing case_dir")
        if "test_variants" not in profile or not profile["test_variants"]:
            raise ValueError(
                f"grid_profiles.{grid_key} must define non-empty test_variants"
            )
        if "val_h5" not in profile or not profile["val_h5"]:
            raise ValueError(
                f"grid_profiles.{grid_key} must define non-empty val_h5"
            )

    for method in matrix["methods"]:
        if method not in config["method_profiles"]:
            raise ValueError(f"method_profiles missing entry for method {method!r}")
        profile = config["method_profiles"][method]
        if "entrypoint" not in profile or "case_suffix" not in profile:
            raise ValueError(f"method_profiles.{method} needs entrypoint and case_suffix")
        if uses_train_variants and not profile.get("ablation_case_token"):
            raise ValueError(
                f"method_profiles.{method} must define ablation_case_token "
                "when matrix.train_variants is set"
            )


def _grid_defaults(grid_profile: dict[str, Any]) -> dict[str, Any]:
    """Shared grid-level training args excluding method-specific overrides."""
    skip = {"case_dir", "resource_profile", "test_variants", "method_overrides"}
    return {key: value for key, value in grid_profile.items() if key not in skip}


def _resolve_job_args(
    grid_profile: dict[str, Any],
    method: str,
    method_profile: dict[str, Any],
) -> dict[str, Any]:
    """Merge grid defaults, method defaults, and per-grid method overrides."""
    args: dict[str, Any] = {}
    args.update(_grid_defaults(grid_profile))
    args.update(method_profile.get("args", {}))
    overrides = grid_profile.get("method_overrides", {}).get(method, {})
    args.update(overrides)
    return args


def _dataset_paths(case_dir: str, train_suffix: str = "") -> tuple[str, str]:
    """Resolve training HDF5 and network JSON paths for one grid.

    ``train_suffix=""`` -> ``{case}_training.h5``
    ``train_suffix=060_035_005`` -> ``{case}_training_060_035_005.h5``
    """
    if train_suffix:
        train_stem = f"{case_dir}_training_{train_suffix}"
    else:
        train_stem = f"{case_dir}_training"
    train_h5 = f"{DATASET_ROOT}/{case_dir}/{train_stem}.h5"
    data_json = f"{DATASET_ROOT}/{case_dir}/network_config.json"
    return train_h5, data_json


def _build_case_name(
    case_dir: str,
    case_suffix: str,
    *,
    train_suffix: str = "",
    ablation_case_token: str | None = None,
) -> str:
    if train_suffix:
        if not ablation_case_token:
            raise ValueError(
                "ablation_case_token is required when train_suffix is non-empty"
            )
        return f"{case_dir}_{ablation_case_token}_{train_suffix}"
    return f"{case_dir}_{case_suffix}"


def _args_to_cli(args: dict[str, Any], test_variants: Sequence[str]) -> list[str]:
    """Convert resolved args dict to train.py CLI tokens."""
    cli: list[str] = []
    for key, value in sorted(args.items()):
        flag = f"--{key}"
        if key in _BOOLEAN_FLAGS:
            if value:
                cli.append(flag)
            continue
        if key in _BOOLEAN_OPTIONAL_FLAGS:
            cli.append(flag if value else f"--no-{key}")
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if isinstance(value, list):
            if not value:
                continue
            cli.append(flag)
            cli.extend(str(item) for item in value)
            continue
        cli.extend([flag, str(value)])

    if test_variants:
        cli.append("--test_h5_variants")
        cli.extend(str(variant) for variant in test_variants)
    return cli


def build_train_command(
    *,
    entrypoint: str,
    train_h5: str,
    data_json: str,
    case_name: str,
    results_dir: str,
    seed: int,
    args: dict[str, Any],
    test_variants: Sequence[str],
) -> list[str]:
    """Build the full python command for one training job (repo-relative paths)."""
    command = [
        "python",
        entrypoint,
        "--train_h5",
        train_h5,
        "--data_json",
        data_json,
        "--case_name",
        case_name,
        "--results_dir",
        results_dir,
        "--seed",
        str(seed),
    ]
    command.extend(_args_to_cli(args, test_variants))
    return command


def expand_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand matrix into job records.

    Default: ``methods × grids × seeds``.
    With ``matrix.train_variants``: ``methods × grids × seeds × train_variants``
    (train variant is the innermost loop).
    """
    matrix = config["matrix"]
    results_dir = config.get("results_dir", "results")
    train_variants = _normalize_train_variants(matrix)
    jobs: list[dict[str, Any]] = []

    for method in matrix["methods"]:
        method_profile = config["method_profiles"][method]
        entrypoint = method_profile["entrypoint"]
        case_suffix = method_profile["case_suffix"]
        ablation_case_token = method_profile.get("ablation_case_token")

        for grid in matrix["grids"]:
            grid_key = str(grid)
            grid_profile = config["grid_profiles"][grid_key]
            case_dir = grid_profile["case_dir"]
            test_variants = list(grid_profile["test_variants"])
            resource_profile = grid_profile.get("resource_profile", "medium")
            resolved_args = _resolve_job_args(grid_profile, method, method_profile)

            for seed in matrix["seeds"]:
                seed_int = int(seed)
                for variant in train_variants:
                    train_suffix = variant["suffix"]
                    train_h5, data_json = _dataset_paths(case_dir, train_suffix)
                    case_name = _build_case_name(
                        case_dir,
                        case_suffix,
                        train_suffix=train_suffix,
                        ablation_case_token=ablation_case_token,
                    )
                    command = build_train_command(
                        entrypoint=entrypoint,
                        train_h5=train_h5,
                        data_json=data_json,
                        case_name=case_name,
                        results_dir=results_dir,
                        seed=seed_int,
                        args=resolved_args,
                        test_variants=test_variants,
                    )
                    jobs.append(
                        {
                            "method": method,
                            "grid": grid_key,
                            "seed": seed_int,
                            "train_suffix": train_suffix,
                            "case_name": case_name,
                            "case_dir": case_dir,
                            "results_dir": results_dir,
                            "results_root": make_results_root(
                                results_dir, case_name, seed_int
                            ),
                            "resource_profile": resource_profile,
                            "train_h5": train_h5,
                            "data_json": data_json,
                            "test_variants": test_variants,
                            "val_h5": grid_profile["val_h5"],
                            "args": resolved_args,
                            "command": command,
                        }
                    )

    for index, job in enumerate(jobs):
        job["index"] = index
    return jobs


def _check_train_h5_exist(jobs: Sequence[dict[str, Any]]) -> None:
    """Fail fast when any referenced training HDF5 is missing on disk."""
    missing: list[str] = []
    for job in jobs:
        train_h5 = str(job["train_h5"])
        if not (PROJECT_ROOT / train_h5).is_file():
            missing.append(train_h5)
    if missing:
        lines = "\n".join(f"  - {path}" for path in sorted(set(missing)))
        raise FileNotFoundError(
            "Missing training HDF5 file(s) referenced by config:\n" + lines
        )


def _resource_rank(profile_name: str) -> int:
    try:
        return _RESOURCE_ORDER.index(profile_name)
    except ValueError:
        return len(_RESOURCE_ORDER)


def _select_dominant_resource_profile(
    jobs: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    slurm_cfg = config.get("slurm", {})
    default_name = slurm_cfg.get("default_resource_profile")
    if default_name:
        return str(default_name)
    if not jobs:
        return "medium"
    return max(
        (str(job["resource_profile"]) for job in jobs),
        key=_resource_rank,
    )


def _format_gpus(profile: Mapping[str, Any], slurm_cfg: Mapping[str, Any]) -> str:
    """Build ``-G [type:]count`` from a resource profile."""
    gpus = int(profile["gpus"])
    gpu_type = profile.get("gpu_type") or slurm_cfg.get("gpu_type")
    if gpu_type:
        return f"{gpu_type}:{gpus}"
    return f"{gpus}"


def _sbatch_directives(
    *,
    config: dict[str, Any],
    resource_profile: str,
    job_name: str,
    log_dir: str,
    array_spec: str,
) -> list[str]:
    resources = config["resources"]
    if resource_profile not in resources:
        raise ValueError(f"Unknown resource profile: {resource_profile!r}")
    profile = resources[resource_profile]
    slurm_cfg = config.get("slurm", {})
    lines = [
        "#!/bin/bash",
        "#SBATCH --requeue",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/%A_%a.out",
        f"#SBATCH --error={log_dir}/%A_%a.err",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --time={profile['time']}",
        f"#SBATCH --cpus-per-task={profile['cpus_per_task']}",
        f"#SBATCH --mem={profile['mem']}",
        f"#SBATCH -G {_format_gpus(profile, slurm_cfg)}",
    ]
    gpu_constraint = profile.get("gpu_constraint") or slurm_cfg.get("gpu_constraint")
    if gpu_constraint:
        lines.append(f"#SBATCH --constraint={gpu_constraint}")
    for key in ("partition", "account", "qos"):
        if key in profile:
            lines.append(f"#SBATCH --{key}={profile[key]}")
    for extra in profile.get("extra_sbatch", []) or slurm_cfg.get("extra_sbatch", []) or []:
        lines.append(str(extra))
    return lines


def _conda_activation_lines(conda_env: str) -> list[str]:
    # Conda activate/deactivate hooks (e.g. cuda-nvcc) may read unset variables.
    return [
        "set +u",
        'if [ -f "${CONDA_EXE%/*}/../etc/profile.d/conda.sh" ]; then',
        '  source "${CONDA_EXE%/*}/../etc/profile.d/conda.sh"',
        'elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then',
        '  source "$HOME/miniconda3/etc/profile.d/conda.sh"',
        'elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then',
        '  source "$HOME/anaconda3/etc/profile.d/conda.sh"',
        "else",
        '  echo "ERROR: conda.sh not found" >&2',
        "  exit 1",
        "fi",
        f"conda activate {conda_env}",
        "set -u",
    ]


def _jobs_file_bash_line(submit_script: Path) -> str:
    """Bash assignment for jobs.jsonl next to the generated submit script."""
    jobs_path = submit_script.parent / "jobs.jsonl"
    try:
        jobs_relpath = jobs_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # Test artifacts may live outside the repo; resolve beside the script.
        return 'JOBS_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/jobs.jsonl"'
    return f'JOBS_FILE="$PROJECT_DIR/{jobs_relpath}"'


_SUBMIT_METADATA_PREFIX = "# train_jobs:"


def _array_spec_from_indices(
    job_indices: Sequence[int],
    *,
    max_parallel: int,
) -> str:
    """Build a Slurm ``#SBATCH --array`` value from jobs.jsonl indices."""
    if not job_indices:
        raise ValueError("job_indices must be non-empty")
    array_max = max(job_indices)
    array_min = min(job_indices)
    ordered = [int(index) for index in job_indices]
    is_contiguous = ordered == list(range(array_min, array_max + 1))
    if is_contiguous:
        if array_min == array_max:
            array_spec = str(array_min)
        else:
            array_spec = f"{array_min}-{array_max}"
    else:
        array_spec = ",".join(str(index) for index in ordered)
    if max_parallel and array_spec:
        array_spec = f"{array_spec}%{max_parallel}"
    return array_spec


def parse_slurm_array_spec(spec: str) -> list[int]:
    """Parse a Slurm ``--array`` value into sorted unique jobs.jsonl indices."""
    body = spec.strip()
    if "%" in body:
        body = body.split("%", 1)[0]
    indices: list[int] = []
    for part in body.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid Slurm array range: {token!r}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(token))
    if not indices:
        raise ValueError(f"Empty Slurm array spec: {spec!r}")
    return sorted(set(indices))


def _parse_submit_script_metadata(script_text: str) -> dict[str, str]:
    """Extract machine-readable metadata comments from a generated submit script."""
    metadata: dict[str, str] = {}
    for line in script_text.splitlines():
        if not line.startswith(_SUBMIT_METADATA_PREFIX):
            continue
        key, _, value = line[len(_SUBMIT_METADATA_PREFIX) :].strip().partition("=")
        if key and value:
            metadata[key.strip()] = value.strip()
    return metadata


def _submit_script_header_lines(
    *,
    resource_profile: str,
    job_indices: Sequence[int],
    jobs_by_index: Mapping[int, Mapping[str, Any]],
) -> list[str]:
    """Human- and machine-readable index mapping for generated submit scripts."""
    lines = [
        "# Generated by train_jobs.py — regenerate instead of editing array ranges.",
        f"{_SUBMIT_METADATA_PREFIX} resource_profile={resource_profile}",
        (
            f"{_SUBMIT_METADATA_PREFIX} job_indices="
            f"{','.join(str(index) for index in job_indices)}"
        ),
        "#",
        "# SLURM_ARRAY_TASK_ID selects jobs.jsonl index (not feeder/grid id):",
    ]
    for index in job_indices:
        job = jobs_by_index[int(index)]
        suffix = str(job.get("train_suffix", "") or "default")
        lines.append(
            f"#   [{index}] grid={job['grid']} suffix={suffix}  {job['case_name']}"
        )
    lines.append(
        "# Submit without overriding --array; use train_jobs.py verify after regenerate."
    )
    return lines


def _submit_script_index_guard_lines(job_indices: Sequence[int]) -> list[str]:
    """Fail fast when sbatch --array overrides point at the wrong jobs.jsonl indices."""
    allowed = "|".join(str(index) for index in job_indices)
    script_name = '$(basename "${BASH_SOURCE[0]}")'
    return [
        f'case "$TASK_ID" in',
        f"  {allowed}) ;;",
        "  *)",
        f'    echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID is not allowed for {script_name}" >&2',
        f'    echo "Allowed jobs.jsonl indices: {" ".join(str(index) for index in job_indices)}" >&2',
        '    echo "Regenerate with train_jobs.py generate and sbatch this script without --array overrides." >&2',
        "    exit 1",
        "    ;;",
        "esac",
        "",
    ]


def verify_artifacts(
    output_dir: Path,
    jobs: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Return validation errors for jobs.jsonl vs generated submit_*.sh scripts."""
    jobs_path = output_dir / "jobs.jsonl"
    if jobs is None:
        if not jobs_path.exists():
            return [f"Missing jobs file: {jobs_path}"]
        jobs = load_jobs(jobs_path)

    jobs_by_index = {int(job["index"]): job for job in jobs}
    expected_by_profile: dict[str, list[int]] = {}
    for job in jobs:
        profile = str(job["resource_profile"])
        expected_by_profile.setdefault(profile, []).append(int(job["index"]))
    for profile in expected_by_profile:
        expected_by_profile[profile].sort()

    submit_scripts = sorted(output_dir.glob("submit*.sh"))
    if not submit_scripts:
        return [f"No submit scripts found in {output_dir}"]

    errors: list[str] = []
    seen_profiles: set[str] = set()
    for script_path in submit_scripts:
        script_text = script_path.read_text(encoding="utf-8")
        metadata = _parse_submit_script_metadata(script_text)
        profile = metadata.get("resource_profile")
        if profile is None:
            stem = script_path.stem
            if stem.startswith("submit_") and stem != "submit":
                profile = stem.removeprefix("submit_")
            else:
                errors.append(
                    f"{script_path.name}: missing {_SUBMIT_METADATA_PREFIX} resource_profile "
                    "(regenerate submit scripts)"
                )
                continue

        array_line = next(
            (line for line in script_text.splitlines() if line.startswith("#SBATCH --array=")),
            None,
        )
        if array_line is None:
            errors.append(f"{script_path.name}: missing #SBATCH --array directive")
            continue

        array_spec = array_line.removeprefix("#SBATCH --array=").strip()
        try:
            actual_indices = parse_slurm_array_spec(array_spec)
        except ValueError as exc:
            errors.append(f"{script_path.name}: {exc}")
            continue

        metadata_indices = metadata.get("job_indices")
        if metadata_indices is not None:
            declared_indices = [int(part) for part in metadata_indices.split(",") if part]
            if declared_indices != actual_indices:
                errors.append(
                    f"{script_path.name}: metadata job_indices={declared_indices} "
                    f"does not match #SBATCH --array={array_spec!r}"
                )

        expected_indices = expected_by_profile.get(profile)
        if expected_indices is None:
            errors.append(
                f"{script_path.name}: resource profile {profile!r} is not used by any job"
            )
            continue
        if actual_indices != expected_indices:
            errors.append(
                f"{script_path.name}: array indices {actual_indices} do not match "
                f"jobs.jsonl indices for profile {profile!r}: {expected_indices}"
            )

        for index in actual_indices:
            job = jobs_by_index.get(index)
            if job is None:
                errors.append(f"{script_path.name}: jobs.jsonl has no index {index}")
                continue
            if str(job["resource_profile"]) != profile:
                errors.append(
                    f"{script_path.name}: index {index} ({job['case_name']}) uses profile "
                    f"{job['resource_profile']!r}, not {profile!r}"
                )

        if profile in seen_profiles:
            errors.append(f"Duplicate submit script for resource profile {profile!r}")
        seen_profiles.add(profile)

    for profile, expected_indices in expected_by_profile.items():
        if profile not in seen_profiles:
            errors.append(
                f"Missing submit script for resource profile {profile!r} "
                f"(expected indices {expected_indices})"
            )

    return errors


def _project_dir_bash_lines() -> list[str]:
    """Resolve repo root inside Slurm batch scripts.

    Slurm may copy the batch script to a spool directory (e.g. under /var), so
    ``BASH_SOURCE[0]`` is unreliable. ``SLURM_SUBMIT_DIR`` is the cwd from which
    ``sbatch`` was invoked and is preferred when it contains the launcher.
    """
    return [
        'if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/slurm_runs/train_jobs.py" ]; then',
        '  PROJECT_DIR="$SLURM_SUBMIT_DIR"',
        "else",
        '  SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        '  PROJECT_DIR="$(cd "$SUBMIT_DIR/../../.." && pwd)"',
        "fi",
        'cd "$PROJECT_DIR"',
        "",
        'if [ ! -f "$PROJECT_DIR/scripts/slurm_runs/train_jobs.py" ]; then',
        '  echo "ERROR: repository root not found at $PROJECT_DIR" >&2',
        '  echo "Run sbatch from the repository root (see scripts/slurm_runs/README.md)." >&2',
        "  exit 1",
        "fi",
    ]


def _write_submit_script(
    *,
    path: Path,
    config: dict[str, Any],
    resource_profile: str,
    job_indices: Sequence[int],
    jobs_by_index: Mapping[int, Mapping[str, Any]],
) -> None:
    max_parallel = int(config.get("max_parallel", 8))
    slurm_cfg = config.get("slurm", {})
    env_cfg = config.get("env", {})
    conda_env = env_cfg.get("conda_env", "new_opt")
    config_name = config["name"]
    log_dir = str(Path(slurm_cfg.get("log_dir", "logs/training")) / config_name).replace("\\", "/")
    ordered_indices = [int(index) for index in job_indices]
    array_spec = _array_spec_from_indices(ordered_indices, max_parallel=max_parallel)

    sbatch_lines = _sbatch_directives(
        config=config,
        resource_profile=resource_profile,
        job_name=slurm_cfg.get("job_name", "train"),
        log_dir=log_dir,
        array_spec=array_spec,
    )
    lines = [sbatch_lines[0]]
    lines.extend(
        _submit_script_header_lines(
            resource_profile=resource_profile,
            job_indices=ordered_indices,
            jobs_by_index=jobs_by_index,
        )
    )
    lines.extend(sbatch_lines[1:])
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            *_project_dir_bash_lines(),
            "",
            *(_conda_activation_lines(conda_env)),
            "",
            _jobs_file_bash_line(path),
            'TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"',
            "",
            *_submit_script_index_guard_lines(ordered_indices),
            (
                f'python "$PROJECT_DIR/{LAUNCHER_RELPATH}" run --jobs "$JOBS_FILE" '
                f'--index "$TASK_ID" --expect-resource-profile {resource_profile}'
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _write_readme(
    path: Path,
    *,
    config_name: str,
    output_dir: Path,
    submit_scripts: Sequence[tuple[str, Path, int]],
    total_jobs: int,
    results_dir: str,
) -> None:
    lines = [
        f"# Generated training jobs: {config_name}",
        "",
        f"Total jobs: {total_jobs}",
        "",
        "## Results layout",
        "",
        f"Each job writes to `{{results_dir}}/{{case_name}}/seed_{{seed}}/`.",
        f"With `results_dir: {results_dir}`, example:",
        f"`{results_dir}/IEEE13Nodeckt_penalty_pf/seed_1/`.",
        "",
        "## Submit to Slurm",
        "",
        "Run `sbatch` from the **repository root** (Slurm log paths are relative to that directory):",
        "",
        "```bash",
        "cd /path/to/ACOPF-Multiphase-Distribution-topo",
        "```",
        "",
    ]
    for label, script_path, n_jobs in submit_scripts:
        try:
            display = str(script_path.relative_to(PROJECT_ROOT))
        except ValueError:
            display = str(script_path)
        lines.append(f"- `{label}` ({n_jobs} jobs): `sbatch {display}`")
    try:
        jobs_rel = str((output_dir / "jobs.jsonl").relative_to(PROJECT_ROOT))
    except ValueError:
        jobs_rel = str(output_dir / "jobs.jsonl")
    lines.extend(
        [
            "",
            "Generated `submit.sh` discovers the repo root from its own location;",
            "no machine-specific paths are embedded.",
            "",
            "## Run one job locally",
            "",
            "```bash",
            f"python scripts/slurm_runs/train_jobs.py run --jobs {jobs_rel} --index 0 --dry-run",
            "```",
            "",
            "## Regenerate and verify",
            "",
            "```bash",
            "python scripts/slurm_runs/train_jobs.py generate --config scripts/slurm_runs/train_jobs_<method>.yaml",
            "python scripts/slurm_runs/train_jobs.py verify --jobs-dir "
            + (
                str(output_dir.relative_to(PROJECT_ROOT))
                if output_dir.is_relative_to(PROJECT_ROOT)
                else str(output_dir)
            ),
            "```",
            "",
            "Do not pass custom `sbatch --array=...` overrides; each submit script",
            "embeds the jobs.jsonl indices it is allowed to run.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_artifacts(
    config_path: Path,
    output_dir: Path | None = None,
    *,
    check_train_h5: bool = False,
) -> Path:
    """Expand config and write jobs.jsonl, submit.sh, and README.md."""
    config = load_config(config_path)
    jobs = expand_jobs(config)
    if check_train_h5:
        _check_train_h5_exist(jobs)
    config_name = config["name"]
    out_dir = output_dir or (DEFAULT_OUTPUT_ROOT / config_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs_path = out_dir / "jobs.jsonl"
    with jobs_path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, sort_keys=True))
            handle.write("\n")

    jobs_by_index = {int(job["index"]): job for job in jobs}
    group_by_resource = bool(config.get("group_by_resource", False))
    submit_scripts: list[tuple[str, Path, int]] = []

    if group_by_resource:
        by_profile: dict[str, list[int]] = {}
        for job in jobs:
            profile = str(job["resource_profile"])
            by_profile.setdefault(profile, []).append(int(job["index"]))
        for profile_name, indices in sorted(by_profile.items()):
            script_path = out_dir / f"submit_{profile_name}.sh"
            _write_submit_script(
                path=script_path,
                config=config,
                resource_profile=profile_name,
                job_indices=indices,
                jobs_by_index=jobs_by_index,
            )
            submit_scripts.append((f"submit_{profile_name}.sh", script_path, len(indices)))
    else:
        dominant = _select_dominant_resource_profile(jobs, config)
        script_path = out_dir / "submit.sh"
        indices = [int(job["index"]) for job in jobs]
        _write_submit_script(
            path=script_path,
            config=config,
            resource_profile=dominant,
            job_indices=indices,
            jobs_by_index=jobs_by_index,
        )
        submit_scripts.append(("submit.sh", script_path, len(jobs)))

    verify_errors = verify_artifacts(out_dir, jobs)
    if verify_errors:
        joined = "\n".join(f"  - {error}" for error in verify_errors)
        raise RuntimeError(f"Generated artifacts failed verification:\n{joined}")

    _write_readme(
        out_dir / "README.md",
        config_name=config_name,
        output_dir=out_dir,
        submit_scripts=submit_scripts,
        total_jobs=len(jobs),
        results_dir=config.get("results_dir", "results"),
    )
    return out_dir


def load_jobs(jobs_path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with jobs_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def find_job(
    jobs: Sequence[dict[str, Any]],
    *,
    index: int | None = None,
    method: str | None = None,
    grid: str | None = None,
    seed: int | None = None,
    train_suffix: str | None = None,
) -> dict[str, Any]:
    if index is not None:
        for job in jobs:
            if int(job["index"]) == index:
                return job
        raise KeyError(f"No job with index {index}")

    matches = [
        job
        for job in jobs
        if (method is None or job["method"] == method)
        and (grid is None or str(job["grid"]) == str(grid))
        and (seed is None or int(job["seed"]) == seed)
        and (
            train_suffix is None
            or str(job.get("train_suffix", "")) == train_suffix
        )
    ]
    if not matches:
        raise KeyError(
            f"No job matching method={method!r} grid={grid!r} seed={seed!r} "
            f"train_suffix={train_suffix!r}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"Multiple jobs match method={method!r} grid={grid!r} seed={seed!r} "
            f"train_suffix={train_suffix!r}; use --index"
        )
    return matches[0]


def _validate_job_resource_profile(
    job: Mapping[str, Any],
    *,
    expect_resource_profile: str | None,
) -> None:
    if expect_resource_profile is None:
        return
    actual_profile = str(job["resource_profile"])
    if actual_profile != expect_resource_profile:
        suffix = job.get("train_suffix", "")
        suffix_label = f" suffix={suffix}" if suffix else ""
        raise ValueError(
            f"Job index {job['index']} is grid={job['grid']}{suffix_label} "
            f"({job['case_name']}) with resource_profile={actual_profile!r}, but the "
            f"submit script expects {expect_resource_profile!r}. Regenerate with "
            "train_jobs.py generate and sbatch without overriding --array."
        )


def run_job_record(
    job: dict[str, Any],
    *,
    dry_run: bool = False,
    expect_resource_profile: str | None = None,
) -> int:
    _validate_job_resource_profile(
        job,
        expect_resource_profile=expect_resource_profile,
    )
    command = [str(part) for part in job["command"]]
    print(
        f"Running job index={job['index']} method={job['method']} "
        f"grid={job['grid']} seed={job['seed']}"
        + (
            f" train_suffix={job['train_suffix']}"
            if job.get("train_suffix")
            else ""
        )
    )
    print(f"Results root: {job.get('results_root', '(unknown)')}")
    print("Command:", " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return 0
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(result.returncode)


def _cmd_generate(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    out = generate_artifacts(
        config_path,
        output_dir,
        check_train_h5=bool(args.check_train_h5),
    )
    print(f"Generated {out / 'jobs.jsonl'} ({len(load_jobs(out / 'jobs.jsonl'))} jobs)")
    for name in ("submit.sh", "README.md"):
        candidate = out / name
        if candidate.exists():
            print(f"  {candidate}")
    for script in sorted(out.glob("submit_*.sh")):
        if script.name != "submit.sh":
            print(f"  {script}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    jobs_path = Path(args.jobs).resolve()
    jobs = load_jobs(jobs_path)
    job = find_job(
        jobs,
        index=args.index,
        method=args.method,
        grid=args.grid,
        seed=args.seed,
        train_suffix=args.train_suffix,
    )
    return run_job_record(
        job,
        dry_run=args.dry_run,
        expect_resource_profile=args.expect_resource_profile,
    )


def _cmd_verify(args: argparse.Namespace) -> int:
    jobs_dir = Path(args.jobs_dir).resolve()
    jobs = load_jobs(jobs_dir / "jobs.jsonl") if (jobs_dir / "jobs.jsonl").exists() else None
    errors = verify_artifacts(jobs_dir, jobs)
    if errors:
        print(f"Verification failed for {jobs_dir}:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    job_count = len(jobs) if jobs is not None else len(load_jobs(jobs_dir / "jobs.jsonl"))
    print(f"OK: {jobs_dir} ({job_count} jobs, submit scripts match jobs.jsonl)")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    jobs = expand_jobs(config)
    for job in jobs:
        suffix = job.get("train_suffix", "")
        suffix_label = f"  suffix={suffix}" if suffix else ""
        print(
            f"{job['index']:3d}  {job['method']:14s}  grid={job['grid']:>4s}  "
            f"seed={job['seed']}{suffix_label}  {job['results_root']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and run training jobs from YAML config.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="Expand YAML config into jobs.jsonl and submit.sh"
    )
    generate_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"Path to YAML config (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)})",
    )
    generate_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: scripts/slurm_runs/<config.name>)",
    )
    generate_parser.add_argument(
        "--check-train-h5",
        action="store_true",
        help="Fail if any referenced training HDF5 is missing on disk",
    )
    generate_parser.set_defaults(func=_cmd_generate)

    run_parser = subparsers.add_parser("run", help="Run one expanded job")
    run_parser.add_argument("--jobs", type=str, required=True, help="Path to jobs.jsonl")
    run_parser.add_argument("--index", type=int, default=None, help="Job index")
    run_parser.add_argument("--method", type=str, default=None)
    run_parser.add_argument("--grid", type=str, default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument(
        "--train-suffix",
        type=str,
        default=None,
        help="Filter by matrix.train_variants suffix (e.g. 060_035_005)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    run_parser.add_argument(
        "--expect-resource-profile",
        type=str,
        default=None,
        help="Reject jobs whose resource_profile does not match the submit script",
    )
    run_parser.set_defaults(func=_cmd_run)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Check jobs.jsonl indices match generated submit_*.sh array ranges",
    )
    verify_parser.add_argument(
        "--jobs-dir",
        type=str,
        required=True,
        help="Directory containing jobs.jsonl and submit_*.sh",
    )
    verify_parser.set_defaults(func=_cmd_verify)

    list_parser = subparsers.add_parser("list", help="List expanded jobs from config")
    list_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
    )
    list_parser.set_defaults(func=_cmd_list)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate" and args.config is None:
        args.config = str(DEFAULT_CONFIG)
    if args.command == "list" and args.config is None:
        args.config = str(DEFAULT_CONFIG)
    if args.command == "run" and args.index is None and not (
        args.method and args.grid and args.seed is not None
    ):
        parser.error("run requires --index or (--method, --grid, --seed)")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
