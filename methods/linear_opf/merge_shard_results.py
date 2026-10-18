"""Merge per-shard linear OPF benchmark JSON into canonical variant results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_SUM_KEYS = (
    "num_batches",
    "num_samples",
    "num_skipped",
    "num_evaluated",
    "num_qp_failed",
    "num_qp_solved",
    "total_inverse_time",
    "total_qp_time",
    "total_linearization_time",
    "total_qp_solver_time",
    "total_qp_wall_time",
    "total_qp_overhead_time",
    "total_pf_time",
    "total_e2e_time",
    "total_solve_time",
    "total_solver_pf_time",
)

_WEIGHTED_MEAN_KEYS = (
    "cost",
    "cost_target",
    "distance",
    "violation",
    "violation_target",
    "cost_error",
    "abs_cost_error",
    "optimality_gap",
    "speedup",
    "solver_speedup",
    "solver_pf_speedup",
    "pder_l1_norm",
    "pder_l2_norm",
    "qder_l1_norm",
    "qder_l2_norm",
)

_MAX_KEYS = (
    "max_violation",
    "max_violation_target",
    "max_abs_cost_error",
    "pder_l_inf_norm",
    "qder_l_inf_norm",
)


def _weighted_mean(
    shards: list[dict[str, Any]], key: str, weight_key: str = "num_evaluated"
) -> float:
    total_weight = sum(int(s.get(weight_key, 0)) for s in shards)
    if total_weight == 0:
        return 0.0
    return sum(float(s[key]) * int(s.get(weight_key, 0)) for s in shards) / total_weight


def _max_over_shards(shards: list[dict[str, Any]], key: str) -> float:
    return max(float(s.get(key, 0.0)) for s in shards)


def merge_shard_metrics(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge shard summary dicts using the same weighting as ``benchmark_linear_opf``."""
    if not shards:
        raise ValueError("cannot merge empty shard list")

    num_shards = {int(s["num_shards"]) for s in shards if "num_shards" in s}
    if len(num_shards) != 1:
        raise ValueError(f"inconsistent num_shards metadata: {num_shards}")
    expected = num_shards.pop()

    shard_ids = sorted(int(s["shard_id"]) for s in shards)
    if shard_ids != list(range(expected)):
        raise ValueError(
            f"expected shard_id 0..{expected - 1}, got {shard_ids}"
        )

    batch_size = shards[0].get("batch_size")
    if batch_size is None:
        raise ValueError("shard metrics missing batch_size")
    if any(s.get("batch_size") != batch_size for s in shards):
        raise ValueError("all shards must share the same batch_size")

    merged: dict[str, Any] = {"batch_size": batch_size}

    for key in _SUM_KEYS:
        if key in shards[0]:
            merged[key] = sum(int(s.get(key, 0)) for s in shards)

    if "num_evaluated" not in merged and "num_samples" in merged:
        merged["num_evaluated"] = int(merged["num_samples"]) - int(
            merged.get("num_skipped", 0)
        )
    if "num_qp_solved" not in merged and "num_evaluated" in merged:
        merged["num_qp_solved"] = int(merged["num_evaluated"]) - int(
            merged.get("num_qp_failed", 0)
        )

    num_batches = int(merged.get("num_batches", 0))
    num_samples = int(merged.get("num_samples", 0))
    num_evaluated = int(merged.get("num_evaluated", 0))

    for key in _WEIGHTED_MEAN_KEYS:
        if key in shards[0]:
            merged[key] = _weighted_mean(shards, key)

    for key in _MAX_KEYS:
        if key in shards[0]:
            merged[key] = _max_over_shards(shards, key)

    total_inverse = float(merged.get("total_inverse_time", 0.0))
    total_qp = float(merged.get("total_qp_time", 0.0))
    total_lin = float(merged.get("total_linearization_time", 0.0))
    total_qp_solver = float(merged.get("total_qp_solver_time", 0.0))
    total_qp_wall = float(merged.get("total_qp_wall_time", 0.0))
    total_qp_overhead = float(merged.get("total_qp_overhead_time", 0.0))
    total_pf = float(merged.get("total_pf_time", 0.0))
    total_e2e = float(merged.get("total_e2e_time", 0.0))
    total_solve = float(merged.get("total_solve_time", 0.0))
    total_solver_pf = float(merged.get("total_solver_pf_time", 0.0))

    merged["avg_inverse_time_per_batch"] = (
        total_inverse / num_batches if num_batches else 0.0
    )
    merged["avg_inverse_time_per_sample"] = (
        total_inverse / num_samples if num_samples else 0.0
    )
    merged["avg_qp_time_per_batch"] = total_qp / num_batches if num_batches else 0.0
    merged["avg_qp_time_per_sample"] = total_qp / num_samples if num_samples else 0.0
    merged["avg_linearization_time_per_sample"] = (
        total_lin / num_samples if num_samples else 0.0
    )
    merged["avg_qp_solver_time_per_sample"] = (
        total_qp_solver / num_samples if num_samples else 0.0
    )
    merged["avg_qp_wall_time_per_sample"] = (
        total_qp_wall / num_samples if num_samples else 0.0
    )
    merged["avg_qp_overhead_time_per_sample"] = (
        total_qp_overhead / num_samples if num_samples else 0.0
    )
    merged["avg_pf_time_per_batch"] = total_pf / num_batches if num_batches else 0.0
    merged["avg_pf_time_per_sample"] = total_pf / num_evaluated if num_evaluated else 0.0
    merged["avg_e2e_time_per_batch"] = total_e2e / num_batches if num_batches else 0.0
    merged["avg_e2e_time_per_sample"] = total_e2e / num_samples if num_samples else 0.0
    merged["mean_qp_time"] = total_qp / num_samples if num_samples else 0.0
    merged["mean_pf_time"] = total_pf / num_evaluated if num_evaluated else 0.0
    merged["avg_solver_pf_time_per_sample"] = (
        total_solver_pf / num_evaluated if num_evaluated else 0.0
    )
    merged["avg_solve_time_per_sample"] = (
        total_solve / num_evaluated if num_evaluated else 0.0
    )
    merged["avg_ipopt_solve_time_per_sample"] = merged["avg_solve_time_per_sample"]
    merged["avg_clarabel_solve_time_per_sample"] = (
        total_qp_solver / num_samples if num_samples else 0.0
    )

    merged["num_shards"] = expected
    merged["merged_from_shards"] = True
    return merged


def load_shard_summaries(variant_dir: Path, batch_size: int) -> list[dict[str, Any]]:
    """Load contiguous shard summary JSON files from ``variant_dir/shards/``."""
    shards_root = variant_dir / "shards"
    if not shards_root.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shards_root}")

    shard_dirs = sorted(
        p for p in shards_root.iterdir() if p.is_dir() and p.name.startswith("shard_")
    )
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_* directories under {shards_root}")

    summaries: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / f"testing_results_batch_{batch_size}.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        with open(summary_path) as f:
            summaries.append(json.load(f))
    return summaries


def merge_shard_batch_details(
    variant_dir: Path, batch_size: int
) -> list[dict[str, Any]] | None:
    """Concatenate per-shard batch detail lists when all shards provide them."""
    shards_root = variant_dir / "shards"
    if not shards_root.is_dir():
        return None

    all_details: list[dict[str, Any]] = []
    found_any = False
    for shard_dir in sorted(
        p for p in shards_root.iterdir() if p.is_dir() and p.name.startswith("shard_")
    ):
        details_path = shard_dir / f"testing_batch_details_batch_{batch_size}.json"
        if not details_path.is_file():
            return None
        found_any = True
        with open(details_path) as f:
            all_details.extend(json.load(f))

    return all_details if found_any else None


def merge_variant_shards(variant_dir: Path, batch_size: int) -> tuple[Path, Path | None]:
    """Merge shard JSONs and write canonical files under ``variant_dir``."""
    shards = load_shard_summaries(variant_dir, batch_size)
    merged = merge_shard_metrics(shards)

    summary_path = variant_dir / f"testing_results_batch_{batch_size}.json"
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)

    details_path = None
    batch_details = merge_shard_batch_details(variant_dir, batch_size)
    if batch_details is not None:
        details_path = variant_dir / f"testing_batch_details_batch_{batch_size}.json"
        with open(details_path, "w") as f:
            json.dump(batch_details, f, indent=2)

    return summary_path, details_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sharded linear OPF test JSON into canonical variant results"
    )
    parser.add_argument(
        "--variant_dir",
        type=str,
        required=True,
        help="Path to testing/<variant_name>/ containing shards/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size key for testing_results_batch_<N>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant_dir = Path(args.variant_dir)
    summary_path, details_path = merge_variant_shards(variant_dir, args.batch_size)
    print(f"Merged summary written to {summary_path}")
    if details_path is not None:
        print(f"Merged batch details written to {details_path}")


if __name__ == "__main__":
    main()
