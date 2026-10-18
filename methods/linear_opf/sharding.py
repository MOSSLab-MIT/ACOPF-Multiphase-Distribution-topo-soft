"""Sample sharding helpers for parallel linear OPF benchmark jobs."""

from __future__ import annotations

import math
from pathlib import Path

import h5py


def count_h5_samples(h5_path: str | Path) -> int:
    """Return the number of samples in an OPF HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        return int(f["input"]["pd"].shape[0])


def compute_shard_range(
    total_samples: int,
    shard_id: int,
    num_shards: int,
    *,
    max_samples: int | None = None,
) -> tuple[int, int]:
    """Return ``(sample_offset, sample_count)`` for a shard index.

    Shards partition ``[0, total)`` (optionally capped by ``max_samples``) into
  ``num_shards`` contiguous blocks with sizes differing by at most one sample.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(
            f"shard_id must be in [0, {num_shards}), got {shard_id}"
        )
    if total_samples < 0:
        raise ValueError(f"total_samples must be non-negative, got {total_samples}")

    effective_total = total_samples
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError(f"max_samples must be non-negative, got {max_samples}")
        effective_total = min(total_samples, max_samples)

    if effective_total == 0:
        raise ValueError(
            f"empty sample range (total_samples={total_samples}, "
            f"max_samples={max_samples})"
        )

    shard_size = math.ceil(effective_total / num_shards)
    offset = shard_id * shard_size
    if offset >= effective_total:
        raise ValueError(
            f"shard {shard_id} has no samples "
            f"(effective_total={effective_total}, num_shards={num_shards})"
        )
    count = min(shard_size, effective_total - offset)
    return offset, count


def resolve_shard_test_range(
    h5_path: str | Path,
    *,
    test_samples: int | None,
    shard_id: int | None,
    num_shards: int | None,
    test_sample_offset: int | None = None,
) -> tuple[int, int, int, int]:
    """Resolve shard metadata and the ``(offset, count)`` range for one job.

    Returns ``(sample_offset, sample_count, shard_id, num_shards)``.
    When sharding is disabled (``num_shards`` is 1 or unset), ``shard_id`` is 0.
    """
    if shard_id is not None and num_shards is not None:
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(
                f"shard_id must be in [0, {num_shards}), got {shard_id}"
            )
    elif shard_id is not None or num_shards is not None:
        raise ValueError("shard_id and num_shards must both be set or both omitted")

    total = count_h5_samples(h5_path)
    resolved_num_shards = num_shards if num_shards is not None else 1
    resolved_shard_id = shard_id if shard_id is not None else 0

    if test_sample_offset is not None:
        if test_sample_offset < 0:
            raise ValueError(
                f"test_sample_offset must be non-negative, got {test_sample_offset}"
            )
        if resolved_num_shards != 1:
            raise ValueError(
                "test_sample_offset cannot be combined with num_shards > 1"
            )
        count = test_samples if test_samples is not None else total - test_sample_offset
        if count <= 0 or test_sample_offset + count > total:
            raise ValueError(
                f"invalid manual range offset={test_sample_offset} count={count} "
                f"for total={total}"
            )
        return test_sample_offset, count, 0, 1

    offset, count = compute_shard_range(
        total,
        resolved_shard_id,
        resolved_num_shards,
        max_samples=test_samples,
    )
    return offset, count, resolved_shard_id, resolved_num_shards


def shard_results_dir(
    testing_root: Path,
    variant_name: str,
    shard_id: int,
    num_shards: int,
) -> Path:
    """Directory for shard JSON output (canonical dir when ``num_shards == 1``)."""
    variant_dir = testing_root / variant_name
    if num_shards <= 1:
        return variant_dir
    return variant_dir / "shards" / f"shard_{shard_id:02d}"
