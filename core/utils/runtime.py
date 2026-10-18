"""Runtime utilities and bootstrap helpers for training scripts."""

import gc
import os
import warnings

import jax
import numpy as np
import optax
import tensorflow as tf

_BYTES_PER_GB = 1024**3
_GPU_PEAK_KEYS = ("peak_bytes_in_use_gb", "peak_bytes_reserved_gb")


def _bytes_to_gb(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return float(num_bytes) / _BYTES_PER_GB


def _device_memory_stats_gb(mem_stats: dict) -> dict[str, float | None]:
    """Normalize JAX ``device.memory_stats()`` into GB fields."""
    stats = {
        "bytes_in_use_gb": _bytes_to_gb(mem_stats.get("bytes_in_use")) or 0.0,
        "peak_bytes_in_use_gb": _bytes_to_gb(mem_stats.get("peak_bytes_in_use")) or 0.0,
        "bytes_reserved_gb": _bytes_to_gb(mem_stats.get("bytes_reserved")),
        "peak_bytes_reserved_gb": _bytes_to_gb(mem_stats.get("peak_bytes_reserved")),
        "bytes_limit_gb": _bytes_to_gb(mem_stats.get("bytes_limit")),
    }
    return stats


def get_gpu_memory_usage():
    """Return best-effort JAX device memory statistics.

    Values come from ``jax.Device.memory_stats()`` and reflect the JAX/XLA
    allocator on each device. ``peak_*`` counters are cumulative since the
    last ``device.clear_memory_stats()`` call (see ``clear_gpu_memory``).

    ``peak_bytes_reserved_gb`` is usually the better OOM proxy than
    ``peak_bytes_in_use_gb`` because it includes allocator reservations.
    """
    memory_stats = {}
    try:
        devices = jax.local_devices()
        for i, device in enumerate(devices):
            if device.platform == "gpu":
                try:
                    mem_stats = device.memory_stats()
                    device_stats = _device_memory_stats_gb(mem_stats)
                    device_stats["device_kind"] = str(device.device_kind)
                    memory_stats[f"device_{i}"] = device_stats
                except Exception as exc:
                    memory_stats[f"device_{i}"] = {
                        "device_kind": str(device.device_kind),
                        "error": f"Could not get memory stats: {exc}",
                    }
            elif device.platform == "cpu":
                memory_stats[f"device_{i}"] = {
                    "device_kind": "cpu",
                    "note": "CPU memory tracking not implemented",
                }
    except Exception as exc:
        memory_stats["error"] = str(exc)
    return memory_stats


def merge_peak_gpu_memory(
    peak_tracker: dict[str, dict[str, float]],
    memory_usage: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """Update per-device peak tracker from ``get_gpu_memory_usage()`` output."""
    for device_name, stats in memory_usage.items():
        if "bytes_in_use_gb" not in stats:
            continue
        device_peaks = peak_tracker.setdefault(device_name, {})
        for key in _GPU_PEAK_KEYS:
            if key not in stats or stats[key] is None:
                continue
            value = float(stats[key])
            device_peaks[key] = max(device_peaks.get(key, 0.0), value)
    return peak_tracker


def format_gpu_memory_line(device_name: str, stats: dict) -> str | None:
    """Format one GPU device memory line for logging."""
    if "bytes_in_use_gb" not in stats:
        return None
    line = (
        f"GPU Memory ({device_name}): {stats['bytes_in_use_gb']:.3f} GB "
        f"(peak in use: {stats['peak_bytes_in_use_gb']:.3f} GB"
    )
    peak_reserved = stats.get("peak_bytes_reserved_gb")
    if peak_reserved is not None:
        line += f", peak reserved: {peak_reserved:.3f} GB"
    return line + ")"


def print_gpu_memory_usage(memory_usage: dict, *, prefix: str = "  ") -> None:
    """Print current and cumulative peak GPU memory for all devices."""
    for device_name, stats in memory_usage.items():
        line = format_gpu_memory_line(device_name, stats)
        if line is not None:
            print(f"{prefix}{line}")


def clear_gpu_memory():
    """Clear Python garbage and JAX caches before large training/eval runs.

    Also resets per-device JAX allocator peak counters so subsequent
    ``peak_*`` readings measure only memory after this call.
    """
    gc.collect()
    jax.clear_caches()
    try:
        for device in jax.local_devices():
            if device.platform == "gpu":
                device.clear_memory_stats()
    except AttributeError:
        pass
    print("  Cleared GPU memory and caches")


def set_random_seeds(seed: int):
    """Set non-JAX random seeds; JAX PRNG keys remain explicit."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def lr_schedule_params(
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.5,
) -> dict[str, float | int]:
    """Resolved warmup-cosine schedule parameters for Optax and startup logging."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    decay_steps = max(total_steps - warmup_steps, 1)
    min_lr = min_lr_ratio * base_lr
    init_lr = 0.0 if warmup_steps > 0 else base_lr
    return {
        "peak_lr": base_lr,
        "init_lr": init_lr,
        "min_lr": min_lr,
        "min_lr_ratio": min_lr_ratio,
        "warmup_epochs": warmup_epochs,
        "warmup_steps": warmup_steps,
        "total_epochs": total_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "decay_steps": decay_steps,
    }


def create_lr_schedule(
    base_lr,
    warmup_epochs,
    total_epochs,
    steps_per_epoch,
    min_lr_ratio: float = 0.5,
):
    """Create warmup-cosine learning-rate schedule."""
    params = lr_schedule_params(
        base_lr, warmup_epochs, total_epochs, steps_per_epoch, min_lr_ratio
    )

    return optax.warmup_cosine_decay_schedule(
        init_value=params["init_lr"],
        peak_value=params["peak_lr"],
        warmup_steps=params["warmup_steps"],
        decay_steps=params["decay_steps"],
        end_value=params["min_lr"],
    )


def set_protobuf_implementation() -> None:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


def configure_tensorflow_for_data_loading() -> None:
    """Hide TensorFlow GPUs so JAX owns the accelerator."""
    try:
        tf.config.set_visible_devices([], "GPU")
        for device in tf.config.get_visible_devices():
            if device.device_type == "GPU":
                raise RuntimeError("TensorFlow still sees a GPU after disable.")
    except Exception:
        pass
