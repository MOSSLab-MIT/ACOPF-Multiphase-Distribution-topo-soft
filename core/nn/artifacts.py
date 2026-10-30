"""Checkpoint-side persistence of the NN input layout and normalization statistics."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import jax.numpy as jnp
import numpy as np

from core.nn.input import NNInputLayout

if TYPE_CHECKING:
    from core.grids.data import OPFData

_LAYOUT_JSON = "nn_input_layout.json"
_STATS_NPZ = "nn_input_stats.npz"


def _layout_to_json(layout: NNInputLayout) -> dict[str, Any]:
    return {
        **layout.summary_dict(),
        "load_idx": np.asarray(layout.load_idx, dtype=np.int32).tolist(),
        "tail_start": layout.tail_start,
    }


def _layout_from_json(payload: dict[str, Any]) -> NNInputLayout:
    return NNInputLayout(
        load_idx=jnp.asarray(payload["load_idx"], dtype=jnp.int32),
        din_full=int(payload["din_full"]),
        din_compact=int(payload["din_compact"]),
        n_load=int(payload["n_load"]),
        n_net=int(payload["n_net"]),
        n_der_unit=int(payload["n_der_unit"]),
        n_obj_weights=int(payload["n_obj_weights"]),
        n_changing_switches=int(payload["n_changing_switches"]),
        tail_start=int(payload["tail_start"]),
        tail_end=int(payload.get("tail_end", payload["din_full"])),
    )


def save_nn_input_artifacts(
    checkpoint_dir: str | Path,
    layout: NNInputLayout,
    input_stat: Tuple[jnp.ndarray, jnp.ndarray],
) -> None:
    """Persist layout JSON and mean/std NPZ next to a checkpoint directory."""
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / _LAYOUT_JSON, "w", encoding="utf-8") as f:
        json.dump(_layout_to_json(layout), f, indent=2)
    mean, std = input_stat
    np.savez(
        root / _STATS_NPZ,
        mean=np.asarray(mean, dtype=np.float64),
        std=np.asarray(std, dtype=np.float64),
    )


def load_nn_input_artifacts(
    checkpoint_dir: str | Path,
) -> Tuple[NNInputLayout, Tuple[jnp.ndarray, jnp.ndarray]] | None:
    """Load layout + stats from a checkpoint directory, or None if legacy checkpoint."""
    root = Path(checkpoint_dir)
    layout_path = root / _LAYOUT_JSON
    stats_path = root / _STATS_NPZ
    if not layout_path.is_file() or not stats_path.is_file():
        return None
    with open(layout_path, encoding="utf-8") as f:
        layout = _layout_from_json(json.load(f))
    with np.load(stats_path) as z:
        mean = jnp.asarray(z["mean"])
        std = jnp.asarray(z["std"])
    valid_widths = (layout.din_compact, layout.din_full)
    if mean.shape != std.shape or mean.shape[0] not in valid_widths:
        raise ValueError(
            f"nn_input_stats shapes {mean.shape}/{std.shape} do not match "
            f"din_compact {layout.din_compact} or din_full {layout.din_full}"
        )
    return layout, (mean, std)


def resolve_nn_input_for_checkpoint(
    session: "OPFData",
    checkpoint_dir: str | Path,
    *,
    recompute_stats_fn=None,
) -> Tuple[NNInputLayout, Tuple[jnp.ndarray, jnp.ndarray], bool]:
    """Load artifacts from checkpoint or fall back to full-width legacy inputs.

    Returns:
        (layout, input_stat, use_compact) — ``use_compact`` is False for legacy
        checkpoints (no gather; ``din_full`` MLP and stats).
    """
    loaded = load_nn_input_artifacts(checkpoint_dir)
    if loaded is not None:
        layout, input_stat = loaded
        use_compact = input_stat[0].shape[0] == layout.din_compact
        return layout, input_stat, use_compact

    print(
        "WARNING: checkpoint missing nn_input_layout.json / nn_input_stats.npz; "
        "using full HDF5 input width (legacy). Recompute stats from training data."
    )
    if recompute_stats_fn is None:
        raise ValueError("recompute_stats_fn required when artifacts are missing")
    input_stat = recompute_stats_fn(use_compact=False)
    return session.nn_layout, input_stat, False
