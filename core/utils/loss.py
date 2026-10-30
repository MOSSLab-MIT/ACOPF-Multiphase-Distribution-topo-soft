"""Cost scaling for the training loss and cost-normalization statistics."""

from pathlib import Path
from typing import Literal, Optional

import h5py
import jax.numpy as jnp
import numpy as np

from core.grids.data import OPFData
from core.utils.types import CostNormStats

CostNormMode = Literal["robust", "max", "none"]
_EPS = 1e-8
_PRIMAL_OBJECTIVE_META_KEY = "primal_objective_value"


def normalize_cost_for_loss(
    cost: jnp.ndarray, cost_scale: Optional[jnp.ndarray]
) -> jnp.ndarray:
    """Affine scale for loss; pass ``None`` to disable."""
    if cost_scale is None:
        return cost
    eps = jnp.asarray(_EPS, dtype=cost.dtype)
    return cost / (cost_scale + eps)


def read_primal_objective_values(
    session: OPFData,
    *,
    h5_path: str | Path | None = None,
    samples: int | None = None,
) -> np.ndarray | None:
    """Read ``meta/primal_objective_value`` from eval HDF5, or ``None`` if absent."""
    path = Path(h5_path) if h5_path is not None else Path(session.test_h5_path)
    with h5py.File(path, "r") as f:
        meta = f.get("meta")
        if meta is None or _PRIMAL_OBJECTIVE_META_KEY not in meta:
            return None
        arr = np.asarray(meta[_PRIMAL_OBJECTIVE_META_KEY], dtype=np.float64).reshape(-1)
    if samples is not None:
        arr = arr[: int(samples)]
    return arr


def _cost_norm_stats_from_costs(
    all_costs: np.ndarray, mode: CostNormMode
) -> CostNormStats:
    if all_costs.size == 0:
        raise ValueError("Cost normalization requires at least one eval sample.")
    cost_max = float(np.max(all_costs))
    cost_p95 = float(np.percentile(all_costs, 95))
    if mode == "max":
        cost_scale = max(cost_max, _EPS)
    else:
        cost_scale = max(cost_p95, _EPS)
    return CostNormStats(
        cost_max=cost_max,
        cost_p95=cost_p95,
        mode=mode,
        cost_scale=cost_scale,
        n_samples=int(all_costs.size),
    )


def compute_cost_norm_stats_from_eval(
    session: OPFData,
    *,
    mode: CostNormMode = "robust",
    samples: int | None = None,
    h5_path: str | Path | None = None,
) -> CostNormStats:
    """Build cost normalization stats from ``meta/primal_objective_value`` in eval HDF5."""
    if mode == "none":
        return CostNormStats(
            cost_max=1.0,
            cost_p95=1.0,
            mode="none",
            cost_scale=1.0,
            n_samples=0,
        )

    stored = read_primal_objective_values(session, h5_path=h5_path, samples=samples)
    if stored is None:
        path = Path(h5_path) if h5_path is not None else Path(session.test_h5_path)
        raise ValueError(
            f"Eval HDF5 missing meta/{_PRIMAL_OBJECTIVE_META_KEY!r} at {path}. "
            "Regenerate datasets with datagen_topo so primal objective values are stored."
        )
    return _cost_norm_stats_from_costs(stored, mode)
