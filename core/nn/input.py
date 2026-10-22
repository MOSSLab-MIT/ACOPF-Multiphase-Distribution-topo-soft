"""Compact MLP input layout: gather nonzero load features from full HDF5 vectors."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Tuple

import h5py
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from core.grids.data import OPFData

LOAD_SUPPORT_TOL = 1e-12
_LAYOUT_JSON = "nn_input_layout.json"
_STATS_NPZ = "nn_input_stats.npz"


class NNInputLayout(NamedTuple):
    """Static per-case indexing for MLP inputs (compact pd/qd + full tail)."""

    load_idx: jnp.ndarray  # (n_load,) int32 node indices into pd/qd
    din_full: int
    din_compact: int
    n_load: int
    n_net: int
    n_der_unit: int
    n_obj_weights: int
    n_changing_switches: int
    tail_start: int  # start column in full concatenated input (pder_avai block)
    tail_end: int  # exclusive end column for NN tail gather
    include_topo_nn_input: bool = True

    @property
    def load_sparsity_frac(self) -> float:
        """Fraction of non-slack nodes without load (dropped from the compact input)."""
        if self.n_net <= 0:
            return 0.0
        return 1.0 - float(self.n_load) / float(self.n_net)

    @property
    def din_reduction_frac(self) -> float:
        """Relative input-width reduction of the compact layout versus the full HDF5 layout."""
        if self.din_full <= 0:
            return 0.0
        return 1.0 - float(self.din_compact) / float(self.din_full)

    def summary_dict(self) -> dict[str, Any]:
        """JSON-serializable layout summary for logs and checkpoints."""
        return {
            "din_full": self.din_full,
            "din_compact": self.din_compact,
            "din_reduction_frac": self.din_reduction_frac,
            "n_load": self.n_load,
            "n_net": self.n_net,
            "load_sparsity_frac": self.load_sparsity_frac,
            "n_der_unit": self.n_der_unit,
            "n_obj_weights": self.n_obj_weights,
            "n_changing_switches": self.n_changing_switches,
            "tail_end": self.tail_end,
            "include_topo_nn_input": self.include_topo_nn_input,
        }


def load_idx_from_refs(
    p_load_ref: np.ndarray,
    q_load_ref: np.ndarray,
    *,
    tol: float = LOAD_SUPPORT_TOL,
) -> np.ndarray:
    """Return sorted bus-phase indices where reference load is nonzero."""
    p = np.asarray(p_load_ref, dtype=np.float64).reshape(-1)
    q = np.asarray(q_load_ref, dtype=np.float64).reshape(-1)
    if p.shape != q.shape:
        raise ValueError(
            f"p_load_ref length {p.shape[0]} != q_load_ref length {q.shape[0]}"
        )
    mask = (np.abs(p) + np.abs(q)) > tol
    return np.flatnonzero(mask).astype(np.int32)


def build_nn_input_layout(session: "OPFData") -> NNInputLayout:
    """Build compact layout from session slices and network_config load references."""
    net = session.net
    raw = dict(session.config)
    for key in ("p_load_ref", "q_load_ref"):
        if key not in raw:
            raise ValueError(
                f"network_config missing {key!r}; required for compact NN inputs."
            )
    idx_np = load_idx_from_refs(raw["p_load_ref"], raw["q_load_ref"])
    if idx_np.shape[0] == 0:
        raise ValueError("load_idx is empty; expected at least one load bus-phase.")
    if int(idx_np.max()) >= net.n_net or int(idx_np.min()) < 0:
        raise ValueError(
            f"load_idx out of range [0, {net.n_net}); got min={idx_np.min()}, "
            f"max={idx_np.max()}"
        )

    slices = session.slices["input"]
    pd_sl = slices["pd"]
    qd_sl = slices["qd"]
    if pd_sl.stop - pd_sl.start != net.n_net:
        raise ValueError("pd slice width does not match n_net")
    if qd_sl.stop - qd_sl.start != net.n_net:
        raise ValueError("qd slice width does not match n_net")

    n_load = int(idx_np.shape[0])
    tail_start = slices["pder_avai"].start
    din_full = max(s.stop for s in slices.values())
    n_der = slices["pder_avai"].stop - slices["pder_avai"].start
    n_obj = slices["obj_weights"].stop - slices["obj_weights"].start
    topo_key = session.topology_input_key
    n_topo = slices[topo_key].stop - slices[topo_key].start
    include_topo = getattr(session, "include_topo_nn_input", True)
    tail_end = (
        slices[topo_key].stop if include_topo else slices["obj_weights"].stop
    )
    din_compact = 2 * n_load + (tail_end - tail_start)

    return NNInputLayout(
        load_idx=jnp.asarray(idx_np, dtype=jnp.int32),
        din_full=din_full,
        din_compact=din_compact,
        n_load=n_load,
        n_net=net.n_net,
        n_der_unit=n_der,
        n_obj_weights=n_obj,
        n_changing_switches=n_topo,
        tail_start=tail_start,
        tail_end=tail_end,
        include_topo_nn_input=include_topo,
    )


def gather_nn_features(
    inputs: np.ndarray | jnp.ndarray,
    layout: NNInputLayout,
) -> jnp.ndarray:
    """Extract compact MLP features from a full HDF5 input batch ``(B, din_full)``."""
    x = jnp.asarray(inputs)
    if x.ndim == 1:
        x = x[None, :]
    n_net = layout.n_net
    pd = x[:, :n_net]
    qd = x[:, n_net : 2 * n_net]
    tail = x[:, layout.tail_start : layout.tail_end]
    pd_c = pd[:, layout.load_idx]
    qd_c = qd[:, layout.load_idx]
    return jnp.concatenate([pd_c, qd_c, tail], axis=1)


def gather_nn_features_from_session(
    inputs: np.ndarray | jnp.ndarray,
    session: "OPFData",
) -> jnp.ndarray:
    """Gather compact features using ``session.slices``."""
    layout = session.nn_layout
    x = jnp.asarray(inputs)
    if x.ndim == 1:
        x = x[None, :]
    sl = session.slices["input"]
    pd = x[:, sl["pd"]]
    qd = x[:, sl["qd"]]
    tail = x[:, layout.tail_start : layout.tail_end]
    pd_c = pd[:, layout.load_idx]
    qd_c = qd[:, layout.load_idx]
    return jnp.concatenate([pd_c, qd_c, tail], axis=1)


def scatter_load_to_full(
    pd_compact: np.ndarray | jnp.ndarray,
    qd_compact: np.ndarray | jnp.ndarray,
    layout: NNInputLayout,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Embed compact load vectors into length-``n_net`` arrays (zeros elsewhere)."""
    pd_c = jnp.asarray(pd_compact)
    qd_c = jnp.asarray(qd_compact)
    if pd_c.ndim == 1:
        pd_c = pd_c[None, :]
        qd_c = qd_c[None, :]
    batch = pd_c.shape[0]
    pd = jnp.zeros((batch, layout.n_net), dtype=pd_c.dtype)
    qd = jnp.zeros((batch, layout.n_net), dtype=qd_c.dtype)
    pd = pd.at[:, layout.load_idx].set(pd_c)
    qd = qd.at[:, layout.load_idx].set(qd_c)
    return pd, qd


def validate_load_support(
    session: "OPFData",
    h5_path: str | None = None,
    *,
    n_check: int = 32,
    tol: float = LOAD_SUPPORT_TOL,
) -> None:
    """Fail if HDF5 samples have load outside the reference support."""
    path = h5_path or session.train_h5_path
    layout = session.nn_layout
    load_set = set(int(i) for i in np.asarray(layout.load_idx))

    with h5py.File(path, "r") as f:
        pd_ds = f["input/pd"]
        qd_ds = f["input/qd"]
        n_rows = int(pd_ds.shape[0])
        if n_rows == 0:
            raise ValueError(f"No rows in {path}")
        rng = np.random.default_rng(0)
        indices = (
            np.arange(min(n_check, n_rows))
            if n_rows <= n_check
            else rng.choice(n_rows, size=n_check, replace=False)
        )
        for row in indices:
            pd = np.asarray(pd_ds[int(row)], dtype=np.float64).reshape(-1)
            qd = np.asarray(qd_ds[int(row)], dtype=np.float64).reshape(-1)
            for name, vec in (("pd", pd), ("qd", qd)):
                bad = np.flatnonzero(np.abs(vec) > tol)
                extra = [int(j) for j in bad if int(j) not in load_set]
                if extra:
                    raise ValueError(
                        f"{path} row {row}: {name} nonzero outside load_idx "
                        f"(first extras: {extra[:5]})"
                    )


def log_nn_input_layout(
    layout: NNInputLayout,
    *,
    n_train_samples: int | None = None,
) -> None:
    """Print a structured summary of the compact NN input layout."""
    print("NN input layout (compact loads)")
    print(
        f"  din_full={layout.din_full}  din_compact={layout.din_compact}  "
        f"reduction={100.0 * layout.din_reduction_frac:.1f}%"
    )
    print(
        f"  load nodes: n_load={layout.n_load} / n_net={layout.n_net} "
        f"(sparsity {100.0 * layout.load_sparsity_frac:.1f}%)"
    )
    topo_in_nn = (
        layout.n_changing_switches
        if layout.include_topo_nn_input
        else 0
    )
    print(
        f"  tail: pder_avai={layout.n_der_unit}  obj_weights={layout.n_obj_weights}  "
        f"changing_switch_status={topo_in_nn}/{layout.n_changing_switches} "
        f"(include_topo_nn_input={layout.include_topo_nn_input})"
    )
    if n_train_samples is not None:
        print(f"  normalization: stats from {n_train_samples} training rows")


def log_input_stats(
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
    layout: NNInputLayout,
) -> None:
    """Log compact normalization vector diagnostics."""
    mean = np.asarray(input_mean)
    std = np.asarray(input_std)
    print(
        f"  NN input stats (din_compact={layout.din_compact}): "
        f"mean in [{mean.min():.4g}, {mean.max():.4g}], "
        f"std in [{std.min():.4g}, {std.max():.4g}]"
    )


def _layout_to_json(layout: NNInputLayout) -> dict[str, Any]:
    return {
        **layout.summary_dict(),
        "load_idx": np.asarray(layout.load_idx, dtype=np.int32).tolist(),
        "tail_start": layout.tail_start,
        "tail_end": layout.tail_end,
        "include_topo_nn_input": layout.include_topo_nn_input,
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
        include_topo_nn_input=bool(payload.get("include_topo_nn_input", True)),
    )


def save_nn_input_artifacts(
    checkpoint_dir: str | Path,
    layout: NNInputLayout,
    input_stat: Tuple[jnp.ndarray, jnp.ndarray],
) -> None:
    """Persist layout JSON and mean/std NPZ next to a checkpoint directory."""
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    layout_path = root / _LAYOUT_JSON
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(_layout_to_json(layout), f, indent=2)
    mean, std = input_stat
    stats_path = root / _STATS_NPZ
    np.savez(
        stats_path,
        mean=np.asarray(mean, dtype=np.float64),
        std=np.asarray(std, dtype=np.float64),
    )
    # print(f"  NN input artifacts: {layout_path.name}, {stats_path.name}")


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
    layout = session.nn_layout
    if recompute_stats_fn is None:
        raise ValueError("recompute_stats_fn required when artifacts are missing")
    input_stat = recompute_stats_fn(use_compact=False)
    return layout, input_stat, False


def nn_features_for_apply(
    inputs: np.ndarray | jnp.ndarray,
    session: "OPFData",
    *,
    use_compact: bool = True,
) -> jnp.ndarray:
    """Full or compact NN features depending on training/checkpoint mode."""
    if use_compact:
        return gather_nn_features_from_session(inputs, session)
    return jnp.asarray(inputs)
