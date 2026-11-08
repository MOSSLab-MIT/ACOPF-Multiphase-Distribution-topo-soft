"""NN input features: compact load layout, feature gathering, support checks, normalization stats."""

from typing import TYPE_CHECKING, Any, NamedTuple

import h5py
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from core.grids.data import OPFData

LOAD_SUPPORT_TOL = 1e-12
# A training column is constant when std <= CONST_STD_RTOL * max(|mean|, 1). Float64
# round-off on a truly constant column is ~1e-16 * |mean|; the smallest real feature std
# on the 123/240/8500 datasets is 1.55e-6 (240-bus qd), so the threshold separates the two.
CONST_STD_RTOL = 1e-12


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

    @property
    def load_sparsity_frac(self) -> float:
        if self.n_net <= 0:
            return 0.0
        return 1.0 - float(self.n_load) / float(self.n_net)

    @property
    def din_reduction_frac(self) -> float:
        if self.din_full <= 0:
            return 0.0
        return 1.0 - float(self.din_compact) / float(self.din_full)

    def summary_dict(self) -> dict[str, Any]:
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
        }


# --- Layout construction ---


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
    tail_end = slices[topo_key].stop
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
    )


# --- Feature gathering ---


def gather_nn_features_from_session(
    inputs: np.ndarray | jnp.ndarray,
    session: "OPFData",
) -> jnp.ndarray:
    """Compact MLP features ``[pd[load_idx], qd[load_idx], tail]`` from full inputs ``(B, din_full)``."""
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


# --- Normalization statistics ---


def constant_input_mask(input_mean, input_std):
    """True for feature columns that are constant over the training set."""
    mean = jnp.asarray(input_mean)
    std = jnp.asarray(input_std)
    tol = CONST_STD_RTOL * jnp.maximum(jnp.abs(mean), 1.0)
    return std <= tol


def input_scale(input_mean, input_std):
    """Per-column divisor for z-scoring: training std, or 1 for constant columns.

    With scale 1, a constant column maps its training value to 0 (exactly for values such
    as 0/1 switch bits; otherwise up to the float64 round-off of the training mean) and any
    other value ``x`` to the finite ``x - mean`` instead of dividing by ~0.
    """
    std = jnp.asarray(input_std)
    return jnp.where(constant_input_mask(input_mean, std), jnp.ones_like(std), std)


def compute_input_stats(
    train_dataset,
    session: "OPFData",
    *,
    use_compact: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mean and std of the NN features over the whole training dataset."""
    layout = session.nn_layout
    n = len(train_dataset)
    chunks = []
    for inputs_batch in train_dataset.batch(n):
        full = inputs_batch.numpy()
        if use_compact:
            full = gather_nn_features_from_session(full, session)
        chunks.append(np.asarray(full, dtype=np.float64))
    all_data = np.concatenate(chunks, axis=0)
    input_mean = jnp.mean(all_data, axis=0)
    input_std = jnp.std(all_data, axis=0)
    din = int(all_data.shape[1])
    expected = layout.din_compact if use_compact else layout.din_full
    if din != expected:
        raise ValueError(f"Input stats width {din} != expected {expected}")
    _log_input_stats(input_mean, input_std, layout)
    return input_mean, input_std


# --- Logging ---


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
    print(
        f"  tail: pder_avai={layout.n_der_unit}  obj_weights={layout.n_obj_weights}  "
        f"changing_switch_status={layout.n_changing_switches}"
    )
    if n_train_samples is not None:
        print(f"  normalization: stats from {n_train_samples} training rows")


def _log_input_stats(
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
    layout: NNInputLayout,
) -> None:
    mean = np.asarray(input_mean)
    std = np.asarray(input_std)
    const_cols = np.flatnonzero(np.asarray(constant_input_mask(mean, std)))
    print(
        f"  NN input stats (din={mean.shape[0]}, din_compact={layout.din_compact}): "
        f"mean in [{mean.min():.4g}, {mean.max():.4g}], "
        f"std in [{std.min():.4g}, {std.max():.4g}]"
    )
    if const_cols.size:
        print(
            f"  constant input columns (scale=1): {const_cols.size} "
            f"(first: {const_cols[:10].tolist()})"
        )
