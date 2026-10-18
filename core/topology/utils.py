"""Topology data packing and matrix-slicing helpers."""

from typing import Any

import jax.numpy as jnp

from core.utils.types import TopologyData


def matrices_first_to_y_0row(
    first: jnp.ndarray, n_net: int, n_total: int
) -> jnp.ndarray:
    """Normalize ``topology_matrices*`` first element to ``Y_0row``."""
    n_slk = n_total - n_net
    if first.ndim == 2:
        if first.shape == (n_total, n_total):
            return first[n_net:n_total, :]
        if first.shape == (n_slk, n_total):
            return first
    elif first.ndim == 3:
        if first.shape[1:] == (n_total, n_total):
            return first[:, n_net:n_total, :]
        if first.shape[1:] == (n_slk, n_total):
            return first
    raise ValueError(
        f"expected first element shape (n_total, n_total) or (n_slk, n_total), "
        f"or batched (B, n_total, n_total) or (B, n_slk, n_total); "
        f"got ndim={first.ndim}, shape={first.shape}, "
        f"n_net={n_net}, n_total={n_total}, n_slk={n_slk}"
    )


def y_0row_from_scaled(Y_scaled: jnp.ndarray, n_net: int, n_total: int) -> jnp.ndarray:
    """Slack rows ``Y_scaled[n_net:n_total, :]`` for packing into ``TopologyData``."""
    del n_total
    if Y_scaled.ndim == 2:
        return Y_scaled[n_net:, :]
    return Y_scaled[..., n_net:, :]


def Y_LL_L0_from_scaled(Y_scaled: jnp.ndarray, n_net: int):
    """Partition net-net and net-slack blocks from a full stamped admittance."""
    if Y_scaled.ndim == 2:
        return Y_scaled[:n_net, :n_net], Y_scaled[:n_net, n_net:]
    return Y_scaled[..., :n_net, :n_net], Y_scaled[..., :n_net, n_net:]


def make_topology_data(
    Y_0row,
    Y_LL_inv,
    w,
    v_init=None,
    branch_status=None,
) -> TopologyData:
    """Build ``TopologyData`` from topology-backend outputs."""
    if v_init is None:
        v_init = w
    if branch_status is None:
        branch_status = jnp.zeros((0,), dtype=jnp.int32)
    return TopologyData(
        v_init=v_init,
        Y_0row=Y_0row,
        Y_LL_inv=Y_LL_inv,
        w=w,
        branch_status=branch_status,
    )


def topology_data_from_matrices(
    Y_scaled: jnp.ndarray,
    Y_LL_inv: jnp.ndarray,
    w: jnp.ndarray,
    branch_status: jnp.ndarray,
    n_net: int,
    n_total: int,
) -> TopologyData:
    """Pack a ``topology_matrices*`` 4-tuple into ``TopologyData``."""
    return make_topology_data(
        Y_0row=matrices_first_to_y_0row(Y_scaled, n_net, n_total),
        Y_LL_inv=Y_LL_inv,
        w=w,
        v_init=w,
        branch_status=branch_status,
    )


def gather_topology_by_index(topo_unique: TopologyData, inverse: Any) -> TopologyData:
    """Gather per-unique ``TopologyData`` back to full batch order."""
    idx = jnp.asarray(inverse, dtype=jnp.int32)

    def _gather(x):
        return x[idx]

    return TopologyData(
        v_init=_gather(topo_unique.v_init),
        Y_0row=_gather(topo_unique.Y_0row),
        Y_LL_inv=_gather(topo_unique.Y_LL_inv),
        w=_gather(topo_unique.w),
        branch_status=_gather(topo_unique.branch_status),
    )


def expand_topology_batch(topo: TopologyData, B: int) -> TopologyData:
    """Broadcast a single-topology ``TopologyData`` to batch size ``B``."""

    def _expand(x):
        return jnp.broadcast_to(x[None], (B,) + x.shape)

    return TopologyData(
        v_init=_expand(topo.v_init),
        Y_0row=_expand(topo.Y_0row),
        Y_LL_inv=_expand(topo.Y_LL_inv),
        w=_expand(topo.w),
        branch_status=_expand(topo.branch_status),
    )
