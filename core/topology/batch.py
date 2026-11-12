"""Canonical host-side topology precompute for training and evaluation.

``precompute_topology`` deduplicates compact switch statuses on the host, then
calls ``build_topology_batch``. Pass the same ``session`` object used as a
``static_argnums`` entry in downstream JIT (``train_step``, etc.) to avoid
recompilation from backend swaps.

Dedup policy (``dedup=True``): see ``precompute_topology``.
"""
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

from core.grids.data import OPFData
from core.utils.types import TopologyData
from core.topology.utils import (
    expand_topology_batch,
    gather_topology_by_index,
    topology_data_from_matrices,
)


def _host_int32_matrix(arr: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
    """Compact status on host: no-op for NumPy, at most one device→host sync for JAX."""
    if isinstance(arr, np.ndarray):
        out = np.asarray(arr, dtype=np.int32)
    else:
        out = np.asarray(jax.device_get(arr), dtype=np.int32)
    if out.ndim == 1:
        return out.reshape(1, -1)
    return out


def _topology_batch_via_session(
    branch_status_compact: np.ndarray, session: OPFData
) -> TopologyData:
    """Batch topology via backend ``topology_data_batch`` (stamp path)."""
    return session.topology_backend.topology_data_batch(branch_status_compact)


def build_topology_batch_jit(
    branch_status_compact: jnp.ndarray, session: OPFData
) -> TopologyData:
    """JIT batch topology (Woodbury backend only).

    Delegates to the already-JIT'd ``backend.batch_matrices_jit`` (which takes
    ``backend.static`` as a traced arg, avoiding captured-constant re-capture)
    rather than wrapping ``session`` as a static argnum. The previous
    ``@partial(jax.jit, static_argnums=(1,))`` made ``backend.static`` concrete
    inside the trace and re-baked its arrays as HLO constants.
    """
    backend = session.topology_backend
    if not hasattr(backend, "batch_matrices_jit"):
        raise TypeError(
            "build_topology_batch_jit requires WoodburyTopologyBackend; "
            f"got {type(backend).__name__}. Use build_topology_batch."
        )
    compact = jnp.asarray(branch_status_compact, dtype=jnp.int32)
    Y_scaled, Y_LL_inv, w, branch_status = backend.batch_matrices_jit(
        backend.static, compact
    )
    n_net, n_total = session.net.n_net, session.net.n_total
    return topology_data_from_matrices(Y_scaled, Y_LL_inv, w, branch_status, n_net, n_total)


def build_topology_batch_vmap(
    branch_status_compact: jnp.ndarray, session: OPFData
) -> TopologyData:
    """JIT batch topology via vmap (Woodbury branchless backend only).

    Uses ``backend.batch_vmap_jit`` (``jax.vmap`` of the branchless slim fn) for
    all supported grids. Faster than ``lax.map`` but higher VRAM; fall back
    to ``--woodbury_batch_mode map`` if GPU OOM occurs.
    """
    backend = session.topology_backend
    compact = jnp.asarray(branch_status_compact, dtype=jnp.int32)
    Y_0row, Y_LL_inv, w, branch_status = backend.batch_vmap_jit(
        backend.static, compact
    )
    n_net, n_total = session.net.n_net, session.net.n_total
    return topology_data_from_matrices(Y_0row, Y_LL_inv, w, branch_status, n_net, n_total)


def build_topology_batch(
    branch_status_compact: Union[np.ndarray, jnp.ndarray], session: OPFData
) -> TopologyData:
    """Build batched ``TopologyData`` from compact status (outside train JIT).

    Dispatch order:
    1. ``batch_mode == "map"`` -> ``lax.map`` of branchless fn (lower VRAM).
    2. ``batch_mode == "vmap"`` (default) and ``batch_vmap_jit`` exists -> vmap
       of branchless fn (fast, all supported grids).
    3. Unresolved ``"auto"`` fallback -> ``lax.map`` with adaptive fn.
    """
    compact_host = _host_int32_matrix(branch_status_compact)
    backend = session.topology_backend
    if getattr(backend, "batch_mode", "vmap") == "map":
        return build_topology_batch_jit(
            jnp.asarray(compact_host, dtype=jnp.int32), session
        )
    if getattr(backend, "batch_vmap_jit", None) is not None:
        return build_topology_batch_vmap(
            jnp.asarray(compact_host, dtype=jnp.int32), session
        )
    if hasattr(backend, "batch_matrices_jit"):
        return build_topology_batch_jit(
            jnp.asarray(compact_host, dtype=jnp.int32), session
        )
    return _topology_batch_via_session(compact_host, session)


def _topology_one_row(compact_row: np.ndarray, session: OPFData) -> TopologyData:
    """Single-sample topology (no batch axis); Woodbury uses single-row JIT, not ``lax.map``."""
    row = _host_int32_matrix(compact_row)[0]
    backend = session.topology_backend
    n_net, n_total = session.net.n_net, session.net.n_total
    if hasattr(backend, "_topology_matrices_jit"):
        row_j = jnp.asarray(row, dtype=jnp.int32)
        mats = backend._topology_matrices_jit(backend.static, row_j)
        return topology_data_from_matrices(*mats, n_net, n_total)
    return backend.topology_data(row)


def _broadcast_one_topology(compact_row: np.ndarray, batch_size: int, session: OPFData) -> TopologyData:
    return expand_topology_batch(_topology_one_row(compact_row, session), batch_size)


def compact_switch_status_from_inputs(
    session: OPFData, inputs_batch: Union[np.ndarray, jnp.ndarray]
) -> np.ndarray:
    """Slice compact switch statuses from a batched input matrix (host NumPy)."""
    bs_slice = session.slices["input"][session.topology_input_key]
    return _host_int32_matrix(inputs_batch[:, bs_slice])


def precompute_topology(
    session: OPFData,
    compact_or_inputs: Union[np.ndarray, jnp.ndarray],
    *,
    dedup: bool = True,
) -> TopologyData:
    """Precompute topology matrices outside train/eval JIT graphs.

    Prefer NumPy loader batches; JAX inputs sync once via ``_host_int32_matrix``.

    Args:
        session: OPFData session (backend on ``session.topology_backend``).
        compact_or_inputs: Either compact status ``(B, n_changing)`` or full
            input batch ``(B, din)``.
        dedup: If True, deduplicate identical compact rows on the host before
            batch build; if False, build all ``B`` rows without dedup.
    """
    n_changing = session.net.n_changing_switches
    if (
        compact_or_inputs.ndim == 1
        or compact_or_inputs.shape[-1] == n_changing
    ):
        compact_np = _host_int32_matrix(compact_or_inputs)
    else:
        compact_np = compact_switch_status_from_inputs(session, compact_or_inputs)

    batch_size = int(compact_np.shape[0])
    net = session.net

    if not dedup or net.n_changing_switches == 0:
        if net.n_changing_switches == 0:
            return _broadcast_one_topology(compact_np[0:1], batch_size, session)
        return build_topology_batch(compact_np, session)

    unique, inverse = np.unique(compact_np, axis=0, return_inverse=True)
    n_unique = int(unique.shape[0])

    if n_unique == 1:
        return _broadcast_one_topology(unique[0:1], batch_size, session)

    if n_unique == batch_size:
        return build_topology_batch(compact_np, session)

    topo_u = build_topology_batch(unique, session)
    return gather_topology_by_index(topo_u, inverse)
