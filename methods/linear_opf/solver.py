"""Reusable CVXPY problem cache for ``core.grids.linear_opf_solver``."""

from __future__ import annotations

import numpy as np

from core.grids.data import OPFData
from core.grids.linear_opf_solver import (
    LinearOPFProblemCache,
    build_linear_opf_cache,
    release_linear_opf_cache_state,
    solve_linear_opf,
)
from core.topology.utils import make_topology_data
from core.utils.types import TopologyData

# One compiled CVXPY problem per (network object, PF solver) in this process.
_CACHE: dict[tuple[int, str], LinearOPFProblemCache] = {}


def solve_linear_opf_batch(
    data: OPFData,
    topo: TopologyData,
    s_load_batch: np.ndarray,
    pder_avai_batch: np.ndarray,
    obj_weights_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Solve one linear OPF per batch row; reuse one CVXPY graph per grid."""
    cache_key = (id(data.net), data.power_flow_solver)
    cache = _CACHE.get(cache_key)
    if cache is None:
        cache = build_linear_opf_cache(
            data.net,
            _topo_for_batch_row(topo, 0),
            {},
            power_flow_solver=data.power_flow_solver,
        )
        _CACHE[cache_key] = cache

    batch_size = int(s_load_batch.shape[0])
    pder_out = np.zeros((batch_size, data.net.n_der_phase), dtype=np.float64)
    qder_out = np.zeros((batch_size, data.net.n_der_phase), dtype=np.float64)
    infos: list[dict] = []
    for i in range(batch_size):
        topo_i = _topo_for_batch_row(topo, i)
        pder, qder, info = solve_linear_opf(
            cache,
            s_load_batch[i],
            pder_avai_batch[i],
            obj_weights_batch[i],
            net=data.net,
            topo=topo_i,
            power_flow_solver=data.power_flow_solver,
        )
        pder_out[i] = pder
        qder_out[i] = qder
        infos.append(info)
        release_linear_opf_cache_state(cache)
    return pder_out, qder_out, infos


def _topo_for_batch_row(topo: TopologyData, row: int) -> TopologyData:
    """Slice batched topology matrices to a single-sample ``TopologyData``."""
    if topo.Y_LL_inv.ndim == 2:
        return topo
    return make_topology_data(
        Y_0row=topo.Y_0row[row],
        Y_LL_inv=topo.Y_LL_inv[row],
        w=topo.w[row],
        v_init=topo.v_init[row] if topo.v_init.ndim > 1 else topo.v_init,
        branch_status=topo.branch_status[row],
    )
