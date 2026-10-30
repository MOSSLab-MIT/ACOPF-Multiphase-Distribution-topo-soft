"""Batch extraction: physics tensors and OPF labels from flat HDF5 input/target batches."""

from functools import partial

import jax
import jax.numpy as jnp

from core.grids.data import OPFData
from core.utils.types import BatchData, TopologyData


@partial(jax.jit, static_argnums=(2,))
def prepare_batch_data(
    inputs_batch,
    targets_batch,
    session: OPFData,
    topo: TopologyData | None = None,
) -> BatchData:
    """Extract physics tensors and OPF labels from flat HDF5 batches."""
    net = session.net
    pd = inputs_batch[:, session.slices["input"]["pd"]]
    qd = inputs_batch[:, session.slices["input"]["qd"]]
    pder_avai = inputs_batch[:, session.slices["input"]["pder_avai"]]
    obj_weights = inputs_batch[:, session.slices["input"]["obj_weights"]]
    s_load = jax.lax.complex(pd, qd)
    if topo is not None:
        branch_status = topo.branch_status
        switchable_status = branch_status[:, net.changing_switches]
    else:
        topo_key = session.topology_input_key
        branch_status_compact = jnp.asarray(
            inputs_batch[:, session.slices["input"][topo_key]], dtype=jnp.int32
        )
        branch_status = session.expand_branch_status(branch_status_compact)
        switchable_status = branch_status[:, net.changing_switches]

    pder_target = targets_batch[:, session.slices["target"]["pder"]]
    qder_target = targets_batch[:, session.slices["target"]["qder"]]
    vr_target = targets_batch[:, session.slices["target"]["vr"]]
    vi_target = targets_batch[:, session.slices["target"]["vi"]]
    v_target = jax.lax.complex(vr_target, vi_target)

    if topo is not None:
        v_init = (
            topo.w
            if topo.w.ndim > 1
            else jnp.broadcast_to(topo.w[None, :], (pd.shape[0], net.n_net))
        )
    else:
        v_init = jnp.broadcast_to(net.w[None, :], (pd.shape[0], net.n_net))

    return BatchData(
        s_load=s_load,
        pder_avai=pder_avai,
        obj_weights=obj_weights,
        branch_status=branch_status,
        switchable_status=switchable_status,
        v_init=v_init,
        v_target=v_target,
        pder_target=pder_target,
        qder_target=qder_target,
    )
