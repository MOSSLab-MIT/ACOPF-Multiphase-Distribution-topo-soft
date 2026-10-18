"""Fused AC violation kernel matches separate max and square reductions."""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.grids.data import OPFData
from core.grids.metrics import (
    ac_constraint_violation_vector,
    calc_ineq_violation,
)
from core.topology.batch import precompute_topology
from core.utils.workflow import prepare_batch_data

DATASET_ROOT = Path(__file__).resolve().parents[1] / "datagen_topo" / "datasets"
IEEE13 = {
    "train": DATASET_ROOT / "IEEE13Nodeckt" / "IEEE13Nodeckt_training.h5",
    "test": DATASET_ROOT / "IEEE13Nodeckt" / "IEEE13Nodeckt_test_feasible.h5",
    "json": DATASET_ROOT / "IEEE13Nodeckt" / "network_config.json",
}


@partial(jax.jit, static_argnums=(0,))
def _direct_violation_reductions(net, pder, qder, pder_avai, v_full, y_0row):
    x = jnp.concatenate([pder, qder], axis=1)

    def one(x_i, v_i, y_i, av_i):
        vio = ac_constraint_violation_vector(net, x_i, v_i, y_i, av_i)
        return jnp.max(vio), jnp.mean(vio**2)

    return jax.vmap(one, in_axes=(0, 0, 0, 0))(x, v_full, y_0row, pder_avai)


@pytest.fixture(scope="module")
def ieee13_batch():
    if not IEEE13["test"].is_file():
        pytest.skip("IEEE13 feasible eval H5 not available")
    session = OPFData(str(IEEE13["train"]), str(IEEE13["test"]), str(IEEE13["json"]))
    inputs_batch, targets_batch = next(
        session.make_test_dataset(samples=8).batch(8).as_numpy_iterator()
    )
    inputs_np = np.asarray(inputs_batch)
    topo = precompute_topology(session, inputs_np)
    batch = prepare_batch_data(
        jnp.asarray(inputs_batch, dtype=jnp.float64),
        jnp.asarray(targets_batch, dtype=jnp.float64),
        session,
        topo=topo,
    )
    net = session.net
    v_full = jnp.concatenate(
        [batch.v_target, jnp.broadcast_to(net.v_slack_local, (8, net.n_slk))],
        axis=1,
    )
    return net, batch.pder_target, batch.qder_target, batch.pder_avai, v_full, topo.Y_0row


def test_fused_violation_matches_separate(ieee13_batch):
    net, pder, qder, pder_avai, v_full, y_0row = ieee13_batch
    vio_max, vio_sq = calc_ineq_violation(
        net,
        pder,
        qder,
        pder_avai,
        v_full,
        Y_0row=y_0row,
        return_square=True,
    )
    vio_ac = calc_ineq_violation(net, pder, qder, pder_avai, v_full, Y_0row=y_0row)
    direct_max, direct_sq = _direct_violation_reductions(
        net, pder, qder, pder_avai, v_full, y_0row
    )
    np.testing.assert_allclose(np.asarray(vio_max), np.asarray(vio_ac), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(vio_max), np.asarray(direct_max), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(vio_sq), np.asarray(direct_sq), rtol=0, atol=0)
