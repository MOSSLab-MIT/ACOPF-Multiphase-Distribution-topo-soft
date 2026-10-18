"""Regression: fixed-topology networks (``changing_switches=[]``) through Woodbury JIT."""
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from core.grids.data import OPFData
from core.topology.batch import precompute_topology

IEEE13_TEST_ROOT = (
    Path(__file__).resolve().parents[1] / "datagen_topo" / "datasets" / "IEEE13_test"
)


def _read_h5_inputs(data: OPFData, n: int) -> np.ndarray:
    with h5py.File(data.train_h5_path, "r") as f:
        parts = []
        for key in data.input_keys:
            arr = np.array(f["input"][key][:n])
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            parts.append(arr)
        return np.concatenate(parts, axis=1).astype(np.float64)


@pytest.mark.skipif(
    not (IEEE13_TEST_ROOT / "network_config.json").is_file(),
    reason="IEEE13_test fixed-topology dataset not available",
)
def test_precompute_topology_zero_changing_switches():
    """``changing_switches=[]`` must not call ``jnp.linalg.cond`` on empty Woodbury K."""
    train_h5 = IEEE13_TEST_ROOT / "IEEE13_test_training.h5"
    test_h5 = IEEE13_TEST_ROOT / "IEEE13_test_test_feasible.h5"
    json_path = IEEE13_TEST_ROOT / "network_config.json"
    data = OPFData(str(train_h5), str(test_h5), str(json_path))
    assert data.net.n_changing_switches == 0

    inputs = _read_h5_inputs(data, 4)
    topo = precompute_topology(data, inputs, dedup=True)
    n_net = data.net.n_net
    assert topo.Y_LL_inv.shape == (4, n_net, n_net)
    assert topo.w.shape == (4, n_net)

    eye = np.eye(n_net, dtype=np.complex128)
    resid = np.max(np.abs(np.asarray(topo.Y_LL_inv[0]) @ eye - eye))
    assert resid < 1e-8
