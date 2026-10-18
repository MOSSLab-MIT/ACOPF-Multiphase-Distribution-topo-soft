import os
import sys
import unittest
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf

jax.config.update("jax_enable_x64", True)
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conftest import CASE_ROOTS, case_paths
from core.grids.data import OPFData
from core.topology.batch import precompute_topology
from core.feasibility.slfs import forward_pass_core, forward_pass_inference_core
from core.utils.workflow import prepare_batch_data
from core.feasibility.slfs_config import FeasibilityConfig


def _available_multiphase_cases():
    """FeederFlow multiphase cases with local HDF5 + network_config.json."""
    cases = []
    for name in ("IEEE13Nodeckt", "IEEE123Master"):
        if name not in CASE_ROOTS:
            continue
        paths = case_paths(name)
        if paths["json"].is_file() and paths["test_feasible"].is_file():
            train = paths["train"] if paths["train"].is_file() else paths["test_feasible"]
            cases.append(
                {
                    "name": name,
                    "train_h5": str(train),
                    "test_h5": str(paths["test_feasible"]),
                    "json": str(paths["json"]),
                }
            )
    return cases


class _DummyState:
    def __init__(self, n_der):
        self.n_der_phase = n_der

    def apply_fn(self, params, x):
        p = 0.5 * (jnp.tanh(x[:, : self.n_der_phase]) + 1.0)
        q = 0.25 * jnp.tanh(x[:, self.n_der_phase : 2 * self.n_der_phase])
        return jnp.concatenate([p, q], axis=1)


class TestInferenceParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _available_multiphase_cases()
        if not cls.cases:
            raise unittest.SkipTest(
                "No multiphase datasets found (IEEE13Nodeckt / IEEE123Master under "
                "datagen_topo/datasets/). Run Julia datagen first."
            )

    def test_forward_inference_matches_training_forward(self):
        fs_config = FeasibilityConfig(
            fs_tol=1e-6,
            fs_maxiter=100,
            lin_steps=2,
            diff_steps=0,
            fs_rho=1e-3,
            fs_damp=1.0,
        ).validate()

        for case in self.cases:
            data = OPFData(case["train_h5"], case["test_h5"], case["json"])
            nn_inputs, targets = next(
                data.make_test_dataset(samples=4).batch(4).as_numpy_iterator()
            )
            nn_inputs = jnp.asarray(nn_inputs)
            targets = jnp.asarray(targets)

            inputs_np = np.asarray(nn_inputs)
            topo = precompute_topology(data, inputs_np)
            nn_inputs = jnp.asarray(inputs_np)
            targets = jnp.asarray(targets)
            batch = prepare_batch_data(nn_inputs, targets, data, topo=topo)

            state = _DummyState(data.n_der_phase)
            out_ref = forward_pass_core(
                None, state, nn_inputs, batch, data.net, topo, fs_config
            )
            out_inf = forward_pass_inference_core(
                None, state, nn_inputs, batch, data.net, topo, fs_config
            )

            p_diff = float(jnp.max(jnp.abs(out_ref.pder_final - out_inf.pder_final)))
            q_diff = float(jnp.max(jnp.abs(out_ref.qder_final - out_inf.qder_final)))
            v_diff = float(jnp.max(jnp.abs(out_ref.v_hat - out_inf.v_hat)))

            self.assertLessEqual(p_diff, 1e-6, f"{case['name']} p diff {p_diff:.3e}")
            self.assertLessEqual(q_diff, 1e-6, f"{case['name']} q diff {q_diff:.3e}")
            self.assertLessEqual(v_diff, 1e-6, f"{case['name']} v diff {v_diff:.3e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
