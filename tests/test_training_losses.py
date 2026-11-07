"""Exercise projected training losses and optimizer/evaluation paths on real feeder data."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from conftest import case_paths
from core.grids.data import OPFData
from core.grids.der_projection import project_der_controls_batch
from core.topology.batch import precompute_topology
from core.utils.batch import prepare_batch_data
from methods.penalty_direct import trainer as direct
from methods.penalty_pf import trainer as pf


@pytest.fixture(scope="module")
def feeder_batch():
    paths = case_paths("IEEE13Nodeckt")
    if not paths["test_feasible"].exists():
        pytest.skip("IEEE13 feasible dataset is unavailable")
    data = OPFData(str(paths["train"]), str(paths["test_feasible"]),
                   str(paths["json"]), woodbury_ir_steps="0")
    inputs = jnp.asarray(data._load_inputs(paths["test_feasible"], samples=4))
    targets = jnp.asarray(data._load_targets(paths["test_feasible"], samples=4))
    topo = precompute_topology(data, np.asarray(inputs))
    return data, inputs, targets, topo


def _state(data, method):
    # A trainable bias-only predictor isolates the physical loss from MLP details.
    params = jnp.concatenate([0.2 * data.net.pdermax, 0.1 * data.net.sdermax])
    if method == "direct":
        params = jnp.concatenate([params, jnp.full(data.net.n_net, 0.98),
                                  jnp.full(data.net.n_net, -0.01)])

    def predict(parameters, features):
        controls = parameters["controls"]
        return jnp.broadcast_to(controls, (features.shape[0], controls.size))

    return TrainState.create(apply_fn=predict, params={"controls": params}, tx=optax.sgd(1e-9))


def _loss(state, feeder, method, scale, params=None):
    data, inputs, targets, topo = feeder
    params = state.params if params is None else params
    if method == "pf":
        return pf.compute_penalty_pf_loss(params, state, inputs, targets, data, topo,
                                           cost_scale=scale)
    return direct.compute_penalty_direct_loss(params, state, inputs, targets, data, 2.0, topo,
                                               cost_scale=scale)


@pytest.mark.parametrize("method", ["pf", "direct"])
@pytest.mark.parametrize("scale", [None, 1.0, 2.0])
def test_training_loss_matches_physical_objective_and_retains_diagnostics(feeder_batch, method, scale):
    data, inputs, targets, topo = feeder_batch
    state = _state(data, method)
    loss, metrics, internals = _loss(state, feeder_batch, method, scale)
    denom = 1.0 if scale is None else scale
    expected = np.mean(np.asarray(internals["cost"])) / denom
    if method == "direct":
        # Independently evaluate the complex Z-bus residual at predicted voltages.
        batch = prepare_batch_data(inputs, targets, data, topo=topo)
        n = data.net.n_der_phase
        controls = state.params["controls"]
        p, q = project_der_controls_batch(
            data.net, jnp.broadcast_to(controls[:n], (len(inputs), n)),
            jnp.broadcast_to(controls[n:2*n], (len(inputs), n)), batch.pder_avai, 1e-12
        )
        s_net = np.asarray(jax.vmap(data.net.compute_net_injection)(batch.s_load, p + 1j*q))
        v = np.asarray(internals["v_final"])
        residual = np.asarray(topo.w) + np.einsum(
            "bij,bj->bi", np.asarray(topo.Y_LL_inv), np.conj(s_net / v)
        ) - v
        eq_mse = np.mean(np.abs(residual)**2)
        assert eq_mse > 1e-6, "Exercise an active equality penalty"
        expected += 2.0 * eq_mse
    np.testing.assert_allclose(loss, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(metrics["cost_norm"], np.mean(internals["cost"]) / denom,
                               atol=1e-12, rtol=1e-12)
    assert float(metrics["ineq_violation"]) <= 1e-12
    assert "ineq_violation_sq" in metrics
    assert "eq_violation" in metrics


@pytest.mark.parametrize("method", ["pf", "direct"])
def test_training_epoch_decreases_loss_and_evaluation_agrees(feeder_batch, method):
    data, inputs, targets, topo = feeder_batch
    state = _state(data, method)
    loss_fn = lambda params: _loss(state, feeder_batch, method, 2.0, params)[0]
    before, grads = jax.value_and_grad(loss_fn)(state.params)
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grads))
    grad_norm = float(optax.global_norm(grads))
    assert grad_norm > 0
    # Keep the update local regardless of the feeder objective's scale.
    state = state.replace(tx=optax.sgd(1e-5 / max(grad_norm, 1.0)))
    if method == "pf":
        updated, train_metrics = pf.train_epoch_penalty_pf(state, [inputs], data, cost_scale=2.0)
        eval_metrics = pf.eval_epoch_penalty_pf(updated, [(inputs, targets)], data, cost_scale=2.0)
    else:
        updated, train_metrics, rho_eq = direct.train_epoch_penalty_direct(
            state, [inputs], data, 2.0, rho_rate=0.0, violation_threshold=1e-6,
            decay_rate=1.0, rho_max=10.0, cost_scale=2.0
        )
        assert rho_eq == 2.0
        eval_metrics = direct.eval_epoch_penalty_direct(
            updated, [(inputs, targets)], data, rho_eq, cost_scale=2.0
        )
    after, _, _ = _loss(updated, feeder_batch, method, 2.0)
    assert float(after) < float(before)
    assert int(updated.step) == int(state.step) + 1
    np.testing.assert_allclose(train_metrics["loss"], before, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(eval_metrics["loss"], after, atol=1e-12, rtol=1e-12)
    assert "ineq_violation_sq" in eval_metrics
