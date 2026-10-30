"""NN input layout, checkpoint artifacts, and train-state factories on the IEEE13 feeder."""

import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow as tf
from flax import nnx

from conftest import case_paths
from core.grids.data import OPFData
from core.nn.artifacts import resolve_nn_input_for_checkpoint, save_nn_input_artifacts
from core.nn.input import compute_input_stats, gather_nn_features_from_session, nn_features_for_apply
from core.nn.state import create_train_state, create_train_state_full, update_model


@pytest.fixture(scope="module")
def feeder():
    paths = case_paths("IEEE13Nodeckt")
    if not paths["test_feasible"].exists():
        pytest.skip("IEEE13 feasible dataset is unavailable")
    data = OPFData(str(paths["train"]), str(paths["test_feasible"]),
                   str(paths["json"]), woodbury_ir_steps="0")
    x = np.asarray(data._load_inputs(paths["test_feasible"], samples=8))
    return data, x, tf.data.Dataset.from_tensor_slices(x)


def test_compact_features_select_load_columns_then_tail(feeder):
    data, x, _ = feeder
    layout, sl = data.nn_layout, data.slices["input"]
    feats = np.asarray(gather_nn_features_from_session(x, data))
    idx = np.asarray(layout.load_idx)
    expected = np.concatenate(
        [x[:, sl["pd"]][:, idx], x[:, sl["qd"]][:, idx], x[:, layout.tail_start:layout.tail_end]],
        axis=1,
    )
    np.testing.assert_array_equal(feats, expected)
    assert feats.shape[1] == layout.din_compact
    np.testing.assert_array_equal(nn_features_for_apply(x, data, use_compact=False), x)


def test_input_stats_match_numpy_and_check_width(feeder):
    data, x, ds = feeder
    mean, std = compute_input_stats(ds, data, use_compact=True)
    feats = np.asarray(gather_nn_features_from_session(x, data), dtype=np.float64)
    np.testing.assert_allclose(mean, feats.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(std, feats.std(axis=0), rtol=1e-12, atol=1e-12)
    mean_f, _ = compute_input_stats(ds, data, use_compact=False)
    assert mean_f.shape[0] == data.nn_layout.din_full


def test_artifacts_round_trip_and_legacy_fallback(feeder, tmp_path):
    data, _, ds = feeder
    stat = compute_input_stats(ds, data, use_compact=True)
    save_nn_input_artifacts(tmp_path, data.nn_layout, stat)
    layout, (mean, std), use_compact = resolve_nn_input_for_checkpoint(data, tmp_path)
    assert use_compact
    for field in layout._fields:
        np.testing.assert_array_equal(getattr(layout, field), getattr(data.nn_layout, field))
    np.testing.assert_array_equal(mean, stat[0])
    np.testing.assert_array_equal(std, stat[1])

    empty = tmp_path / "legacy"
    empty.mkdir()
    with pytest.raises(ValueError, match="recompute_stats_fn"):
        resolve_nn_input_for_checkpoint(data, empty)
    full_stat = compute_input_stats(ds, data, use_compact=False)
    layout, got, use_compact = resolve_nn_input_for_checkpoint(
        data, empty, recompute_stats_fn=lambda use_compact: full_stat
    )
    assert not use_compact and got is full_stat and layout is data.nn_layout


def test_train_state_outputs_respect_der_and_voltage_bounds(feeder):
    data, x, ds = feeder
    net, layout = data.net, data.nn_layout
    stat = compute_input_stats(ds, data, use_compact=True)
    feats = gather_nn_features_from_session(x, data)
    kw = dict(dhid=16, num_layers=3, scheduler=1e-3, input_stat=stat, data=data, rng=nnx.Rngs(0))

    state = create_train_state(din=layout.din_compact, dout=2 * net.n_der_phase, **kw)
    y = np.asarray(state.apply_fn(state.params, feats))
    assert y.shape == (x.shape[0], 2 * net.n_der_phase)
    n = net.n_der_phase
    assert np.all(y[:, :n] >= np.asarray(net.pdermin)) and np.all(y[:, :n] <= np.asarray(net.pdermax))
    assert np.all(y[:, n:] >= np.asarray(net.qdermin)) and np.all(y[:, n:] <= np.asarray(net.qdermax))

    dout_full = 2 * n + 2 * net.n_net
    state_f = create_train_state_full(din=layout.din_compact, dout=dout_full, **kw)
    yf = np.asarray(state_f.apply_fn(state_f.params, feats))
    assert yf.shape == (x.shape[0], dout_full)
    vr, vi = yf[:, 2 * n:2 * n + net.n_net], yf[:, 2 * n + net.n_net:]
    assert np.all(vr >= np.asarray(net.vrmin)) and np.all(vr <= np.asarray(net.vrmax))
    assert np.all(vi >= np.asarray(net.vimin)) and np.all(vi <= np.asarray(net.vimax))

    grads = {k: jnp.ones_like(v) for k, v in state.params.items()} if isinstance(state.params, dict) else None
    if grads is None:
        import jax
        grads = jax.tree_util.tree_map(jnp.ones_like, state.params)
    stepped = update_model(state, grads)
    assert int(stepped.step) == 1


def test_graph_models_require_full_inputs_and_unknown_type_rejected(feeder):
    data, _, ds = feeder
    stat = compute_input_stats(ds, data, use_compact=False)
    kw = dict(dhid=8, num_layers=3, scheduler=1e-3, input_stat=stat, data=data, rng=nnx.Rngs(0))
    data.use_compact_nn = True
    with pytest.raises(ValueError, match="full NN inputs"):
        create_train_state(din=data.nn_layout.din_full, dout=2, model_type="gcn", **kw)
    with pytest.raises(ValueError, match="model_type must be one of"):
        create_train_state(din=data.nn_layout.din_full, dout=2, model_type="transformer", **kw)
