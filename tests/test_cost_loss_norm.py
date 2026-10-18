"""Tests for eval-oracle cost scaling used only in loss."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from core.grids.data import OPFData
from core.grids.metrics import compute_ac_metrics_from_voltage
from core.grids.metrics import with_slack_voltage
from core.utils.loss import (
    CostNormStats,
    compute_cost_norm_stats_from_eval,
    normalize_cost_for_loss,
    read_primal_objective_values,
)
from core.topology.batch import precompute_topology
from core.utils.workflow import prepare_batch_data

DATASET_ROOT = Path(__file__).resolve().parents[1] / "datagen_topo" / "datasets"
IEEE13 = {
    "train": DATASET_ROOT / "IEEE13Nodeckt" / "IEEE13Nodeckt_training.h5",
    "test": DATASET_ROOT / "IEEE13Nodeckt" / "IEEE13Nodeckt_test_feasible.h5",
    "json": DATASET_ROOT / "IEEE13Nodeckt" / "network_config.json",
}


def _cost_at_stored_voltage(net, batch, topo):
    v_full = with_slack_voltage(net, batch.v_target)
    metrics = compute_ac_metrics_from_voltage(
        net,
        batch.s_load,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        v_full,
        batch.obj_weights,
        topo,
    )
    return metrics.cost


@pytest.fixture(scope="module")
def ieee13_session():
    if not IEEE13["test"].is_file():
        pytest.skip("IEEE13 feasible eval H5 not available")
    return OPFData(str(IEEE13["train"]), str(IEEE13["test"]), str(IEEE13["json"]))


@pytest.fixture(scope="module")
def ieee13_eval_dataset(ieee13_session):
    return ieee13_session.make_test_dataset(samples=64)


def test_normalize_cost_disabled():
    cost = jnp.array([1.0, 2.0, 3.0])
    out = normalize_cost_for_loss(cost, None)
    assert jnp.allclose(out, cost)


def test_normalize_cost_divide():
    scale = jnp.asarray(10.0)
    cost = jnp.array([2.0, 5.0, 20.0])
    out = normalize_cost_for_loss(cost, scale)
    assert jnp.allclose(out, cost / 10.0)


def test_read_primal_objective_values(ieee13_session):
    stored = read_primal_objective_values(ieee13_session, samples=64)
    assert stored is not None
    assert stored.shape == (64,)
    assert np.all(np.isfinite(stored))


def test_cost_norm_from_stored_matches_recompute(ieee13_session, ieee13_eval_dataset):
    """Stored ``meta/primal_objective_value`` must match recomputed target costs."""
    stored_stats = compute_cost_norm_stats_from_eval(
        ieee13_session, mode="robust", samples=64, h5_path=IEEE13["test"]
    )
    # Compare stored HDF5 objectives to recomputed target costs.
    stored = read_primal_objective_values(ieee13_session, samples=64)
    recomputed = []
    for inputs_batch, targets_batch in ieee13_eval_dataset.batch(16).as_numpy_iterator():
        inputs_np = np.asarray(inputs_batch)
        topo = precompute_topology(ieee13_session, inputs_np)
        inputs_j = jnp.asarray(inputs_batch, dtype=jnp.float64)
        targets_j = jnp.asarray(targets_batch, dtype=jnp.float64)
        batch = prepare_batch_data(inputs_j, targets_j, ieee13_session, topo=topo)
        recomputed.append(
            np.asarray(
                _cost_at_stored_voltage(ieee13_session.net, batch, topo),
                dtype=np.float64,
            )
        )
    all_recomputed = np.concatenate(recomputed)
    assert stored.shape == all_recomputed.shape
    np.testing.assert_allclose(stored, all_recomputed, rtol=0, atol=1e-8)
    assert stored_stats.n_samples == 64


def test_robust_scale_below_max(ieee13_session, ieee13_eval_dataset):
    stats = compute_cost_norm_stats_from_eval(ieee13_session, mode="robust")
    assert stats.cost_scale <= stats.cost_max + 1e-12
    assert stats.cost_scale >= max(stats.cost_p95, 1e-8) - 1e-12
    assert stats.n_samples > 0


def test_max_mode_uses_max(ieee13_session, ieee13_eval_dataset):
    stats = compute_cost_norm_stats_from_eval(ieee13_session, mode="max")
    assert stats.cost_scale == pytest.approx(max(stats.cost_max, 1e-8))


def test_target_cost_at_p95_scale(ieee13_session, ieee13_eval_dataset):
    """At p95 reference, normalized target costs should have max ~ 1."""
    stats = compute_cost_norm_stats_from_eval(ieee13_session, mode="robust")
    scale = stats.cost_scale_jax()
    norms = []
    for inputs_batch, targets_batch in (
        ieee13_eval_dataset.batch(16).as_numpy_iterator()
    ):
        inputs_np = np.asarray(inputs_batch)
        topo = precompute_topology(ieee13_session, inputs_np)
        inputs_j = jnp.asarray(inputs_batch, dtype=jnp.float64)
        targets_j = jnp.asarray(targets_batch, dtype=jnp.float64)
        batch = prepare_batch_data(inputs_j, targets_j, ieee13_session, topo=topo)
        cost_target = _cost_at_stored_voltage(ieee13_session.net, batch, topo)
        norms.append(np.asarray(normalize_cost_for_loss(cost_target, scale)))
    all_norm = np.concatenate(norms)
    assert float(np.max(all_norm)) <= 1.0 + 1e-6
    assert float(np.percentile(all_norm, 95)) <= 1.0 + 1e-5


def test_cost_norm_none_when_no_val_h5(ieee13_session):
    stats = compute_cost_norm_stats_from_eval(ieee13_session, mode="none")
    assert stats.mode == "none"
    assert stats.n_samples == 0
    assert stats.cost_scale == 1.0


def test_cost_norm_explicit_h5_path_respects_samples(ieee13_session):
    stats_all = compute_cost_norm_stats_from_eval(
        ieee13_session, mode="robust", h5_path=IEEE13["test"]
    )
    stats_subset = compute_cost_norm_stats_from_eval(
        ieee13_session, mode="robust", samples=32, h5_path=IEEE13["test"]
    )
    assert stats_subset.n_samples == 32
    assert stats_all.n_samples > stats_subset.n_samples


def test_extrapolation_above_one():
    scale = jnp.asarray(5.0)
    cost = jnp.asarray([12.5])
    out = float(jnp.asarray(normalize_cost_for_loss(cost, scale)).reshape(-1)[0])
    assert out == pytest.approx(2.5)
    assert out > 1.0
