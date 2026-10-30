"""PF-consistent target metrics for fair eval vs NN+PF predictions."""

import os
import sys

import numpy as np
import pytest
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.utils.runtime import configure_tensorflow_for_data_loading

configure_tensorflow_for_data_loading()
jax.config.update("jax_enable_x64", True)

from conftest import case_paths
from core.grids.data import OPFData
from core.grids.metrics import (
    calc_ac_metrics_pf,
    calc_ineq_violation,
    compute_ac_metrics_from_voltage,
)
from core.grids.metrics import with_slack_voltage
from core.topology.batch import precompute_topology
from core.utils.batch import prepare_batch_data


def _load_batch(case_name: str, split: str, n_samples: int = 8):
    paths = case_paths(case_name)
    if split == "feasible":
        h5_path = paths["test_feasible"]
    else:
        h5_path = paths["root"] / f"{case_name}_test_{split}.h5"
    if not h5_path.exists():
        pytest.skip(f"No {split} test HDF5 for {case_name}")
    data = OPFData(str(paths["train"]), str(h5_path), str(paths["json"]))
    inputs = data._load_inputs(h5_path, samples=n_samples)
    targets = data._load_targets(h5_path, samples=n_samples)
    inputs_np = np.asarray(inputs)
    topo = precompute_topology(data, inputs_np)
    batch = prepare_batch_data(
        jnp.asarray(inputs_np), jnp.asarray(targets), data, topo=topo
    )
    return data, batch, topo


def _stored_ineq_violation(net, batch, topo):
    v_full = with_slack_voltage(net, batch.v_target)
    return calc_ineq_violation(
        net,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        v_full,
        Y_0row=topo.Y_0row,
    )


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
def feasible_batch():
    return _load_batch("IEEE13Nodeckt", "feasible")


@pytest.fixture(scope="module")
def infeasible_batch():
    return _load_batch("IEEE13Nodeckt", "infeasible")


def test_pf_target_matches_stored_on_feasible(feasible_batch):
    """Feasible OPF primals are PF-consistent; inequality metrics should agree."""
    data, batch, topo = feasible_batch
    net = data.net
    pf_metrics = calc_ac_metrics_pf(
        net,
        batch.s_load,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        batch.obj_weights,
        topo,
        v_init=batch.v_target,
    )
    vio_stored = _stored_ineq_violation(net, batch, topo)
    np.testing.assert_allclose(
        np.asarray(pf_metrics.ineq_violation),
        np.asarray(vio_stored),
        atol=1e-7,
        err_msg="PF target ineq violation should match stored primal on feasible data",
    )


def test_cost_target_matches_stored_on_feasible(feasible_batch):
    """Stored OPF voltages are PF-consistent on feasible IEEE13; target cost matches."""
    data, batch, topo = feasible_batch
    net = data.net
    pf_metrics = calc_ac_metrics_pf(
        net,
        batch.s_load,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        batch.obj_weights,
        topo,
        v_init=batch.v_target,
    )
    cost_stored = _cost_at_stored_voltage(net, batch, topo)
    np.testing.assert_allclose(
        np.asarray(pf_metrics.cost),
        np.asarray(cost_stored),
        atol=1e-6,
        err_msg="PF target cost should match stored v_target cost on feasible data",
    )


def test_feasible_target_eq_residual_small(feasible_batch):
    """PF-consistent feasible targets should have small Z-bus equality residual."""
    data, batch, topo = feasible_batch
    net = data.net
    pf_metrics = calc_ac_metrics_pf(
        net,
        batch.s_load,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        batch.obj_weights,
        topo,
        v_init=batch.v_target,
    )
    assert float(pf_metrics.eq_violation.max()) < 1e-6


def test_total_violation_is_max_of_components(feasible_batch):
    """Reported violation must equal max(eq, ineq) per sample."""
    data, batch, topo = feasible_batch
    net = data.net
    pf_metrics = calc_ac_metrics_pf(
        net,
        batch.s_load,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        batch.obj_weights,
        topo,
        v_init=batch.v_target,
    )
    expected = jnp.maximum(pf_metrics.eq_violation, pf_metrics.ineq_violation)
    np.testing.assert_allclose(
        np.asarray(pf_metrics.violation),
        np.asarray(expected),
        rtol=0,
        atol=0,
    )


def test_cost_target_pf_differs_from_stored_on_infeasible(infeasible_batch):
    """Infeasible Ipopt primals: PF at target DER diverges from stored v_target."""
    data, batch, topo = infeasible_batch
    net = data.net
    pf_metrics = calc_ac_metrics_pf(
        net,
        batch.s_load,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        batch.obj_weights,
        topo,
        v_init=batch.v_target,
    )
    cost_stored = _cost_at_stored_voltage(net, batch, topo)
    rel = (pf_metrics.cost - cost_stored) / (jnp.abs(cost_stored) + 1e-6)
    assert float(rel.mean()) > 0.1
    # Warm-starting from v_target does not pin the FP root on infeasible samples.
    pf_metrics_topo_w = calc_ac_metrics_pf(
        net,
        batch.s_load,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        batch.obj_weights,
        topo,
        v_init=topo.w,
    )
    np.testing.assert_allclose(
        np.asarray(pf_metrics.cost),
        np.asarray(pf_metrics_topo_w.cost),
        rtol=1e-4,
        err_msg="v_target vs topo.w PF init should give the same target cost",
    )


def test_pf_target_exceeds_stored_on_infeasible(infeasible_batch):
    """Infeasible Ipopt primals can disagree with Python PF at the same DER setpoint."""
    data, batch, topo = infeasible_batch
    net = data.net
    pf_metrics = calc_ac_metrics_pf(
        net,
        batch.s_load,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        batch.obj_weights,
        topo,
        v_init=batch.v_target,
    )
    vio_stored = _stored_ineq_violation(net, batch, topo)
    assert float(vio_stored.max()) < 1e-6
    assert float(pf_metrics.violation.mean()) > 0.05
    assert float(pf_metrics.violation.max()) > float(vio_stored.max()) * 1000.0
