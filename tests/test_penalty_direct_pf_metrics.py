"""PF-repaired Penalty Direct metrics: PF at the predicted DERs, predicted voltages discarded.

Checks on real feeder data that the repaired evaluation (1) is the Penalty PF deploy pipeline,
(2) does not depend on the predicted voltages, (3) lands on a power-flow solution, and (4) gives
zero optimality gap for a perfect DER prediction.
"""

import numpy as np
import pytest
import yaml
import jax
import jax.numpy as jnp

from core.utils.runtime import configure_tensorflow_for_data_loading

configure_tensorflow_for_data_loading()
jax.config.update("jax_enable_x64", True)

from conftest import CASE_ROOTS, ROOT
from core.grids.data import OPFData
from core.grids.metrics import calc_ac_metrics_pf, compute_optimality_metrics
from core.topology.batch import precompute_topology
from core.utils.batch import prepare_batch_data
from core.utils.datasets import apply_power_flow_solver_override
from methods.penalty_direct.trainer import pf_repaired_metrics, solve_pf_at_predicted_der
from methods.penalty_pf.trainer import _prediction_metrics, compute_voltage

TRAIN_TAG = "_060_020_020"
N_SAMPLES = 8
# case_dir -> Slurm run config, whose ``args.power_flow_solver`` is the solver that grid is run with
# (e.g. 240 bus uses newton_gmres: fixed-point PF does not converge there).
RUN_CONFIGS = {"IEEE123Master": "123.yaml", "240bus_Master": "240.yaml"}


def _production_pf_solver(name: str) -> str:
    config = yaml.safe_load((ROOT / "scripts/slurm_runs/configs" / RUN_CONFIGS[name]).read_text())
    assert config["case_dir"] == name
    return config["args"]["power_flow_solver"]


@pytest.fixture(scope="module", params=list(RUN_CONFIGS))
def case(request):
    name = request.param
    root = CASE_ROOTS[name]
    train_h5 = root / f"{name}_training{TRAIN_TAG}.h5"
    test_h5 = root / f"{name}_test_feasible.h5"
    if not (train_h5.exists() and test_h5.exists()):
        pytest.skip(f"No training/feasible test HDF5 for {name}")
    data = OPFData(str(train_h5), str(test_h5), str(root / "network_config.json"))
    apply_power_flow_solver_override(data, _production_pf_solver(name))
    inputs = np.asarray(data._load_inputs(test_h5, samples=N_SAMPLES))
    targets = data._load_targets(test_h5, samples=N_SAMPLES)
    topo = precompute_topology(data, inputs)
    batch = prepare_batch_data(jnp.asarray(inputs), jnp.asarray(targets), data, topo=topo)
    return data, batch, topo


def _direct_x(batch, vr, vi):
    """Direct-method output layout ``[pder, qder, vr, vi]`` with the target DER setpoints."""
    return jnp.concatenate([batch.pder_target, batch.qder_target, vr, vi], axis=1)


def _garbage_voltages(net, batch, seed):
    key_r, key_i = jax.random.split(jax.random.PRNGKey(seed))
    shape = (batch.pder_target.shape[0], net.n_net)
    return jax.random.uniform(key_r, shape, minval=-1.0, maxval=1.0), jax.random.uniform(
        key_i, shape, minval=-1.0, maxval=1.0
    )


def test_matches_penalty_pf_pipeline(case):
    """Same voltages, cost and violation as Penalty PF's deploy path at the same DER setpoints."""
    data, batch, topo = case
    net = data.net
    x_direct = _direct_x(batch, *_garbage_voltages(net, batch, 0))
    x_pf = jnp.concatenate([batch.pder_target, batch.qder_target], axis=1)

    v_direct = solve_pf_at_predicted_der(net, batch.s_load, x_direct, topo)
    v_pf = compute_voltage(net, batch, x_pf, topo)
    np.testing.assert_array_equal(np.asarray(v_direct), np.asarray(v_pf))

    m = pf_repaired_metrics(net, batch, x_direct, v_direct, topo)
    cost_pf, violation_pf = _prediction_metrics(net, batch, x_pf, v_pf, topo)
    np.testing.assert_array_equal(np.asarray(m.cost), np.asarray(cost_pf))
    np.testing.assert_array_equal(np.asarray(m.violation), np.asarray(violation_pf))


def test_ignores_predicted_voltages(case):
    """Two different predicted-voltage blocks give bitwise-identical repaired results."""
    data, batch, topo = case
    net = data.net
    results = []
    for seed in (1, 2):
        x = _direct_x(batch, *_garbage_voltages(net, batch, seed))
        v = solve_pf_at_predicted_der(net, batch.s_load, x, topo)
        results.append((v, pf_repaired_metrics(net, batch, x, v, topo)))
    (v_a, m_a), (v_b, m_b) = results
    np.testing.assert_array_equal(np.asarray(v_a), np.asarray(v_b))
    for field in ("cost", "violation", "eq_violation", "ineq_violation", "soft_penalty"):
        np.testing.assert_array_equal(np.asarray(getattr(m_a, field)), np.asarray(getattr(m_b, field)))


def test_repaired_point_satisfies_power_flow(case):
    """The repaired voltages solve AC power flow (same threshold as the PF target tests)."""
    data, batch, topo = case
    net = data.net
    x = _direct_x(batch, *_garbage_voltages(net, batch, 3))
    v = solve_pf_at_predicted_der(net, batch.s_load, x, topo)
    m = pf_repaired_metrics(net, batch, x, v, topo)
    assert float(m.eq_violation.max()) < 1e-6


def test_perfect_der_prediction_has_zero_gap(case):
    """Target DERs + repair reproduce the target cost (scored as in testing), so the gap is ~0."""
    data, batch, topo = case
    net = data.net
    x = _direct_x(batch, *_garbage_voltages(net, batch, 4))
    v = solve_pf_at_predicted_der(net, batch.s_load, x, topo)
    m = pf_repaired_metrics(net, batch, x, v, topo)
    target = calc_ac_metrics_pf(
        net,
        batch.s_load,
        batch.pder_target,
        batch.qder_target,
        batch.pder_avai,
        batch.obj_weights,
        topo,
        v_init=batch.v_target,
    )
    np.testing.assert_allclose(np.asarray(m.cost), np.asarray(target.cost), atol=1e-6)
    _, _, gap = compute_optimality_metrics(m.cost, target.cost)
    assert float(jnp.max(jnp.abs(gap))) < 1e-6
