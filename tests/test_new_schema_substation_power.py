import h5py
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import case_paths
from core.grids.data import OPFData
from core.grids.metrics import calc_substation_power
from core.grids.metrics import calc_substation_power_per_phase
from core.grids.metrics import substation_power_from_voltage as _substation_power_from_voltage

# python -m pytest -q tests/test_new_schema_substation_power.py

CASES = ["IEEE13Nodeckt"]  # TODO: add IEEE123Master when we have more samples in the HDF5 files.
EXPR_ATOL = 1e-10

# Stored psub/qsub labels come from nonlinear solver variables written to HDF5.
# They are close to, but not exactly equal to, voltage-recomputed values.
CASE_PSUB_LABEL_ATOL = {
    "IEEE13Nodeckt": 1e-6,
    "IEEE123Master": 1e-6,
}
CASE_QSUB_LABEL_ATOL = {
    "IEEE13Nodeckt": 1e-6,
    "IEEE123Master": 1e-6,
}
CASE_PSUB_PHASE_LABEL_ATOL = {
    "IEEE13Nodeckt": 1e-6,
    "IEEE123Master": 1e-6,
}


def _load_case_batch(case_name: str, n_samples: int = 10):
    paths = case_paths(case_name)
    data = OPFData(str(paths["train"]), str(paths["test_feasible"]), str(paths["json"]))

    with h5py.File(paths["test_feasible"], "r") as f:
        n = min(n_samples, f["primal"]["vr"].shape[0])
        vr = np.asarray(f["primal"]["vr"][:n], dtype=np.float64)
        vi = np.asarray(f["primal"]["vi"][:n], dtype=np.float64)
        psub = np.asarray(f["primal"]["psub"][:n], dtype=np.float64).reshape(n)
        qsub = np.asarray(f["primal"]["qsub"][:n], dtype=np.float64).reshape(n)
        psub_phase = np.asarray(f["primal"]["psub_phase"][:n], dtype=np.float64)
        qsub_phase = np.asarray(f["primal"]["qsub_phase"][:n], dtype=np.float64)
        branch_status = np.asarray(f["input"]["branch_status"][:n], dtype=np.int32)

    return data, vr, vi, psub, qsub, psub_phase, qsub_phase, branch_status


def _compose_v_full(data: OPFData, vr: np.ndarray, vi: np.ndarray) -> jnp.ndarray:
    v_non_slack = jnp.asarray(vr) + 1j * jnp.asarray(vi)
    v_slack = jnp.broadcast_to(data.net.v_slack_local, (v_non_slack.shape[0], data.net.n_slk))
    return jnp.concatenate([v_non_slack, v_slack], axis=1)


def _reference_substation_power(data: OPFData, v_full: jnp.ndarray, y_scaled: jnp.ndarray):
    v = np.asarray(v_full, dtype=np.complex128)
    y = np.asarray(y_scaled, dtype=np.complex128)
    v_slack = np.asarray(data.net.v_slack_local, dtype=np.complex128)

    if v.ndim == 1:
        i_slack = y[data.net.n_net : data.net.n_total, :] @ v
        return np.sum(v_slack * np.conj(i_slack))

    i_slack = np.einsum("bij,bj->bi", y[:, data.net.n_net : data.net.n_total, :], v)
    return np.sum(v_slack[None, :] * np.conj(i_slack), axis=1)


def _reference_substation_power_per_phase(data: OPFData, v_full: jnp.ndarray, y_scaled: jnp.ndarray):
    v = np.asarray(v_full, dtype=np.complex128)
    y = np.asarray(y_scaled, dtype=np.complex128)
    v_slack = np.asarray(data.net.v_slack_local, dtype=np.complex128)

    if v.ndim == 1:
        i_slack = y[data.net.n_net : data.net.n_total, :] @ v
        return v_slack * np.conj(i_slack)

    i_slack = np.einsum("bij,bj->bi", y[:, data.net.n_net : data.net.n_total, :], v)
    return v_slack[None, :] * np.conj(i_slack)


@pytest.mark.parametrize("case_name", CASES)
def test_calc_substation_power_matches_solver_psub_qsub(case_name):
    data, vr, vi, psub_ref, qsub_ref, _, _, branch_status = _load_case_batch(case_name)

    Y_scaled, *_ = data.net.topology_matrices_batch(jnp.asarray(branch_status))
    v_full = _compose_v_full(data, vr, vi)
    s_sub = np.asarray(calc_substation_power(data.net, v_full, Y_scaled))
    s_ref = _reference_substation_power(data, v_full, Y_scaled)

    np.testing.assert_allclose(s_sub, s_ref, atol=EXPR_ATOL, rtol=0.0)

    np.testing.assert_allclose(s_sub.real, psub_ref, atol=CASE_PSUB_LABEL_ATOL[case_name], rtol=0.0)
    np.testing.assert_allclose(s_sub.imag, qsub_ref, atol=CASE_QSUB_LABEL_ATOL[case_name], rtol=0.0)


@pytest.mark.parametrize("case_name", CASES)
def test_compute_psub_from_v_full_matches_solver_psub(case_name):
    data, vr, vi, psub_ref, _, _, _, branch_status = _load_case_batch(case_name)

    Y_scaled, *_ = data.net.topology_matrices_batch(jnp.asarray(branch_status))
    v_full = _compose_v_full(data, vr, vi)
    psub = np.asarray(_substation_power_from_voltage(data.net, v_full, Y_scaled).real)
    s_ref = _reference_substation_power(data, v_full, Y_scaled)

    np.testing.assert_allclose(psub, s_ref.real, atol=EXPR_ATOL, rtol=0.0)
    np.testing.assert_allclose(psub, psub_ref, atol=CASE_PSUB_LABEL_ATOL[case_name], rtol=0.0)


@pytest.mark.parametrize("case_name", CASES)
def test_per_phase_substation_power_matches_solver_labels(case_name):
    data, vr, vi, psub_ref, qsub_ref, psub_phase_ref, qsub_phase_ref, branch_status = _load_case_batch(
        case_name
    )

    Y_scaled, *_ = data.net.topology_matrices_batch(jnp.asarray(branch_status))
    v_full = _compose_v_full(data, vr, vi)
    s_phase = np.asarray(calc_substation_power_per_phase(data.net, v_full, Y_scaled))
    s_ref = _reference_substation_power_per_phase(data, v_full, Y_scaled)

    np.testing.assert_allclose(s_phase, s_ref, atol=EXPR_ATOL, rtol=0.0)
    np.testing.assert_allclose(
        s_phase.real, psub_phase_ref, atol=CASE_PSUB_PHASE_LABEL_ATOL[case_name], rtol=0.0
    )
    np.testing.assert_allclose(
        s_phase.imag, qsub_phase_ref, atol=CASE_PSUB_PHASE_LABEL_ATOL[case_name], rtol=0.0
    )
    np.testing.assert_allclose(np.sum(psub_phase_ref, axis=1), psub_ref, atol=CASE_PSUB_LABEL_ATOL[case_name])
    np.testing.assert_allclose(np.sum(qsub_phase_ref, axis=1), qsub_ref, atol=CASE_QSUB_LABEL_ATOL[case_name])
