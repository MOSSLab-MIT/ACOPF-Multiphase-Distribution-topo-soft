"""Targeted regression tests for unified topology precompute (post-review fixes)."""
import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import CASE_ROOTS, case_paths
from core.grids.data import OPFData
from core.grids.metrics import calc_ineq_violation
from core.topology.batch import (
    precompute_topology,
    build_topology_batch,
    build_topology_batch_jit,
)
from core.topology.utils import Y_LL_L0_from_scaled

CASE = "IEEE13Nodeckt" if "IEEE13Nodeckt" in CASE_ROOTS else None
ATOL = 1e-8
RTOL = 1e-8


def _inverse_residual(Y_LL, Y_inv):
    eye = np.eye(Y_LL.shape[-1], dtype=np.complex128)
    return float(np.max(np.abs(Y_LL @ Y_inv - eye)))


def _load_compact_rows(case_name: str, n_rows: int) -> np.ndarray:
    paths = case_paths(case_name)
    with h5py.File(paths["train"], "r") as f:
        return np.asarray(f["input"]["changing_switch_status"][:n_rows], dtype=np.int32)


def _read_h5_inputs(data: OPFData, n: int) -> np.ndarray:
    """Load first ``n`` input rows from HDF5 (avoids TensorFlow dataset)."""
    with h5py.File(data.test_h5_path, "r") as f:
        parts = []
        for key in data.input_keys:
            arr = np.array(f["input"][key][:n])
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            parts.append(arr)
        return np.concatenate(parts, axis=1).astype(np.float64)


def _read_h5_targets(data: OPFData, n: int) -> np.ndarray:
    with h5py.File(data.test_h5_path, "r") as f:
        parts = [np.array(f["primal"][key][:n]) for key in data.output_keys]
        return np.concatenate(parts, axis=1).astype(np.float64)


def _inputs_with_compact(data: OPFData, compact: np.ndarray) -> np.ndarray:
    n = int(compact.shape[0])
    inputs = _read_h5_inputs(data, n)
    if inputs.shape[0] < n:
        inputs = np.tile(inputs[:1], (n, 1))
    inputs = inputs.copy()
    inputs[:, data.slices["input"][data.topology_input_key]] = compact
    return inputs


@pytest.mark.skipif(CASE is None, reason="IEEE13 case not available")
def test_partial_dedup_gather_matches_full_batch():
    paths = case_paths(CASE)
    data = OPFData(str(paths["train"]), str(paths["test_feasible"]), str(paths["json"]))
    compact = _load_compact_rows(CASE, 8)
    # Force 3 unique rows in B=8
    compact[2:4] = compact[0:2]
    compact[4:6] = compact[0:2]
    compact[6:8] = compact[1:3]

    inputs = _inputs_with_compact(data, compact)
    topo_dedup = precompute_topology(data, inputs, dedup=True)
    topo_full = build_topology_batch(compact, data)

    for name in ("Y_0row", "Y_LL_inv", "w", "branch_status"):
        np.testing.assert_allclose(
            np.asarray(getattr(topo_dedup, name)),
            np.asarray(getattr(topo_full, name)),
            atol=ATOL,
            rtol=RTOL,
            err_msg=name,
        )


@pytest.mark.skipif(CASE is None, reason="IEEE13 case not available")
def test_woodbury_precompute_matches_stamp_host():
    paths = case_paths(CASE)
    data_wb = OPFData(
        str(paths["train"]), str(paths["test_feasible"]), str(paths["json"]), topology_backend="woodbury"
    )
    data_st = OPFData(
        str(paths["train"]), str(paths["test_feasible"]), str(paths["json"]), topology_backend="stamp"
    )
    compact = _load_compact_rows(CASE, 4)
    inputs = _inputs_with_compact(data_wb, compact)

    topo_wb = precompute_topology(data_wb, inputs)
    topo_st = precompute_topology(data_st, inputs)

    for i in range(compact.shape[0]):
        Y_full_i, _, _, _ = data_st.net.topology_matrices(
            jnp.asarray(compact[i], dtype=jnp.int32)
        )
        Y_LL = np.asarray(Y_LL_L0_from_scaled(Y_full_i, data_wb.net.n_net)[0])
        y_inv_wb = np.asarray(topo_wb.Y_LL_inv[i])
        y_inv_st = np.asarray(topo_st.Y_LL_inv[i])
        res_wb = _inverse_residual(Y_LL, y_inv_wb)
        res_st = _inverse_residual(Y_LL, y_inv_st)
        assert res_wb < 1e-6, f"woodbury inverse residual {res_wb}"
        assert res_st < 1e-6, f"stamp inverse residual {res_st}"
        cond = float(np.linalg.cond(Y_LL))
        if cond > 1e12:
            # Ill-conditioned: both inverses must be valid; skip element-wise parity.
            continue
        np.testing.assert_allclose(
            y_inv_wb,
            y_inv_st,
            atol=1e-6,
            rtol=1e-6,
            err_msg=f"sample {i} Y_LL_inv (cond={cond:.2e})",
        )


@pytest.mark.skipif(CASE is None, reason="IEEE13 case not available")
def test_violation_respects_topo_y_0row():
    """The soft substation penalty must honor the ``Y_0row`` argument; hard violation must not use it.

    The substation limit is soft: its Y-dependence lives in ``calc_soft_penalties`` (``pen_sub``), while
    ``calc_ineq_violation`` covers only DER limits and is therefore independent of ``Y_0row``.
    """
    import jax.numpy as jnp

    from core.grids.metrics import calc_soft_penalties, calc_substation_power_per_phase
    from core.utils.batch import prepare_batch_data

    paths = case_paths(CASE)
    data = OPFData(str(paths["train"]), str(paths["test_feasible"]), str(paths["json"]))
    net = data.net

    compact = _load_compact_rows(CASE, 1)
    inputs = _inputs_with_compact(data, compact)
    targets = _read_h5_targets(data, 1)
    topo = precompute_topology(data, inputs)

    batch = prepare_batch_data(jnp.asarray(inputs), jnp.asarray(targets), data, topo=topo)
    p = batch.pder_target
    q = batch.qder_target
    v_full = jnp.concatenate(
        [batch.v_target, jnp.broadcast_to(net.v_slack_local, (1, net.n_slk))], axis=1
    )

    y_0row_base = net.Y_scaled_base[net.n_net : net.n_total, :]
    Y_base = jnp.broadcast_to(y_0row_base, topo.Y_0row.shape)
    Y_pert = topo.Y_0row * jnp.asarray(1.1 + 0.0j, dtype=jnp.complex128)
    Y_huge = topo.Y_0row * jnp.asarray(1000.0 + 0.0j, dtype=jnp.complex128)

    # Substation power is Y-dependent.
    ssub_topo = jnp.abs(calc_substation_power_per_phase(net, v_full, topo.Y_0row))
    ssub_pert = jnp.abs(calc_substation_power_per_phase(net, v_full, Y_pert))
    assert float(jnp.max(jnp.abs(ssub_topo - ssub_pert))) > 1e-12

    # pen_sub follows the Y_0row passed in; V / I / θ penalties do not depend on it.
    pen_topo = np.asarray(calc_soft_penalties(net, v_full, topo.Y_0row))
    pen_huge = np.asarray(calc_soft_penalties(net, v_full, Y_huge))
    assert float(np.max(pen_huge[:, 3] - pen_topo[:, 3])) > 1e-6
    np.testing.assert_array_equal(pen_huge[:, :3], pen_topo[:, :3])
    # A 2-D (unbatched) Y_0row is broadcast over the batch.
    np.testing.assert_array_equal(
        np.asarray(calc_soft_penalties(net, v_full, y_0row_base)),
        np.asarray(calc_soft_penalties(net, v_full, Y_base)),
    )

    vio_topo = calc_ineq_violation(net, p, q, batch.pder_avai, v_full, Y_0row=topo.Y_0row)
    vio_huge = calc_ineq_violation(net, p, q, batch.pder_avai, v_full, Y_0row=Y_huge)
    vio_implicit = calc_ineq_violation(net, p, q, batch.pder_avai, v_full, Y_0row=None)
    np.testing.assert_array_equal(np.asarray(vio_huge), np.asarray(vio_topo))
    np.testing.assert_array_equal(np.asarray(vio_implicit), np.asarray(vio_topo))


# ---------------------------------------------------------------------------
# Woodbury IR-step + vmap/map batch mode tests
# ---------------------------------------------------------------------------

CASE_906 = "906bus_Master" if "906bus_Master" in CASE_ROOTS else None


@pytest.mark.skipif(CASE_906 is None, reason="906bus case not available")
def test_precompute_topology_vmap_matches_map_906():
    """vmap and map batch modes produce identical precompute_topology output."""
    paths = case_paths(CASE_906)
    train_h5 = str(paths["test_feasible"]) if paths["test_feasible"].exists() else str(paths["train"])
    try:
        with h5py.File(train_h5, "r") as f:
            compact = np.asarray(f["input"]["changing_switch_status"][:4], dtype=np.int32)
    except (OSError, KeyError):
        pytest.skip("No HDF5 with changing_switch_status for 906bus")

    data_vmap = OPFData(
        train_h5, str(paths["test_feasible"]), str(paths["json"]),
        woodbury_ir_steps="0", woodbury_batch_mode="vmap",
    )
    data_map = OPFData(
        train_h5, str(paths["test_feasible"]), str(paths["json"]),
        woodbury_ir_steps="0", woodbury_batch_mode="map",
    )

    topo_vmap = precompute_topology(data_vmap, compact, dedup=False)
    topo_map = precompute_topology(data_map, compact, dedup=False)

    for name in ("Y_0row", "Y_LL_inv", "w", "branch_status"):
        np.testing.assert_allclose(
            np.asarray(getattr(topo_vmap, name)),
            np.asarray(getattr(topo_map, name)),
            atol=1e-10, rtol=1e-10,
            err_msg=name,
        )


@pytest.mark.skipif(CASE is None, reason="IEEE13 case not available")
def test_precompute_topology_auto_resolves_ir_steps():
    """OPFData with woodbury_ir_steps='auto' resolves to 1 on IEEE13."""
    paths = case_paths(CASE)
    data = OPFData(
        str(paths["train"]), str(paths["test_feasible"]), str(paths["json"]),
        woodbury_ir_steps="auto",
    )
    assert data.topology_backend._resolved_ir_steps == 1
