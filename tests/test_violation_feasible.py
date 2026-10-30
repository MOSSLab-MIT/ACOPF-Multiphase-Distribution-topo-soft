"""
Test violation calculation on test_feasible datasets.

Feasible datasets contain solutions from a verified AC OPF solver.
All constraint violations should be near-zero (within numerical tolerance).

Tests both IEEE13Nodeckt and 906bus_Master cases.
"""

import sys
from pathlib import Path
from functools import partial

import numpy as np
import pytest
import jax
import jax.numpy as jnp

# ── project root on sys.path ──
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

jax.config.update("jax_enable_x64", True)

import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

from core.grids.data import OPFData
from core.grids.network import map_to_node
from core.grids.der_projection import unit_active_power_violation_batch
from core.grids.metrics import (
    calc_ineq_violation,
    calc_substation_power_per_phase,
    constraint_bounds,
    constraint_row_layout,
    constraint_values_from_voltage,
    _group_phase_sum,
)
from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow, power_flow_residual

# ── Dataset paths ──
from conftest import DATASET_ROOT

CASES = {
    "IEEE13Nodeckt": {
        "train": DATASET_ROOT / "IEEE13Nodeckt" / "IEEE13Nodeckt_training.h5",
        "test": DATASET_ROOT / "IEEE13Nodeckt" / "IEEE13Nodeckt_test.h5",
        "test_feasible": DATASET_ROOT / "IEEE13Nodeckt" / "IEEE13Nodeckt_test_feasible.h5",
        "json": DATASET_ROOT / "IEEE13Nodeckt" / "network_config.json",
    },
    "906bus_Master": {
        "train": DATASET_ROOT / "906bus_Master" / "906bus_Master_training.h5",
        # 906bus has no _test.h5; use _test_feasible.h5 as test path for OPFData init
        "test": DATASET_ROOT / "906bus_Master" / "906bus_Master_test_feasible.h5",
        "test_feasible": DATASET_ROOT / "906bus_Master" / "906bus_Master_test_feasible.h5",
        "json": DATASET_ROOT / "906bus_Master" / "network_config.json",
    },
}


def _load_feasible_data(case_name: str):
    """Load OPFData and all test_feasible samples."""
    paths = CASES[case_name]
    import h5py

    # OPFData needs train/test/json paths; use test_feasible as test if no _test.h5
    data = OPFData(
        train_h5_path=str(paths["train"]),
        test_h5_path=str(paths["test"]),
        json_path=str(paths["json"]),
    )

    # Load feasible samples directly from HDF5
    with h5py.File(str(paths["test_feasible"]), "r") as f:
        pd = np.array(f["input"]["pd"])
        qd = np.array(f["input"]["qd"])
        pder_avai = np.array(f["input"]["pder_avai"])
        obj_weights = np.array(f["input"]["obj_weights"])
        branch_status_compact = np.array(f["input"]["branch_status"])

        pder = np.array(f["primal"]["pder"])
        qder = np.array(f["primal"]["qder"])
        vr = np.array(f["primal"]["vr"])
        vi = np.array(f["primal"]["vi"])

    return data, pd, qd, pder_avai, obj_weights, branch_status_compact, pder, qder, vr, vi


def _build_v_full(data, vr, vi):
    """Build full local voltage vector [non-slack; slack]."""
    v_net = vr + 1j * vi  # (B, n_net)
    v_slack = np.broadcast_to(
        np.asarray(data.v_slack_local), (v_net.shape[0], data.n_slk)
    )
    return np.concatenate([v_net, v_slack], axis=1)  # (B, n_total)


# ════════════════════════════════════════════════════════════════════
#  Parametrize over both cases
# ════════════════════════════════════════════════════════════════════

@pytest.fixture(params=["IEEE13Nodeckt", "906bus_Master"], scope="module")
def feasible_bundle(request):
    """Load feasible data for a given case. Cached per-module for speed."""
    case_name = request.param
    data, pd, qd, pder_avai, obj_weights, branch_status_compact, pder, qder, vr, vi = \
        _load_feasible_data(case_name)
    v_full = _build_v_full(data, vr, vi)

    # Expand branch status to full width
    branch_status_full = jnp.asarray(
        data.expand_branch_status_np(branch_status_compact)
    )

    # Build Y_0row for each sample via topology_matrices_batch
    from core.topology.utils import y_0row_from_scaled

    Y_scaled_batch = data.topology_matrices_batch(branch_status_full)[0]
    Y_0row_batch = y_0row_from_scaled(
        Y_scaled_batch, data.net.n_net, data.net.n_total
    )

    return {
        "case_name": case_name,
        "data": data,
        "pd": jnp.asarray(pd),
        "qd": jnp.asarray(qd),
        "pder_avai": jnp.asarray(pder_avai),
        "obj_weights": jnp.asarray(obj_weights),
        "branch_status_full": branch_status_full,
        "pder": jnp.asarray(pder),
        "qder": jnp.asarray(qder),
        "vr": jnp.asarray(vr),
        "vi": jnp.asarray(vi),
        "v_full": jnp.asarray(v_full),
        "Y_0row_batch": Y_0row_batch,
        "n_samples": pd.shape[0],
    }


# ════════════════════════════════════════════════════════════════════
#  Test 1: calc_ineq_violation should return small values on feasible data
# ════════════════════════════════════════════════════════════════════

def test_violation_ac_small(feasible_bundle):
    """calc_ineq_violation must return near-zero violations for feasible targets."""
    b = feasible_bundle
    data = b["data"]

    violations = calc_ineq_violation(
        data.net,
        b["pder"],
        b["qder"],
        b["pder_avai"],
        b["v_full"],
        Y_0row=b["Y_0row_batch"],
    )

    violations_np = np.asarray(violations)
    print(f"\n[{b['case_name']}] calc_ineq_violation:")
    print(f"  mean = {violations_np.mean():.6e}")
    print(f"  max  = {violations_np.max():.6e}")
    print(f"  min  = {violations_np.min():.6e}")
    print(f"  std  = {violations_np.std():.6e}")

    # Feasible solutions should have violations below 1e-4 (solver tolerance)
    assert violations_np.max() < 1e-6, (
        f"[{b['case_name']}] Max violation {violations_np.max():.6e} exceeds 1e-4"
    )


# ════════════════════════════════════════════════════════════════════
#  Test 2: Decomposed per-constraint-type violations
# ════════════════════════════════════════════════════════════════════

def test_violation_decomposed(feasible_bundle):
    """Break down violations into individual constraint types and check each."""
    b = feasible_bundle
    data = b["data"].net
    tol = 1e-6

    results = {}

    for i in range(b["n_samples"]):
        x_i = jnp.concatenate([b["pder"][i], b["qder"][i]])
        v_i = b["v_full"][i]
        Y_i = b["Y_0row_batch"][i]
        avail_i = b["pder_avai"][i]

        # ── Constraint values ──
        vals = constraint_values_from_voltage(data, x_i, v_i, Y_i)
        l, u = constraint_bounds(data)
        row_vio = jnp.maximum(jnp.maximum(l - vals, 0.0), jnp.maximum(vals - u, 0.0))

        layout = constraint_row_layout(data)
        vmag_vio = row_vio[layout["sl_v"]]
        angle_vio = row_vio[layout["sl_angle"]]
        ssub_vio = row_vio[layout["sl_sub"]]
        branch_i_vio = row_vio[layout["sl_branch"]]

        # ── DER box + disk + per-unit availability (primal, not linear rows) ──
        p = x_i[: data.n_der_phase]
        q = x_i[data.n_der_phase :]
        p_vio = jnp.maximum(
            jnp.maximum(data.pdermin - p, 0.0),
            jnp.maximum(p - data.pdermax, 0.0),
        )
        q_vio = jnp.maximum(
            jnp.maximum(data.qdermin - q, 0.0),
            jnp.maximum(q - data.qdermax, 0.0),
        )
        s_vio = jnp.maximum(p**2 + q**2 - data.sdermax**2, 0.0)
        unit_avail_vio = float(
            unit_active_power_violation_batch(
                data,
                p[None, :],
                avail_i[None, :],
            )
        )

        # Accumulate per-type max
        sample_results = {
            "vmag": float(jnp.max(vmag_vio)),
            "angle": float(jnp.max(angle_vio)) if n_angle > 0 else 0.0,
            "ssub": float(jnp.max(ssub_vio)),
            "branch_i": float(jnp.max(branch_i_vio)) if n_branch_i > 0 else 0.0,
            "unit_avail": unit_avail_vio,
            "pder_box": float(jnp.max(p_vio)),
            "qder_box": float(jnp.max(q_vio)),
            "sder_disk": float(jnp.max(s_vio)),
        }

        for key, val in sample_results.items():
            if key not in results:
                results[key] = []
            results[key].append(val)

    # Print summary
    print(f"\n[{b['case_name']}] Per-constraint-type max violations (over {b['n_samples']} samples):")
    for key, vals in results.items():
        arr = np.array(vals)
        print(f"  {key:12s}: mean={arr.mean():.6e}  max={arr.max():.6e}")

    # Assert each type is within tolerance
    for key, vals in results.items():
        max_val = np.max(vals)
        assert max_val < tol, (
            f"[{b['case_name']}] {key} violation max={max_val:.6e} exceeds {tol}"
        )


# ════════════════════════════════════════════════════════════════════
#  Test 3: Power flow residual consistency
# ════════════════════════════════════════════════════════════════════

# def test_power_flow_residual_small(feasible_bundle):
#     """Verify that the target voltages satisfy the AC power flow equations."""
#     b = feasible_bundle
#     data = b["data"]

#     s_load = jax.lax.complex(b["pd"], b["qd"])
#     s_der = b["pder"] + 1j * b["qder"]
#     s_injection = jax.vmap(data.compute_net_injection, in_axes=(0, 0))(s_load, s_der)

#     v_net = jax.lax.complex(b["vr"], b["vi"])

#     # Process sample-by-sample to avoid OOM on large cases (e.g. 906 bus)
#     residuals = []
#     for i in range(b["n_samples"]):
#         # Get topology matrices for this single sample
#         topo_i = data.net.topology_matrices(b["branch_status_full"][i])
#         Y_LL_i = topo_i[1]
#         Y_L0_i = topo_i[2]
#         r = power_flow_residual(
#             v_net[i], s_injection[i], Y_LL_i, Y_L0_i, data.v_slack_local
#         )
#         residuals.append(float(jnp.max(jnp.abs(r))))

#     residuals = np.array(residuals)
#     print(f"\n[{b['case_name']}] Power flow residual |u*conj(Yu) - s|:")
#     print(f"  mean = {residuals.mean():.6e}")
#     print(f"  max  = {residuals.max():.6e}")

#     # PF residual should be very small for a solved AC OPF
#     assert residuals.max() < 1e-6, (
#         f"[{b['case_name']}] Max PF residual {residuals.max():.6e} exceeds 1e-6"
#     )


# ════════════════════════════════════════════════════════════════════
#  Test 4: optional square violation path
# ════════════════════════════════════════════════════════════════════

# def test_violation_square_consistency(feasible_bundle):
#     """calc_ineq_violation(return_square=True) must return a valid square penalty."""
#     b = feasible_bundle
#     data = b["data"]

#     s_load = jax.lax.complex(b["pd"], b["qd"])
#     s_der = b["pder"] + 1j * b["qder"]
#     s_injection = jax.vmap(data.compute_net_injection, in_axes=(0, 0))(s_load, s_der)

#     vio, vio_sq = calc_ineq_violation(
#         data.net,
#         b["pder"],
#         b["qder"],
#         b["pder_avai"],
#         b["v_full"],
#         Y_0row=b["Y_0row_batch"],
#         return_square=True,
#     )

#     np.testing.assert_allclose(
#         np.asarray(vio),
#         np.asarray(calc_ineq_violation(data.net, b["pder"], b["qder"], b["pder_avai"], b["v_full"], Y_0row=b["Y_0row_batch"])),
#         rtol=0,
#         atol=0,
#     )
#     assert np.all(np.asarray(vio_sq) >= 0.0)


# ════════════════════════════════════════════════════════════════════
#  Test 5: Re-solved PF voltage gives same small violations
# ════════════════════════════════════════════════════════════════════

def test_violation_with_resolved_voltage(feasible_bundle):
    """Re-solve PF from target DER setpoints, verify violations stay small.
    
    This catches cases where the stored voltage might diverge from
    what our PF solver computes, which would make violations look
    artificially large/small.
    """
    b = feasible_bundle
    data = b["data"]

    s_load = jax.lax.complex(b["pd"], b["qd"])
    s_der = b["pder"] + 1j * b["qder"]
    s_injection = jax.vmap(data.compute_net_injection, in_axes=(0, 0))(s_load, s_der)

    from core.grids.power_flow import solve_power_flow_fixed_point as solve_power_flow

    # Re-solve PF sample-by-sample to avoid OOM on large cases
    v_resolved_list = []
    from core.topology.utils import y_0row_from_scaled

    Y_0row_list = []
    for i in range(b["n_samples"]):
        topo_i = data.net.topology_matrices(b["branch_status_full"][i])
        Y_scaled_i, Y_LL_inv_i, w_i, _ = topo_i
        v_i = solve_power_flow(s_injection[i], w_i, Y_LL_inv_i, w_i)
        v_resolved_list.append(v_i)
        Y_0row_list.append(
            y_0row_from_scaled(Y_scaled_i, data.net.n_net, data.net.n_total)
        )

    v_resolved = jnp.stack(v_resolved_list)
    Y_0row_resolved = jnp.stack(Y_0row_list)

    # Build v_full from resolved voltage
    v_slack = jnp.broadcast_to(data.v_slack_local, (b["n_samples"], data.n_slk))
    v_full_resolved = jnp.concatenate([v_resolved, v_slack], axis=-1)

    # Check resolved vs stored voltage agreement
    v_net_stored = jax.lax.complex(b["vr"], b["vi"])
    v_diff = jnp.max(jnp.abs(v_resolved - v_net_stored), axis=1)
    print(f"\n[{b['case_name']}] Voltage diff (resolved vs stored):")
    print(f"  mean = {float(v_diff.mean()):.6e}")
    print(f"  max  = {float(v_diff.max()):.6e}")

    # Compute violations with resolved voltage
    violations_resolved = calc_ineq_violation(
        data.net,
        b["pder"],
        b["qder"],
        b["pder_avai"],
        v_full_resolved,
        Y_0row=Y_0row_resolved,
    )

    vio_np = np.asarray(violations_resolved)
    print(f"  Violations with resolved V: mean={vio_np.mean():.6e}, max={vio_np.max():.6e}")

    assert vio_np.max() < 1e-4, (
        f"[{b['case_name']}] Max violation with resolved voltage {vio_np.max():.6e} exceeds 1e-4"
    )


# ════════════════════════════════════════════════════════════════════
#  Test 6: Voltage magnitude bounds
# ════════════════════════════════════════════════════════════════════

def test_voltage_magnitude_bounds(feasible_bundle):
    """Direct check that |v| is within [vmin, vmax] for non-slack nodes."""
    b = feasible_bundle
    data = b["data"].net

    v_net = jax.lax.complex(b["vr"], b["vi"])
    vmag = jnp.abs(v_net)  # (B, n_net)

    vmin_vio = jnp.maximum(data.vmin[None, :] - vmag, 0.0)
    vmax_vio = jnp.maximum(vmag - data.vmax[None, :], 0.0)

    print(f"\n[{b['case_name']}] Voltage magnitude bounds:")
    print(f"  |v| range: [{float(vmag.min()):.6f}, {float(vmag.max()):.6f}]")
    print(f"  vmin violation max: {float(vmin_vio.max()):.6e}")
    print(f"  vmax violation max: {float(vmax_vio.max()):.6e}")

    assert float(vmin_vio.max()) < 1e-6, (
        f"[{b['case_name']}] vmin violation {float(vmin_vio.max()):.6e}"
    )
    assert float(vmax_vio.max()) < 1e-6, (
        f"[{b['case_name']}] vmax violation {float(vmax_vio.max()):.6e}"
    )


# ════════════════════════════════════════════════════════════════════
#  Test 7: DER operational constraints
# ════════════════════════════════════════════════════════════════════

def test_der_constraints(feasible_bundle):
    """Check DER box constraints (p, q bounds) and disk constraint (|s| <= smax)."""
    b = feasible_bundle
    data = b["data"].net

    p = b["pder"]
    q = b["qder"]

    # Box constraints
    p_lo_vio = jnp.max(jnp.maximum(data.pdermin - p, 0.0))
    p_hi_vio = jnp.max(jnp.maximum(p - data.pdermax, 0.0))
    q_lo_vio = jnp.max(jnp.maximum(data.qdermin - q, 0.0))
    q_hi_vio = jnp.max(jnp.maximum(q - data.qdermax, 0.0))

    # Apparent power disk
    s_mag_sq = p**2 + q**2
    s_vio = jnp.max(jnp.maximum(s_mag_sq - data.sdermax**2, 0.0))

    # Unit availability
    unit_p = jax.vmap(_group_phase_sum, in_axes=(0, None, None))(
        p, data.der_unit_to_phase, data.der_unit_to_phase_mask
    )
    avail_vio = jnp.max(jnp.maximum(unit_p - b["pder_avai"], 0.0))

    print(f"\n[{b['case_name']}] DER constraints:")
    print(f"  pder_lo_vio: {float(p_lo_vio):.6e}")
    print(f"  pder_hi_vio: {float(p_hi_vio):.6e}")
    print(f"  qder_lo_vio: {float(q_lo_vio):.6e}")
    print(f"  qder_hi_vio: {float(q_hi_vio):.6e}")
    print(f"  sder_disk:   {float(s_vio):.6e}")
    print(f"  unit_avail:  {float(avail_vio):.6e}")

    tol = 1e-4
    assert float(p_lo_vio) < tol
    assert float(p_hi_vio) < tol
    assert float(q_lo_vio) < tol
    assert float(q_hi_vio) < tol
    assert float(s_vio) < tol
    assert float(avail_vio) < tol


# ════════════════════════════════════════════════════════════════════
#  Test 8: Substation apparent power constraint
# ════════════════════════════════════════════════════════════════════

def test_substation_power_constraint(feasible_bundle):
    """Check |s_k| <= sub_smax_phase[k] for each slack phase."""
    b = feasible_bundle
    data = b["data"].net

    ssub = jax.vmap(
        lambda v, Y: jnp.abs(calc_substation_power_per_phase(data, v, Y)),
        in_axes=(0, 0),
    )(b["v_full"], b["Y_0row_batch"])

    ssub_vio = jnp.maximum(ssub - data.sub_smax_phase, 0.0)

    print(f"\n[{b['case_name']}] Substation power:")
    print(f"  |s_k| range: [{float(ssub.min()):.6f}, {float(ssub.max()):.6f}]")
    print(f"  sub_smax_phase: {np.asarray(data.sub_smax_phase)}")
    print(f"  violation max: {float(ssub_vio.max()):.6e}")

    assert float(ssub_vio.max()) < 1e-6, (
        f"[{b['case_name']}] Substation power violation {float(ssub_vio.max()):.6e}"
    )
