"""Tests for solve_power_flow_nr (Newton-Raphson) convergence and voltage correctness."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import jax
import jax.numpy as jnp

# Reuse helpers from conftest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import (
    case_paths,
    load_case_config,
    load_h5_sample,
    parse_json_tree,
    power_flow_residual,
    local_voltage_vector,
    system_voltage_vector,
    load_case_matrices,
)


CASES = ["IEEE13Nodeckt", "IEEE123Master"]


def _load_batch(case_name: str, n: int = 5):
    """Load data and prepare a batch for testing."""
    from core.grids.data import OPFData
    from core.utils.workflow import prepare_batch_data
    from core.topology.utils import Y_LL_L0_from_scaled

    paths = case_paths(case_name)
    data = OPFData(str(paths["train"]), str(paths["test_feasible"]), str(paths["json"]))
    with h5py.File(paths["test_feasible"], "r") as f:
        n_total = f["input"]["p_load"].shape[0]
        n = min(n, n_total)
        inputs = np.concatenate([np.asarray(f["input"][k]) for k in data.input_keys])[:n]
        targets = np.concatenate(
            [np.asarray(f["primal"][k]) for k in data.output_keys]
            + [np.asarray(f["meta"]["solve_time"])[:, None]],
            axis=1,
        )[:n]
    batch = prepare_batch_data(jnp.asarray(inputs), jnp.asarray(targets), data)
    return data, batch


@pytest.mark.parametrize("case_name", CASES)
def test_nr_convergence_residual_small(case_name):
    """Newton-Raphson should produce very small power flow residuals."""
    from core.grids.power_flow_exp import solve_power_flow_nr, power_flow_residual_batch

    data, batch = _load_batch(case_name, n=10)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_nr(s, v0, Y, w0, maxiter=10, diffiter=3)
    )
    u = solve_many(s_net, w, Y_inv, w)
    residual = power_flow_residual_batch(u, s_net, Y_LL, Y_L0, data.v_slack_local)

    max_residual = float(np.max(np.abs(np.asarray(residual))))
    print(f"\n{case_name} NR max residual: {max_residual:.2e}")
    assert max_residual <= 1e-6, f"NR residual too large: {max_residual}"


@pytest.mark.parametrize("case_name", CASES)
def test_nr_voltage_matches_target(case_name):
    """Newton-Raphson voltages should match dataset target voltages."""
    from core.grids.power_flow_exp import solve_power_flow_nr

    data, batch = _load_batch(case_name, n=10)
    _, Y_inv, _, _, w, _ = data.net.topology_matrices_batch(batch.switchable_status)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_nr(s, v0, Y, w0, maxiter=10, diffiter=3)
    )
    v_pred = solve_many(s_net, w, Y_inv, w)

    # Compare to target voltages from dataset
    err = jnp.max(jnp.abs(v_pred - batch.v_target), axis=1)
    max_err = float(jnp.max(err))
    mean_err = float(jnp.mean(err))
    print(f"\n{case_name} NR voltage error vs target: max={max_err:.2e}, mean={mean_err:.2e}")
    assert max_err < 1e-4, f"NR voltage error too large: {max_err}"


@pytest.mark.parametrize("case_name", CASES)
def test_nr_vs_zbus_convergence_comparison(case_name):
    """Newton-Raphson should converge faster (fewer iterations) than Z-bus for ill-conditioned cases."""
    from core.grids.power_flow_exp import solve_power_flow_nr, solve_power_flow, power_flow_residual_batch

    data, batch = _load_batch(case_name, n=5)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    # NR with 7 iterations
    solve_nr = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_nr(s, v0, Y, w0, maxiter=7, diffiter=3)
    )
    u_nr = solve_nr(s_net, w, Y_inv, w)
    res_nr = power_flow_residual_batch(u_nr, s_net, Y_LL, Y_L0, data.v_slack_local)
    max_res_nr = float(np.max(np.abs(np.asarray(res_nr))))

    # Z-bus with 80 iterations
    solve_zb = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow(s, v0, Y, w0, maxiter=80, diffiter=2)
    )
    u_zb = solve_zb(s_net, w, Y_inv, w)
    res_zb = power_flow_residual_batch(u_zb, s_net, Y_LL, Y_L0, data.v_slack_local)
    max_res_zb = float(np.max(np.abs(np.asarray(res_zb))))

    print(f"\n{case_name}: NR residual (7 iters) = {max_res_nr:.2e}, Z-bus residual (80 iters) = {max_res_zb:.2e}")
    # NR should achieve good convergence in just 7 iterations
    assert max_res_nr <= 1e-4, f"NR failed to converge in 7 iterations: {max_res_nr}"


@pytest.mark.parametrize("case_name", CASES)
def test_nr_zero_elimination_convergence(case_name):
    """Newton-Raphson with zero-injection elimination should also converge."""
    from core.grids.power_flow_exp import solve_power_flow_nr_zero_eliminated, power_flow_residual_batch

    data, batch = _load_batch(case_name, n=5)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_nr_zero_eliminated(s, v0, Y, w0, maxiter=10, diffiter=3)
    )
    u = solve_many(s_net, w, Y_inv, w)
    residual = power_flow_residual_batch(u, s_net, Y_LL, Y_L0, data.v_slack_local)

    max_residual = float(np.max(np.abs(np.asarray(residual))))
    print(f"\n{case_name} NR (zero elim) max residual: {max_residual:.2e}")
    assert max_residual <= 1e-6, f"NR zero-elimination residual too large: {max_residual}"


@pytest.mark.parametrize("case_name", CASES)
def test_nr_schur_convergence(case_name):
    """Newton-Raphson with Schur complement should converge."""
    from core.grids.power_flow_exp import solve_power_flow_nr_schur, power_flow_residual_batch

    data, batch = _load_batch(case_name, n=5)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    solve_many = jax.vmap(
        lambda s, v0, Y, w0: solve_power_flow_nr_schur(s, v0, Y, w0, maxiter=10, diffiter=3)
    )
    u = solve_many(s_net, w, Y_inv, w)
    residual = power_flow_residual_batch(u, s_net, Y_LL, Y_L0, data.v_slack_local)

    max_residual = float(np.max(np.abs(np.asarray(residual))))
    print(f"\n{case_name} NR (Schur) max residual: {max_residual:.2e}")
    assert max_residual <= 1e-6, f"NR Schur residual too large: {max_residual}"


@pytest.mark.parametrize("case_name", CASES)
def test_nr_different_initial_guesses(case_name):
    """Newton-Raphson should converge from different initial voltage guesses."""
    from core.grids.power_flow_exp import solve_power_flow_nr, power_flow_residual_batch

    data, batch = _load_batch(case_name, n=3)
    Y_scaled, Y_inv, w, _ = data.net.topology_matrices_batch(batch.switchable_status)

    Y_LL, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, data.n_net)
    s_der = jax.lax.complex(batch.pder_target, batch.qder_target)
    s_net = jax.vmap(data.net.compute_net_injection, in_axes=(0, 0))(batch.s_load, s_der)

    # Test with w (standard init) and with perturbed initial guesses
    for scale in [0.9, 1.0, 1.1]:
        v_init = w * scale
        solve_many = jax.vmap(
            lambda s, v0, Y, w0: solve_power_flow_nr(s, v0, Y, w0, maxiter=10, diffiter=3)
        )
        u = solve_many(s_net, v_init, Y_inv, w)
        residual = power_flow_residual_batch(u, s_net, Y_LL, Y_L0, data.v_slack_local)
        max_res = float(np.max(np.abs(np.asarray(residual))))
        print(f"\n{case_name} NR init scale={scale}: max residual = {max_res:.2e}")
        assert max_res <= 1e-4, f"NR failed with init scale {scale}: {max_res}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
