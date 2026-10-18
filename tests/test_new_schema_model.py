import numpy as np
import pytest
import jax
import jax.numpy as jnp

from conftest import (
    expand_compact_status,
    load_case_config,
    load_case_matrices,
    load_h5_sample,
    local_voltage_vector,
    parse_json_tree,
    power_flow_residual,
    stamp_switch_topology,
)


CASES = ["IEEE13Nodeckt"]


def _load_data(case_name: str):
    from pathlib import Path
    from core.grids.data import OPFData

    case_root = Path(__file__).resolve().parents[1] / "datagen_topo" / "datasets" / case_name
    return OPFData(
        str(case_root / f"{case_name}_training.h5"),
        str(case_root / f"{case_name}_test_feasible.h5"),
        str(case_root / "network_config.json"),
    )


def _reference_cost(data, psub, pder, pder_avai, obj_weights):
    psub = np.asarray(psub, dtype=np.float64).reshape(-1)
    pder = np.asarray(pder, dtype=np.float64)
    pder_avai = np.asarray(pder_avai, dtype=np.float64)
    obj_weights = np.asarray(obj_weights, dtype=np.float64)

    sub_cost = np.asarray(data.sub_cost, dtype=np.float64)
    if sub_cost.size >= 2:
        sub_quad = float(sub_cost[0])
        sub_lin = float(sub_cost[1])
    elif sub_cost.size == 1:
        sub_lin, sub_quad = float(sub_cost[0]), 0.0
    else:
        sub_lin, sub_quad = 0.0, 0.0

    der_cost = np.asarray(data.der_cost, dtype=np.float64)
    if der_cost.shape[1] >= 2:
        der_quad = der_cost[:, 0]
        der_lin = der_cost[:, 1]
    else:
        der_lin = der_cost[:, 0] if der_cost.shape[1] == 1 else np.zeros(der_cost.shape[0], dtype=np.float64)
        der_quad = np.zeros_like(der_lin)

    obj_cost = sub_lin * psub + sub_quad * psub**2 + np.sum(der_lin * pder + der_quad * pder**2, axis=1)
    unit_p = []
    for phase_ids, mask in zip(np.asarray(data.der_unit_to_phase), np.asarray(data.der_unit_to_phase_mask)):
        unit_p.append(np.sum(pder[:, phase_ids] * mask, axis=1))
    unit_p = np.stack(unit_p, axis=1)
    obj_curtailment = np.sum(pder_avai - unit_p, axis=1)
    return obj_weights[:, 0] * obj_cost + obj_weights[:, 1] * obj_curtailment


def _reference_violation(cfg, data, sample):
    n_net = int(cfg["n_net"])
    n_slk = int(cfg["n_slk"])
    n_total = int(cfg["n_total"])

    pder = np.asarray(sample["primal"]["pder"], dtype=np.float64)
    qder = np.asarray(sample["primal"]["qder"], dtype=np.float64)
    pd = np.asarray(sample["input"]["pd"], dtype=np.float64)
    qd = np.asarray(sample["input"]["qd"], dtype=np.float64)
    pder_avai = np.asarray(sample["input"]["pder_avai"], dtype=np.float64)

    v_local = local_voltage_vector(cfg, sample["primal"]["vr"], sample["primal"]["vi"])
    v_non_slack = v_local[:n_net]

    vmin = np.asarray(cfg["vmin"], dtype=np.float64)[:n_net]
    vmax = np.asarray(cfg["vmax"], dtype=np.float64)[:n_net]
    vmag = np.abs(v_non_slack)
    v_vio = np.max(np.maximum(vmin - vmag, 0.0) + np.maximum(vmag - vmax, 0.0))

    branch_from_idx = [np.asarray(row, dtype=np.int64) - 1 for row in cfg["branch_from_idx"]]
    branch_to_idx = [np.asarray(row, dtype=np.int64) - 1 for row in cfg["branch_to_idx"]]
    branch_valid = [np.asarray(row, dtype=bool) for row in cfg["branch_valid"]]
    branch_y_series = [np.asarray(parse_json_tree(row), dtype=np.complex128) for row in cfg["branch_y_series"]]
    branch_imax = np.asarray(cfg["branch_imax"], dtype=np.float64)
    v_scale_full = np.asarray(cfg["v_scale_full"], dtype=np.float64)

    def system_parts(idx: int):
        if idx < n_net:
            return v_scale_full[idx] * v_non_slack[idx].real, v_scale_full[idx] * v_non_slack[idx].imag
        slack_idx = idx - n_net
        return v_scale_full[idx] * v_local[n_net + slack_idx].real, v_scale_full[idx] * v_local[n_net + slack_idx].imag

    angle_vio = 0.0
    branch_current_vio = 0.0
    for line_idx, from_idx in enumerate(branch_from_idx):
        to_idx = branch_to_idx[line_idx]
        valid = branch_valid[line_idx]
        y_series = branch_y_series[line_idx]
        for r in range(len(from_idx)):
            if not valid[r]:
                continue
            i_f = int(from_idx[r])
            i_t = int(to_idx[r])
            vr_f = v_local[i_f].real
            vi_f = v_local[i_f].imag
            vr_t = v_local[i_t].real
            vi_t = v_local[i_t].imag
            angle_dot = vr_f * vr_t + vi_f * vi_t
            angle_cross = vr_f * vi_t - vi_f * vr_t
            angle_vio = max(
                angle_vio,
                angle_cross - cfg["dvamax"] * angle_dot,
                -angle_cross - cfg["dvamax"] * angle_dot,
                0.0,
            )

            ir = 0.0
            ii = 0.0
            for c in range(len(from_idx)):
                if not valid[c]:
                    continue
                c_f = int(from_idx[c])
                c_t = int(to_idx[c])
                dv_re = system_parts(c_f)[0] - system_parts(c_t)[0]
                dv_im = system_parts(c_f)[1] - system_parts(c_t)[1]
                y = y_series[r, c]
                ir += np.real(y) * dv_re - np.imag(y) * dv_im
                ii += np.imag(y) * dv_re + np.real(y) * dv_im
            branch_current_vio = max(branch_current_vio, np.hypot(ir, ii) - branch_imax[line_idx], 0.0)

    v_slack_local = v_local[n_net:]
    y_scaled = stamp_switch_topology(cfg, sample["input"]["branch_status"])
    Y_LL = y_scaled[:n_net, :n_net]
    y_ns = y_scaled[:n_net, n_net:]
    s_net = np.zeros(n_net, dtype=np.complex128)
    der_node = np.asarray(cfg["der_node"], dtype=np.int64) - 1
    s_der_phase = pder + 1j * qder
    np.add.at(s_net, der_node, s_der_phase)
    s_net = s_net - (pd + 1j * qd)
    sub_smax_phase = np.asarray(cfg["sub_smax_phase"], dtype=np.float64)
    sub_vio = 0.0
    for k in range(n_slk):
        row = y_scaled[n_net + k, :]
        s_k = v_slack_local[k] * np.conjugate(row @ v_local)
        sub_vio = max(sub_vio, abs(s_k) - sub_smax_phase[k], 0.0)

    unit_vio = 0.0
    for g, phase_ids in enumerate(cfg["der_unit_to_phase"]):
        idx = np.asarray(phase_ids, dtype=np.int64) - 1
        unit_vio = max(unit_vio, np.sum(pder[idx]) - pder_avai[g], 0.0)

    pmin = np.zeros_like(pder)
    pmax = np.asarray(cfg["der_pmax"], dtype=np.float64)
    qmin = np.asarray(cfg["der_qmin"], dtype=np.float64)
    qmax = np.asarray(cfg["der_qmax"], dtype=np.float64)
    smax = np.asarray(cfg["der_smax"], dtype=np.float64)
    p_vio = np.max(np.maximum(pmin - pder, 0.0) + np.maximum(pder - pmax, 0.0))
    q_vio = np.max(np.maximum(qmin - qder, 0.0) + np.maximum(qder - qmax, 0.0))
    s_vio = np.max(np.maximum(pder**2 + qder**2 - smax**2, 0.0))

    return max(v_vio, angle_vio, branch_current_vio, sub_vio, unit_vio, p_vio, q_vio, s_vio)


@pytest.mark.parametrize("case_name", CASES)
def test_calc_cost_matches_two_weight_julia_objective(case_name):
    from core.grids.metrics import calc_cost

    cfg = load_case_config(case_name)
    data = _load_data(case_name)
    sample = load_h5_sample(case_name, split="test_feasible", index=0)

    psub = np.asarray(sample["primal"]["psub"], dtype=np.float64)[None]
    pder = np.asarray(sample["primal"]["pder"], dtype=np.float64)[None, :]
    pder_avai = np.asarray(sample["input"]["pder_avai"], dtype=np.float64)[None, :]
    obj_weights = np.asarray(sample["input"]["obj_weights"], dtype=np.float64)[None, :]

    expected = _reference_cost(data, psub, pder, pder_avai, obj_weights)
    actual = np.asarray(calc_cost(data, psub, pder, pder_avai, obj_weights))

    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert actual.shape == (1,)


@pytest.mark.parametrize("case_name", CASES)
def test_calc_violation_matches_reference_equations(case_name):
    from core.grids.metrics import calc_ineq_violation

    cfg = load_case_config(case_name)
    data = _load_data(case_name)
    sample = load_h5_sample(case_name, split="test_feasible", index=0)
    topo_branch_status = np.asarray(sample["input"]["branch_status"], dtype=np.int32)[None, :]

    v_local = local_voltage_vector(cfg, sample["primal"]["vr"], sample["primal"]["vi"])
    v_full = v_local[None, :]
    pder = np.asarray(sample["primal"]["pder"], dtype=np.float64)[None, :]
    qder = np.asarray(sample["primal"]["qder"], dtype=np.float64)[None, :]
    pder_avai = np.asarray(sample["input"]["pder_avai"], dtype=np.float64)[None, :]

    actual = np.asarray(
        calc_ineq_violation(
            data,
            pder,
            qder,
            pder_avai,
            v_full,
            branch_status=data.expand_branch_status(topo_branch_status),
        )
    )
    expected = np.asarray([_reference_violation(cfg, data, sample)])

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert actual.shape == (1,)
    assert actual[0] <= 1e-5


@pytest.mark.parametrize("case_name", ["IEEE13Nodeckt"])
def test_linearization_has_first_order_accuracy(case_name):
    from core.grids.linear_model import create_models
    from core.topology.batch import build_topology_batch_jit

    data = _load_data(case_name)
    cfg = load_case_config(case_name)
    sample = load_h5_sample(case_name, split="test_feasible", index=0)
    compact = np.asarray(sample["input"]["branch_status"], dtype=np.int32)
    topo_batch = build_topology_batch_jit(compact[None, :], data)
    from core.topology.utils import make_topology_data

    topo = make_topology_data(
        Y_0row=topo_batch.Y_0row[0],
        Y_LL_inv=topo_batch.Y_LL_inv[0],
        w=topo_batch.w[0],
        v_init=topo_batch.w[0],
        branch_status=topo_batch.branch_status[0],
    )

    s_load = jnp.asarray(sample["input"]["pd"], dtype=jnp.float64) + 1j * jnp.asarray(
        sample["input"]["qd"], dtype=jnp.float64
    )
    s_der0 = jnp.asarray(sample["primal"]["pder"], dtype=jnp.float64) + 1j * jnp.asarray(
        sample["primal"]["qder"], dtype=jnp.float64
    )
    pder_avai = jnp.asarray(sample["input"]["pder_avai"], dtype=jnp.float64)

    def scalar_fn(z):
        n = len(cfg["der_node"])
        s_der = jax.lax.complex(z[:n], z[n:])
        models = create_models(data.net, s_load, s_der, pder_avai, topo.v_init, topo)
        return jnp.sum(models["linear_model"]["f0"])

    z0 = jnp.concatenate([jnp.real(s_der0), jnp.imag(s_der0)])
    direction = jax.random.normal(jax.random.PRNGKey(7), z0.shape, dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)

    grad = jax.jacfwd(scalar_fn)(z0)
    fd_eps = 1e-5
    fd = (scalar_fn(z0 + fd_eps * direction) - scalar_fn(z0 - fd_eps * direction)) / (2.0 * fd_eps)
    directional = jnp.vdot(grad, direction)

    np.testing.assert_allclose(np.asarray(directional), np.asarray(fd), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("case_name", ["IEEE13Nodeckt"])
def test_linear_model_matches_frozen_zbus_reference(case_name):
    import core.grids.linear_model as linear_model
    from core.topology.batch import build_topology_batch_jit

    data = _load_data(case_name)
    sample = load_h5_sample(case_name, split="test_feasible", index=0)
    compact = np.asarray(sample["input"]["branch_status"], dtype=np.int32)
    topo_batch = build_topology_batch_jit(compact[None, :], data)
    from core.topology.utils import make_topology_data

    topo = make_topology_data(
        Y_0row=topo_batch.Y_0row[0],
        Y_LL_inv=topo_batch.Y_LL_inv[0],
        w=topo_batch.w[0],
        v_init=topo_batch.w[0],
        branch_status=topo_batch.branch_status[0],
    )

    s_load = jnp.asarray(sample["input"]["pd"], dtype=jnp.float64) + 1j * jnp.asarray(
        sample["input"]["qd"], dtype=jnp.float64
    )
    s_der0 = jnp.asarray(sample["primal"]["pder"], dtype=jnp.float64) + 1j * jnp.asarray(
        sample["primal"]["qder"], dtype=jnp.float64
    )
    x0 = jnp.concatenate([s_der0.real, s_der0.imag])

    lm = linear_model.create_linear_model(
        data.net,
        s_load,
        s_der0,
        topo.v_init,
        topo,
    )
    u_hat = lm["u_hat"]

    def frozen_values(x):
        p = x[: data.n_der_phase]
        q = x[data.n_der_phase :]
        s_der = jax.lax.complex(p, q)
        s_net = data.net.compute_net_injection(s_load, s_der)
        u = topo.w + topo.Y_LL_inv @ (jnp.conj(s_net) / jnp.conj(u_hat))
        from core.grids.metrics import constraint_values_from_voltage

        return constraint_values_from_voltage(
            data.net,
            x,
            linear_model._full_local_voltage(data.net, u),
            topo.Y_0row,
        )

    from core.grids.metrics import constraint_values_from_voltage

    f0_ref = constraint_values_from_voltage(
        data.net,
        x0,
        linear_model._full_local_voltage(data.net, u_hat),
        topo.Y_0row,
    )
    A_ref = jax.jacfwd(frozen_values)(x0)

    np.testing.assert_allclose(np.asarray(lm["f0"]), np.asarray(f0_ref), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(lm["A"] @ lm["x0"] + lm["c"]),
        np.asarray(lm["f0"]),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(np.asarray(lm["A"]), np.asarray(A_ref), atol=1e-8, rtol=1e-8)
