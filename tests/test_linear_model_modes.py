import numpy as np
import jax
import jax.numpy as jnp

from core.topology.utils import make_topology_data


jax.config.update("jax_enable_x64", True)


def _load_ieee13_point():
    from pathlib import Path
    from core.grids.data import OPFData
    from core.topology.batch import precompute_topology

    case_root = Path(__file__).resolve().parents[1] / "datagen_topo" / "datasets" / "IEEE13Nodeckt"
    data = OPFData(
        str(case_root / "IEEE13Nodeckt_training.h5"),
        str(case_root / "IEEE13Nodeckt_test_feasible.h5"),
        str(case_root / "network_config.json"),
    )
    test_ds = data.make_test_dataset(samples=1)
    inputs_batch, targets_batch = next(iter(test_ds.batch(1)))
    inputs_np = np.asarray(inputs_batch)
    topo_batch = precompute_topology(data, inputs_np)
    topo = make_topology_data(
        Y_0row=topo_batch.Y_0row[0],
        Y_LL_inv=topo_batch.Y_LL_inv[0],
        w=topo_batch.w[0],
        v_init=topo_batch.v_init[0],
        branch_status=topo_batch.branch_status[0],
    )
    targets_np = np.asarray(targets_batch)
    s_load = jnp.asarray(inputs_np[0, data.slices["input"]["pd"]]) + 1j * jnp.asarray(
        inputs_np[0, data.slices["input"]["qd"]]
    )
    s_der = jnp.asarray(targets_np[0, data.slices["target"]["pder"]]) + 1j * jnp.asarray(
        targets_np[0, data.slices["target"]["qder"]]
    )
    pder_avai = jnp.asarray(inputs_np[0, data.slices["input"]["pder_avai"]])
    return data, s_load, s_der, pder_avai, topo


def test_frozen_zbus_linear_model_builds_reduced_system():
    from core.grids.linear_model import create_models

    data, s_load, s_der, pder_avai, topo = _load_ieee13_point()
    models = create_models(
        data.net,
        s_load,
        s_der,
        pder_avai,
        topo.v_init,
        topo,
    )

    a_reduced = models["reduced_system"]["A_reduced"]
    assert a_reduced.shape[1] == 2 * data.n_der_phase
    assert np.isfinite(np.asarray(a_reduced)).all()


def test_create_models_fixed_voltage_matches_pf():
    from core.grids.linear_model import create_models, create_models_batch

    data, s_load, s_der, pder_avai, topo = _load_ieee13_point()
    pf_models = create_models(
        data.net, s_load, s_der, pder_avai, topo.v_init, topo
    )
    fixed_models = create_models(
        data.net,
        s_load,
        s_der,
        pder_avai,
        topo.v_init,
        topo,
        u_hat=pf_models["v_hat"],
    )
    np.testing.assert_allclose(
        fixed_models["linear_model"]["A"],
        pf_models["linear_model"]["A"],
        atol=1e-9,
        rtol=0.0,
    )

    s_load_b = jnp.stack([s_load, s_load])
    s_der_b = jnp.stack([s_der, s_der])
    pder_b = jnp.stack([pder_avai, pder_avai])
    batch_fixed = create_models_batch(
        data.net,
        s_load_b,
        s_der_b,
        pder_b,
        jnp.stack([topo.v_init, topo.v_init]),
        topo,
        u_hat_batch=jnp.stack([pf_models["v_hat"], pf_models["v_hat"]]),
    )
    np.testing.assert_allclose(
        batch_fixed["linear_model"]["A"][0],
        fixed_models["linear_model"]["A"],
        atol=1e-9,
        rtol=0.0,
    )
