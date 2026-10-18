from types import SimpleNamespace

import jax
import jax.numpy as jnp

import core.feasibility.slfs as trainer
from core.feasibility.slfs_config import FeasibilityConfig


def _install_slfs_test_doubles(monkeypatch):
    def fake_create_models_batch(
        data,
        s_load,
        s_der,
        pder_avai,
        v_init,
        topo,
        u_hat_batch=None,
        solve_pf_fn=None,
        power_flow_solver="fixed_point",
    ):
        del data, s_load, s_der, pder_avai, u_hat_batch, solve_pf_fn, power_flow_solver, topo
        if v_init is None:
            v_init = topo.v_init
        return {"reduced_system": None, "v_hat": v_init, "linear_model": None}

    def fake_feasibility_seeking_batch(
        data,
        reduced_system,
        x_der,
        tol,
        maxiter,
        rho,
        adaptive_step=True,
    ):
        del data, reduced_system, tol, maxiter, rho, adaptive_step
        # Linear map with known Jacobian so gradient expectations are exact.
        return 2.0 * x_der

    monkeypatch.setattr(
        trainer,
        "create_models_batch",
        fake_create_models_batch,
    )
    monkeypatch.setattr(
        trainer,
        "feasibility_seeking_batch",
        fake_feasibility_seeking_batch,
    )


def _grad_wrt_initial_p(diff_steps: int, lin_steps: int = 2):
    p0 = jnp.asarray([[1.0, -0.5]], dtype=jnp.float64)

    def loss_fn(p_init):
        data = SimpleNamespace(n_der_phase=p_init.shape[1])
        topo = SimpleNamespace(v_init=jnp.zeros((p_init.shape[0], 1), dtype=jnp.complex128))
        s_load = jnp.zeros((p_init.shape[0], 1), dtype=jnp.complex128)
        pder_avai = jnp.ones_like(p_init)
        s_der_init = jax.lax.complex(p_init, jnp.zeros_like(p_init))
        cfg = FeasibilityConfig(
            fs_tol=1e-8,
            fs_maxiter=5,
            lin_steps=lin_steps,
            diff_steps=diff_steps,
            fs_rho=0.0,
        ).validate()

        s_der_final, _, _, _ = trainer.slfs_core(
            data,
            s_load,
            s_der_init,
            pder_avai,
            topo,
            cfg,
        )
        return jnp.sum(s_der_final.real)

    return jax.grad(loss_fn)(p0)


def test_slfs_non_diff_steps_stop_gradient(monkeypatch):
    _install_slfs_test_doubles(monkeypatch)
    grad = _grad_wrt_initial_p(diff_steps=1, lin_steps=2)
    assert jnp.allclose(grad, 0.0, atol=1e-12, rtol=0.0)


def test_slfs_all_steps_differentiable(monkeypatch):
    _install_slfs_test_doubles(monkeypatch)
    grad = _grad_wrt_initial_p(diff_steps=2, lin_steps=2)
    assert jnp.allclose(grad, 4.0, atol=1e-12, rtol=0.0)


def test_slfs_zero_diff_steps_stop_gradient(monkeypatch):
    _install_slfs_test_doubles(monkeypatch)
    grad = _grad_wrt_initial_p(diff_steps=0, lin_steps=2)
    assert jnp.allclose(grad, 0.0, atol=1e-12, rtol=0.0)
