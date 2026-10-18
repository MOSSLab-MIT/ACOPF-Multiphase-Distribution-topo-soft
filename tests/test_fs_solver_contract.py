import jax
import jax.numpy as jnp

from core.feasibility.fs import feasibility_seeking_batch


jax.config.update("jax_enable_x64", True)


class _TinyData:
    n_der = 1
    n_der_phase = 1
    n_der_unit = 1
    pdermin = jnp.array([-10.0], dtype=jnp.float64)
    pdermax = jnp.array([10.0], dtype=jnp.float64)
    qdermin = jnp.array([-10.0], dtype=jnp.float64)
    qdermax = jnp.array([10.0], dtype=jnp.float64)
    sdermax = jnp.array([10.0], dtype=jnp.float64)
    der_phase_to_unit = jnp.array([0], dtype=jnp.int32)
    der_unit_to_phase = jnp.array([[0]], dtype=jnp.int32)
    der_unit_to_phase_mask = jnp.array([[True]])

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def _one_sided_system(*, batched_bounds):
    bounds_shape = (1, 2) if batched_bounds else (2,)
    avail_shape = (1, 1) if batched_bounds else (1,)
    lx = jnp.full(bounds_shape, -10.0, dtype=jnp.float64)
    ux = jnp.full(bounds_shape, 10.0, dtype=jnp.float64)
    return {
        "A_reduced": jnp.array([[[1.0, 0.0]]], dtype=jnp.float64),
        "l_reduced": jnp.array([[-jnp.inf]], dtype=jnp.float64),
        "u_reduced": jnp.array([[0.0]], dtype=jnp.float64),
        "lx": lx,
        "ux": ux,
        "pder_avai": jnp.full(avail_shape, 10.0, dtype=jnp.float64),
    }


def test_training_solver_respects_loose_tolerance_freeze():
    data = _TinyData()
    reduced = _one_sided_system(batched_bounds=True)
    x_init = jnp.array([[1.0, 0.0]], dtype=jnp.float64)

    one_iter = feasibility_seeking_batch(
        data, reduced, x_init, tol=2.0, maxiter=1, rho=0.0
    )
    many_iter = feasibility_seeking_batch(
        data, reduced, x_init, tol=2.0, maxiter=50, rho=0.0
    )

    assert jnp.allclose(many_iter, one_iter, rtol=0.0, atol=0.0)
