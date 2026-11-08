"""NN input z-score normalizer: exact z-scores, constant columns, no clipping."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from core.nn.input import CONST_STD_RTOL, constant_input_mask, input_scale  # noqa: E402
from core.nn.state import _make_input_normalizer  # noqa: E402


def _stats(train):
    train = jnp.asarray(train, dtype=jnp.float64)
    return jnp.mean(train, axis=0), jnp.std(train, axis=0)


def _train_matrix(rng):
    """Columns: varying pd-like, varying binary, constant 1.0, constant 0.0, constant 0.1."""
    n = 1000
    pd = 1.7e-5 + 8.8e-6 * rng.standard_normal(n)  # 240-bus scale per-unit load
    sw = (rng.random(n) < 0.3).astype(np.float64)
    return np.stack([pd, sw, np.ones(n), np.zeros(n), np.full(n, 0.1)], axis=1)


def test_varying_columns_are_exact_zscores_without_clip():
    rng = np.random.default_rng(0)
    train = _train_matrix(rng)
    mean, std = _stats(train)
    norm = _make_input_normalizer((mean, std))

    m, s = np.asarray(mean), np.asarray(std)
    x = np.array(train[:4])
    # Push the first two (varying) columns to +-10 sigma: old code clipped these to +-5.
    x[0, 0] = m[0] + 10.0 * s[0]
    x[1, 1] = m[1] - 10.0 * s[1]
    z = np.asarray(norm(jnp.asarray(x)))

    expected = (x[:, :2] - m[:2]) / s[:2]
    np.testing.assert_allclose(z[:, :2], expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(z[0, 0], 10.0, rtol=1e-12)
    np.testing.assert_allclose(z[1, 1], -10.0, rtol=1e-12)
    # Training data normalized with its own stats has mean 0, std 1 per varying column.
    zt = np.asarray(norm(jnp.asarray(train)))[:, :2]
    np.testing.assert_allclose(zt.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(zt.std(axis=0), 1.0, rtol=1e-12)


def test_constant_columns_map_to_zero_and_stay_finite_off_training_value():
    rng = np.random.default_rng(1)
    train = _train_matrix(rng)
    mean, std = _stats(train)
    np.testing.assert_array_equal(
        np.asarray(constant_input_mask(mean, std)), [False, False, True, True, True]
    )
    norm = _make_input_normalizer((mean, std))

    # Training values of constant columns -> 0. Exactly for 1.0 / 0.0 (switch bits; their
    # float64 sum is exact); for 0.1 the mean itself carries summation round-off, bounded by
    # log2(n) * eps * |x| for pairwise summation, so z = x - mean is that small, not 0.
    z_train = np.asarray(norm(jnp.asarray(train[:3])))
    np.testing.assert_array_equal(z_train[:, 2:4], 0.0)
    roundoff = np.log2(train.shape[0]) * np.finfo(np.float64).eps * 0.1
    assert np.all(np.abs(z_train[:, 4]) <= roundoff), z_train[:, 4]

    # An always-closed switch seen open (1 -> 0), and unseen values elsewhere: x - mean.
    x = np.array(train[:1])
    x[0, 2:] = [0.0, 1.0, 0.35]
    z = np.asarray(norm(jnp.asarray(x)))
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(z[0, 2:], x[0, 2:] - np.asarray(mean)[2:], rtol=0, atol=1e-15)


def test_raw_zero_std_from_checkpoint_is_handled():
    """Checkpoints store raw std (exact zeros possible); the normalizer must not divide by 0."""
    mean = jnp.array([1.0, 0.0, 2.0])
    std = jnp.array([0.0, 0.0, 0.5])
    norm = _make_input_normalizer((mean, std))
    z = np.asarray(norm(jnp.array([[0.0, 3.0, 3.0]])))
    np.testing.assert_array_equal(z, [[-1.0, 3.0, 2.0]])


def test_constant_threshold_separates_roundoff_from_smallest_real_std():
    # Round-off std of a constant non-dyadic column is classified constant.
    for value in (0.1, 1.7e-5, 144.3506):
        col = np.full(200_000, value)
        mean, std = _stats(col[:, None])
        assert bool(constant_input_mask(mean, std)[0]), (value, float(std[0]))
        np.testing.assert_array_equal(np.asarray(input_scale(mean, std)), [1.0])

    # Smallest real std on the 123/240/8500 datasets (240-bus qd) is kept as a real scale.
    mean = jnp.array([3.0e-6])
    std = jnp.array([1.55e-6])
    assert not bool(constant_input_mask(mean, std)[0])
    np.testing.assert_array_equal(np.asarray(input_scale(mean, std)), [1.55e-6])

    # Boundary: std exactly at the tolerance is constant, just above is not.
    tol = CONST_STD_RTOL * 1.0
    assert bool(constant_input_mask(jnp.array([0.5]), jnp.array([tol]))[0])
    assert not bool(constant_input_mask(jnp.array([0.5]), jnp.array([2.0 * tol]))[0])
    tol_big = CONST_STD_RTOL * 200.0
    assert bool(constant_input_mask(jnp.array([-200.0]), jnp.array([tol_big]))[0])
    assert not bool(constant_input_mask(jnp.array([-200.0]), jnp.array([2.0 * tol_big]))[0])


def test_normalizer_is_jittable_and_matches_eager():
    rng = np.random.default_rng(2)
    train = _train_matrix(rng)
    norm = _make_input_normalizer(_stats(train))
    x = jnp.asarray(train[:16])
    # XLA may fuse subtract/divide differently under jit: allow a few ulp, nothing more.
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(
        np.asarray(jax.jit(norm)(x)), np.asarray(norm(x)), rtol=4 * eps, atol=4 * eps
    )
