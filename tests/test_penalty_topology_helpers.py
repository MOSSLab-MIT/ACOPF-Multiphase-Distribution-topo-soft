import numpy as np
import jax.numpy as jnp

from core.grids.metrics import substation_power_from_voltage, with_slack_voltage


class _DummyData:
    def __init__(self) -> None:
        self.n_net = 2
        self.n_slk = 1
        self.n_total = 3
        self.v_slack_local = jnp.asarray([2.0 + 0.0j], dtype=jnp.complex128)

        # Slack row picks only bus-0 current contribution.
        self.Y_scaled_base = jnp.asarray(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ],
            dtype=jnp.complex128,
        )


def test_with_slack_voltage_orders_non_slack_then_slack():
    data = _DummyData()
    v_net = jnp.asarray([[3.0 + 4.0j, 1.0 - 2.0j]], dtype=jnp.complex128)

    v_full = with_slack_voltage(data, v_net)

    expected = np.asarray([[3.0 + 4.0j, 1.0 - 2.0j, 2.0 + 0.0j]], dtype=np.complex128)
    np.testing.assert_allclose(np.asarray(v_full), expected, atol=1e-12, rtol=1e-12)


def test_substation_power_from_voltage_uses_base_slack_rows_when_no_y_0row():
    data = _DummyData()
    v_full = jnp.asarray(
        [
            [3.0 + 4.0j, 0.0 + 0.0j, 2.0 + 0.0j],
            [1.0 - 1.0j, 0.0 + 0.0j, 2.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )

    psub = substation_power_from_voltage(data, v_full).real

    # s_sub = v_slack * conj(i_slack), i_slack = Y_0row_row @ v_full = v_bus0.
    # Real power is 2 * Re(v_bus0).
    expected = np.asarray([6.0, 2.0], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(psub), expected, atol=1e-12, rtol=1e-12)


def test_substation_power_from_voltage_respects_explicit_y_0row():
    data = _DummyData()
    v_full = jnp.asarray(
        [
            [3.0 + 0.0j, 0.0 + 0.0j, 2.0 + 0.0j],
            [3.0 + 0.0j, 0.0 + 0.0j, 2.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    y_open = data.Y_scaled_base[data.n_net : data.n_total, :]
    y_closed = y_open.at[0, 0].set(2.0 + 0.0j)
    Y_0row = jnp.stack([y_open, y_closed], axis=0)

    psub = substation_power_from_voltage(data, v_full, Y_0row=Y_0row).real

    # coefficient 1 => 2 * 3 = 6; coefficient 2 => 2 * 6 = 12.
    expected = np.asarray([6.0, 12.0], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(psub), expected, atol=1e-12, rtol=1e-12)
