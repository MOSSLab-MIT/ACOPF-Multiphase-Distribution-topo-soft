import numpy as np
import pytest

from conftest import load_case_config, load_h5_sample, local_voltage_vector, parse_json_tree


CASES = ["IEEE13Nodeckt", "IEEE123Master"]
LEGACY_IMAX_SENTINEL = 10.0


def _branch_terminal_currents_and_limits(cfg: dict, sample: dict):
    """Compute per-terminal branch current magnitudes in system pu and their limits."""
    v_local = local_voltage_vector(cfg, sample["primal"]["vr"], sample["primal"]["vi"])
    v_system = np.asarray(cfg["v_scale_full"], dtype=np.float64) * v_local

    branch_from_idx = [np.asarray(row, dtype=np.int64) - 1 for row in cfg["branch_from_idx"]]
    branch_to_idx = [np.asarray(row, dtype=np.int64) - 1 for row in cfg["branch_to_idx"]]
    branch_valid = [np.asarray(row, dtype=bool) for row in cfg["branch_valid"]]
    branch_y_series = [np.asarray(parse_json_tree(row), dtype=np.complex128) for row in cfg["branch_y_series"]]
    branch_imax = np.asarray(cfg["branch_imax"], dtype=np.float64)

    current_magnitudes = []
    current_limits = []

    for line_idx, from_idx in enumerate(branch_from_idx):
        to_idx = branch_to_idx[line_idx]
        valid = branch_valid[line_idx]
        y_series = branch_y_series[line_idx]
        imax = branch_imax[line_idx]

        for r in range(len(from_idx)):
            if not valid[r]:
                continue

            ir = 0.0
            ii = 0.0
            for c in range(len(from_idx)):
                if not valid[c]:
                    continue
                i_from = int(from_idx[c])
                i_to = int(to_idx[c])
                dv = v_system[i_from] - v_system[i_to]
                y = y_series[r, c]
                ir += np.real(y) * np.real(dv) - np.imag(y) * np.imag(dv)
                ii += np.imag(y) * np.real(dv) + np.real(y) * np.imag(dv)

            current_magnitudes.append(np.hypot(ir, ii))
            current_limits.append(imax)

    return np.asarray(current_magnitudes, dtype=np.float64), np.asarray(current_limits, dtype=np.float64)


@pytest.mark.parametrize("case_name", CASES)
def test_branch_imax_is_pu_and_no_legacy_sentinel(case_name):
    cfg = load_case_config(case_name)
    branch_imax = np.asarray(cfg["branch_imax"], dtype=np.float64)

    assert branch_imax.size > 0
    assert np.all(np.isfinite(branch_imax))
    assert np.all(branch_imax > 0.0)
    # These feeder datasets store branch limits in per-unit current.
    assert np.max(branch_imax) < 5.0
    assert not np.any(np.isclose(branch_imax, LEGACY_IMAX_SENTINEL))


@pytest.mark.parametrize("case_name", CASES)
def test_feasible_sample_branch_currents_respect_branch_imax(case_name):
    cfg = load_case_config(case_name)
    sample = load_h5_sample(case_name, split="test_feasible", index=0)

    currents, limits = _branch_terminal_currents_and_limits(cfg, sample)

    assert currents.size == limits.size
    assert currents.size > 0

    max_violation = float(np.max(currents - limits))
    assert max_violation <= 1e-5
