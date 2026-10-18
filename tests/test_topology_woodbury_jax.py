"""Correctness and speed tests for JAX Woodbury topology (``topology_woodbury_jax``).

Ground truth for the inverse is dense ``jnp.linalg.solve(Y_LL, I)`` on the same
stamped ``Y_LL``. NumPy ``StampTopologyBackend._topology_matrices_from_stamping`` cross-checks
stamping + inversion on the host.

Speed tests JIT both Woodbury and direct-inverse batched paths (warmup included).
On small feeders Woodbury can be slower; the test records timings and only
requires a speedup when ``n_net`` is large enough that rank is small vs. cubic cost.

python -m pytest tests/test_topology_woodbury_jax.py -v
use conda env: new_opt
"""
import functools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import CASE_ROOTS
from core.grids.data import (
    _complex_array,
    _idx_array_0,
    _pad_complex_matrices,
    _pad_vectors_0,
)
from core.topology.stamp import StampTopologyBackend
from core.topology.utils import matrices_first_to_y_0row, topology_data_from_matrices, Y_LL_L0_from_scaled, y_0row_from_scaled
from core.topology.woodbury import (
    WoodburyTopologyBackend,
    WoodburyStatic,
    _build_lowrank_padded,
    _build_one_switch_uv,
    _expand_compact,
    _stamp_Y_scaled,
    _woodbury_inverse,
    _woodbury_solve,
    woodbury_static_from_net,
    woodbury_topology_matrices,
    woodbury_topology_matrices_slim,
)

ROOT = Path(__file__).resolve().parents[1]

STRICT_MAX_ABS_TOL = 1e-8
INVERSE_RESIDUAL_TOL = 1e-6  # matches ``_INVERSE_RESIDUAL_TOL`` in topology_woodbury_jax
# When cond(Y_LL) exceeds this, Y_LL_inv is non-unique in floating point and NumPy/JAX
# solve backends can disagree while both satisfy ||Y Y^{-1} - I|| ~ 1e-9 (e.g. IEEE123 bus 610).
COND_SKIP_ELEMENTWISE_INV = 1e12
MAX_STATUS_PER_CASE = 12
LARGE_N_NET_THRESHOLD = 1000000  # reduce suite size for big feeders (123/240/906)

N_WARMUP = 1
N_REPEATS = 5
SPEED_BATCH_SIZES = (1, 8, 16)
# Woodbury should beat O(n^3) direct inv when n_net is modest and rank stays small.
MIN_N_NET_FOR_SPEEDUP_ASSERT = 40
MIN_BATCH_FOR_SPEEDUP_ASSERT = 8
MIN_WOODBURY_SPEEDUP = 1.1


def _available_cases() -> list[str]:
    out = []
    for name, root in CASE_ROOTS.items():
        if (root / "network_config.json").exists():
            out.append(name)
    return sorted(out)


def case_paths(case_name: str) -> dict:
    case_root = CASE_ROOTS[case_name]
    train_path = case_root / f"{case_name}_training.h5"
    test_path = case_root / f"{case_name}_test.h5"
    if not train_path.exists():
        train_path = test_path
    return {
        "root": case_root,
        "json": case_root / "network_config.json",
        "train": train_path,
        "test": test_path,
        "test_feasible": case_root / f"{case_name}_test_feasible.h5",
    }


def _load_net_numpy(case_name: str) -> SimpleNamespace:
    """Build a ``net`` object with numpy arrays only (mirrors ``OPFData`` switch fields)."""
    paths = case_paths(case_name)
    raw = json.loads(paths["json"].read_text())
    n_branch = len(raw.get("branch_status", []))

    base_branch_status = np.asarray(
        raw.get("branch_status", [True] * n_branch), dtype=np.int32
    )
    switch_idx = np.asarray(_idx_array_0(raw.get("switch_idx", [])), dtype=np.int32).reshape(-1)
    changing_switches = np.asarray(
        _idx_array_0(raw.get("changing_switches", [])), dtype=np.int32
    ).reshape(-1)
    switch_is_closed_base = np.asarray(
        raw.get("switch_is_closed_base", []), dtype=np.int32
    ).reshape(-1)

    switch_from_np, switch_valid_from_np = _pad_vectors_0(raw.get("switch_from_idx", []))
    switch_to_np, switch_valid_to_np = _pad_vectors_0(
        raw.get("switch_to_idx", []), width=switch_from_np.shape[1]
    )
    switch_valid_raw = raw.get("switch_valid", [])
    if not switch_valid_raw:
        switch_valid_np = switch_valid_from_np & switch_valid_to_np
    else:
        _, switch_valid_shape_np = _pad_vectors_0(raw.get("switch_from_idx", []))
        switch_valid_np = np.zeros_like(switch_valid_shape_np)
        for i, row in enumerate(switch_valid_raw):
            n = min(len(row), switch_valid_np.shape[1])
            switch_valid_np[i, :n] = np.asarray(row[:n], dtype=bool)
        switch_valid_np &= switch_valid_from_np & switch_valid_to_np

    max_switch_terms = int(switch_from_np.shape[1])
    switch_y_series = _pad_complex_matrices(raw.get("switch_y_series", []), max_switch_terms)
    switch_y_shunt = _pad_complex_matrices(raw.get("switch_y_shunt", []), max_switch_terms)

    v_scale_full = np.asarray(raw["v_scale_full"], dtype=np.float64).reshape(-1)
    v_sub_ref = _complex_array(raw["v_sub_ref"])
    v_scale_slack = np.asarray(raw["v_scale_slack"], dtype=np.float64).reshape(-1)
    v_slack_local = (v_sub_ref / v_scale_slack).reshape(-1)

    Y_scaled_base = _complex_array(raw["Y_scaled_base"])

    n_net = int(raw["n_net"])
    n_slk = int(raw["n_slk"])
    n_total = int(raw["n_total"])
    n_switch = len(raw.get("switch_idx", []))
    n_changing_switches = int(changing_switches.shape[0])

    return SimpleNamespace(
        n_net=n_net,
        n_slk=n_slk,
        n_total=n_total,
        n_branch=n_branch,
        n_switch=n_switch,
        n_changing_switches=n_changing_switches,
        max_switch_terms=max_switch_terms,
        base_branch_status=base_branch_status,
        changing_switches=changing_switches,
        switch_idx=switch_idx,
        switch_is_closed_base=switch_is_closed_base,
        switch_from_idx=switch_from_np,
        switch_to_idx=switch_to_np,
        switch_valid=switch_valid_np,
        switch_y_series=switch_y_series,
        switch_y_shunt=switch_y_shunt,
        v_scale_full=v_scale_full,
        v_slack_local=v_slack_local,
        Y_scaled_base=Y_scaled_base,
    )


def _load_data(case_name: str):
    return _load_net_numpy(case_name), case_paths(case_name)


def _n_net_for_case(case_name: str) -> int:
    return int(json.loads(case_paths(case_name)["json"].read_text())["n_net"])


CASES = _available_cases()
if not CASES:
    pytest.skip("No topology datasets with network_config.json found.", allow_module_level=True)

# NumPy loop stamping is cross-checked only on smaller feeders (runtime + host limits).
CASES_STAMP_NUMPY = [c for c in CASES if _n_net_for_case(c) < LARGE_N_NET_THRESHOLD]


@pytest.mark.parametrize("case_name", CASES)
def test_changing_switch_row_mapping(case_name):
    """Each compact slot maps to the switch catalog row for its branch line."""
    net, _ = _load_data(case_name)
    s = woodbury_static_from_net(net)
    assert int(s.changing_switch_row.shape[0]) == int(s.n_changing_switches)
    switch_idx = np.asarray(s.switch_idx)
    changing = np.asarray(s.changing_switches)
    rows = np.asarray(s.changing_switch_row)
    if s.n_changing_switches:
        counts = np.array([(switch_idx == branch).sum() for branch in changing], dtype=np.int32)
        np.testing.assert_array_equal(
            counts,
            np.ones_like(counts),
            err_msg="Every compact changing switch must appear exactly once in switch_idx.",
        )
        for j in range(int(s.n_changing_switches)):
            assert int(switch_idx[rows[j]]) == int(changing[j])


@pytest.mark.skipif("IEEE13Nodeckt" not in CASES, reason="IEEE13 dataset required")
def test_changing_switch_row_mapping_fails_when_switch_missing():
    """Static build must fail fast if a compact switch is not present in the full catalog."""
    net, _ = _load_data("IEEE13Nodeckt")
    if net.n_changing_switches == 0:
        pytest.skip("No changing switches in test case.")

    bad_net = SimpleNamespace(**vars(net))
    missing_branch = int(net.changing_switches[0])
    drop_row = int(np.flatnonzero(np.asarray(net.switch_idx) == missing_branch)[0])
    keep_rows = np.ones(int(net.n_switch), dtype=bool)
    keep_rows[drop_row] = False

    bad_net.switch_idx = np.asarray(net.switch_idx[keep_rows], dtype=np.int32)
    bad_net.n_switch = int(bad_net.switch_idx.shape[0])
    bad_net.switch_is_closed_base = np.asarray(net.switch_is_closed_base[keep_rows], dtype=np.int32)
    bad_net.switch_from_idx = np.asarray(net.switch_from_idx[keep_rows], dtype=np.int32)
    bad_net.switch_to_idx = np.asarray(net.switch_to_idx[keep_rows], dtype=np.int32)
    bad_net.switch_valid = np.asarray(net.switch_valid[keep_rows], dtype=bool)
    bad_net.switch_y_series = np.asarray(net.switch_y_series[keep_rows], dtype=np.complex128)
    bad_net.switch_y_shunt = np.asarray(net.switch_y_shunt[keep_rows], dtype=np.complex128)

    with pytest.raises(ValueError, match="changing_switches\\[0\\].*not a switch line"):
        woodbury_static_from_net(bad_net)


def _suite_limits(case_name: str) -> int:
    """Smaller status suites on large networks keep pytest runtime bounded."""
    if _n_net_for_case(case_name) >= LARGE_N_NET_THRESHOLD:
        return 6
    return MAX_STATUS_PER_CASE


def _load_compact_status_matrix(paths) -> np.ndarray:
    status_key = "changing_switch_status"

    def _read(path) -> np.ndarray:
        with h5py.File(path, "r") as f:
            grp = f["input"]
            key = status_key if status_key in grp else "branch_status"
            return np.asarray(grp[key][:], dtype=np.int32)

    if paths["test_feasible"].exists():
        compact = _read(paths["test_feasible"])
        if compact.shape[0] > 0:
            return compact
    return _read(paths["train"])


def _build_status_suite(
    tm,
    compact_dataset: np.ndarray,
    *,
    max_status: int = MAX_STATUS_PER_CASE,
) -> list[np.ndarray]:
    """Build a deterministic suite from connected statuses stored in datasets.

    Training generation uses ``sample_topology_connected`` and testing generation
    writes feasible topologies to ``*_test_feasible.h5``. To keep this suite on
    physically valid connected topologies, do not synthesize all-zero/all-one or
    random toggle patterns here; those can violate the topology policy even when
    the datasets are connectivity-filtered.
    """
    statuses: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    def add(status: np.ndarray) -> None:
        status_i32 = np.asarray(status, dtype=np.int32).reshape(-1)
        key = tuple(int(x) for x in status_i32.tolist())
        if key in seen:
            return
        seen.add(key)
        statuses.append(status_i32)

    add(tm.base_compact_status)

    if tm.n_changing_switches == 0:
        return statuses

    compact_dataset = np.asarray(compact_dataset, dtype=np.int32)
    if compact_dataset.ndim == 1:
        compact_dataset = compact_dataset.reshape(1, -1)

    for sample in compact_dataset:
        add(sample)
        if len(statuses) >= max_status:
            return statuses

    return statuses


def woodbury_topology_matrices_direct(
    s: WoodburyStatic,
    compact: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Same stamping as Woodbury; replace update with dense ``solve(Y_LL, I)``."""
    compact = jnp.asarray(compact, dtype=jnp.int32).reshape(-1)
    full_status = _expand_compact(s, compact)
    Y_scaled = _stamp_Y_scaled(s, full_status)
    Y_LL = Y_scaled[: s.n_net, : s.n_net]
    Y_L0 = Y_scaled[: s.n_net, s.n_net : s.n_total]
    eye = jnp.eye(s.n_net, dtype=jnp.complex128)
    Y_LL_inv = jnp.linalg.solve(Y_LL, eye)
    w = -(Y_LL_inv @ (Y_L0 @ s.v_slack_local))
    return Y_scaled, Y_LL_inv, w, full_status


def _max_abs_diff(a, b) -> float:
    """Host-safe max abs diff (JAX arrays compared on device, scalar returned)."""
    diff = jnp.max(jnp.abs(a - b))
    jax.block_until_ready(diff)
    return float(diff)


def _host_cond(Y_LL: jnp.ndarray) -> float:
    """Condition number of ``Y_LL`` on the host (avoids JAX GPU cond on large matrices)."""
    Y = np.asarray(jax.device_get(Y_LL))
    return float(np.linalg.cond(Y))


def _infer_n_total(first, n_net: int) -> int:
    """Infer ``n_total`` from full ``Y_scaled`` or slim ``Y_0row`` first tuple element."""
    if first.ndim == 2:
        if first.shape[0] == first.shape[1]:
            return int(first.shape[0])
        return int(first.shape[1])
    if first.shape[1] == first.shape[2]:
        return int(first.shape[1])
    return int(first.shape[2])


def _assert_vs_ref(
    tag: str,
    got,
    ref,
    *,
    skip_elementwise_inverse: bool = False,
) -> None:
    """Compare topology 4-tuples via packed ``TopologyData`` (full or slim first element)."""
    y_inv_g = got[1]
    n_net = int(y_inv_g.shape[-1])
    n_total = _infer_n_total(ref[0], n_net)
    topo_g = topology_data_from_matrices(*got, n_net, n_total)
    topo_r = topology_data_from_matrices(*ref, n_net, n_total)

    full_g_np = np.asarray(jax.device_get(topo_g.branch_status))
    full_r_np = np.asarray(jax.device_get(topo_r.branch_status))
    np.testing.assert_array_equal(
        full_g_np, full_r_np, err_msg=f"{tag}: full status mismatch"
    )

    y_0row_diff = _max_abs_diff(topo_g.Y_0row, topo_r.Y_0row)
    y_inv_diff = _max_abs_diff(topo_g.Y_LL_inv, topo_r.Y_LL_inv)
    w_diff = _max_abs_diff(topo_g.w, topo_r.w)

    assert y_0row_diff < STRICT_MAX_ABS_TOL, (
        f"{tag}: Y_0row max abs diff {y_0row_diff:.3e} exceeds {STRICT_MAX_ABS_TOL:.1e}"
    )

    Y_LL_ref, _ = Y_LL_L0_from_scaled(ref[0], n_net)

    if skip_elementwise_inverse:
        res_g = _inverse_residual(Y_LL_ref, topo_g.Y_LL_inv)
        res_r = _inverse_residual(Y_LL_ref, topo_r.Y_LL_inv)
        assert res_g < INVERSE_RESIDUAL_TOL, (
            f"{tag}: JAX inverse residual {res_g:.3e} exceeds {INVERSE_RESIDUAL_TOL:.1e} "
            f"(cond(Y_LL) too large for element-wise Y_LL_inv parity)"
        )
        assert res_r < INVERSE_RESIDUAL_TOL, (
            f"{tag}: reference inverse residual {res_r:.3e} exceeds {INVERSE_RESIDUAL_TOL:.1e} "
            f"(cond(Y_LL) too large for element-wise Y_LL_inv parity)"
        )
        return

    assert y_inv_diff < STRICT_MAX_ABS_TOL, (
        f"{tag}: Y_LL_inv max abs diff {y_inv_diff:.3e} exceeds {STRICT_MAX_ABS_TOL:.1e}"
    )
    assert w_diff < STRICT_MAX_ABS_TOL, (
        f"{tag}: w max abs diff {w_diff:.3e} exceeds {STRICT_MAX_ABS_TOL:.1e}"
    )


def _inverse_residual(Y_LL: jnp.ndarray, Y_LL_inv: jnp.ndarray) -> float:
    n = Y_LL.shape[0]
    eye = jnp.eye(n, dtype=jnp.complex128)
    res = jnp.max(jnp.abs(Y_LL @ Y_LL_inv - eye))
    jax.block_until_ready(res)
    return float(res)


def _timed_jit(fn, *args, n_warmup: int = N_WARMUP, n_repeats: int = N_REPEATS) -> tuple[float, float]:
    """Return (mean_ms, std_ms) after warmup."""
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times) * 1e3
    return float(arr.mean()), float(arr.std())


@pytest.mark.parametrize("case_name", CASES)
def test_jax_woodbury_matches_direct_inverse(case_name):
    """Woodbury update must match dense inv(Y_LL) on the same stamped admittance."""
    net, paths = _load_data(case_name)
    s = woodbury_static_from_net(net)
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    max_s = _suite_limits(case_name)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=max_s)

    for i, status in enumerate(statuses):
        compact = jnp.asarray(status, dtype=jnp.int32)
        got = woodbury_topology_matrices(s, compact)
        ref = woodbury_topology_matrices_direct(s, compact)
        Y_LL_chk, _ = Y_LL_L0_from_scaled(got[0], s.n_net)
        cond = _host_cond(Y_LL_chk)
        skip_inv = cond > COND_SKIP_ELEMENTWISE_INV
        _assert_vs_ref(
            f"{case_name}/direct/{i}",
            got,
            ref,
            skip_elementwise_inverse=skip_inv,
        )


@pytest.mark.parametrize("case_name", CASES_STAMP_NUMPY)
def test_jax_woodbury_matches_numpy_stamping_reference(case_name):
    """Cross-check JAX stamping + Woodbury against NumPy direct stamping path."""
    net, paths = _load_data(case_name)
    ref_tm = StampTopologyBackend(net)
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    max_s = _suite_limits(case_name)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=max_s)

    for i, status in enumerate(statuses):
        want = ref_tm._topology_matrices_from_stamping(status, sample_index=i)
        got = jax_tm.topology_matrices_eager(status, sample_index=i)
        n_net = int(got[1].shape[-1])
        # cond(Y_LL) from the stamped reference (got[0] may be slim Y_0row).
        Y_LL_chk, _ = Y_LL_L0_from_scaled(want[0], n_net)
        cond = _host_cond(Y_LL_chk)
        skip_inv = cond > COND_SKIP_ELEMENTWISE_INV
        _assert_vs_ref(
            f"{case_name}/numpy/{i}",
            got,
            want,
            skip_elementwise_inverse=skip_inv,
        )


@pytest.mark.parametrize("case_name", CASES)
def test_jax_woodbury_inverse_residual_identity(case_name):
    """Y_LL @ Y_LL_inv ≈ I for every topology sample in the suite."""
    net, paths = _load_data(case_name)
    s = woodbury_static_from_net(net)
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    max_s = _suite_limits(case_name)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=max_s)

    for i, status in enumerate(statuses):
        compact = jnp.asarray(status, dtype=jnp.int32)
        Y_scaled, Y_LL_inv, _, _ = woodbury_topology_matrices(s, compact)
        _, Y_LL_inv_ref, _, _ = woodbury_topology_matrices_direct(s, compact)
        Y_LL, _ = Y_LL_L0_from_scaled(Y_scaled, s.n_net)
        res_ref = _inverse_residual(Y_LL, Y_LL_inv_ref)
        if res_ref > INVERSE_RESIDUAL_TOL:
            pytest.skip(
                f"{case_name}/sample={i}: dense solve residual {res_ref:.3e} > "
                f"{INVERSE_RESIDUAL_TOL:.1e} (cond(Y_LL) too large for absolute tol)"
            )
        res = _inverse_residual(Y_LL, Y_LL_inv)
        assert res < INVERSE_RESIDUAL_TOL, (
            f"{case_name}/sample={i}: inverse residual {res:.3e} > {INVERSE_RESIDUAL_TOL:.1e}"
        )


def test_jax_woodbury_base_status_reuses_base_inverse():
    """No switch movement → return precomputed ``Y_LL_base_inv`` exactly."""
    net, _ = _load_data(CASES[0])
    s = woodbury_static_from_net(net)
    jax_tm = WoodburyTopologyBackend(net)
    _, y_inv, _, _ = woodbury_topology_matrices(
        s, jnp.asarray(jax_tm.base_compact_status, dtype=jnp.int32)
    )
    diff = float(jnp.max(jnp.abs(y_inv - s.Y_LL_base_inv)))
    assert diff == 0.0, f"base inverse mismatch max abs {diff}"


@pytest.mark.parametrize("case_name", CASES)
def test_woodbury_solve_matches_inverse_matvec(case_name):
    """``_woodbury_solve`` on slack rhs must match ``_woodbury_inverse @ rhs``."""
    net, paths = _load_data(case_name)
    s = woodbury_static_from_net(net)
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    max_s = _suite_limits(case_name)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=max_s)

    for i, status in enumerate(statuses):
        compact = jnp.asarray(status, dtype=jnp.int32)
        full = _expand_compact(s, compact)
        Y_scaled = _stamp_Y_scaled(s, full)
        _, Y_L0 = Y_LL_L0_from_scaled(Y_scaled, s.n_net)
        rhs = Y_L0 @ s.v_slack_local
        U, V = _build_lowrank_padded(s, full)
        w_solve = -_woodbury_solve(s, U, V, rhs)
        w_inv = -(_woodbury_inverse(s, U, V) @ rhs)
        diff = float(jnp.max(jnp.abs(w_solve - w_inv)))
        assert diff < STRICT_MAX_ABS_TOL, (
            f"{case_name}/sample={i}: solve vs inverse@rhs diff {diff:.3e}"
        )


@pytest.mark.skipif("IEEE13Nodeckt" not in CASES, reason="IEEE13 dataset required")
def test_jax_woodbury_smw_ir_stiff_switch_topology():
    """SMW-IR fixes stiff-tie topology where plain Woodbury drifts from dense inverse."""
    if "IEEE13Nodeckt" not in CASES:
        pytest.skip("IEEE13Nodeckt not in CASES")
    net, _ = _load_data("IEEE13Nodeckt")
    s = woodbury_static_from_net(net)
    compact = jnp.asarray([1, 0], dtype=jnp.int32)
    Y_scaled, y_inv, _, _ = woodbury_topology_matrices(s, compact)
    ref = woodbury_topology_matrices_direct(s, compact)
    y_inv_ref = ref[1]
    Y_LL, _ = Y_LL_L0_from_scaled(Y_scaled, s.n_net)
    diff = float(jnp.max(jnp.abs(y_inv - y_inv_ref)))
    res = _inverse_residual(Y_LL, y_inv)
    assert diff < STRICT_MAX_ABS_TOL, f"stiff topology inverse diff {diff:.3e}"
    assert res < INVERSE_RESIDUAL_TOL, f"stiff topology residual {res:.3e}"


@pytest.mark.parametrize("case_name", CASES)
def test_jax_woodbury_jit_matches_eager(case_name):
    net, paths = _load_data(case_name)
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    max_s = _suite_limits(case_name)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=max_s)

    for i, status in enumerate(statuses):
        g_e = jax_tm.topology_matrices_eager(status, sample_index=i)
        g_j = jax_tm.topology_matrices(status, sample_index=i)
        _assert_vs_ref(f"{case_name}/jit_eager/{i}", g_j, g_e)


@pytest.mark.parametrize("case_name", CASES)
def test_jax_woodbury_vmap_batch_matches_eager(case_name):
    net, paths = _load_data(case_name)
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    max_s = _suite_limits(case_name)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=max_s)
    status_batch = np.asarray(statuses, dtype=np.int32)

    got_batch = jax_tm.topology_matrices_batch(status_batch)
    for k, status in enumerate(statuses):
        got = jax_tm.topology_matrices_eager(status, sample_index=k)
        ref = tuple(arr[k] for arr in got_batch)
        _assert_vs_ref(f"{case_name}/batch/{k}", got, ref)


@pytest.mark.parametrize("case_name", CASES)
def test_jax_woodbury_pure_fn_matches_lax_map_batch(case_name):
    """Production batch path (lax.map) must match per-sample eager evaluation."""
    net, paths = _load_data(case_name)
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    max_s = _suite_limits(case_name)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=max_s)
    batch = np.asarray(statuses, dtype=np.int32)

    got_batch = jax_tm.topology_matrices_batch(batch)
    for k in range(batch.shape[0]):
        got = jax_tm.topology_matrices_eager(statuses[k], sample_index=k)
        ref = tuple(arr[k] for arr in got_batch)
        _assert_vs_ref(f"{case_name}/lax_map_batch/{k}", got, ref)


@pytest.mark.parametrize("case_name", CASES)
def test_woodbury_slim_matches_full_topology_data(case_name):
    """Slim production path must pack to the same ``TopologyData`` as full ``Y_scaled``."""
    net, paths = _load_data(case_name)
    s = woodbury_static_from_net(net)
    n_net, n_total = int(net.n_net), int(net.n_total)
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    max_s = _suite_limits(case_name)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=max_s)

    for i, status in enumerate(statuses):
        compact = jnp.asarray(status, dtype=jnp.int32)
        full = woodbury_topology_matrices(s, compact)
        slim = woodbury_topology_matrices_slim(s, compact)
        topo_full = topology_data_from_matrices(*full, n_net, n_total)
        topo_slim = topology_data_from_matrices(*slim, n_net, n_total)
        np.testing.assert_allclose(
            topo_full.Y_0row, topo_slim.Y_0row, rtol=0, atol=0,
            err_msg=f"{case_name}/slim/{i}: Y_0row",
        )
        np.testing.assert_allclose(
            topo_full.Y_LL_inv, topo_slim.Y_LL_inv, rtol=0, atol=0,
            err_msg=f"{case_name}/slim/{i}: Y_LL_inv",
        )
        np.testing.assert_allclose(
            topo_full.w, topo_slim.w, rtol=0, atol=0,
            err_msg=f"{case_name}/slim/{i}: w",
        )


@pytest.mark.skipif("IEEE13Nodeckt" not in CASES, reason="IEEE13 dataset required")
def test_matrices_first_to_y_0row_both_shapes():
    """Full Y_scaled and slim Y_0row first elements pack to identical Y_0row."""
    case_name = "IEEE13Nodeckt"
    net, paths = _load_data(case_name)
    n_net, n_total = int(net.n_net), int(net.n_total)
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=3)
    status_batch = np.asarray(statuses, dtype=np.int32)

    # --- single sample ---
    compact = jnp.asarray(statuses[0], dtype=jnp.int32)
    Y_full, Y_LL_inv, w, full_status = jax_tm.topology_matrices_eager_full(compact)
    Y_0row_ref = y_0row_from_scaled(Y_full, n_net, n_total)
    Y_0row_slim = matrices_first_to_y_0row(Y_full, n_net, n_total)

    # The slim branch returns the input unchanged; verify against the
    # full-branch slice to confirm both paths agree on the slack rows.
    np.testing.assert_allclose(Y_0row_slim, Y_0row_ref, rtol=0, atol=0)

    topo_full = topology_data_from_matrices(
        Y_full, Y_LL_inv, w, full_status, n_net, n_total
    )
    topo_slim = topology_data_from_matrices(
        Y_0row_slim, Y_LL_inv, w, full_status, n_net, n_total
    )
    np.testing.assert_allclose(topo_full.Y_0row, topo_slim.Y_0row, rtol=0, atol=0)
    np.testing.assert_allclose(topo_full.Y_LL_inv, topo_slim.Y_LL_inv, rtol=0, atol=0)
    np.testing.assert_allclose(topo_full.w, topo_slim.w, rtol=0, atol=0)

    topo_backend = jax_tm.topology_data(compact)
    np.testing.assert_allclose(topo_backend.Y_0row, topo_full.Y_0row, rtol=0, atol=0)

    # --- batched (production slim path) ---
    Y_batch, Y_inv_batch, w_batch, status_batch_j = jax_tm.topology_matrices_batch(
        status_batch
    )
    full_rows = [
        jax_tm.topology_matrices_eager_full(statuses[k])
        for k in range(status_batch.shape[0])
    ]
    Y_batch_full = jnp.stack([r[0] for r in full_rows])
    Y_0row_from_full = matrices_first_to_y_0row(Y_batch_full, n_net, n_total)
    np.testing.assert_allclose(Y_batch, Y_0row_from_full, rtol=0, atol=0)

    topo_batch_slim = topology_data_from_matrices(
        Y_batch, Y_inv_batch, w_batch, status_batch_j, n_net, n_total
    )
    topo_batch_full = topology_data_from_matrices(
        Y_batch_full, Y_inv_batch, w_batch, status_batch_j, n_net, n_total
    )
    np.testing.assert_allclose(
        topo_batch_full.Y_0row, topo_batch_slim.Y_0row, rtol=0, atol=0
    )

    topo_backend_batch = jax_tm.topology_data_batch(status_batch)
    np.testing.assert_allclose(
        topo_backend_batch.Y_0row, topo_batch_full.Y_0row, rtol=0, atol=0
    )


def _speed_batch_sizes(case_name: str) -> tuple[int, ...]:
    if _n_net_for_case(case_name) >= LARGE_N_NET_THRESHOLD:
        return (1, 4)
    return SPEED_BATCH_SIZES


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_jax_woodbury_speed_vs_direct_inverse(case_name, batch_size, capsys):
    """JIT batched Woodbury vs JIT batched dense inv; record speedup (assert on large n_net)."""
    if batch_size not in _speed_batch_sizes(case_name):
        pytest.skip(f"{case_name}: batch_size {batch_size} not in speed grid")

    net, paths = _load_data(case_name)
    s = woodbury_static_from_net(net)
    jax_tm = WoodburyTopologyBackend(net)
    compact_dataset = _load_compact_status_matrix(paths)
    max_s = _suite_limits(case_name)
    statuses = _build_status_suite(jax_tm, compact_dataset, max_status=max_s)

    # if len(statuses) < batch_size:
    #     pytest.skip(f"{case_name}: need at least {batch_size} unique statuses")

    batch = np.asarray(statuses[:batch_size], dtype=np.int32)
    batch_j = jnp.asarray(batch, dtype=jnp.int32)

    woodbury_fn = jax_tm.topology_matrices_batch
    direct_fn = jax.jit(
        lambda status_batch: jax.lax.map(
            functools.partial(woodbury_topology_matrices_direct, s), status_batch
        )
    )

    w_mean, w_std = _timed_jit(woodbury_fn, batch)
    d_mean, d_std = _timed_jit(direct_fn, batch_j)
    speedup = d_mean / w_mean if w_mean > 0 else float("inf")

    print(
        f"\n[{case_name}] n_net={s.n_net} batch={batch_size}: "
        f"woodbury={w_mean:.3f}±{w_std:.3f} ms  "
        f"direct_inv={d_mean:.3f}±{d_std:.3f} ms  speedup={speedup:.2f}x"
    )

    # Correctness on the timed batch (no tolerance relaxation).
    out_w = woodbury_fn(batch_j)
    out_d = direct_fn(batch_j)
    for k in range(batch_size):
        got = tuple(x[k] for x in out_w)
        ref = tuple(x[k] for x in out_d)
        Y_LL_chk, _ = Y_LL_L0_from_scaled(ref[0], s.n_net)
        cond = _host_cond(Y_LL_chk)
        skip_inv = cond > COND_SKIP_ELEMENTWISE_INV
        _assert_vs_ref(
            f"{case_name}/speed_batch/{k}",
            got,
            ref,
            skip_elementwise_inverse=skip_inv,
        )

    if s.n_net >= MIN_N_NET_FOR_SPEEDUP_ASSERT and batch_size >= MIN_BATCH_FOR_SPEEDUP_ASSERT:
        assert speedup >= MIN_WOODBURY_SPEEDUP, (
            f"{case_name} batch={batch_size}: expected Woodbury >= {MIN_WOODBURY_SPEEDUP:.2f}x "
            f"faster than direct inv, got {speedup:.2f}x "
            f"(woodbury {w_mean:.3f} ms, direct {d_mean:.3f} ms)"
        )


@pytest.mark.skip(reason="slow; kept for manual verification of _build_one_switch_uv factorization")
@pytest.mark.parametrize("case_name", CASES)
def test_build_one_switch_uv_factorization(case_name):
    """Low-rank factors U_s, V_s reconstruct per-switch admittance perturbation.

    For each switch at delta = +1, -1 (when achievable), and 0, verify that
    U_s @ V_s.T matches the Y_LL difference from direct stamping
    (``_stamp_Y_scaled`` with and without that single switch toggle).
    """
    net, _ = _load_data(case_name)
    s = woodbury_static_from_net(net)

    if s.n_switch == 0:
        pytest.skip(f"{case_name}: no switches")

    base_compact = s.base_branch_status[s.changing_switches]
    base_full = _expand_compact(s, base_compact)
    # Use stamping to get base Y_LL (not s.Y_LL_base which may differ for non-base topologies).
    # Actually, base_full == s.base_branch_status always (since base_compact comes from
    # s.base_branch_status[s.changing_switches]), but we compute via stamping to be consistent.
    base_Y = _stamp_Y_scaled(s, base_full)[:s.n_net, :s.n_net]

    for sidx in range(int(s.n_switch)):
        branch_idx = int(s.switch_idx[sidx])
        closed_base = int(s.switch_is_closed_base[sidx])

        for desired_delta in [1, -1]:
            desired_full = closed_base + desired_delta
            if desired_full not in (0, 1):
                continue  # delta would produce invalid branch status

            U_s, V_s = _build_one_switch_uv(s, sidx, jnp.asarray(float(desired_delta)))

            assert U_s.shape == (s.n_net, 3 * s.max_switch_terms), (
                f"U_s shape {U_s.shape}"
            )
            assert V_s.shape == (s.n_net, 3 * s.max_switch_terms), (
                f"V_s shape {V_s.shape}"
            )

            Delta_uv = U_s @ V_s.T  # (n_net, n_net)

            modified_full = base_full.at[branch_idx].set(desired_full)
            Y_mod = _stamp_Y_scaled(s, modified_full)[:s.n_net, :s.n_net]
            Delta_expected = Y_mod - base_Y

            diff = float(jnp.max(jnp.abs(Delta_uv - Delta_expected)))
            assert diff < 2e-9, (
                f"{case_name} switch {sidx} (branch {branch_idx}) delta={desired_delta:+d}: "
                f"max abs diff {diff:.3e} — U_s @ V_s.T does not match stamped delta"
            )

        # delta = 0: factors must be identically zero (no contribution)
        U_s, V_s = _build_one_switch_uv(s, sidx, jnp.asarray(0.0))
        Delta_uv = U_s @ V_s.T
        diff = float(jnp.max(jnp.abs(Delta_uv)))
        assert diff == 0.0, (
            f"{case_name} switch {sidx} (branch {branch_idx}) delta=0: "
            f"expected zero contribution, got max abs {diff:.3e}"
        )


# ---------------------------------------------------------------------------
# Branchless vmap batch parity test
# ---------------------------------------------------------------------------

# Expected IR steps per case (experiment-derived, tol = 1e-6).
_BRANCHLESS_IR_STEPS = {
    "IEEE13Nodeckt": 1,
    "IEEE123Master": 1,
    "240bus_Master": 2,
    "906bus_Master": 0,
}


@pytest.mark.parametrize("case_name", CASES)
def test_branchless_vmap_batch_matches_eager(case_name):
    """vmap of ``woodbury_topology_matrices_slim_branchless(ir_steps=N)`` matches
    per-sample eager evaluation."""
    ir_steps = _BRANCHLESS_IR_STEPS.get(case_name, 1)
    net, paths = _load_data(case_name)
    s = woodbury_static_from_net(net)
    tm = WoodburyTopologyBackend(net, ir_steps=ir_steps)
    compact_dataset = _load_compact_status_matrix(paths)
    max_s = _suite_limits(case_name)
    statuses = _build_status_suite(tm, compact_dataset, max_status=max_s)
    batch = np.asarray(statuses, dtype=np.int32)
    batch_j = jnp.asarray(batch, dtype=jnp.int32)

    # vmap batch via backend's batch_vmap_jit.
    assert tm.batch_vmap_jit is not None
    out_batch = tm.batch_vmap_jit(s, batch_j)

    for k in range(batch.shape[0]):
        compact = jnp.asarray(statuses[k], dtype=jnp.int32)
        out_eager = tm._fn_slim(s, compact)
        for name, a, b in [("Y_0row", out_batch[0][k], out_eager[0]),
                           ("Y_LL_inv", out_batch[1][k], out_eager[1]),
                           ("w", out_batch[2][k], out_eager[2])]:
            abs_diff = float(jnp.max(jnp.abs(a - b)))
            max_abs = float(jnp.max(jnp.abs(b)))
            rel_diff = abs_diff / max_abs if max_abs > 0 else abs_diff
            assert rel_diff < 1e-8, (
                f"{case_name} sample {k} {name}: "
                f"vmap vs eager rel_diff {rel_diff:.3e}"
            )
