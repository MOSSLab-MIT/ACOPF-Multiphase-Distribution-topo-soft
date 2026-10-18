import numpy as np
import pytest

from conftest import (
    CASE_ROOTS,
    case_paths,
    expand_compact_status,
    load_case_config,
    load_case_matrices,
    load_h5_sample,
    parse_json_tree,
    stamp_switch_topology,
)


CASES = [name for name in ("IEEE13Nodeckt", "IEEE123Master") if name in CASE_ROOTS]


@pytest.mark.parametrize("case_name", CASES)
def test_topology_stamping_matches_direct_reference(case_name):
    from core.grids.data import OPFData
    from core.topology.batch import build_topology_batch_jit

    cfg = load_case_config(case_name)
    paths = case_paths(case_name)
    data = OPFData(str(paths["train"]), str(paths["test_feasible"]), str(paths["json"]))

    sample = load_h5_sample(case_name, split="test_feasible", index=0)
    inp = sample["input"]
    topo_key = "changing_switch_status" if "changing_switch_status" in inp else "branch_status"
    compact = np.asarray(inp[topo_key], dtype=np.int32)
    if compact.size:
        compact = compact.copy()
        compact[0] = 1 - compact[0]

    topo = build_topology_batch_jit(compact[None, :], data)
    cfg, y_scaled, Y_LL, y_ns = load_case_matrices(case_name)
    stamped = stamp_switch_topology(cfg, compact)
    n_net = int(cfg["n_net"])
    v_slack_local = np.asarray(parse_json_tree(cfg["v_sub_ref"]), dtype=np.complex128) / np.asarray(
        cfg["v_scale_slack"], dtype=np.float64
    )
    Y_LL_ref = stamped[:n_net, :n_net]
    y_ns_ref = stamped[:n_net, n_net:]
    Y_LL_inv_ref = np.linalg.inv(Y_LL_ref)
    w_ref = -Y_LL_inv_ref @ (y_ns_ref @ v_slack_local)

    expanded_ref = expand_compact_status(cfg, compact)
    np.testing.assert_array_equal(np.asarray(topo.branch_status[0]), expanded_ref)
    np.testing.assert_allclose(np.asarray(topo.Y_LL_inv[0]), Y_LL_inv_ref, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(topo.Y_0row[0]), stamped[n_net:, :], atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(np.asarray(topo.w[0]), w_ref, atol=1e-10, rtol=1e-10)


