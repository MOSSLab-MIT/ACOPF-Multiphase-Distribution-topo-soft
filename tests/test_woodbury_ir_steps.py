"""Tests for branchless Woodbury IR-step variants and vmap batch path.

Validates that:
- Branchless ``woodbury_topology_matrices_slim_branchless(ir_steps=N)`` produces
  accurate inverses (residual < 1e-6) iff N >= the case's required IR steps.
- ``resolve_ir_steps()`` returns the expected per-case step count.
- Branchless ir_steps=0 matches the adaptive lax.map baseline on 906bus (where 0
  IR steps is exact).
- vmap of the branchless fn matches per-sample eager evaluation.

Run with conda env ``new_opt`` (see ``AGENTS.md``):

    python -m pytest tests/test_woodbury_ir_steps.py -v
"""
import functools
import sys
from pathlib import Path

import h5py
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import CASE_ROOTS, case_paths
from core.topology.woodbury import (
    WoodburyTopologyBackend,
    WoodburyStatic,
    woodbury_static_from_net,
    woodbury_topology_matrices_slim_branchless,
    _expand_compact,
    _build_lowrank_padded,
    _woodbury_inverse,
    _woodbury_inverse_with_ir,
    _woodbury_solve,
)
from core.topology.slim import _stamp_Y_LL_slim_bl

from test_topology_woodbury_jax import _load_net_numpy, _load_compact_status_matrix, _build_status_suite

DATASET_ROOT = ROOT / "datagen_topo" / "datasets"

_ALL_IR_CASES = ["IEEE13Nodeckt", "IEEE123Master", "240bus_Master", "906bus_Master"]
IR_TEST_CASES = [
    name for name in _ALL_IR_CASES
    if (DATASET_ROOT / name / "network_config.json").exists()
]

# Experiment-derived IR-step needs per case (tol = 1e-6).
# These are verified by test_resolve_ir_steps at runtime.
EXPECTED_IR_STEPS = {
    "IEEE13Nodeckt": 1,
    "IEEE123Master": 1,
    "240bus_Master": 2,
    "906bus_Master": 0,
}

INVERSE_RESIDUAL_TOL = 1e-6


def _load_unique_compacts(case_name: str, max_unique: int = 50) -> np.ndarray:
    """Load unique compact statuses from training data (capped for speed)."""
    paths = case_paths(case_name)
    compact_dataset = _load_compact_status_matrix(paths)
    unique = np.unique(compact_dataset, axis=0)
    if unique.shape[0] > max_unique:
        # Subsample evenly to keep diversity.
        idx = np.linspace(0, unique.shape[0] - 1, max_unique, dtype=int)
        unique = unique[idx]
    return unique.astype(np.int32)


def _inverse_residual(Y_LL, y_inv, n_net):
    eye = jnp.eye(n_net, dtype=jnp.complex128)
    res = jnp.max(jnp.abs(Y_LL @ y_inv - eye))
    jax.block_until_ready(res)
    return float(res)


@pytest.mark.parametrize("case_name", IR_TEST_CASES)
@pytest.mark.parametrize("ir_steps", [0, 1, 2])
def test_branchless_ir_steps_residual(case_name, ir_steps):
    """Inverse residual < tol for ALL compacts iff ir_steps >= required.

    When ir_steps < required, at least ONE compact must have residual >= tol
    (the worst-case topology that motivated the required step count).
    """
    net = _load_net_numpy(case_name)
    s = woodbury_static_from_net(net)
    required = EXPECTED_IR_STEPS[case_name]

    unique_compacts = _load_unique_compacts(case_name)
    fn = jax.jit(functools.partial(woodbury_topology_matrices_slim_branchless, ir_steps=ir_steps))

    residuals = []
    for row in unique_compacts:
        compact = jnp.asarray(row, dtype=jnp.int32)
        _, y_inv, _, _ = fn(s, compact)
        fs = _expand_compact(s, compact)
        Y_LL = _stamp_Y_LL_slim_bl(s, fs)
        res = _inverse_residual(Y_LL, y_inv, s.n_net)
        residuals.append(res)

    max_res = max(residuals)
    if ir_steps >= required:
        assert max_res < INVERSE_RESIDUAL_TOL, (
            f"{case_name} ir_steps={ir_steps} (required={required}): "
            f"max residual {max_res:.3e} >= {INVERSE_RESIDUAL_TOL:.1e}"
        )
    else:
        assert max_res >= INVERSE_RESIDUAL_TOL, (
            f"{case_name} ir_steps={ir_steps} (required={required}): "
            f"max residual {max_res:.3e} < {INVERSE_RESIDUAL_TOL:.1e} "
            f"(expected at least one insufficient, but all accurate)"
        )


@pytest.mark.skipif("906bus_Master" not in IR_TEST_CASES, reason="906bus dataset required")
def test_branchless_ir0_matches_adaptive_906():
    """Branchless ir_steps=0 output matches adaptive baseline on 906bus."""
    case_name = "906bus_Master"
    net = _load_net_numpy(case_name)
    s = woodbury_static_from_net(net)

    # Default backend (auto, unresolved) uses adaptive fn.
    tm_default = WoodburyTopologyBackend(net)
    # Branchless ir_steps=0 backend.
    tm_bl = WoodburyTopologyBackend(net, ir_steps=0)

    unique_compacts = _load_unique_compacts(case_name, max_unique=6)

    for i, row in enumerate(unique_compacts):
        compact = jnp.asarray(row, dtype=jnp.int32)
        out_default = tm_default._topology_matrices_jit(s, compact)
        out_bl = tm_bl._topology_matrices_jit(s, compact)

        for name, a, b in [("Y_0row", out_default[0], out_bl[0]),
                           ("Y_LL_inv", out_default[1], out_bl[1]),
                           ("w", out_default[2], out_bl[2])]:
            diff = float(jnp.max(jnp.abs(a - b)))
            assert diff < 1e-8, (
                f"{case_name} sample {i} {name}: "
                f"branchless ir=0 vs adaptive diff {diff:.3e}"
            )


@pytest.mark.parametrize("case_name", IR_TEST_CASES)
def test_resolve_ir_steps(case_name):
    """resolve_ir_steps() returns the expected per-case step count."""
    net = _load_net_numpy(case_name)
    tm = WoodburyTopologyBackend(net, ir_steps="auto")

    unique_compacts = _load_unique_compacts(case_name)

    resolved = tm.resolve_ir_steps(unique_compacts)
    expected = EXPECTED_IR_STEPS[case_name]
    assert resolved == expected, (
        f"{case_name}: resolve_ir_steps returned {resolved}, expected {expected}"
    )
    assert tm._resolved_ir_steps == expected
    if tm.n_changing_switches > 0:
        assert tm.batch_vmap_jit is not None


@pytest.mark.parametrize("case_name", IR_TEST_CASES)
def test_branchless_vmap_matches_eager(case_name):
    """vmap of branchless fn matches per-sample eager evaluation."""
    net = _load_net_numpy(case_name)
    s = woodbury_static_from_net(net)
    required = EXPECTED_IR_STEPS[case_name]

    tm = WoodburyTopologyBackend(net, ir_steps=required)
    unique_compacts = _load_unique_compacts(case_name, max_unique=6)
    batch = np.asarray(unique_compacts, dtype=np.int32)
    batch_j = jnp.asarray(batch, dtype=jnp.int32)

    assert tm.batch_vmap_jit is not None
    out_batch = tm.batch_vmap_jit(s, batch_j)

    for k in range(batch.shape[0]):
        compact = jnp.asarray(unique_compacts[k], dtype=jnp.int32)
        out_eager = tm._fn_slim(s, compact)
        for name, a, b in [("Y_0row", out_batch[0][k], out_eager[0]),
                           ("Y_LL_inv", out_batch[1][k], out_eager[1]),
                           ("w", out_batch[2][k], out_eager[2])]:
            abs_diff = float(jnp.max(jnp.abs(a - b)))
            # vmap batches BLAS calls; summation reordering produces diffs
            # ~machine_eps * max_abs.  Use relative tolerance to accommodate.
            max_abs = float(jnp.max(jnp.abs(b)))
            rel_diff = abs_diff / max_abs if max_abs > 0 else abs_diff
            assert rel_diff < 1e-8, (
                f"{case_name} sample {k} {name}: "
                f"vmap vs eager rel_diff {rel_diff:.3e} (abs={abs_diff:.3e}, max_abs={max_abs:.3e})"
            )


@pytest.mark.skipif("906bus_Master" not in IR_TEST_CASES, reason="906bus dataset required")
def test_vmap_map_batch_parity_906():
    """vmap and map batch modes produce identical output on 906bus."""
    case_name = "906bus_Master"
    net = _load_net_numpy(case_name)
    s = woodbury_static_from_net(net)

    tm = WoodburyTopologyBackend(net, ir_steps=0)
    unique_compacts = _load_unique_compacts(case_name, max_unique=6)
    batch = np.asarray(unique_compacts, dtype=np.int32)
    batch_j = jnp.asarray(batch, dtype=jnp.int32)

    out_vmap = tm.batch_vmap_jit(s, batch_j)
    out_map = tm.batch_matrices_jit(s, batch_j)

    for name, a, b in [("Y_0row", out_vmap[0], out_map[0]),
                       ("Y_LL_inv", out_vmap[1], out_map[1]),
                       ("w", out_vmap[2], out_map[2])]:
        abs_diff = float(jnp.max(jnp.abs(a - b)))
        max_abs = float(jnp.max(jnp.abs(b)))
        rel_diff = abs_diff / max_abs if max_abs > 0 else abs_diff
        assert rel_diff < 1e-8, (
            f"{case_name} {name}: vmap vs map rel_diff {rel_diff:.3e} "
            f"(abs={abs_diff:.3e}, max_abs={max_abs:.3e})"
        )


def test_woodbury_inverse_with_ir_matches_slim():
    """``_woodbury_inverse_with_ir`` returns the same ``Y_LL_inv`` as the slim fn."""
    if "IEEE13Nodeckt" not in IR_TEST_CASES:
        pytest.skip("IEEE13 dataset required")
    case_name = "IEEE13Nodeckt"
    net = _load_net_numpy(case_name)
    s = woodbury_static_from_net(net)
    ir_steps = EXPECTED_IR_STEPS[case_name]

    unique_compacts = _load_unique_compacts(case_name, max_unique=4)

    ir_fn = jax.jit(functools.partial(_woodbury_inverse_with_ir, ir_steps=ir_steps))
    slim_fn = jax.jit(functools.partial(woodbury_topology_matrices_slim_branchless, ir_steps=ir_steps))

    for row in unique_compacts:
        compact = jnp.asarray(row, dtype=jnp.int32)
        full_status = _expand_compact(s, compact)
        _, y_inv_ir, _, _ = ir_fn(s, full_status)
        _, y_inv_slim, _, _ = slim_fn(s, compact)
        diff = float(jnp.max(jnp.abs(y_inv_ir - y_inv_slim)))
        assert diff < 1e-12, (
            f"{case_name}: _woodbury_inverse_with_ir vs slim fn Y_LL_inv diff {diff:.3e}"
        )
