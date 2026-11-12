"""JAX-native Woodbury topology: JIT- and vmap-compatible with fixed shapes.

See ``TopologyManagerWoodbury`` in ``core/topology_new.py`` for the reference
NumPy/SciPy implementation. This module keeps the same math and IO contract.
"""
import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from core.topology.utils import make_topology_data

_INVERSE_RESIDUAL_TOL = 1e-6
# SMW-IR entry gate: cond(K) for K = I + V^T A^{-1} U (see experiments/compare_k_gate_cond_vs_solve.py).
_K_COND_THRESH = 1e8


def _build_changing_switch_row_map(
    switch_idx: np.ndarray,
    changing_switches: np.ndarray,
) -> np.ndarray:
    """Map each compact changing-switch slot to its switch-catalog row index.

    ``changing_switches[j]`` and ``switch_idx[row]`` are both branch-line indices.
    """
    switch_idx = np.asarray(switch_idx, dtype=np.int32).reshape(-1)
    changing_switches = np.asarray(changing_switches, dtype=np.int32).reshape(-1)
    row_by_branch: dict[int, int] = {}
    for row, branch in enumerate(switch_idx.tolist()):
        branch = int(branch)
        if branch in row_by_branch:
            raise ValueError(
                f"switch_idx maps branch {branch} to multiple catalog rows "
                f"({row_by_branch[branch]} and {row})."
            )
        row_by_branch[branch] = row
    changing_rows = np.empty(changing_switches.shape[0], dtype=np.int32)
    for j, branch in enumerate(changing_switches.tolist()):
        branch = int(branch)
        if branch not in row_by_branch:
            raise ValueError(
                f"changing_switches[{j}]={branch} is not a switch line in switch_idx."
            )
        changing_rows[j] = row_by_branch[branch]
    return changing_rows


def _validate_static_contracts(
    *,
    n_net: int,
    n_total: int,
    n_switch: int,
    n_changing_switches: int,
    max_switch_terms: int,
    base_branch_status: np.ndarray,
    changing_switches: np.ndarray,
    changing_switch_row: np.ndarray,
    switch_idx: np.ndarray,
    switch_is_closed_base: np.ndarray,
    switch_from_idx: np.ndarray,
    switch_to_idx: np.ndarray,
    switch_valid: np.ndarray,
    switch_y_series: np.ndarray,
    switch_y_shunt: np.ndarray,
    v_scale_full: np.ndarray,
    Y_scaled_base: np.ndarray,
) -> None:
    """Validate static topology tensors before tracing JAX computations."""
    n_branch = int(base_branch_status.shape[0])
    if changing_switches.shape != (n_changing_switches,):
        raise ValueError(
            "changing_switches width does not match n_changing_switches "
            f"({changing_switches.shape[0]} != {n_changing_switches})."
        )
    if changing_switch_row.shape != (n_changing_switches,):
        raise ValueError(
            "changing_switch_row width does not match n_changing_switches "
            f"({changing_switch_row.shape[0]} != {n_changing_switches})."
        )
    if switch_idx.shape != (n_switch,):
        raise ValueError(
            "switch_idx width does not match n_switch "
            f"({switch_idx.shape[0]} != {n_switch})."
        )
    if switch_is_closed_base.shape != (n_switch,):
        raise ValueError(
            "switch_is_closed_base width does not match n_switch "
            f"({switch_is_closed_base.shape[0]} != {n_switch})."
        )

    expected_switch_shape = (n_switch, max_switch_terms)
    if switch_from_idx.shape != expected_switch_shape:
        raise ValueError(
            "switch_from_idx has invalid shape "
            f"{switch_from_idx.shape}; expected {expected_switch_shape}."
        )
    if switch_to_idx.shape != expected_switch_shape:
        raise ValueError(
            "switch_to_idx has invalid shape "
            f"{switch_to_idx.shape}; expected {expected_switch_shape}."
        )
    if switch_valid.shape != expected_switch_shape:
        raise ValueError(
            "switch_valid has invalid shape "
            f"{switch_valid.shape}; expected {expected_switch_shape}."
        )

    expected_switch_matrix_shape = (n_switch, max_switch_terms, max_switch_terms)
    if switch_y_series.shape != expected_switch_matrix_shape:
        raise ValueError(
            "switch_y_series has invalid shape "
            f"{switch_y_series.shape}; expected {expected_switch_matrix_shape}."
        )
    if switch_y_shunt.shape != expected_switch_matrix_shape:
        raise ValueError(
            "switch_y_shunt has invalid shape "
            f"{switch_y_shunt.shape}; expected {expected_switch_matrix_shape}."
        )
    if v_scale_full.shape != (n_total,):
        raise ValueError(
            "v_scale_full width does not match n_total "
            f"({v_scale_full.shape[0]} != {n_total})."
        )
    if Y_scaled_base.shape != (n_total, n_total):
        raise ValueError(
            "Y_scaled_base has invalid shape "
            f"{Y_scaled_base.shape}; expected ({n_total}, {n_total})."
        )
    if Y_scaled_base[:n_net, :n_net].shape != (n_net, n_net):
        raise ValueError("Y_scaled_base[:n_net, :n_net] must have shape (n_net, n_net).")

    if np.any((base_branch_status != 0) & (base_branch_status != 1)):
        raise ValueError("base_branch_status must contain only binary values {0, 1}.")
    if np.any((switch_is_closed_base != 0) & (switch_is_closed_base != 1)):
        raise ValueError("switch_is_closed_base must contain only binary values {0, 1}.")
    if np.any((changing_switches < 0) | (changing_switches >= n_branch)):
        raise ValueError("changing_switches must be 0-based indices within [0, n_branch).")
    if np.any((switch_idx < 0) | (switch_idx >= n_branch)):
        raise ValueError("switch_idx must be 0-based indices within [0, n_branch).")
    if np.any((changing_switch_row < 0) | (changing_switch_row >= n_switch)):
        raise ValueError("changing_switch_row must be 0-based indices within [0, n_switch).")
    if not np.array_equal(switch_idx[changing_switch_row], changing_switches):
        raise ValueError(
            "changing_switch_row must map each compact slot back to the same branch line in "
            "switch_idx."
        )


@dataclass(frozen=True)
class WoodburyStatic:
    """Immutable tensors shared across all topology samples for one network."""

    n_net: int
    n_total: int
    n_switch: int
    n_changing_switches: int
    max_switch_terms: int

    base_branch_status: jnp.ndarray  # (n_branch,) int32
    changing_switches: jnp.ndarray  # (n_changing_switches,) int32 branch lines
    # Compact slot j -> switch catalog row s.changing_switch_row[j] (switch_idx[s] == changing_switches[j])
    changing_switch_row: jnp.ndarray  # (n_changing_switches,) int32
    switch_idx: jnp.ndarray  # (n_switch,) int32 branch line per catalog row
    switch_is_closed_base: jnp.ndarray  # (n_switch,) int32
    switch_from_idx: jnp.ndarray  # (n_switch, m)
    switch_to_idx: jnp.ndarray
    switch_valid: jnp.ndarray  # (n_switch, m) bool
    switch_y_series: jnp.ndarray  # complex
    switch_y_shunt: jnp.ndarray
    v_scale_full: jnp.ndarray  # float64 (n_total,)
    v_slack_local: jnp.ndarray  # complex (n_slk,)
    Y_scaled_base: jnp.ndarray  # (n_total, n_total) complex128 (shared with net)
    Y_LL_base_inv: jnp.ndarray  # (n_net, n_net) precomputed A^{-1}


# --- PyTree registration: arrays = traced leaves, ints = static aux ----------
# Binding a WoodburyStatic via functools.partial into jax.jit bakes its large
# arrays (Y_scaled_base, Y_LL_base_inv ~ N^2 each) as HLO constants,
# which XLA then constant-folds and OOMs on large feeders. Registering
# it as a PyTree lets callers pass it as a positional JIT arg so the arrays stay
# traced (one device copy, no folding) while the int fields remain available at
# trace time as Python ints for shapes / slicing / control flow.
_WOODBURY_ARRAY_FIELDS = (
    "base_branch_status",
    "changing_switches",
    "changing_switch_row",
    "switch_idx",
    "switch_is_closed_base",
    "switch_from_idx",
    "switch_to_idx",
    "switch_valid",
    "switch_y_series",
    "switch_y_shunt",
    "v_scale_full",
    "v_slack_local",
    "Y_scaled_base",
    "Y_LL_base_inv",
)
_WOODBURY_STATIC_FIELDS = (
    "n_net",
    "n_total",
    "n_switch",
    "n_changing_switches",
    "max_switch_terms",
)


def _woodbury_flatten(s: WoodburyStatic):
    children = tuple(getattr(s, f) for f in _WOODBURY_ARRAY_FIELDS)
    aux = tuple(getattr(s, f) for f in _WOODBURY_STATIC_FIELDS)
    return children, aux


def _woodbury_unflatten(aux, children) -> WoodburyStatic:
    kw = dict(zip(_WOODBURY_STATIC_FIELDS, aux))
    kw.update(dict(zip(_WOODBURY_ARRAY_FIELDS, children)))
    return WoodburyStatic(**kw)


jax.tree_util.register_pytree_node(
    WoodburyStatic, _woodbury_flatten, _woodbury_unflatten
)


def woodbury_static_from_net(net: Any) -> WoodburyStatic:
    """Build static buffers from a new-schema ``net`` object (same as topology_new)."""
    required_attrs = [
        "n_net",
        "n_total",
        "n_switch",
        "n_changing_switches",
        "max_switch_terms",
        "base_branch_status",
        "changing_switches",
        "switch_idx",
        "switch_is_closed_base",
        "switch_from_idx",
        "switch_to_idx",
        "switch_valid",
        "switch_y_series",
        "switch_y_shunt",
        "v_scale_full",
        "v_slack_local",
        "Y_scaled_base",
    ]
    missing = [name for name in required_attrs if not hasattr(net, name)]
    if missing:
        raise ValueError(
            "Woodbury JAX requires a new-schema network object; "
            f"missing fields: {missing}"
        )

    n_net = int(net.n_net)
    n_total = int(net.n_total)
    n_switch = int(net.n_switch)
    n_changing_switches = int(net.n_changing_switches)
    m = int(net.max_switch_terms)

    base_branch_status_np = np.asarray(net.base_branch_status, dtype=np.int32).reshape(-1)
    switch_idx_np = np.asarray(net.switch_idx, dtype=np.int32).reshape(-1)
    changing_switches_np = np.asarray(net.changing_switches, dtype=np.int32).reshape(-1)
    changing_switch_row_np = _build_changing_switch_row_map(switch_idx_np, changing_switches_np)
    switch_is_closed_base_np = np.asarray(net.switch_is_closed_base, dtype=np.int32).reshape(-1)
    switch_from_idx_np = np.asarray(net.switch_from_idx, dtype=np.int32)
    switch_to_idx_np = np.asarray(net.switch_to_idx, dtype=np.int32)
    switch_valid_np = np.asarray(net.switch_valid, dtype=bool)
    switch_y_series_np = np.asarray(net.switch_y_series, dtype=np.complex128)
    switch_y_shunt_np = np.asarray(net.switch_y_shunt, dtype=np.complex128)
    v_scale_full_np = np.asarray(net.v_scale_full, dtype=np.float64).reshape(-1)
    v_slack_local_np = np.asarray(net.v_slack_local, dtype=np.complex128).reshape(-1)
    Y_scaled_base_np = np.asarray(net.Y_scaled_base, dtype=np.complex128)
    _validate_static_contracts(
        n_net=n_net,
        n_total=n_total,
        n_switch=n_switch,
        n_changing_switches=n_changing_switches,
        max_switch_terms=m,
        base_branch_status=base_branch_status_np,
        changing_switches=changing_switches_np,
        changing_switch_row=changing_switch_row_np,
        switch_idx=switch_idx_np,
        switch_is_closed_base=switch_is_closed_base_np,
        switch_from_idx=switch_from_idx_np,
        switch_to_idx=switch_to_idx_np,
        switch_valid=switch_valid_np,
        switch_y_series=switch_y_series_np,
        switch_y_shunt=switch_y_shunt_np,
        v_scale_full=v_scale_full_np,
        Y_scaled_base=Y_scaled_base_np,
    )

    # Reuse ``net.Y_scaled_base`` directly to avoid duplicating the
    # (n_total, n_total) complex128 buffer on GPU. ``np.asarray`` is only
    # used to obtain a host copy for shape/contract validation above; the
    # final stored tensor keeps the original JAX array reference so the
    # woodbury JIT captures the same buffer that lives on ``net``.
    Y_scaled_base = jnp.asarray(net.Y_scaled_base, dtype=jnp.complex128)
    Y_LL_base = Y_scaled_base[:n_net, :n_net]
    try:
        Y_LL_base_inv = jnp.linalg.solve(
            Y_LL_base, jnp.eye(n_net, dtype=jnp.complex128)
        )
    except Exception as exc:  # pragma: no cover - jax raises on singular in some builds
        raise ValueError("Base Y_LL is singular; Woodbury updates cannot be initialized.") from exc
    if not bool(jnp.all(jnp.isfinite(Y_LL_base_inv))):
        raise ValueError("Base Y_LL is singular; Woodbury updates cannot be initialized.")
    # ``Y_LL_base`` is only needed to bootstrap ``Y_LL_base_inv``; the field
    # is dropped from the dataclass (it is a view of ``Y_scaled_base`` and
    # never read after construction).

    return WoodburyStatic(
        n_net=n_net,
        n_total=n_total,
        n_switch=n_switch,
        n_changing_switches=n_changing_switches,
        max_switch_terms=m,
        base_branch_status=jnp.asarray(base_branch_status_np, dtype=np.int32),
        changing_switches=jnp.asarray(changing_switches_np, dtype=np.int32),
        changing_switch_row=jnp.asarray(changing_switch_row_np, dtype=np.int32),
        switch_idx=jnp.asarray(switch_idx_np, dtype=np.int32),
        switch_is_closed_base=jnp.asarray(switch_is_closed_base_np, dtype=np.int32),
        switch_from_idx=jnp.asarray(switch_from_idx_np, dtype=np.int32),
        switch_to_idx=jnp.asarray(switch_to_idx_np, dtype=np.int32),
        switch_valid=jnp.asarray(switch_valid_np, dtype=bool),
        switch_y_series=jnp.asarray(switch_y_series_np, dtype=np.complex128),
        switch_y_shunt=jnp.asarray(switch_y_shunt_np, dtype=np.complex128),
        v_scale_full=jnp.asarray(v_scale_full_np, dtype=np.float64),
        v_slack_local=jnp.asarray(v_slack_local_np, dtype=np.complex128),
        Y_scaled_base=Y_scaled_base,
        Y_LL_base_inv=Y_LL_base_inv,
    )


def _expand_compact(s: WoodburyStatic, compact: jnp.ndarray) -> jnp.ndarray:
    """Scatter compact branch bits onto ``base_branch_status`` copy."""
    # compact: (n_changing_switches,) int32
    full = s.base_branch_status.at[s.changing_switches].set(compact)
    return full


def _switch_delta_f64(s: WoodburyStatic, full_status: jnp.ndarray) -> jnp.ndarray:
    """Closed minus base per switch row; shape ``(n_switch,)`` float64. Values in {-1, 0, 1}."""
    return (
        full_status[s.switch_idx].astype(jnp.float64)
        - s.switch_is_closed_base.astype(jnp.float64)
    )


def _changing_switch_delta_f64(s: WoodburyStatic, full_status: jnp.ndarray) -> jnp.ndarray:
    """Closed minus base on the compact changing-switch subset.

    Equivalent to ``_switch_delta_f64(s, full_status)[s.changing_switch_row]``
    because ``switch_idx[changing_switch_row] == changing_switches`` (validated
    at construction), but indexed directly to avoid materializing the full
    ``(n_switch,)`` delta when only the compact subset is needed.
    """
    return (
        full_status[s.changing_switches].astype(jnp.float64)
        - s.switch_is_closed_base[s.changing_switch_row].astype(jnp.float64)
    )


def _switches_moved(s: WoodburyStatic, full_status: jnp.ndarray) -> jnp.ndarray:
    """Scalar bool: any compact changing-switch differs from its mapped switch-catalog row."""
    if s.n_changing_switches == 0:
        return jnp.array(False, dtype=jnp.bool_)
    return jnp.any(_changing_switch_delta_f64(s, full_status) != 0.0)


def _stamp_Y_scaled(s: WoodburyStatic, full_status: jnp.ndarray) -> jnp.ndarray:
    """Apply switch deltas onto ``Y_scaled_base``."""
    if s.n_switch == 0:
        return s.Y_scaled_base

    m = s.max_switch_terms
    delta = _switch_delta_f64(s, full_status)
    no_change = jnp.all(delta == 0)

    def stamp_body(Y: jnp.ndarray) -> jnp.ndarray:
        r = jnp.arange(m * m) // m
        c = jnp.arange(m * m) % m
        
        valid = s.switch_valid[:, r] & s.switch_valid[:, c]
        coeff = delta[:, None] * valid.astype(jnp.float64)
        
        i_from = s.switch_from_idx[:, r]
        i_to = s.switch_to_idx[:, r]
        j_from = s.switch_from_idx[:, c]
        j_to = s.switch_to_idx[:, c]
        
        y_series = s.switch_y_series[:, r, c]
        y_shunt = s.switch_y_shunt[:, r, c]
        
        self_val = y_series + 0.5 * y_shunt
        mutual_val = -y_series
        
        scale_ff = s.v_scale_full[i_from] * s.v_scale_full[j_from]
        scale_tt = s.v_scale_full[i_to] * s.v_scale_full[j_to]
        scale_ft = s.v_scale_full[i_from] * s.v_scale_full[j_to]
        scale_tf = s.v_scale_full[i_to] * s.v_scale_full[j_from]
        
        upd_ff = coeff * self_val * scale_ff
        upd_tt = coeff * self_val * scale_tt
        upd_ft = coeff * mutual_val * scale_ft
        upd_tf = coeff * mutual_val * scale_tf
        
        # We need to accumulate updates carefully since multiple indices might target the same location.
        # We can use flattened indices to use `add` correctly over the entire array
        i_from_flat = i_from.flatten()
        j_from_flat = j_from.flatten()
        i_to_flat = i_to.flatten()
        j_to_flat = j_to.flatten()
        
        Ycur = Y.at[i_from_flat, j_from_flat].add(upd_ff.flatten())
        Ycur = Ycur.at[i_to_flat, j_to_flat].add(upd_tt.flatten())
        Ycur = Ycur.at[i_from_flat, j_to_flat].add(upd_ft.flatten())
        Ycur = Ycur.at[i_to_flat, j_from_flat].add(upd_tf.flatten())
        
        return Ycur

    return jax.lax.cond(no_change, lambda: s.Y_scaled_base, lambda: stamp_body(s.Y_scaled_base))


def _build_one_switch_uv(
    s: WoodburyStatic,
    switch_row: jnp.ndarray,
    delta_switch: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Low-rank factors for one switch's admittance perturbation.

    For full switch-catalog row ``switch_row``, the stamp in ``_stamp_Y_scaled`` adds
    ``δ_s * (y_series + ½ y_shunt)`` on ff/tt diagonals and ``δ_s * (-y_series)`` on
    ft/tf off-diagonals (with per-bus ``v_scale`` products). That update is written as

        ΔY_s = δ_s · B @ M @ B^T   with   B = [A | F | T]   (each block n_net × m),

    then factored for Woodbury as ``U_s V_s^T`` with ``U_s = δ_s B`` and ``V_s = B M^T``.
    Shapes: ``U_s``, ``V_s`` are ``(n_net, 3*m)``.
    """
    m = s.max_switch_terms
    n_net = s.n_net

    # δ_s ∈ {-1, 0, +1}: closed minus base for this switch (0 ⇒ no rank-1/2 update).
    d_s = delta_switch
    from_idx = s.switch_from_idx[switch_row]  # (m,) bus index per switch term (phase/port)
    to_idx = s.switch_to_idx[switch_row]
    valid = s.switch_valid[switch_row]  # (m,) which terms are physically present

    # B = [A | F | T]: three incidence-like blocks, one column per switch term r.
    # Column r stamps scaled bus rows; M couples columns (y_series / y_shunt weights).
    A = jnp.zeros((n_net, m), dtype=jnp.complex128)
    F = jnp.zeros((n_net, m), dtype=jnp.complex128)
    T = jnp.zeros((n_net, m), dtype=jnp.complex128)

    r = jnp.arange(m)
    vr = valid
    i_f = from_idx
    i_t = to_idx

    # Per-term voltage scaling (same factors as scale_ff, scale_tt in _stamp_Y_scaled).
    s_f = jnp.where((i_f >= 0) & (i_f < n_net), s.v_scale_full[i_f], 0.0)
    s_t = jnp.where((i_t >= 0) & (i_t < n_net), s.v_scale_full[i_t], 0.0)
    # Clip indices only for .at[] addressing; contributions gated by mf/mt below.
    i_fs = jnp.clip(i_f, 0, n_net - 1)
    i_ts = jnp.clip(i_t, 0, n_net - 1)
    mf = vr & (i_f >= 0) & (i_f < n_net)
    mt = vr & (i_t >= 0) & (i_t < n_net)

    # Build A, F, T: column r is unique, so no column collisions; .at[].add() sums duplicate rows.
    #   A: +scale at from, −scale at to  (difference / series path in B @ M @ B^T)
    #   F: +scale at from only           (from shunt half in pi model)
    #   T: +scale at to only             (to shunt half in pi model)
    A = A.at[i_fs, r].add(jnp.where(mf, s_f, 0.0))
    A = A.at[i_ts, r].add(jnp.where(mt, -s_t, 0.0))
    F = F.at[i_fs, r].add(jnp.where(mf, s_f, 0.0))
    T = T.at[i_ts, r].add(jnp.where(mt, s_t, 0.0))

    # Admittance weights for term pairs (r, c); invalid pairs zeroed (matches stamp valid mask).
    mask = jnp.outer(valid, valid).astype(jnp.complex128)
    y_series = s.switch_y_series[switch_row] * mask  # (m, m)
    y_shunt = s.switch_y_shunt[switch_row] * mask
    B = jnp.concatenate([A, F, T], axis=1)  # (n_net, 3m)

    # Block-diagonal M: A-block ↔ series; F/T blocks ↔ ½ shunt (self ff/tt in stamping).
    M = jnp.zeros((3 * m, 3 * m), dtype=jnp.complex128)
    M = M.at[:m, :m].set(y_series)
    M = M.at[m : 2 * m, m : 2 * m].set(0.5 * y_shunt)
    M = M.at[2 * m :, 2 * m :].set(0.5 * y_shunt)

    # ΔY_s = δ_s B M B^T = U_s V_s^T; Woodbury needs U, V separately (K = I + V^T A^{-1} U).
    # If δ_s = 0, zero both factors so inactive columns do not pollute the inner solve.
    active = jnp.abs(d_s) > 0
    V_s = jnp.where(active, B @ M.T, jnp.zeros_like(B))
    U_s = jnp.where(active, d_s * B, jnp.zeros_like(B))
    return U_s, V_s


def _build_lowrank_padded(s: WoodburyStatic, full_status: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Stack padded U, V of shape ``(n_net, n_changing_switches * 3 * m)``.

    Only called from ``_inverse_woodbury_jax`` after ``_switches_moved`` is True, so there is
    no separate “no delta” short-circuit here (that case skips Woodbury entirely).
    """
    n_chg = s.n_changing_switches
    m = s.max_switch_terms
    if n_chg == 0:
        return (
            jnp.zeros((s.n_net, 0), dtype=jnp.complex128),
            jnp.zeros((s.n_net, 0), dtype=jnp.complex128),
        )

    delta = _changing_switch_delta_f64(s, full_status)

    def one_slot(j: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        switch_row = s.changing_switch_row[j]
        return _build_one_switch_uv(s, switch_row, delta[j])

    idx = jnp.arange(n_chg, dtype=jnp.int32)
    U_blk, V_blk = jax.vmap(one_slot)(idx)
    U = U_blk.transpose(1, 0, 2).reshape(s.n_net, n_chg * 3 * m)
    V = V_blk.transpose(1, 0, 2).reshape(s.n_net, n_chg * 3 * m)
    return _mask_active_lowrank_columns(U, V)


def _mask_active_lowrank_columns(
    U: jnp.ndarray, V: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Zero padded columns that do not participate in both U and V.

    Woodbury uses ``K = I + V^T A^{-1} U``; columns with ``U_j != 0`` and ``V_j = 0`` (or the
    reverse) reconstruct ``U V^T`` correctly but corrupt the inner solve.

    Kept as a cheap safety guard: a probe over 240/906-bus found it is currently a
    no-op (no valid-but-zero-admittance switch term), but it costs O(n_net·k) <<
    O(n_net²·k) Woodbury inverse and protects future grids.
    """
    u_norm = jnp.linalg.norm(U, axis=0)
    v_norm = jnp.linalg.norm(V, axis=0)
    col_active = (u_norm > 0) & (v_norm > 0)
    return U * col_active, V * col_active


def _woodbury_k_matrix(
    s: WoodburyStatic,
    U: jnp.ndarray,
    V: jnp.ndarray,
) -> jnp.ndarray:
    """Woodbury core ``K = I + V^T A^{-1} U`` with ``A`` the base ``Y_LL``."""
    a_inv_u = s.Y_LL_base_inv @ U
    k = U.shape[1]
    return jnp.eye(k, dtype=jnp.complex128) + (V.T @ a_inv_u)


def _woodbury_solve(
    s: WoodburyStatic,
    U: jnp.ndarray,
    V: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Apply ``(A + U V^T)^{-1} @ rhs`` via Sherman--Morrison--Woodbury (``A = Y_LL`` base).

    For vector or thin-matrix ``rhs`` with ``p << n_net``. Do not pass ``eye`` here;
    use :func:`_woodbury_inverse` to form the full dense inverse without ``A^{-1} @ I``.
    """
    if U.shape[1] == 0:
        return s.Y_LL_base_inv @ rhs
    a_inv_rhs = s.Y_LL_base_inv @ rhs
    a_inv_u = s.Y_LL_base_inv @ U
    k_mat = _woodbury_k_matrix(s, U, V)
    middle = jnp.linalg.solve(k_mat, V.T @ a_inv_rhs)
    return a_inv_rhs - (a_inv_u @ middle)


def _woodbury_inverse(
    s: WoodburyStatic,
    U: jnp.ndarray,
    V: jnp.ndarray,
) -> jnp.ndarray:
    """Dense ``(A + U V^T)^{-1}`` without ``Y_LL_base_inv @ eye`` (avoids O(n^3) multiply).

    ``(A + U V^T)^{-1} = A^{-1} - A^{-1} U (I + V^T A^{-1} U)^{-1} V^T A^{-1}``.
    """
    if U.shape[1] == 0:
        return s.Y_LL_base_inv
    a_inv_u = s.Y_LL_base_inv @ U
    k_mat = _woodbury_k_matrix(s, U, V)
    vt_a_inv = V.T @ s.Y_LL_base_inv
    return s.Y_LL_base_inv - a_inv_u @ jnp.linalg.solve(k_mat, vt_a_inv)


def _inverse_residual(Y_LL: jnp.ndarray, y_inv: jnp.ndarray) -> jnp.ndarray:
    """Scalar ``max |Y_LL @ y_inv - I|`` without materializing identity."""
    n = Y_LL.shape[0]
    m = Y_LL @ y_inv
    m = m.at[jnp.arange(n), jnp.arange(n)].add(-1.0)
    return jnp.max(jnp.abs(m))


def _smw_ir_residual(Y_LL: jnp.ndarray, y_inv: jnp.ndarray) -> jnp.ndarray:
    """``I - Y_LL @ y_inv`` without materializing identity."""
    n = Y_LL.shape[0]
    prod = Y_LL @ y_inv
    return (-prod).at[jnp.arange(n), jnp.arange(n)].add(1.0)


def _inverse_woodbury_jax(
    s: WoodburyStatic, full_status: jnp.ndarray, Y_LL: jnp.ndarray
) -> jnp.ndarray:
    """Woodbury inverse with up to two SMW-IR steps when ``cond(K)`` is large.

    Entry gate uses ``cond(K)`` (k×k, no extra ``Y_LL @ y_inv``). After SMW-IR, the
    second check uses ``max |Y y_inv - I|`` because ``K`` is unchanged by the correction.
    No switch movement returns ``Y_LL_base_inv`` exactly.
    """
    def return_base(_: None) -> jnp.ndarray:
        return s.Y_LL_base_inv

    def woodbury_with_ir(_: None) -> jnp.ndarray:
        u_mat, v_mat = _build_lowrank_padded(s, full_status)
        y_inv = _woodbury_inverse(s, u_mat, v_mat)
        # cond(K) entry gate for SMW-IR (k×k, cheaper than full Y Y^{-1} check).
        # Static guard: jnp.linalg.cond rejects empty (0, 0) matrices; the
        # _switches_moved branch above normally short-circuits n_changing==0,
        # but keep the guard to mirror the old _k_matrix_gate behavior.
        if u_mat.shape[1] == 0:
            need_ir = jnp.array(False)
        else:
            need_ir = (
                jnp.linalg.cond(_woodbury_k_matrix(s, u_mat, v_mat))
                > _K_COND_THRESH
            )

        def after_first_ir(y0: jnp.ndarray) -> jnp.ndarray:
            # One SMW-IR step: y1 = y0 + (Y_LL)^{-1} (I - Y_LL y0) via Woodbury solve.
            y1 = y0 + _woodbury_solve(s, u_mat, v_mat, _smw_ir_residual(Y_LL, y0))
            res1 = _inverse_residual(Y_LL, y1)
            return jax.lax.cond(
                res1 > _INVERSE_RESIDUAL_TOL,
                lambda y1_in: y1_in
                + _woodbury_solve(s, u_mat, v_mat, _smw_ir_residual(Y_LL, y1_in)),
                lambda y1_in: y1_in,
                y1,
            )

        return jax.lax.cond(
            need_ir,
            after_first_ir,
            lambda y0: y0,
            y_inv,
        )

    return jax.lax.cond(
        _switches_moved(s, full_status),
        woodbury_with_ir,
        return_base,
        None,
    )


def _fixed_topology_matrices(
    s: WoodburyStatic,
    compact: jnp.ndarray,
    *,
    slim: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Base topology when ``n_changing_switches == 0`` (``compact`` ignored).

    ``slim=True`` returns ``Y_0row`` (slack rows only); ``slim=False`` returns the
    full ``Y_scaled`` buffer. The remaining blocks (``Y_L0``, ``Y_LL_inv``, ``w``)
    are identical for both paths since no switch perturbs the base admittance.
    """
    del compact
    n_net, n_total = s.n_net, s.n_total
    first = s.Y_scaled_base[n_net:n_total, :] if slim else s.Y_scaled_base
    Y_L0 = s.Y_scaled_base[:n_net, n_net:n_total]
    Y_LL_inv = s.Y_LL_base_inv
    w = -(Y_LL_inv @ (Y_L0 @ s.v_slack_local))
    return first, Y_LL_inv, w, s.base_branch_status


def woodbury_topology_matrices(
    s: WoodburyStatic,
    compact: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure JAX: compact status (n_changing_switches,) int32 -> topology tensors.

    Returns
    -------
    Y_scaled, Y_LL_inv, w, full_status

    Stamps the full ``(n_total, n_total)`` admittance matrix. Prefer
    :func:`woodbury_topology_matrices_slim` in training/eval batch paths on large
    feeders to avoid materializing ``Y_scaled`` on device.
    """
    compact = jnp.asarray(compact, dtype=jnp.int32).reshape(-1)
    full_status = _expand_compact(s, compact)
    Y_scaled = _stamp_Y_scaled(s, full_status)
    Y_LL = Y_scaled[: s.n_net, : s.n_net]
    Y_L0 = Y_scaled[: s.n_net, s.n_net : s.n_total]
    Y_LL_inv = _inverse_woodbury_jax(s, full_status, Y_LL)
    w = -(Y_LL_inv @ (Y_L0 @ s.v_slack_local))
    return Y_scaled, Y_LL_inv, w, full_status


def woodbury_topology_matrices_slim(
    s: WoodburyStatic,
    compact: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Slim production path: stamp only ``Y_0row``, ``Y_L0``, ``Y_LL`` blocks.

    Returns
    -------
    Y_0row, Y_LL_inv, w, full_status

    ``TopologyData`` only needs ``Y_0row`` (not full ``Y_scaled``). This avoids
    an extra dense ``(n_total, n_total)`` buffer per batch row that the uncached
    ``lax.map`` batch path used to stack on GPU.
    """
    # Lazy: ``slim_topology`` imports this module (cycle if hoisted).
    from core.topology.slim import (
        _stamp_Y_LL_slim,
        _stamp_Y_L0_slim,
        _stamp_Y_0row_slim,
    )

    compact = jnp.asarray(compact, dtype=jnp.int32).reshape(-1)
    full_status = _expand_compact(s, compact)
    Y_0row = _stamp_Y_0row_slim(s, full_status)
    Y_L0 = _stamp_Y_L0_slim(s, full_status)
    Y_LL = _stamp_Y_LL_slim(s, full_status)
    Y_LL_inv = _inverse_woodbury_jax(s, full_status, Y_LL)
    w = -(Y_LL_inv @ (Y_L0 @ s.v_slack_local))
    return Y_0row, Y_LL_inv, w, full_status


def _woodbury_inverse_with_ir(
    s: WoodburyStatic,
    full_status: jnp.ndarray,
    ir_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Stamp ``Y_LL`` (branchless), build low-rank factors, Woodbury inverse + IR corrections.

    Shared core of the branchless paths. Returns ``(Y_LL, y_inv, U, V)`` so each
    caller can stamp only the blocks it needs (``Y_0row`` / ``Y_L0``) and discard
    the rest. ``ir_steps`` is a static int (XLA unrolls the loop at trace time).

    Lazy import of ``_stamp_Y_LL_slim_bl`` breaks the ``woodbury <-> slim`` cycle.
    """
    from core.topology.slim import _stamp_Y_LL_slim_bl

    Y_LL = _stamp_Y_LL_slim_bl(s, full_status)
    U, V = _build_lowrank_padded(s, full_status)
    y_inv = _woodbury_inverse(s, U, V)
    for _ in range(ir_steps):
        y_inv = y_inv + _woodbury_solve(s, U, V, _smw_ir_residual(Y_LL, y_inv))
    return Y_LL, y_inv, U, V


def woodbury_topology_matrices_slim_branchless(
    s: WoodburyStatic,
    compact: jnp.ndarray,
    *,
    ir_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Branchless slim topology: cond-free stamp + cond-free Woodbury with fixed IR steps.

    Removes all ``lax.cond`` predicates so the body is fully vmap-able:

    - **Stamp:** always runs the scatter-add. When no switch moved, ``delta`` is
      all-zero so every update contributes zero and the base block is returned
      unchanged (no ``_switches_moved`` cond needed).
    - **Woodbury inverse:** always builds ``U, V`` and applies
      ``A^{-1} - A^{-1} U (I + V^T A^{-1} U)^{-1} V^T A^{-1}``. When no switch
      moved, ``U = V = 0`` so the formula returns ``A^{-1}`` exactly.
    - **SMW-IR:** applies exactly ``ir_steps`` correction steps (static int so
      XLA unrolls at trace time). 0 = skip IR (fastest), 1 = one correction,
      2 = two corrections.
    """
    from core.topology.slim import (
        _stamp_Y_L0_slim_bl,
        _stamp_Y_0row_slim_bl,
    )

    compact = jnp.asarray(compact, dtype=jnp.int32).reshape(-1)
    full_status = _expand_compact(s, compact)
    Y_0row = _stamp_Y_0row_slim_bl(s, full_status)
    Y_L0 = _stamp_Y_L0_slim_bl(s, full_status)
    _, y_inv, _, _ = _woodbury_inverse_with_ir(s, full_status, ir_steps)
    w = -(y_inv @ (Y_L0 @ s.v_slack_local))
    return Y_0row, y_inv, w, full_status


def _resid_n_ir(s: WoodburyStatic, compact: jnp.ndarray, n_ir: int) -> jnp.ndarray:
    """Scalar inverse residual after ``n_ir`` SMW-IR steps (probe kernel for resolve_ir_steps)."""
    full_status = _expand_compact(s, compact)
    Y_LL, y_inv, _, _ = _woodbury_inverse_with_ir(s, full_status, n_ir)
    return _inverse_residual(Y_LL, y_inv)


class WoodburyTopologyBackend:
    """Switch-topology backend: JAX Woodbury (JIT + vmap/map over batch).

    Batch dispatch is controlled by ``batch_mode``:
    - ``"vmap"`` (default): ``jax.vmap`` of the branchless slim fn — fast, higher
      VRAM. Used for all supported grids.
    - ``"map"``: ``jax.lax.map`` of the branchless slim fn — lower VRAM (sequential).
      Fallback when vmap causes GPU memory pressure.

    The slim fn is selected by ``ir_steps``:
    - Explicit ``0``/``1``/``2``: branchless fn with exactly that many SMW-IR
      corrections (static int, XLA unrolls).
    - ``"auto"``: adaptive fn (with ``cond(K)`` gate) until :meth:`resolve_ir_steps`
      is called, then swaps to the resolved branchless variant.
    """

    def __init__(
        self,
        net: Any,
        *,
        ir_steps: int | str = "auto",
        batch_mode: str = "vmap",
    ) -> None:
        self.net = net
        self.static = woodbury_static_from_net(net)
        bbs = np.asarray(net.base_branch_status, dtype=np.int32).reshape(-1)
        swl = np.asarray(net.changing_switches, dtype=np.int32).reshape(-1)
        self.base_compact_status = bbs[swl].astype(np.int32, copy=False)
        self.n_changing_switches = self.static.n_changing_switches

        # Normalize ir_steps: accept "auto", "0", "1", "2" (str from CLI) or int.
        if isinstance(ir_steps, str) and ir_steps in ("0", "1", "2"):
            ir_steps = int(ir_steps)
        self.ir_steps = ir_steps
        self.batch_mode = batch_mode
        self._resolved_ir_steps: Optional[int] = None

        # Select slim fn based on ir_steps.
        if self.static.n_changing_switches == 0:
            self._fn_slim = functools.partial(_fixed_topology_matrices, slim=True)
            self._fn_full = functools.partial(_fixed_topology_matrices, slim=False)
            self._resolved_ir_steps = 0
        elif isinstance(ir_steps, int) and ir_steps in (0, 1, 2):
            self._fn_full = woodbury_topology_matrices
            self._fn_slim = functools.partial(
                woodbury_topology_matrices_slim_branchless, ir_steps=ir_steps
            )
            self._resolved_ir_steps = ir_steps
        else:
            # "auto" — keep adaptive fn until resolve_ir_steps() swaps it.
            self._fn_full = woodbury_topology_matrices
            self._fn_slim = woodbury_topology_matrices_slim

        self._topology_matrices_jit = jax.jit(self._fn_slim)

        # vmap batch JIT — built when branchless fn is selected (explicit or resolved).
        self.batch_vmap_jit: Optional[Callable] = None
        if self._resolved_ir_steps is not None and self.static.n_changing_switches > 0:
            self.batch_vmap_jit = self._make_batch_vmap_jit()

        # ``self.static`` is the first traced arg; the per-row lambda derives its
        # ``s`` from that tracer, so the array fields are never captured constants.
        self.batch_matrices_jit = jax.jit(
            lambda s, status_batch: jax.lax.map(
                lambda c: self._topology_matrices_jit(s, c), status_batch
            )
        )

    def resolve_ir_steps(
        self,
        unique_compacts: np.ndarray,
        tol: float = 1e-6,
    ) -> int:
        """One-time residual probe: pick min IR steps (0/1/2) for accuracy.

        JITs a single-compact residual probe and evaluates it sequentially over
        unique training compacts (peak memory ~ one topology). Picks the minimum
        of {0, 1, 2} such that all residuals < ``tol``, then swaps ``_fn_slim``
        to the resolved branchless variant and rebuilds JITs.

        Returns the resolved step count (0, 1, or 2).
        """
        if self._resolved_ir_steps is not None:
            return self._resolved_ir_steps

        if self.static.n_changing_switches == 0:
            self._resolved_ir_steps = 0
            return 0

        s = self.static
        compacts = np.asarray(unique_compacts, dtype=np.int32)
        if compacts.ndim == 1:
            compacts = compacts.reshape(1, -1)

        resid0_jit = jax.jit(functools.partial(_resid_n_ir, n_ir=0))
        resid1_jit = jax.jit(functools.partial(_resid_n_ir, n_ir=1))

        # Check r0 (no IR) first; stop as soon as one compact fails.
        all_ok = True
        for row in compacts:
            compact = jnp.asarray(row, dtype=jnp.int32)
            if float(jax.device_get(resid0_jit(s, compact))) >= tol:
                all_ok = False
                break

        if all_ok:
            resolved = 0
        else:
            all_ok = True
            for row in compacts:
                compact = jnp.asarray(row, dtype=jnp.int32)
                if float(jax.device_get(resid1_jit(s, compact))) >= tol:
                    all_ok = False
                    break
            resolved = 1 if all_ok else 2

        self._resolved_ir_steps = resolved
        self.ir_steps = resolved
        self._fn_slim = functools.partial(
            woodbury_topology_matrices_slim_branchless, ir_steps=resolved
        )
        self._topology_matrices_jit = jax.jit(self._fn_slim)
        self.batch_vmap_jit = self._make_batch_vmap_jit()
        return resolved

    def _make_batch_vmap_jit(self) -> Callable:
        """JIT'd vmap of the current slim fn over a status batch.

        Shared by :meth:`__init__` (explicit/resolved ir_steps) and
        :meth:`resolve_ir_steps` so both build the identical closure.
        """
        return jax.jit(
            lambda s, status_batch: jax.vmap(
                lambda c: self._fn_slim(s, c)
            )(status_batch)
        )

    def normalize_compact_status(
        self, status: Any, sample_index: Optional[int] = None
    ) -> np.ndarray:
        arr = np.asarray(status, dtype=np.int32).reshape(-1)
        if arr.shape[0] != self.n_changing_switches:
            raise ValueError(
                "Compact-only contract violated: branch_status width must be n_changing_switches "
                f"({self.n_changing_switches}), got {arr.shape[0]}."
            )
        if np.any((arr != 0) & (arr != 1)):
            sample_msg = f" sample={sample_index}" if sample_index is not None else ""
            raise ValueError(
                "Compact branch_status must be binary {0,1}."
                f"{sample_msg}, values={arr.tolist()}"
            )
        return arr

    def normalize_compact_status_matrix(self, statuses: Any) -> np.ndarray:
        arr = np.asarray(statuses, dtype=np.int32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Compact branch_status matrix must be 2D; got shape {arr.shape}.")
        if arr.shape[1] != self.n_changing_switches and arr.shape[0] == self.n_changing_switches:
            arr = arr.T
        if arr.shape[1] != self.n_changing_switches:
            raise ValueError(
                "Compact-only contract violated: branch_status width must be n_changing_switches "
                f"({self.n_changing_switches}), got {arr.shape[1]}."
            )
        invalid_mask = (arr != 0) & (arr != 1)
        if np.any(invalid_mask):
            bad_row = int(np.argmax(np.any(invalid_mask, axis=1)))
            raise ValueError(
                "Compact branch_status matrix must be binary {0,1}; "
                f"first invalid sample index={bad_row}, values={arr[bad_row].tolist()}"
            )
        return arr

    def topology_matrices(
        self, compact_status: Any, sample_index: Optional[int] = None
    ) -> Tuple[jnp.ndarray, ...]:
        compact = self.normalize_compact_status(compact_status, sample_index=sample_index)
        return self._topology_matrices_jit(
            self.static, jnp.asarray(compact, dtype=jnp.int32)
        )

    def topology_matrices_eager(
        self, compact_status: Any, sample_index: Optional[int] = None
    ) -> Tuple[jnp.ndarray, ...]:
        compact = self.normalize_compact_status(compact_status, sample_index=sample_index)
        return self._fn_slim(self.static, jnp.asarray(compact, dtype=jnp.int32))

    def topology_matrices_eager_full(
        self, compact_status: Any, sample_index: Optional[int] = None
    ) -> Tuple[jnp.ndarray, ...]:
        """Eager full ``Y_scaled`` path (tests / diagnostics only)."""
        compact = self.normalize_compact_status(compact_status, sample_index=sample_index)
        return self._fn_full(self.static, jnp.asarray(compact, dtype=jnp.int32))

    def topology_matrices_batch(self, compact_status_batch: Any) -> Tuple[jnp.ndarray, ...]:
        status_batch = self.normalize_compact_status_matrix(compact_status_batch)
        return self.batch_matrices_jit(
            self.static, jnp.asarray(status_batch, dtype=jnp.int32)
        )

    def topology_data(self, compact_status: Any, sample_index: Optional[int] = None):
        # ``_fn_slim`` always returns the slim ``Y_0row`` as its first element
        # (fixed/branchless/adaptive branches all produce the slack-row slice),
        # so the ``matrices_first_to_y_0row`` normalizer is a no-op here.
        Y_0row, Y_LL_inv, w, full_status = self.topology_matrices(
            compact_status, sample_index=sample_index
        )
        return make_topology_data(
            Y_0row=Y_0row,
            Y_LL_inv=Y_LL_inv,
            w=w,
            v_init=w,
            branch_status=full_status,
        )

    def topology_data_batch(self, compact_status_batch: Any):
        Y_0row, Y_LL_inv, w, full_status = self.topology_matrices_batch(
            compact_status_batch
        )
        return make_topology_data(
            Y_0row=Y_0row,
            Y_LL_inv=Y_LL_inv,
            w=w,
            v_init=w,
            branch_status=full_status,
        )