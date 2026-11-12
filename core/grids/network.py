"""Static feeder physics: ``NetworkParams`` and its builder from ``network_config.json``.

Julia indices in the JSON are converted to 0-based in ``_idx_array_0``.
See ``docs/architecture.md``.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from core.grids.power_flow import PowerFlowSolver

# obj_weights layout (HDF5 ``input/obj_weights``), mirrors Julia ``OBJ_WEIGHT_NAMES``:
# [w_cost, w_curtail, ρ_V, ρ_I, ρ_θ, ρ_sub]; the ρ entries weight the soft-limit penalty families.
OBJ_WEIGHT_NAMES = ("w_cost", "w_curtail", "rho_v", "rho_i", "rho_theta", "rho_sub")
N_ECON_WEIGHTS = 2
N_OBJ_WEIGHTS = len(OBJ_WEIGHT_NAMES)


def _complex_array(value, *, dtype=np.complex128) -> np.ndarray:
    """Parse JSON arrays that may store complex numbers as ``{re, im}``."""

    def convert(x):
        if isinstance(x, dict) and "re" in x and "im" in x:
            return complex(float(x["re"]), float(x["im"]))
        if isinstance(x, list):
            return [convert(v) for v in x]
        return x

    return np.asarray(convert(value), dtype=dtype)


def _real_array(value, *, dtype=np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _idx_array_0(value) -> np.ndarray:
    """Convert Julia 1-based positive indices to Python 0-based indices."""

    arr = np.asarray(value, dtype=np.int32)
    return np.where(arr > 0, arr - 1, -1).astype(np.int32)


def _pad_vectors_0(vectors, *, width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if width is None:
        width = max((len(v) for v in vectors), default=0)
    width = max(width, 1)
    padded = np.zeros((len(vectors), width), dtype=np.int32)
    mask = np.zeros((len(vectors), width), dtype=bool)
    for i, vec in enumerate(vectors):
        arr = _idx_array_0(vec)
        n = min(len(arr), width)
        if n:
            padded[i, :n] = np.maximum(arr[:n], 0)
            mask[i, :n] = arr[:n] >= 0
    return padded, mask


def _pad_complex_matrices(mats, width: int) -> np.ndarray:
    width = max(width, 1)
    out = np.zeros((len(mats), width, width), dtype=np.complex128)
    for i, mat in enumerate(mats):
        arr = _complex_array(mat)
        r = min(arr.shape[0], width)
        c = min(arr.shape[1], width)
        out[i, :r, :c] = arr[:r, :c]
    return out


def _pad_index_groups(groups, width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if width is None:
        width = max((len(g) for g in groups), default=0)
    width = max(width, 1)
    padded = np.zeros((len(groups), width), dtype=np.int32)
    mask = np.zeros((len(groups), width), dtype=bool)
    for i, group in enumerate(groups):
        arr = _idx_array_0(group)
        n = min(len(arr), width)
        if n:
            padded[i, :n] = np.maximum(arr[:n], 0)
            mask[i, :n] = arr[:n] >= 0
    return padded, mask


def _linear_quadratic_terms(coeff):
    coeff = jnp.asarray(coeff)
    zero = jnp.asarray(0.0, dtype=coeff.dtype)
    if coeff.shape[0] >= 2:
        c_quad = coeff[0]
        c_lin = coeff[1]
    elif coeff.shape[0] == 1:
        c_lin = coeff[0]
        c_quad = zero
    else:
        c_lin = zero
        c_quad = zero
    return c_lin, c_quad


def _parse_Y_L0(cfg, n_net: int, n_slk: int, Y_scaled_base: jnp.ndarray) -> jnp.ndarray:
    """Read network–slack block from dataset JSON key ``Y_ns`` (legacy name)."""
    if "Y_ns" not in cfg:
        return Y_scaled_base[:n_net, n_net:]
    Y_L0 = _complex_array(cfg["Y_ns"])
    if Y_L0.shape == (n_net, n_slk):
        return jnp.asarray(Y_L0)
    if Y_L0.shape == (n_slk, n_net):
        return jnp.asarray(Y_L0.T)
    raise ValueError(
        f"Y_ns shape {Y_L0.shape} is incompatible with "
        f"(n_net, n_slk)=({n_net}, {n_slk})."
    )


def _changing_switches_from_config(raw_config: dict) -> jnp.ndarray:
    raw = raw_config.get("changing_switches", [])
    return jnp.asarray(_idx_array_0(raw))


def build_network_params(
    raw_config: dict, *, precompute_inverse: bool = True
) -> "NetworkParams":
    """Build computation-only network parameters from FeederFlow JSON config.

    Args:
        raw_config: Parsed FeederFlow network config.
        precompute_inverse: If True, materialize the dense ``(n_net, n_net)``
            ``Y_LL_inv`` on the returned :class:`NetworkParams` for stamp
            consumers and diagnostic scripts. If False (woodbury default),
            the inverse is computed once inside the topology backend and
            ``NetworkParams.Y_LL_inv`` is left as ``None`` to save a
            ~``n_net^2 * 16``-byte device allocation.
    """
    required = [
        "n_net",
        "n_slk",
        "n_total",
        "n_der_unit",
        "der_node",
        "der_phase_to_unit",
        "Y_scaled_base",
        "v_sub_ref",
        "v_scale_slack",
    ]
    missing = [k for k in required if k not in raw_config]
    if missing:
        raise ValueError(
            "OPFData now supports only the FeederFlow new schema; missing "
            f"required keys: {missing}"
        )

    n_net = int(raw_config["n_net"])
    n_slk = int(raw_config["n_slk"])
    n_total = int(raw_config["n_total"])
    n_der_unit = int(raw_config["n_der_unit"])
    n_der_phase = len(raw_config["der_node"])
    n_branch = len(raw_config.get("branch_status", []))
    n_branch_constraints = len(raw_config.get("branch_from_idx", []))
    n_switch = len(raw_config.get("switch_idx", []))
    changing_switches = _changing_switches_from_config(raw_config)
    n_changing_switches = int(changing_switches.shape[0])

    der_node = jnp.asarray(_idx_array_0(raw_config["der_node"]))
    der_phase_to_unit = jnp.asarray(_idx_array_0(raw_config["der_phase_to_unit"]))

    pdermin = jnp.zeros(n_der_phase, dtype=jnp.float64)
    pdermax = jnp.asarray(_real_array(raw_config["der_pmax"]))
    qdermin = jnp.asarray(_real_array(raw_config["der_qmin"]))
    qdermax = jnp.asarray(_real_array(raw_config["der_qmax"]))
    sdermax = jnp.asarray(_real_array(raw_config["der_smax"]))
    der_pavail_ref = jnp.asarray(_real_array(raw_config["der_pavail_ref"]))

    vmin_full = jnp.asarray(_real_array(raw_config["vmin"]))
    vmax_full = jnp.asarray(_real_array(raw_config["vmax"]))
    vmin = vmin_full[:n_net]
    vmax = vmax_full[:n_net]
    vrmin = -jnp.ones(n_net, dtype=jnp.float64)
    vrmax = jnp.ones(n_net, dtype=jnp.float64)
    vimin = -jnp.ones(n_net, dtype=jnp.float64)
    vimax = jnp.ones(n_net, dtype=jnp.float64)

    dvamax = jnp.asarray(float(raw_config["dvamax"]), dtype=jnp.float64)
    if "sub_smax_phase" not in raw_config:
        raise ValueError(
            "OPFData requires sub_smax_phase in network_config.json "
            "(per-phase substation apparent-power limits)."
        )
    sub_smax_phase = jnp.asarray(_real_array(raw_config["sub_smax_phase"]), dtype=jnp.float64)
    if sub_smax_phase.shape != (n_slk,):
        raise ValueError(
            f"sub_smax_phase length {sub_smax_phase.shape[0]} does not match n_slk={n_slk}."
        )
    ssubmax = sub_smax_phase
    branch_imax = jnp.asarray(_real_array(raw_config.get("branch_imax", [])))
    rate_a = branch_imax

    v_scale = jnp.asarray(_real_array(raw_config["v_scale"]))
    v_scale_full = jnp.asarray(_real_array(raw_config["v_scale_full"]))
    v_scale_slack = jnp.asarray(_real_array(raw_config["v_scale_slack"]))
    v_sub_ref = jnp.asarray(_complex_array(raw_config["v_sub_ref"]))
    v_slack_local = v_sub_ref / v_scale_slack

    Y_scaled_base = jnp.asarray(_complex_array(raw_config["Y_scaled_base"]))
    if Y_scaled_base.shape != (n_total, n_total):
        raise ValueError(
            f"Y_scaled_base shape {Y_scaled_base.shape} does not match "
            f"(n_total, n_total)=({n_total}, {n_total})."
        )
    Y_LL_base = Y_scaled_base[:n_net, :n_net]
    Y_L0_base = _parse_Y_L0(raw_config, n_net, n_slk, Y_scaled_base)
    # ``Y_LL_inv`` is the dense (n_net, n_net) complex inverse; only the
    # stamp backend (and a couple of diagnostic scripts) reads it from
    # ``NetworkParams``. For woodbury the inverse lives on
    # ``WoodburyStatic.Y_LL_base_inv``, so the caller opts out via
    # ``precompute_inverse=False`` to avoid materializing a second copy
    # of the dense inverse. ``w`` is computed via ``solve`` so
    # the opt-out path still avoids the explicit inverse.
    Y_LL_inv = (
        jnp.linalg.inv(Y_LL_base) if precompute_inverse else None
    )
    w = -jnp.linalg.solve(Y_LL_base, Y_L0_base @ v_slack_local)

    branch_status = jnp.asarray(
        raw_config.get("branch_status", [True] * n_branch), dtype=jnp.int32
    )
    base_branch_status = branch_status
    switch_idx = jnp.asarray(_idx_array_0(raw_config.get("switch_idx", [])))
    switch_is_closed_base = jnp.asarray(
        raw_config.get("switch_is_closed_base", []), dtype=jnp.int32
    )

    switch_from_np, switch_valid_from_np = _pad_vectors_0(raw_config.get("switch_from_idx", []))
    switch_to_np, switch_valid_to_np = _pad_vectors_0(
        raw_config.get("switch_to_idx", []), width=switch_from_np.shape[1]
    )
    switch_valid_raw = raw_config.get("switch_valid", [])
    if not switch_valid_raw:
        switch_valid_np = switch_valid_from_np & switch_valid_to_np
    else:
        _, switch_valid_shape_np = _pad_vectors_0(raw_config.get("switch_from_idx", []))
        switch_valid_np = np.zeros_like(switch_valid_shape_np)
        for i, row in enumerate(switch_valid_raw):
            n = min(len(row), switch_valid_np.shape[1])
            switch_valid_np[i, :n] = np.asarray(row[:n], dtype=bool)
        switch_valid_np &= switch_valid_from_np & switch_valid_to_np

    max_switch_terms = int(switch_from_np.shape[1])
    switch_from_idx = jnp.asarray(switch_from_np)
    switch_to_idx = jnp.asarray(switch_to_np)
    switch_valid = jnp.asarray(switch_valid_np)
    switch_y_series = jnp.asarray(
        _pad_complex_matrices(raw_config.get("switch_y_series", []), max_switch_terms)
    )
    switch_y_shunt = jnp.asarray(
        _pad_complex_matrices(raw_config.get("switch_y_shunt", []), max_switch_terms)
    )

    branch_from_np, branch_valid_from_np = _pad_vectors_0(raw_config.get("branch_from_idx", []))
    branch_to_np, branch_valid_to_np = _pad_vectors_0(
        raw_config.get("branch_to_idx", []), width=branch_from_np.shape[1]
    )
    branch_valid_np = np.zeros_like(branch_valid_from_np)
    for i, row in enumerate(raw_config.get("branch_valid", [])):
        n = min(len(row), branch_valid_np.shape[1])
        branch_valid_np[i, :n] = np.asarray(row[:n], dtype=bool)
    branch_valid_np &= branch_valid_from_np & branch_valid_to_np

    max_branch_terms = int(branch_from_np.shape[1])
    branch_from_idx = jnp.asarray(branch_from_np)
    branch_to_idx = jnp.asarray(branch_to_np)
    branch_valid = jnp.asarray(branch_valid_np)
    branch_y_series = jnp.asarray(
        _pad_complex_matrices(raw_config.get("branch_y_series", []), max_branch_terms)
    )

    unit_phase_np, unit_phase_mask_np = _pad_index_groups(raw_config["der_unit_to_phase"])
    node_phase_np, node_phase_mask_np = _pad_index_groups(raw_config["node_to_der_phase"])
    der_unit_to_phase = jnp.asarray(unit_phase_np)
    der_unit_to_phase_mask = jnp.asarray(unit_phase_mask_np)
    node_to_der_phase = jnp.asarray(node_phase_np)
    node_to_der_phase_mask = jnp.asarray(node_phase_mask_np)

    sub_cost = jnp.asarray(_real_array(raw_config["sub_cost"]))
    der_cost = jnp.asarray(_real_array(raw_config["der_cost"]))
    sub_lin, sub_quad = _linear_quadratic_terms(sub_cost)
    der_lin_quad = jax.vmap(_linear_quadratic_terms)(der_cost)
    der_lin = der_lin_quad[0]
    der_quad = der_lin_quad[1]

    lx = jnp.concatenate([pdermin, qdermin])
    ux = jnp.concatenate([pdermax, qdermax])

    return NetworkParams(
        n_net=n_net,
        n_slk=n_slk,
        n_total=n_total,
        n_branch=n_branch,
        n_branch_constraints=n_branch_constraints,
        n_der_phase=n_der_phase,
        n_der_unit=n_der_unit,
        n_switch=n_switch,
        n_changing_switches=n_changing_switches,
        changing_switches=changing_switches,
        max_branch_terms=max_branch_terms,
        max_switch_terms=max_switch_terms,
        der_node=der_node,
        der_phase_to_unit=der_phase_to_unit,
        der_unit_to_phase=der_unit_to_phase,
        der_unit_to_phase_mask=der_unit_to_phase_mask,
        node_to_der_phase=node_to_der_phase,
        node_to_der_phase_mask=node_to_der_phase_mask,
        pdermin=pdermin,
        pdermax=pdermax,
        qdermin=qdermin,
        qdermax=qdermax,
        sdermax=sdermax,
        der_pavail_ref=der_pavail_ref,
        vmin=vmin,
        vmax=vmax,
        vrmin=vrmin,
        vrmax=vrmax,
        vimin=vimin,
        vimax=vimax,
        dvamax=dvamax,
        sub_smax_phase=sub_smax_phase,
        ssubmax=ssubmax,
        branch_imax=branch_imax,
        rate_a=rate_a,
        v_scale=v_scale,
        v_scale_full=v_scale_full,
        v_scale_slack=v_scale_slack,
        v_slack_local=v_slack_local,
        Y_scaled_base=Y_scaled_base,
        Y_LL_inv=Y_LL_inv,
        w=w,
        base_branch_status=base_branch_status,
        branch_status=branch_status,
        switch_idx=switch_idx,
        switch_is_closed_base=switch_is_closed_base,
        switch_from_idx=switch_from_idx,
        switch_to_idx=switch_to_idx,
        switch_valid=switch_valid,
        switch_y_series=switch_y_series,
        switch_y_shunt=switch_y_shunt,
        branch_from_idx=branch_from_idx,
        branch_to_idx=branch_to_idx,
        branch_valid=branch_valid,
        branch_y_series=branch_y_series,
        sub_lin=sub_lin,
        sub_quad=sub_quad,
        der_lin=der_lin,
        der_quad=der_quad,
        lx=lx,
        ux=ux,
        power_flow_solver=PowerFlowSolver.FIXED_POINT.value,
    )


class NetworkParams:
    """Static feeder physics for the FeederFlow node/phase schema.

    Built once from ``network_config.json``; shared across all training samples.
    Provides indexing (DER, branches, switches), scaled admittance ``Y_scaled_base``,
    limit vectors, and topology stamping helpers.

    Use ``compute_net_injection(load, der)`` for nodal injections and
    ``expand_branch_status(compact)`` to scatter HDF5 switch status to full width.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            "NetworkParams("
            f"n_net={self.n_net}, n_slk={self.n_slk}, "
            f"n_branch={self.n_branch}, n_der_phase={self.n_der_phase})"
        )

    def compute_net_injection(self, load, der):
        """Return non-slack nodal complex injection: DER generation minus load."""
        der_to_node = map_to_node(der, self.n_net, self.der_node)
        return der_to_node - load

    def expand_branch_status(self, branch_status):
        """Expand changing switch status to full branch-status width."""
        status = branch_status.astype(jnp.int32)
        if status.shape[-1] == self.n_branch:
            return status
        if status.shape[-1] != self.n_changing_switches:
            raise ValueError(
                f"Unexpected branch_status width {status.shape[-1]}; expected "
                f"{self.n_changing_switches} compact or {self.n_branch} full."
            )

        changing = self.changing_switches
        if status.ndim == 1:
            return self.base_branch_status.at[changing].set(status)

        full = jnp.broadcast_to(
            self.base_branch_status, status.shape[:-1] + (self.n_branch,)
        )
        return full.at[..., changing].set(status)

    def _stamp_switches(self, branch_status):
        """Stamp switch-only topology deltas onto ``Y_scaled_base`` in JAX."""
        full_status = self.expand_branch_status(branch_status)

        def no_switches():
            return self.Y_scaled_base

        def with_switches():
            switch_closed = full_status[self.switch_idx].astype(jnp.float64)
            base_closed = self.switch_is_closed_base.astype(jnp.float64)
            delta = (switch_closed - base_closed).astype(jnp.complex128)
            Y = self.Y_scaled_base
            for r in range(self.max_switch_terms):
                for c in range(self.max_switch_terms):
                    valid = self.switch_valid[:, r] & self.switch_valid[:, c]
                    coeff = delta * valid.astype(jnp.complex128)

                    i_from = self.switch_from_idx[:, r]
                    i_to = self.switch_to_idx[:, r]
                    j_from = self.switch_from_idx[:, c]
                    j_to = self.switch_to_idx[:, c]

                    y_series = self.switch_y_series[:, r, c]
                    y_shunt = self.switch_y_shunt[:, r, c]
                    self_val = y_series + 0.5 * y_shunt
                    mutual_val = -y_series

                    scale_ff = self.v_scale_full[i_from] * self.v_scale_full[j_from]
                    scale_tt = self.v_scale_full[i_to] * self.v_scale_full[j_to]
                    scale_ft = self.v_scale_full[i_from] * self.v_scale_full[j_to]
                    scale_tf = self.v_scale_full[i_to] * self.v_scale_full[j_from]

                    Y = Y.at[i_from, j_from].add(coeff * self_val * scale_ff)
                    Y = Y.at[i_to, j_to].add(coeff * self_val * scale_tt)
                    Y = Y.at[i_from, j_to].add(coeff * mutual_val * scale_ft)
                    Y = Y.at[i_to, j_from].add(coeff * mutual_val * scale_tf)
            return Y

        return jax.lax.cond(self.n_switch == 0, no_switches, with_switches)

    def topology_matrices(self, branch_status):
        """Build per-status scaled admittance blocks and fixed-point terms."""
        Y_scaled = self._stamp_switches(branch_status)
        Y_LL = Y_scaled[: self.n_net, : self.n_net]
        Y_L0 = Y_scaled[: self.n_net, self.n_net : self.n_total]
        Y_LL_inv = jnp.linalg.inv(Y_LL)
        w = -(Y_LL_inv @ (Y_L0 @ self.v_slack_local))
        return Y_scaled, Y_LL_inv, w, self.expand_branch_status(branch_status)

    def topology_matrices_batch(self, branch_status_batch):
        return jax.vmap(self.topology_matrices, in_axes=(0,))(branch_status_batch)


def map_to_node(s, n_node, node_map):
    s_node = jnp.zeros(n_node, dtype=s.dtype)
    return s_node.at[node_map].add(s)
