"""Network data loading: ``NetworkParams``, ``OPFData``, HDF5/JSON I/O.

``NetworkParams`` holds static feeder physics; ``OPFData`` adds session state
(HDF5 paths, slice indexing, topology backend). Julia indices in JSON are
converted to 0-based in ``_idx_array_0``. See ``docs/architecture.md``.
"""
import json
from types import MappingProxyType
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

import h5py

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from core.grids.power_flow import PowerFlowSolver


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


def _build_network_params(
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
    # (saves ~1.16 GiB at 8500 nodes). ``w`` is computed via ``solve`` so
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


@runtime_checkable
class TopologyBackend(Protocol):
    """Switch-topology backend: stamping (reference) or Woodbury (default)."""

    def topology_matrices(self, branch_status) -> Tuple[jnp.ndarray, ...]:
        ...

    def topology_matrices_batch(self, branch_status_batch) -> Tuple[jnp.ndarray, ...]:
        ...


def _make_topology_backend(
    name: str,
    net: NetworkParams,
    *,
    ir_steps: int | str = "auto",
    batch_mode: str = "vmap",
) -> TopologyBackend:
    if name == "woodbury":
        # Lazy: avoid pulling Woodbury JAX stack at ``core.grids.data`` import time.
        from core.topology.woodbury import WoodburyTopologyBackend

        return WoodburyTopologyBackend(
            net,
            ir_steps=ir_steps,
            batch_mode=batch_mode,
        )
    if name == "stamp":
        # Lazy: stamp backend is reference-only; defer unless explicitly selected.
        from core.topology.stamp import StampTopologyBackend

        return StampTopologyBackend(net, use_native_batch=False)
    raise ValueError(
        f"topology_backend must be 'woodbury' or 'stamp', got {name!r}"
    )


class OPFData:
    """Training/eval session: HDF5 I/O, slice layout, and topology backend.

    Wraps a ``NetworkParams`` instance as ``self.net``. Unknown attributes
    delegate to ``net`` (e.g. ``data.n_der_phase``). Construct once per feeder
    case; use ``with_test_h5`` to evaluate alternate test splits sharing the
    same ``network_config.json``.

    Attributes:
        net: Static ``NetworkParams`` for this feeder.
        topology_backend: ``woodbury`` (default) or ``stamp`` switch backend.
        slices: Flat input/target index ranges for TensorFlow batch tensors.
    """

    output_keys = ["pder", "qder", "vr", "vi"]
    meta_keys = ["solve_time"]

    @staticmethod
    def _h5_field_width(dataset) -> int:
        if dataset.ndim == 1:
            return 1
        return int(dataset.shape[1])

    @staticmethod
    def _read_h5_field(
        dataset,
        samples: int | None = None,
        sample_offset: int = 0,
    ) -> np.ndarray:
        """Read an HDF5 dataset into NumPy.

        When ``samples`` is set, read ``samples`` rows starting at ``sample_offset``
        (h5py slice), avoiding a full-file load followed by truncation.
        """
        if sample_offset < 0:
            raise ValueError(f"sample_offset must be non-negative, got {sample_offset}")
        if samples is not None:
            if samples < 0:
                raise ValueError(f"samples must be non-negative, got {samples}")
            if samples == 0:
                raise ValueError("samples must be positive when specified")
            data = dataset[sample_offset : sample_offset + samples]
            if int(np.asarray(data).shape[0]) == 0:
                raise ValueError(
                    f"empty HDF5 slice offset={sample_offset} samples={samples}"
                )
        elif sample_offset > 0:
            data = dataset[sample_offset:]
            if int(np.asarray(data).shape[0]) == 0:
                raise ValueError(f"empty HDF5 slice from offset={sample_offset}")
        else:
            data = dataset
        arr = np.asarray(data)
        if arr.ndim == 1:
            return arr[:, np.newaxis]
        return arr

    @staticmethod
    def _read_input_field(
        dataset,
        samples: int | None = None,
        sample_offset: int = 0,
    ) -> np.ndarray:
        return OPFData._read_h5_field(
            dataset, samples=samples, sample_offset=sample_offset
        )

    @staticmethod
    def _input_keys_from_h5(path: str) -> list[str]:
        base_keys = ["pd", "qd", "pder_avai", "obj_weights"]
        with h5py.File(path, "r") as f:
            input_grp = f["input"]
            keys = list(base_keys)
            if "changing_switch_status" not in input_grp:
                raise ValueError(
                    f"HDF5 {path} missing required input/changing_switch_status."
                )
            keys.append("changing_switch_status")
        return keys

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        n = self.net
        return (
            f"OPFData(n_net={n.n_net}, n_slk={n.n_slk}, "
            f"n_der_phase={n.n_der_phase}, n_branch={n.n_branch})"
        )

    def __getattr__(self, name: str):
        """Delegate unknown attrs to ``net``."""
        try:
            return getattr(self.net, name)
        except AttributeError as err:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from err

    def _read_training_compacts(self) -> np.ndarray:
        """Load all training compact switch statuses from the training HDF5."""
        with h5py.File(self.train_h5_path, "r") as f:
            if "changing_switch_status" not in f["input"]:
                raise ValueError(
                    f"HDF5 {self.train_h5_path} missing required "
                    "input/changing_switch_status."
                )
            compacts = self._read_h5_field(f["input"]["changing_switch_status"])
        return np.asarray(compacts, dtype=np.int32)

    def _read_training_compacts_unique(self) -> np.ndarray:
        """Unique training compact statuses for the IR-step residual probe."""
        return np.unique(self._read_training_compacts(), axis=0)

    def __init__(
        self,
        train_h5_path: str,
        test_h5_path: str,
        json_path: str,
        *,
        topology_backend: str = "woodbury",
        include_topo_nn_input: bool = True,
        woodbury_ir_steps: str = "auto",
        woodbury_batch_mode: str = "vmap",
    ) -> None:
        self.train_h5_path = train_h5_path

        with open(json_path, "r") as f:
            raw_config = json.load(f)

        self.case = raw_config.get("case", "")

        config_view = dict(raw_config)
        # Woodbury backend stores its precomputed ``Y_LL_base_inv`` on
        # ``WoodburyStatic``; the dense ``NetworkParams.Y_LL_inv`` would be
        # a duplicate 1.16 GiB on 8500-node networks. Only the stamp backend
        # (and a couple of diagnostic scripts) needs it on the params.
        precompute_inverse = topology_backend != "woodbury"
        net = _build_network_params(
            raw_config, precompute_inverse=precompute_inverse
        )
        # Drop the heavy admittance matrices already materialized on ``net`` /
        # ``WoodburyStatic`` as jnp arrays. ``json.load`` keeps them as nested
        # Python lists (~5x the on-disk JSON size in RAM); retaining them in
        # ``self.config`` wastes ~35-50 GB on the 8500-bus network (11.8 GB JSON).
        # Only small scalar/index keys are ever read back via ``self.config``
        # (e.g. ``p_load_ref``/``q_load_ref`` in ``build_nn_input_layout``).
        for _heavy_key in ("Y_nn", "Y_ns", "Y_scaled", "Y_scaled_base"):
            config_view.pop(_heavy_key, None)
        del raw_config
        config_view.update(
            {
                "der_node": np.asarray(net.der_node).tolist(),
                "der_phase_to_unit": np.asarray(net.der_phase_to_unit).tolist(),
                "switch_idx": np.asarray(net.switch_idx).tolist(),
                "changing_switches": np.asarray(net.changing_switches).tolist(),
            }
        )
        self.config = MappingProxyType(config_view)
        self.net = net
        self.power_flow_solver = net.power_flow_solver
        self.topology_backend_name = topology_backend

        # Resolve Woodbury IR steps when ``woodbury_ir_steps == "auto"``.
        # Init order: tmp backend → resolve auto → real backend with resolved int.
        if topology_backend == "woodbury":
            from core.topology.woodbury import WoodburyTopologyBackend

            tmp = WoodburyTopologyBackend(
                net, ir_steps=woodbury_ir_steps, batch_mode=woodbury_batch_mode
            )
            if woodbury_ir_steps == "auto":
                tmp.resolve_ir_steps(self._read_training_compacts_unique())
            resolved = tmp._resolved_ir_steps  # int 0/1/2

            self.topology_backend = _make_topology_backend(
                topology_backend,
                net,
                ir_steps=resolved,
                batch_mode=woodbury_batch_mode,
            )
        else:
            self.topology_backend = _make_topology_backend(topology_backend, net)
        self.topology_input_key = "changing_switch_status"
        self.include_topo_nn_input = include_topo_nn_input
        self._bind_test_h5(test_h5_path)
        from core.nn.input import build_nn_input_layout

        self.nn_layout = build_nn_input_layout(self)
        self.use_compact_nn = True

    def with_test_h5(self, test_h5_path: str) -> "OPFData":
        """Return a new session that shares ``net`` and ``topology_backend``.

        Only test HDF5 metadata (``input_keys``, ``slices``, ``has_scenario_index``)
        is rebound. Use this for post-training evaluation across dataset variants
        that share the same ``network_config.json`` and tensor layout.

        ``net`` and ``topology_backend`` are shared by reference; do not mutate them
        after creation. ``power_flow_solver`` is copied from this session (including
        any CLI override already applied).

        Raises:
            ValueError: if the new HDF5 slice layout differs from this session.
        """
        expected_layout = self._slice_widths()
        other = object.__new__(OPFData)
        other.train_h5_path = self.train_h5_path
        other.case = self.case
        other.config = self.config
        other.net = self.net
        other.power_flow_solver = self.power_flow_solver
        other.topology_backend_name = self.topology_backend_name
        other.topology_backend = self.topology_backend
        other.topology_input_key = self.topology_input_key
        other.nn_layout = self.nn_layout
        other.use_compact_nn = self.use_compact_nn
        other.include_topo_nn_input = self.include_topo_nn_input
        other._bind_test_h5(test_h5_path)
        actual_layout = other._slice_widths()
        if actual_layout != expected_layout:
            for group in expected_layout:
                for key, width in expected_layout[group].items():
                    if actual_layout.get(group, {}).get(key) != width:
                        raise ValueError(
                            "Incompatible test HDF5 layout for with_test_h5: "
                            f"group={group!r} key={key!r} expected width {width}, "
                            f"got {actual_layout.get(group, {}).get(key)} "
                            f"(parent={self.test_h5_path!r}, variant={test_h5_path!r})"
                        )
            raise ValueError(
                "Incompatible test HDF5 layout for with_test_h5 "
                f"(parent={self.test_h5_path!r}, variant={test_h5_path!r})"
            )
        return other

    def _bind_test_h5(self, test_h5_path: str) -> None:
        """Attach test HDF5 path and rebuild input/target slice indexing."""
        self.test_h5_path = test_h5_path
        self.input_keys = self._input_keys_from_h5(test_h5_path)
        self.has_scenario_index = self._check_scenario_index()
        self._set_slices()

    def _slice_widths(self) -> Dict[str, Dict[str, int]]:
        return {
            group: {key: sl.stop - sl.start for key, sl in slices.items()}
            for group, slices in self.slices.items()
        }

    def compute_net_injection(self, load, der):
        return self.net.compute_net_injection(load, der)

    def expand_branch_status(self, branch_status):
        return self.net.expand_branch_status(branch_status)

    def topology_matrices(self, branch_status):
        return self.topology_backend.topology_matrices(branch_status)

    def topology_matrices_batch(self, branch_status_batch):
        return self.topology_backend.topology_matrices_batch(branch_status_batch)

    def expand_branch_status_np(self, branch_status):
        """NumPy variant of compact/full branch status expansion."""
        net = self.net
        status = np.asarray(branch_status, dtype=np.int32)
        if status.shape[-1] == net.n_branch:
            return status
        if status.shape[-1] != net.n_changing_switches:
            raise ValueError(
                f"Unexpected branch_status width {status.shape[-1]}; expected "
                f"{net.n_changing_switches} compact or {net.n_branch} full."
            )
        changing = np.asarray(net.changing_switches, dtype=np.int32)
        if status.ndim == 1:
            full = np.asarray(net.base_branch_status, dtype=np.int32).copy()
            full[changing] = status
            return full
        full = np.broadcast_to(
            np.asarray(net.base_branch_status, dtype=np.int32),
            status.shape[:-1] + (net.n_branch,),
        ).copy()
        full[..., changing] = status
        return full

    def _check_scenario_index(self) -> bool:
        try:
            with h5py.File(self.test_h5_path, "r") as f:
                return "scenario_index" in f.get("input", {})
        except Exception:
            return False

    def _set_slices(self) -> None:
        self.slices: Dict[str, Dict[str, slice]] = {"input": {}, "target": {}}
        with h5py.File(self.test_h5_path, "r") as f:
            start = 0
            for key in self.input_keys:
                width = self._h5_field_width(f["input"][key])
                self.slices["input"][key] = slice(start, start + width)
                start += width

            start = 0
            for key in self.output_keys:
                width = f["primal"][key].shape[1]
                self.slices["target"][key] = slice(start, start + width)
                start += width

            for key in self.meta_keys:
                meta = f["meta"][key]
                width = 1 if meta.ndim == 1 else meta.shape[1]
                self.slices["target"][key] = slice(start, start + width)
                start += width

    def make_train_dataset(self, samples: int = None) -> tf.data.Dataset:
        inputs = self._load_inputs(self.train_h5_path, samples=samples)
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        return dataset.shuffle(buffer_size=min(len(inputs), 10000))

    def make_test_dataset(
        self,
        samples: int = None,
        sample_offset: int = 0,
    ) -> tf.data.Dataset:
        inputs = self._load_inputs(
            self.test_h5_path, samples=samples, sample_offset=sample_offset
        )
        targets = self._load_targets(
            self.test_h5_path, samples=samples, sample_offset=sample_offset
        )
        return tf.data.Dataset.from_tensor_slices((inputs, targets))

    def _load_inputs(self, path, samples=None, sample_offset: int = 0):
        with h5py.File(path, "r") as f:
            arrays = [
                self._read_input_field(
                    f["input"][key], samples=samples, sample_offset=sample_offset
                )
                for key in self.input_keys
            ]
        return np.concatenate(arrays, axis=1)

    def _load_targets(self, path, samples=None, sample_offset: int = 0):
        arrays = []
        with h5py.File(path, "r") as f:
            for key in self.output_keys:
                arrays.append(
                    self._read_h5_field(
                        f["primal"][key], samples=samples, sample_offset=sample_offset
                    )
                )
            for key in self.meta_keys:
                arrays.append(
                    self._read_h5_field(
                        f["meta"][key], samples=samples, sample_offset=sample_offset
                    )
                )
        return np.concatenate(arrays, axis=1)


def map_to_node(s, n_node, node_map):
    s_node = jnp.zeros(n_node, dtype=s.dtype)
    return s_node.at[node_map].add(s)
