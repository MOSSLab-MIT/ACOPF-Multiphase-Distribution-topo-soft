"""Reference switch-topology backend via admittance restamping.

Production training uses ``core/topology/woodbury.py`` (Woodbury updates).
This module provides ``StampTopologyBackend`` as a correctness reference and
for diagnostics. See ``docs/architecture.md`` (Topology changes).
"""
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from core.topology.utils import make_topology_data, y_0row_from_scaled


class StampTopologyBackend:
    """Canonical switch-only topology backend for the new FeederFlow schema.

    Contract:
    - Input branch status is compact only with width n_changing_switches.
    - Values are binary {0, 1}.
    - Indices are new-schema 0-based indices.
    - Topology perturbation is switch-only by stamping deltas onto Y_scaled_base.
    """

    def __init__(self, net, *, use_native_batch: bool = False) -> None:
        required_attrs = [
            "n_net",
            "n_slk",
            "n_total",
            "n_branch",
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
                f"StampTopologyBackend requires a new-schema network object; missing fields: {missing}"
            )

        self.net = net

        self.n_net = int(net.n_net)
        self.n_slk = int(net.n_slk)
        self.n_total = int(net.n_total)
        self.n_branch = int(net.n_branch)
        self.n_switch = int(net.n_switch)
        self.n_changing_switches = int(net.n_changing_switches)
        self.max_switch_terms = int(net.max_switch_terms)

        self.base_branch_status = np.asarray(net.base_branch_status, dtype=np.int32).reshape(-1)
        self.changing_switches = np.asarray(net.changing_switches, dtype=np.int32).reshape(-1)
        self.switch_idx = np.asarray(net.switch_idx, dtype=np.int32).reshape(-1)
        self.switch_is_closed_base = np.asarray(
            net.switch_is_closed_base, dtype=np.int32
        ).reshape(-1)

        self.switch_from_idx = np.asarray(net.switch_from_idx, dtype=np.int32)
        self.switch_to_idx = np.asarray(net.switch_to_idx, dtype=np.int32)
        self.switch_valid = np.asarray(net.switch_valid, dtype=bool)
        self.switch_y_series = np.asarray(net.switch_y_series, dtype=np.complex128)
        self.switch_y_shunt = np.asarray(net.switch_y_shunt, dtype=np.complex128)

        self.v_scale_full = np.asarray(net.v_scale_full, dtype=np.float64).reshape(-1)
        self.v_slack_local = np.asarray(net.v_slack_local, dtype=np.complex128).reshape(-1)
        self.Y_scaled_base = np.asarray(net.Y_scaled_base, dtype=np.complex128)

        self._validate_static_contracts()
        self.base_compact_status = self.base_branch_status[self.changing_switches]
        self._native_topology_batch_fn = None
        if use_native_batch and hasattr(self.net, "topology_matrices_batch"):
            self._native_topology_batch_fn = jax.jit(self.net.topology_matrices_batch)

    def _validate_static_contracts(self) -> None:
        if self.base_branch_status.shape[0] != self.n_branch:
            raise ValueError(
                "base_branch_status width does not match n_branch "
                f"({self.base_branch_status.shape[0]} != {self.n_branch})."
            )

        if self.changing_switches.shape[0] != self.n_changing_switches:
            raise ValueError(
                "changing_switches width does not match n_changing_switches "
                f"({self.changing_switches.shape[0]} != {self.n_changing_switches})."
            )

        if self.n_switch != self.switch_idx.shape[0]:
            raise ValueError(
                "switch_idx width does not match n_switch "
                f"({self.switch_idx.shape[0]} != {self.n_switch})."
            )

        if self.n_switch != self.switch_is_closed_base.shape[0]:
            raise ValueError(
                "switch_is_closed_base width does not match n_switch "
                f"({self.switch_is_closed_base.shape[0]} != {self.n_switch})."
            )

        if self.switch_from_idx.shape != (self.n_switch, self.max_switch_terms):
            raise ValueError(
                "switch_from_idx has invalid shape "
                f"{self.switch_from_idx.shape}; expected "
                f"({self.n_switch}, {self.max_switch_terms})."
            )

        if self.switch_to_idx.shape != (self.n_switch, self.max_switch_terms):
            raise ValueError(
                "switch_to_idx has invalid shape "
                f"{self.switch_to_idx.shape}; expected "
                f"({self.n_switch}, {self.max_switch_terms})."
            )

        if self.switch_valid.shape != (self.n_switch, self.max_switch_terms):
            raise ValueError(
                "switch_valid has invalid shape "
                f"{self.switch_valid.shape}; expected "
                f"({self.n_switch}, {self.max_switch_terms})."
            )

        expected_switch_matrix_shape = (
            self.n_switch,
            self.max_switch_terms,
            self.max_switch_terms,
        )
        if self.switch_y_series.shape != expected_switch_matrix_shape:
            raise ValueError(
                "switch_y_series has invalid shape "
                f"{self.switch_y_series.shape}; expected {expected_switch_matrix_shape}."
            )

        if self.switch_y_shunt.shape != expected_switch_matrix_shape:
            raise ValueError(
                "switch_y_shunt has invalid shape "
                f"{self.switch_y_shunt.shape}; expected {expected_switch_matrix_shape}."
            )

        if self.Y_scaled_base.shape != (self.n_total, self.n_total):
            raise ValueError(
                "Y_scaled_base has invalid shape "
                f"{self.Y_scaled_base.shape}; expected ({self.n_total}, {self.n_total})."
            )

        if self.v_scale_full.shape[0] != self.n_total:
            raise ValueError(
                "v_scale_full width does not match n_total "
                f"({self.v_scale_full.shape[0]} != {self.n_total})."
            )

        if self.v_slack_local.shape[0] != self.n_slk:
            raise ValueError(
                "v_slack_local width does not match n_slk "
                f"({self.v_slack_local.shape[0]} != {self.n_slk})."
            )

        if np.any(self.changing_switches < 0) or np.any(self.changing_switches >= self.n_branch):
            raise ValueError("changing_switches must be 0-based indices within [0, n_branch).")

        if np.any(self.switch_idx < 0) or np.any(self.switch_idx >= self.n_branch):
            raise ValueError("switch_idx must be 0-based indices within [0, n_branch).")

        if np.any((self.base_branch_status != 0) & (self.base_branch_status != 1)):
            raise ValueError("base_branch_status must contain only binary values {0, 1}.")

        if np.any((self.switch_is_closed_base != 0) & (self.switch_is_closed_base != 1)):
            raise ValueError("switch_is_closed_base must contain only binary values {0, 1}.")

    def normalize_compact_status(self, status, sample_index: Optional[int] = None) -> np.ndarray:
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

    def normalize_compact_status_matrix(self, statuses) -> np.ndarray:
        arr = np.asarray(statuses, dtype=np.int32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(
                f"Compact branch_status matrix must be 2D; got shape {arr.shape}."
            )

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

    def expand_compact_status(self, compact_status, sample_index: Optional[int] = None) -> np.ndarray:
        compact = self.normalize_compact_status(compact_status, sample_index=sample_index)
        full = self.base_branch_status.copy()
        if self.n_changing_switches > 0:
            full[self.changing_switches] = compact
        return full

    def _stamp_switches(self, compact_status, sample_index: Optional[int] = None):
        full_status = self.expand_compact_status(compact_status, sample_index=sample_index)
        Y = self.Y_scaled_base.copy()

        if self.n_switch == 0:
            return Y, full_status

        switch_closed = full_status[self.switch_idx].astype(np.float64)
        base_closed = self.switch_is_closed_base.astype(np.float64)
        delta = switch_closed - base_closed

        if not np.any(delta):
            return Y, full_status

        for r in range(self.max_switch_terms):
            for c in range(self.max_switch_terms):
                valid = self.switch_valid[:, r] & self.switch_valid[:, c]
                if not np.any(valid):
                    continue

                coeff = delta * valid.astype(np.float64)

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

                np.add.at(Y, (i_from, j_from), coeff * self_val * scale_ff)
                np.add.at(Y, (i_to, j_to), coeff * self_val * scale_tt)
                np.add.at(Y, (i_from, j_to), coeff * mutual_val * scale_ft)
                np.add.at(Y, (i_to, j_from), coeff * mutual_val * scale_tf)

        return Y, full_status

    def _topology_matrices_from_stamping(self, compact_status, sample_index: Optional[int] = None):
        """Direct inversion path when a native batched topology builder is unavailable."""
        compact = self.normalize_compact_status(compact_status, sample_index=sample_index)
        Y_scaled_np, full_status_np = self._stamp_switches(
            compact,
            sample_index=sample_index,
        )
        Y_LL_np = Y_scaled_np[: self.n_net, : self.n_net]
        Y_L0_np = Y_scaled_np[: self.n_net, self.n_net : self.n_total]

        try:
            Y_LL_inv_np = np.linalg.inv(Y_LL_np)
        except np.linalg.LinAlgError as exc:
            sample_msg = f"sample {sample_index}" if sample_index is not None else "sample <unknown>"
            raise ValueError(
                "Singular Y_LL encountered while building topology matrices "
                f"for {sample_msg}; compact_status={compact.tolist()}"
            ) from exc

        w_np = -(Y_LL_inv_np @ (Y_L0_np @ self.v_slack_local))

        if (not np.all(np.isfinite(Y_LL_inv_np))) or (not np.all(np.isfinite(w_np))):
            sample_msg = f"sample {sample_index}" if sample_index is not None else "sample <unknown>"
            raise ValueError(
                "Non-finite topology solve encountered while building topology matrices "
                f"for {sample_msg}; compact_status={compact.tolist()}"
            )

        Y_scaled = jnp.asarray(Y_scaled_np)
        Y_LL_inv = jnp.asarray(Y_LL_inv_np)
        w = jnp.asarray(w_np)
        full_status = jnp.asarray(full_status_np, dtype=jnp.int32)

        return Y_scaled, Y_LL_inv, w, full_status

    def topology_matrices(self, compact_status, sample_index: Optional[int] = None):
        """Return switch-stamped topology matrices for one compact status sample."""
        compact = self.normalize_compact_status(compact_status, sample_index=sample_index)

        if self._native_topology_batch_fn is None:
            return self._topology_matrices_from_stamping(compact, sample_index=sample_index)

        compact_batch = jnp.asarray(compact[None, :], dtype=jnp.int32)
        Y_scaled_b, Y_LL_inv_b, w_b, full_status_b = self._native_topology_batch_fn(
            compact_batch
        )

        Y_scaled = Y_scaled_b[0]
        Y_LL_inv = Y_LL_inv_b[0]
        w = w_b[0]
        full_status = full_status_b[0].astype(jnp.int32)

        if not bool(jnp.isfinite(Y_LL_inv).all()):
            sample_msg = f"sample {sample_index}" if sample_index is not None else "sample <unknown>"
            raise ValueError(
                "Singular Y_LL encountered while building topology matrices "
                f"for {sample_msg}; compact_status={compact.tolist()}"
            )

        return Y_scaled, Y_LL_inv, w, full_status

    def topology_data(self, compact_status, sample_index: Optional[int] = None):
        Y_scaled, Y_LL_inv, w, full_status = self.topology_matrices(
            compact_status,
            sample_index=sample_index,
        )
        return make_topology_data(
            Y_0row=y_0row_from_scaled(Y_scaled, self.n_net, self.n_total),
            Y_LL_inv=Y_LL_inv,
            w=w,
            v_init=w,
            branch_status=full_status,
        )

    def topology_matrices_batch(self, compact_status_batch):
        status_batch = self.normalize_compact_status_matrix(compact_status_batch)

        if self._native_topology_batch_fn is None:
            Y_scaled_list = []
            Y_LL_inv_list = []
            w_list = []
            full_status_list = []
            for idx, compact in enumerate(status_batch):
                Y_scaled, Y_LL_inv, w, full_status = self._topology_matrices_from_stamping(
                    compact,
                    sample_index=idx,
                )
                Y_scaled_list.append(Y_scaled)
                Y_LL_inv_list.append(Y_LL_inv)
                w_list.append(w)
                full_status_list.append(full_status)
            return (
                jnp.stack(Y_scaled_list, axis=0),
                jnp.stack(Y_LL_inv_list, axis=0),
                jnp.stack(w_list, axis=0),
                jnp.stack(full_status_list, axis=0),
            )

        compact_batch = jnp.asarray(status_batch, dtype=jnp.int32)
        Y_scaled, Y_LL_inv, w, full_status = self._native_topology_batch_fn(
            compact_batch
        )

        finite_mask = np.asarray(jnp.isfinite(Y_LL_inv).all(axis=(1, 2)))
        if not np.all(finite_mask):
            bad_idx = int(np.flatnonzero(~finite_mask)[0])
            bad_status = status_batch[bad_idx].tolist()
            raise ValueError(
                "Singular Y_LL encountered while building topology matrices "
                f"for sample {bad_idx}; compact_status={bad_status}"
            )

        return Y_scaled, Y_LL_inv, w, full_status.astype(jnp.int32)

    def topology_data_batch(self, compact_status_batch):
        Y_scaled, Y_LL_inv, w, full_status = self.topology_matrices_batch(
            compact_status_batch
        )
        return make_topology_data(
            Y_0row=y_0row_from_scaled(Y_scaled, self.n_net, self.n_total),
            Y_LL_inv=Y_LL_inv,
            w=w,
            v_init=w,
            branch_status=full_status,
        )
