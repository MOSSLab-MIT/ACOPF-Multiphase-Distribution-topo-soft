"""Training/eval session: ``OPFData`` (HDF5 slice layout and I/O, topology backend).

``NetworkParams`` (static feeder physics) lives in ``core.grids.network``; ``OPFData`` wraps it
with session state (HDF5 paths, slice indexing, topology backend). See ``docs/architecture.md``.
"""
import json
from types import MappingProxyType
from typing import Dict, Protocol, Tuple, runtime_checkable

import h5py
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from core.grids.network import (
    N_OBJ_WEIGHTS,
    OBJ_WEIGHT_NAMES,
    NetworkParams,
    build_network_params,
)


def _check_obj_weights_width(path: str) -> None:
    """Reject HDF5 files whose ``input/obj_weights`` width does not match ``OBJ_WEIGHT_NAMES``."""
    with h5py.File(path, "r") as f:
        ds = f["input"]["obj_weights"]
        width = 1 if ds.ndim == 1 else ds.shape[1]
    if width != N_OBJ_WEIGHTS:
        raise ValueError(
            f"{path}: input/obj_weights has {width} columns; expected {N_OBJ_WEIGHTS} "
            f"{list(OBJ_WEIGHT_NAMES)}. Regenerate the dataset with the current datagen_topo."
        )


# HDF5 ``metadata/<key>`` -> ``NetworkParams`` attribute holding the same price term (pu).
_COST_METADATA_ATTRS = MappingProxyType({
    "sub_cost_lin": "sub_lin", "sub_cost_quad": "sub_quad",
    "der_cost_lin": "der_lin", "der_cost_quad": "der_quad",
})
_COST_METADATA_KEYS = tuple(_COST_METADATA_ATTRS)


def _check_cost_metadata(path: str, net: NetworkParams) -> None:
    """Reject an HDF5 whose recorded price differs from the price in ``network_config.json``.

    Datagen writes ``metadata/{sub,der}_cost_{lin,quad}`` (pu) from the price its config used; labels
    in that file were solved with it. Files without these keys predate the ``cost`` config block (all
    were generated with the FeederFlow defaults) and are accepted.
    """
    with h5py.File(path, "r") as f:
        meta = f.get("metadata")
        present = [k for k in _COST_METADATA_KEYS if meta is not None and k in meta]
        if not present:
            return
        if len(present) != len(_COST_METADATA_KEYS):
            raise ValueError(f"{path}: incomplete price metadata, has {present}, expected {list(_COST_METADATA_KEYS)}.")
        for key, attr in _COST_METADATA_ATTRS.items():
            got = np.asarray(meta[key][()], dtype=np.float64).reshape(-1)
            want = np.asarray(getattr(net, attr), dtype=np.float64).reshape(-1)
            if not np.array_equal(got, want):
                raise ValueError(
                    f"{path}: metadata/{key} = {np.unique(got)} differs from network_config.json "
                    f"({np.unique(want)}). The dataset was generated with a different `cost` config; "
                    "regenerate it or use the matching network_config.json."
                )


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
        woodbury_ir_steps: str = "auto",
        woodbury_batch_mode: str = "vmap",
    ) -> None:
        self.train_h5_path = train_h5_path
        _check_obj_weights_width(train_h5_path)

        with open(json_path, "r") as f:
            raw_config = json.load(f)

        self.case = raw_config.get("case", "")

        config_view = dict(raw_config)
        # Woodbury backend stores its precomputed ``Y_LL_base_inv`` on
        # ``WoodburyStatic``; the dense ``NetworkParams.Y_LL_inv`` would be
        # a duplicate 1.16 GiB on 8500-node networks. Only the stamp backend
        # (and a couple of diagnostic scripts) needs it on the params.
        precompute_inverse = topology_backend != "woodbury"
        net = build_network_params(
            raw_config, precompute_inverse=precompute_inverse
        )
        _check_cost_metadata(train_h5_path, net)
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
        self._bind_test_h5(test_h5_path)
        from core.nn.input import build_nn_input_layout

        self.nn_layout = build_nn_input_layout(self)
        self.use_compact_nn = True

    def with_test_h5(self, test_h5_path: str) -> "OPFData":
        """Return a new session that shares ``net`` and ``topology_backend``.

        Only test HDF5 metadata (``input_keys``, ``slices``) is rebound. Use this for post-training evaluation across dataset variants
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
        _check_obj_weights_width(test_h5_path)
        _check_cost_metadata(test_h5_path, self.net)
        self.test_h5_path = test_h5_path
        self.input_keys = self._input_keys_from_h5(test_h5_path)
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
