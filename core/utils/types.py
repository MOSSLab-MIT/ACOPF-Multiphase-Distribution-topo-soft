"""Shared dataclasses, NamedTuples, and type aliases for train/test workflow."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Sequence

import jax.numpy as jnp

TestH5Paths = Dict[str, str]


class BatchData(NamedTuple):
    """Per-batch tensors extracted from HDF5 via ``prepare_batch_data``."""

    s_load: jnp.ndarray
    pder_avai: jnp.ndarray
    obj_weights: jnp.ndarray
    branch_status: jnp.ndarray
    switchable_status: jnp.ndarray
    v_init: jnp.ndarray
    v_target: jnp.ndarray
    pder_target: jnp.ndarray
    qder_target: jnp.ndarray


class TopologyData(NamedTuple):
    """Topology-dependent matrices for one sample or a batch."""

    v_init: jnp.ndarray
    Y_0row: jnp.ndarray
    Y_LL_inv: jnp.ndarray
    w: jnp.ndarray
    branch_status: jnp.ndarray


@dataclass(frozen=True)
class CostNormStats:
    """Eval-oracle target cost statistics for loss scaling."""

    cost_max: float
    cost_p95: float
    mode: str
    cost_scale: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "cost_max": self.cost_max,
            "cost_p95": self.cost_p95,
            "mode": self.mode,
            "cost_scale": self.cost_scale,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CostNormStats":
        return cls(
            cost_max=float(data["cost_max"]),
            cost_p95=float(data["cost_p95"]),
            mode=str(data["mode"]),
            cost_scale=float(data["cost_scale"]),
            n_samples=int(data["n_samples"]),
        )

    def cost_scale_jax(self, dtype=jnp.float64) -> jnp.ndarray:
        return jnp.asarray(self.cost_scale, dtype=dtype)


@dataclass(frozen=True)
class OPFDataBuildOptions:
    topology_backend: str = "woodbury"
    power_flow_solver: str = "fixed_point"
    include_topo_nn_input: bool = True
    woodbury_ir_steps: str = "auto"  # "auto", "0", "1", "2"
    woodbury_batch_mode: str = "vmap"  # "vmap" (fast, higher VRAM), "map" (lower VRAM)


@dataclass(frozen=True)
class CheckpointModelSpec:
    hidden_size: int
    num_layers: int
    learning_rate: float


@dataclass(frozen=True)
class VariantTestSpec:
    test_h5_paths: TestH5Paths
    testing_results_dir: Path
    batch_sizes: Sequence[int]
    test_samples: Optional[int] = None
    gpu_clear_delay_s: float = 0.5
    clear_device_memory: bool = True
    pop_batch_metrics: bool = True
