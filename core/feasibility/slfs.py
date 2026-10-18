"""SLFS algorithm and forward-pass primitives."""

from functools import partial

import jax
import jax.numpy as jnp

from core.utils.types import BatchData, ModelOutput, TopologyData
from core.grids.data import NetworkParams
from core.feasibility.fs import feasibility_seeking_batch
from core.feasibility.fs_inference_weighted import feasibility_seeking_batch_inference
from core.grids.linear_model import create_models_batch
from core.feasibility.slfs_config import FeasibilityConfig


def slfs_core(data: NetworkParams, s_load, s_der_init, pder_avai, topo: TopologyData, fs_config):
    """Successive linearization with feasibility seeking (differentiable core).

    For each of ``fs_config.lin_steps`` iterations:

    1. Linearize AC constraints at current ``s_der_current`` (``create_models_batch``).
    2. Solve reduced PDHG problem for ``x = [pder; qder]`` (``feasibility_seeking_batch``).
    3. Damped update and optional ``stop_gradient`` per ``diff_steps`` (below).
    4. Warm-start ``v_init ← v_hat`` for the next linearization.

    Args:
        data: Static network parameters (``NetworkParams`` / ``OPFData.net``).
        s_load: Nodal load injections, complex ``(B, n_net)``.
        s_der_init: Initial DER setpoints from NN, complex ``(B, n_der_phase)``.
        pder_avai: Per-unit active-power caps ``(B, n_der_unit)``.
        topo: Topology-dependent matrices (``Y_LL_inv``, ``w``, ``v_init``, …).
        fs_config: ``FeasibilityConfig`` (``lin_steps``, ``diff_steps``, FS tolerances).

    Returns:
        ``(s_der_final, pder_final, qder_final, v_hat)`` after the last linearization
        step; ``v_hat`` is the converged voltage from that step's power flow.
    """
    # Bypass SLFS entirely when feasibility projection is disabled: the raw NN
    # output is the final DER setpoint (downstream PF still solves for v_final).
    if not fs_config.use_feasibility:
        return s_der_init, s_der_init.real, s_der_init.imag, topo.v_init
    s_der_current = s_der_init
    v_init = topo.v_init
    n_der_phase = data.n_der_phase
    for step in range(fs_config.lin_steps):
        # Create linearized models around current DER solution
        # Pass dynamic matrices. Return only tensors needed by FS.
        models = create_models_batch(
            data,
            s_load,
            s_der_current,
            pder_avai,
            v_init,
            topo,
            power_flow_solver=fs_config.power_flow_solver,
        )

        # Fast path: when DER buses are unique this is exactly equivalent to
        # map_to_node(...)[der_bus], but avoids per-step scatter/gather.
        # if data.der_bus_is_unique:
        #     x_der = jnp.concatenate([s_der_current.real, s_der_current.imag], axis=1)
        # else:
        #     s_der_bus = jax.vmap(map_to_node, in_axes=(0, None, None))(
        #         s_der_current, data.n_total, data.der_bus
        #     )
        #     x_der = jnp.concatenate([
        #         s_der_bus.real[:, data.der_bus],
        #         s_der_bus.imag[:, data.der_bus]
        #     ], axis=1)
        x_der = jnp.concatenate([s_der_current.real, s_der_current.imag], axis=1)

        
        # Apply feasibility-seeking solver
        _x_fs = feasibility_seeking_batch(data, models['reduced_system'], x_der, fs_config.fs_tol, fs_config.fs_maxiter, fs_config.fs_rho)
        _x_fs = x_der + fs_config.fs_damp * (_x_fs - x_der)

        # Truncated backprop through the first ``diff_steps`` linearization steps only.
        #   diff_steps == 0  → block all SLFS gradients (inference-style forward).
        #   step < diff_steps → keep the full FS update differentiable.
        #   else → block this step entirely; using x + stop_grad(delta)
        #          would leak an identity gradient through non-diff steps.
        if step == 0 and fs_config.diff_steps == 0:
            x_fs = jax.lax.stop_gradient(_x_fs)
        elif step < fs_config.diff_steps:
            x_fs = _x_fs*1
        else:
            x_fs = x_fs + jax.lax.stop_gradient(_x_fs - x_fs) # use straight-through idea here to keep the gradient flow

        # Convert back to DER power injections
        pder_fs = x_fs[:, :n_der_phase]
        qder_fs = x_fs[:, n_der_phase:]
        s_der_current = jax.lax.complex(pder_fs, qder_fs)

        # warm start v_init for next iteration
        v_init = models['v_hat']
    return s_der_current, pder_fs, qder_fs, models['v_hat']


# @partial(jax.jit, static_argnums=(0, 5))
# def slfs(data: NetworkParams, s_load, s_der_init, pder_avai, topo: TopologyData, fs_config):
#     return slfs_core(data, s_load, s_der_init, pder_avai, topo, fs_config)


def _feasibility_seeking_inference_step(data, reduced_system, x_der, fs_config):
    """Single inference PDHG solve with voltage row weighting (``voltage_weight=1`` is unweighted)."""
    n_v = fs_config.n_voltage_rows if fs_config.n_voltage_rows > 0 else data.n_net
    return feasibility_seeking_batch_inference(
        data,
        reduced_system,
        x_der,
        fs_config.fs_tol,
        fs_config.fs_maxiter,
        fs_config.fs_rho,
        n_voltage_rows=n_v,
        voltage_weight=fs_config.voltage_weight,
        step_fraction=fs_config.fs_step_fraction,
    )


def slfs_inference_core(
    data: NetworkParams,
    s_load,
    s_der_init,
    pder_avai,
    topo: TopologyData,
    fs_config,
):
    """SLFS forward pass for inference (no truncated backprop).

    Same outer loop as ``slfs_core`` but uses ``fori_loop`` and inference PDHG
    (``fs_inference_weighted``). No ``stop_gradient`` gating — intended for
    deploy/eval, not training.

    Args:
        data: Static ``NetworkParams``.
        s_load: Load injections, complex ``(B, n_net)``.
        s_der_init: Initial DER setpoints from NN, complex ``(B, n_der_phase)``.
        pder_avai: Per-unit active-power caps ``(B, n_der_unit)``.
        topo: ``TopologyData`` for this batch's switch configuration.
        fs_config: ``FeasibilityConfig`` (``lin_steps``, FS tolerances, inference weights).

    Returns:
        ``(s_der_final, pder_final, qder_final, v_hat)`` after ``lin_steps`` iterations.
    """
    if not fs_config.use_feasibility:
        return s_der_init, s_der_init.real, s_der_init.imag, topo.v_init
    n_der_phase = data.n_der_phase
    x_init = jnp.concatenate([s_der_init.real, s_der_init.imag], axis=1)

    def _body(step_idx, carry):
        del step_idx
        s_der_step, v_init_step, _ = carry
        models = create_models_batch(
            data,
            s_load,
            s_der_step,
            pder_avai,
            v_init_step,
            topo,
            power_flow_solver=fs_config.power_flow_solver,
        )

        x_der = jnp.concatenate([s_der_step.real, s_der_step.imag], axis=1)

        x_fs = _feasibility_seeking_inference_step(
            data, models["reduced_system"], x_der, fs_config
        )
        x_fs = x_der + fs_config.fs_damp * (x_fs - x_der)
        pder_fs = x_fs[:, :n_der_phase]
        qder_fs = x_fs[:, n_der_phase:]
        s_der_next = jax.lax.complex(pder_fs, qder_fs)
        return (s_der_next, models["v_hat"], x_fs)

    #may consider the truncated unrolled differentiation here when we scale up the grid size
    s_der_final, v_hat_final, x_fs_final = jax.lax.fori_loop(
        0,
        fs_config.lin_steps,
        _body,
        (s_der_init, topo.v_init, x_init),
    )
    pder_final = x_fs_final[:, :n_der_phase]
    qder_final = x_fs_final[:, n_der_phase:]
    return s_der_final, pder_final, qder_final, v_hat_final


def forward_pass_core(
    params,
    state,
    inputs_batch,
    batch: BatchData,
    data: NetworkParams,
    topo: TopologyData,
    fs_config: FeasibilityConfig,
) -> ModelOutput:
    """NN forward pass followed by ``slfs_core`` refinement.

    Args:
        params: Flax model parameters.
        state: Training state providing ``apply_fn``.
        inputs_batch: Compact NN features ``(B, din_compact)`` (gathered from HDF5).
        batch: ``BatchData`` with loads, caps, targets, and topology status.
        data: ``NetworkParams`` (static physics; usually ``session.net``).
        topo: Precomputed ``TopologyData`` for this batch's switch configuration.
        fs_config: ``FeasibilityConfig`` controlling SLFS and truncated backprop.

    Returns:
        ``ModelOutput`` with NN prediction, FS-refined controls, and final ``v_hat``.
    """
    n_der_phase = data.n_der_phase
    # Unpack necessary data from batch
    s_load = batch.s_load
    pder_avai = batch.pder_avai
    # Initial prediction from neural network
    x_pred_raw = state.apply_fn(params, inputs_batch)
    pder_pred = x_pred_raw[:, :n_der_phase]
    qder_pred = x_pred_raw[:, n_der_phase:]
    s_der_pred = jax.lax.complex(pder_pred, qder_pred)
    
    # Iterative linearization with feasibility-seeking
    s_der_final, pder_final, qder_final, v_hat = slfs_core(
        data, s_load, s_der_pred, pder_avai, topo, fs_config
    )


    return ModelOutput(
        pder_final=pder_final,
        qder_final=qder_final,
        s_der_final=s_der_final,
        pder_pred=pder_pred,
        qder_pred=qder_pred,
        v_hat=v_hat,
    )


@partial(jax.jit, static_argnums=(4, 6))
def forward_pass(params, state, inputs_batch, batch: BatchData, data: NetworkParams, topo: TopologyData, fs_config: FeasibilityConfig) -> ModelOutput:
    return forward_pass_core(params, state, inputs_batch, batch, data, topo, fs_config)


def forward_pass_inference_core(
    params,
    state,
    inputs_batch,
    batch: BatchData,
    data: NetworkParams,
    topo: TopologyData,
    fs_config: FeasibilityConfig,
) -> ModelOutput:
    """Like ``forward_pass_core`` but calls ``slfs_inference_core`` (deploy PDHG path)."""
    n_der_phase = data.n_der_phase
    x_pred_raw = state.apply_fn(params, inputs_batch)
    pder_pred = x_pred_raw[:, :n_der_phase]
    qder_pred = x_pred_raw[:, n_der_phase:]
    s_der_pred = jax.lax.complex(pder_pred, qder_pred)

    s_der_final, pder_final, qder_final, v_hat = slfs_inference_core(
        data, batch.s_load, s_der_pred, batch.pder_avai, topo, fs_config
    )

    return ModelOutput(
        pder_final=pder_final,
        qder_final=qder_final,
        s_der_final=s_der_final,
        pder_pred=pder_pred,
        qder_pred=qder_pred,
        v_hat=v_hat,
    )


@partial(jax.jit, static_argnums=(4, 6))
def forward_pass_inference(
    params,
    state,
    inputs_batch,
    batch: BatchData,
    data: NetworkParams,
    topo: TopologyData,
    fs_config: FeasibilityConfig,
) -> ModelOutput:
    return forward_pass_inference_core(
        params, state, inputs_batch, batch, data, topo, fs_config
    )


