"""Train-state factories: input normalization, bounded outputs, optimizer, jitted ``apply_fn``."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.training import train_state

from core.nn.input import input_scale
from core.nn.model import GCN_BASE_NODE_FEATURES, MLP, OPFGCN

_GRAD_CLIP_NORM = 1.0
_MODEL_TYPES = ("mlp", "gcn", "modified_gcn")


def _make_input_normalizer(input_stat):
    """Z-score ``(x - mean) / scale`` with ``scale = std``, or 1 on constant columns.

    No epsilon and no clipping: every scale is a real std (or 1), so outputs are finite.
    The raw ``(mean, std)`` is what checkpoints store, so this also covers loaded stats.
    """
    input_mean, input_std = input_stat
    input_mean = jnp.asarray(input_mean)
    scale = input_scale(input_mean, input_std)

    def normalize_input(x):
        return (x - input_mean) / scale

    return normalize_input


def _bounded(raw, lo, hi):
    """Map unconstrained network output into ``[lo, hi]`` with a sigmoid."""
    return nnx.sigmoid(raw) * (hi - lo) + lo


def _make_optimizer(scheduler):
    return optax.chain(
        optax.clip_by_global_norm(_GRAD_CLIP_NORM),
        optax.adam(learning_rate=scheduler),
    )


def create_train_state(din, dhid, dout, num_layers, scheduler, input_stat, data, rng, model_type="mlp"):
    """TrainState predicting DER setpoints ``[pder, qder]`` (``dout = 2 * n_der_phase``).

    ``model_type`` is ``mlp`` or a graph variant (``gcn`` / ``modified_gcn``); the graph
    variants need full, non-compact inputs.
    """
    model_type = model_type.lower()
    if model_type == "mlp":
        model = MLP(din=din, dhid=dhid, dout=dout, num_layers=num_layers, rngs=rng)
    elif model_type in ("gcn", "modified_gcn"):
        if getattr(data, "use_compact_nn", False):
            raise ValueError(
                "model_type gcn/modified_gcn requires full NN inputs; "
                "set data.use_compact_nn = False before create_train_state."
            )
        obj_sl = data.slices["input"]["obj_weights"]
        model = OPFGCN(
            node_features=GCN_BASE_NODE_FEATURES + (obj_sl.stop - obj_sl.start),
            dhid=dhid,
            num_layers=num_layers,
            improved=model_type == "modified_gcn",
            rngs=rng,
        )
    else:
        raise ValueError(f"model_type must be one of: {', '.join(_MODEL_TYPES)}")
    graphdef, params, others = nnx.split(model, nnx.Param, ...)

    normalize_input = _make_input_normalizer(input_stat)
    net = getattr(data, "net", data)

    def normalize_output(y_pred):
        raw_pder = y_pred[:, :net.n_der_phase]
        raw_qder = y_pred[:, net.n_der_phase:]
        pder = _bounded(raw_pder, net.pdermin, net.pdermax)
        qder = _bounded(raw_qder, net.qdermin, net.qdermax)
        return jnp.concatenate([pder, qder], axis=1)

    @jax.jit
    def apply_fn(params, x):
        model_with_params = nnx.merge(graphdef, params, others)
        normalized_x = normalize_input(x)
        if model_type == "mlp":
            y_pred = model_with_params(normalized_x)
        else:
            y_pred = model_with_params(normalized_x, x, data)
        return normalize_output(y_pred)

    return train_state.TrainState.create(
        apply_fn=apply_fn, params=params, tx=_make_optimizer(scheduler)
    )


def create_train_state_full(din, dhid, dout, num_layers, scheduler, input_stat, data, rng):
    """MLP TrainState predicting ``[pder, qder, vr, vi]`` (``dout = 2 * n_der_phase + 2 * n_net``)."""
    model = MLP(din=din, dhid=dhid, dout=dout, num_layers=num_layers, rngs=rng)
    graphdef, params, others = nnx.split(model, nnx.Param, ...)

    normalize_input = _make_input_normalizer(input_stat)
    net = getattr(data, "net", data)

    def normalize_output(y_pred):
        n_der, n_net = net.n_der_phase, net.n_net
        raw_pder = y_pred[:, :n_der]
        raw_qder = y_pred[:, n_der:2 * n_der]
        raw_vr = y_pred[:, 2 * n_der : 2 * n_der + n_net]
        raw_vi = y_pred[:, 2 * n_der + n_net : 2 * n_der + 2 * n_net]
        return jnp.concatenate(
            [
                _bounded(raw_pder, net.pdermin, net.pdermax),
                _bounded(raw_qder, net.qdermin, net.qdermax),
                _bounded(raw_vr, net.vrmin, net.vrmax),
                _bounded(raw_vi, net.vimin, net.vimax),
            ],
            axis=1,
        )

    @jax.jit
    def apply_fn(params, x):
        model_with_params = nnx.merge(graphdef, params, others)
        return normalize_output(model_with_params(normalize_input(x)))

    return train_state.TrainState.create(
        apply_fn=apply_fn, params=params, tx=_make_optimizer(scheduler)
    )


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)
