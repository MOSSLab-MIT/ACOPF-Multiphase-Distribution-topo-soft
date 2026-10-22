"""Flax NNX MLP for DER setpoint prediction."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.training import train_state

from core.nn.input import gather_nn_features_from_session, log_input_stats

from core.grids.data import OPFData

_NORM_EPS = 1e-8
_INPUT_CLIP_MAX = 5.0


class MLP(nnx.Module):
    """MLP optimized for OPF problems with LayerNorm after each hidden activation."""
    def __init__(self, din: int, dhid: int, dout: int, num_layers: int, rngs: nnx.Rngs):
        # kernel_init = nnx.initializers.normal(stddev=0.1)
        kernel_init = nnx.initializers.he_normal()
        # kernel_init = nnx.initializers.variance_scaling(scale=1.0, mode='fan_in', distribution='truncated_normal')
        bias_init = nnx.initializers.zeros

        self.linear_in = nnx.Linear(din, dhid,
                                   kernel_init=kernel_init,
                                   bias_init=bias_init,
                                   rngs=rngs)

        self.linear_hidden = nnx.List([
            nnx.Linear(dhid, dhid,
                      kernel_init=kernel_init,
                      bias_init=bias_init,
                      rngs=rngs)
            for _ in range(num_layers - 2)
        ])

        # Output layer: smaller initialization for stability
        self.linear_out = nnx.Linear(dhid, dout,
                                    kernel_init=kernel_init,
                                    bias_init=nnx.initializers.zeros,
                                    rngs=rngs)

        # LayerNorm after each hidden activation to stabilize training
        self.norm_in = nnx.LayerNorm(dhid, rngs=rngs)
        self.norm_hidden = nnx.List([
            nnx.LayerNorm(dhid, rngs=rngs)
            for _ in range(num_layers - 2)
        ])

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear_in(x)
        activation = nnx.leaky_relu
        x = activation(x)
        x = self.norm_in(x)
        for layer, norm in zip(self.linear_hidden, self.norm_hidden):
            x = layer(x)
            x = activation(x)
            x = norm(x)
        x = self.linear_out(x)
        return x


class GraphConv(nnx.Module):
    """Graph convolution: a linear map of each node plus a linear map of its neighbor-aggregated features (adj: (B, N, N))."""
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        kernel_init = nnx.initializers.he_normal()
        bias_init = nnx.initializers.zeros
        self.self_linear = nnx.Linear(
            din, dout, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
        )
        self.neighbor_linear = nnx.Linear(
            din, dout, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
        )

    def __call__(self, h: jax.Array, adj: jax.Array) -> jax.Array:
        h_neighbor = jnp.einsum("bij,bjf->bif", adj, h)
        return self.self_linear(h) + self.neighbor_linear(h_neighbor)


def _normalized_topology_adjacency(session, raw_x):
    """Symmetric-normalized bus adjacency stamped from per-sample switch status.

    ``raw_x`` must be the full (non-compact) input row so topology status can be
    read via ``session.slices``.
    """
    net = getattr(session, "net", session)
    topo_slice = session.slices["input"][session.topology_input_key]
    compact_status = raw_x[:, topo_slice].astype(jnp.int32)
    full_status = net.expand_branch_status(compact_status).astype(raw_x.dtype)
    branch_status = full_status[:, : net.n_branch_constraints]
    batch_size = raw_x.shape[0]

    adj = jnp.broadcast_to(
        jnp.eye(net.n_net, dtype=raw_x.dtype),
        (batch_size, net.n_net, net.n_net),
    )
    valid = (
        net.branch_valid
        & (net.branch_from_idx < net.n_net)
        & (net.branch_to_idx < net.n_net)
    )
    weights = branch_status[:, :, None] * valid[None, :, :].astype(raw_x.dtype)
    batch_idx = jnp.arange(batch_size)[:, None, None]
    from_idx = jnp.where(valid, net.branch_from_idx, 0)[None, :, :]
    to_idx = jnp.where(valid, net.branch_to_idx, 0)[None, :, :]
    adj = adj.at[batch_idx, from_idx, to_idx].add(weights)
    adj = adj.at[batch_idx, to_idx, from_idx].add(weights)
    adj = jnp.minimum(adj, jnp.asarray(1.0, dtype=raw_x.dtype))

    degree = jnp.sum(adj, axis=-1)
    inv_sqrt_degree = jax.lax.rsqrt(jnp.maximum(degree, jnp.asarray(1.0, dtype=raw_x.dtype)))
    adj_norm = adj * inv_sqrt_degree[:, :, None] * inv_sqrt_degree[:, None, :]
    return adj_norm, degree


def _opf_node_features(session, x_norm, raw_x, degree):
    """Per-bus feature stack: loads, DER availability, and broadcast globals."""
    net = getattr(session, "net", session)
    slices = session.slices["input"]
    pd = x_norm[:, slices["pd"]]
    qd = x_norm[:, slices["qd"]]
    pder_avai = x_norm[:, slices["pder_avai"]]
    obj_weights = x_norm[:, slices["obj_weights"]]
    topo = raw_x[:, slices[session.topology_input_key]]

    pder_avai_phase = pder_avai[:, net.der_phase_to_unit]
    pder_avai_node = jnp.zeros_like(pd).at[:, net.der_node].set(pder_avai_phase)

    status_mean = jnp.mean(topo, axis=1, keepdims=True)
    if "is_radial" in slices:
        is_radial = raw_x[:, slices["is_radial"]]
    else:
        is_radial = jnp.zeros_like(status_mean)

    repeated = [
        jnp.broadcast_to(obj_weights[:, idx : idx + 1], (raw_x.shape[0], net.n_net))
        for idx in range(obj_weights.shape[1])
    ]
    repeated.extend(
        [
            jnp.broadcast_to(status_mean, (raw_x.shape[0], net.n_net)),
            jnp.broadcast_to(is_radial, (raw_x.shape[0], net.n_net)),
            degree / jnp.maximum(jnp.asarray(net.n_net, dtype=degree.dtype), 1.0),
        ]
    )
    return jnp.stack([pd, qd, pder_avai_node, *repeated], axis=-1)


class OPFGCN(nnx.Module):
    """GCN initializer over the feeder graph; requires full (non-compact) NN inputs."""

    def __init__(
        self,
        node_features: int,
        dhid: int,
        dout: int,
        num_layers: int,
        improved: bool,
        rngs: nnx.Rngs,
    ):
        kernel_init = nnx.initializers.he_normal()
        bias_init = nnx.initializers.zeros
        self.improved = improved
        self.input_linear = nnx.Linear(
            node_features, dhid, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
        )
        self.layers = nnx.List([
            GraphConv(dhid, dhid, rngs=rngs) for _ in range(max(num_layers - 1, 1))
        ])
        decoder_in = dhid * 3 if improved else dhid
        self.decoder_hidden = nnx.Linear(
            decoder_in, dhid, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
        )
        self.decoder_out = nnx.Linear(
            dhid, 2, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
        )

    def __call__(self, x_norm: jax.Array, raw_x: jax.Array, session) -> jax.Array:
        net = getattr(session, "net", session)
        adj, degree = _normalized_topology_adjacency(session, raw_x)
        h = _opf_node_features(session, x_norm, raw_x, degree)
        h = nnx.leaky_relu(self.input_linear(h))
        for layer in self.layers:
            h_next = nnx.leaky_relu(layer(h, adj))
            h = h + h_next if self.improved else h_next

        h_der = h[:, net.der_node, :]
        if self.improved:
            h_mean = jnp.mean(h, axis=1, keepdims=True)
            h_max = jnp.max(h, axis=1, keepdims=True)
            h_context = jnp.concatenate([h_mean, h_max], axis=-1)
            h_context = jnp.broadcast_to(
                h_context, (h_der.shape[0], h_der.shape[1], h_context.shape[-1])
            )
            h_der = jnp.concatenate([h_der, h_context], axis=-1)

        y_pair = self.decoder_out(nnx.leaky_relu(self.decoder_hidden(h_der)))
        return jnp.concatenate([y_pair[:, :, 0], y_pair[:, :, 1]], axis=1)


def compute_input_stats(
    train_dataset,
    session: OPFData,
    *,
    use_compact: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute mean and std for input normalization on NN features."""
    layout = session.nn_layout
    n = len(train_dataset)
    chunks = []
    for inputs_batch in train_dataset.batch(n):
        full = inputs_batch.numpy()
        if use_compact:
            compact = np.asarray(
                gather_nn_features_from_session(full, session), dtype=np.float64
            )
            chunks.append(compact)
        else:
            chunks.append(np.asarray(full, dtype=np.float64))
    all_data = np.concatenate(chunks, axis=0)
    input_mean = jnp.mean(all_data, axis=0)
    input_std = jnp.std(all_data, axis=0)
    din = int(all_data.shape[1])
    expected = layout.din_compact if use_compact else layout.din_full
    if din != expected:
        raise ValueError(f"Input stats width {din} != expected {expected}")
    log_input_stats(input_mean, input_std, layout)
    return input_mean, input_std


def create_train_state(din, dhid, dout, num_layers, scheduler, input_stat, data, rng, model_type="mlp"):
    """Creates initial `TrainState`."""
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
        node_features = 6 + (obj_sl.stop - obj_sl.start)
        model = OPFGCN(
            node_features=node_features,
            dhid=dhid,
            dout=dout,
            num_layers=num_layers,
            improved=model_type == "modified_gcn",
            rngs=rng,
        )
    else:
        raise ValueError("model_type must be one of: mlp, gcn, modified_gcn")
    graphdef, params, others = nnx.split(model, nnx.Param, ...)

    input_mean, input_std = input_stat

    def normalize_input(x):
        normalized = (x - input_mean) / (input_std + jnp.asarray(_NORM_EPS, dtype=input_std.dtype))
        return jnp.clip(normalized, -_INPUT_CLIP_MAX, _INPUT_CLIP_MAX)

    net = getattr(data, "net", data)

    def normalize_output(y_pred):
        raw_pder = y_pred[:, :net.n_der_phase]
        raw_qder = y_pred[:, net.n_der_phase:]

        pder = nnx.sigmoid(raw_pder) * (net.pdermax - net.pdermin) + net.pdermin
        qder = nnx.sigmoid(raw_qder) * (net.qdermax - net.qdermin) + net.qdermin
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
        # return y_pred

    # tx = optax.adamw(learning_rate=scheduler, weight_decay=1e-3)
    # tx = optax.adam(learning_rate=scheduler)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients to avoid exploding gradients
        # optax.adamw(learning_rate=scheduler, weight_decay=1e-4)
        optax.adam(learning_rate=scheduler)
        )

    return train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def create_train_state_full(din, dhid, dout, num_layers, scheduler, input_stat, data, rng):
    """Creates initial `TrainState`."""
    model = MLP(din=din, dhid=dhid, dout=dout, num_layers=num_layers, rngs=rng)
    graphdef, params, others = nnx.split(model, nnx.Param, ...)

    input_mean, input_std = input_stat

    def normalize_input(x):
        normalized = (x - input_mean) / (input_std + jnp.asarray(_NORM_EPS, dtype=input_std.dtype))
        return jnp.clip(normalized, -_INPUT_CLIP_MAX, _INPUT_CLIP_MAX)

    net = getattr(data, "net", data)

    def normalize_output(y_pred):
        raw_pder = y_pred[:, :net.n_der_phase]
        raw_qder = y_pred[:, net.n_der_phase:2 * net.n_der_phase]
        raw_vr = y_pred[:, 2 * net.n_der_phase : 2 * net.n_der_phase + net.n_net]
        raw_vi = y_pred[:, 2 * net.n_der_phase + net.n_net : 2 * net.n_der_phase + 2 * net.n_net]

        pder = nnx.sigmoid(raw_pder) * (net.pdermax - net.pdermin) + net.pdermin
        qder = nnx.sigmoid(raw_qder) * (net.qdermax - net.qdermin) + net.qdermin
        vr = nnx.sigmoid(raw_vr) * (net.vrmax - net.vrmin) + net.vrmin
        vi = nnx.sigmoid(raw_vi) * (net.vimax - net.vimin) + net.vimin
        return jnp.concatenate([pder, qder, vr, vi], axis=1)

    @jax.jit
    def apply_fn(params, x):
        model_with_params = nnx.merge(graphdef, params, others)
        normalized_x = normalize_input(x)
        y_pred = model_with_params(normalized_x)
        return normalize_output(y_pred)
        # return y_pred

    # tx = optax.adamw(learning_rate=scheduler, weight_decay=1e-3)
    # tx = optax.adam(learning_rate=scheduler)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients to avoid exploding gradients
        optax.adam(learning_rate=scheduler)
        )

    return train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


@jax.jit
def update_model(state, grads):
    """Apply one optimizer step with grads and return the updated TrainState."""
    return state.apply_gradients(grads=grads)
