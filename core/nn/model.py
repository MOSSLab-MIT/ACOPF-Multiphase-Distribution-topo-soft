"""Flax NNX networks: MLP and the feeder-graph GCN (``OPFGCN``)."""

import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """MLP optimized for OPF problems with LayerNorm after each hidden activation."""

    def __init__(self, din: int, dhid: int, dout: int, num_layers: int, rngs: nnx.Rngs):
        kernel_init = nnx.initializers.he_normal()
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

        self.linear_out = nnx.Linear(dhid, dout,
                                     kernel_init=kernel_init,
                                     bias_init=bias_init,
                                     rngs=rngs)

        self.norm_in = nnx.LayerNorm(dhid, rngs=rngs)
        self.norm_hidden = nnx.List([
            nnx.LayerNorm(dhid, rngs=rngs)
            for _ in range(num_layers - 2)
        ])

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.leaky_relu(self.linear_in(x))
        x = self.norm_in(x)
        for layer, norm in zip(self.linear_hidden, self.norm_hidden):
            x = norm(nnx.leaky_relu(layer(x)))
        return self.linear_out(x)


class GraphConv(nnx.Module):
    """Self + neighbor linear message passing on a batched dense adjacency."""

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


# Per-bus features before the broadcast obj_weights: pd, qd, pder_avai (node-scattered),
# status_mean, is_radial, degree. The obj_weights columns are appended on top of this.
GCN_BASE_NODE_FEATURES = 6


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
