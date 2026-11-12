"""Flax NNX MLP for OPF predictions."""

import jax
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
