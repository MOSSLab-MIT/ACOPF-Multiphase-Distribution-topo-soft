"""Cost scaling for the training loss."""

from typing import Optional

import jax.numpy as jnp


def normalize_cost_for_loss(
    cost: jnp.ndarray, cost_scale: Optional[jnp.ndarray]
) -> jnp.ndarray:
    """Divide the objective by the fixed ``--cost_scale`` hyperparameter; ``None`` disables."""
    if cost_scale is None:
        return cost
    return cost / cost_scale
