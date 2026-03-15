from typing import Dict, Optional, Tuple, Type, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from rlpd.networks import default_init


class FeatureMultiplexer(nn.Module):
    network_cls: Type[nn.Module]

    latent_dim_pixels: int
    latent_dim_state: int

    stop_gradient: bool = False
    feature_keys: Tuple[str, ...] = ("pixels",)

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ):

        observations = FrozenDict(observations)

        xs = []

        for i, key in enumerate(self.feature_keys):

            x = observations[key]     # <-- already [B, F]

            if self.stop_gradient:
                x = jax.lax.stop_gradient(x)

            x = nn.Dense(
                self.latent_dim_pixels,
                kernel_init=default_init(),
                name=f"pixel_proj_{i}",
            )(x)

            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

            xs.append(x)

        x = jnp.concatenate(xs, axis=-1)

        if "state" in observations:

            y = nn.Dense(
                self.latent_dim_state,
                kernel_init=default_init(),
            )(observations["state"])      # [B, state_dim]

            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)

        if actions is None:
            return self.network_cls()(x, training)
        else:
            return self.network_cls()(x, actions, training)