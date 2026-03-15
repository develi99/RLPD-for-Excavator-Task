from functools import partial
from typing import Callable, Sequence, Tuple, Optional

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from rlpd.networks import MLP, Ensemble, StateActionValue, FeatureMultiplexer
from rlpd.agents.sac.sac_learner import SACLearner
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal


@struct.dataclass
class FeatureDrQLearner(SACLearner):
    data_augmentation_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observations,               # Dict[str, np.ndarray] Sample obs
        actions,                    # np.ndarray Sample action
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        latent_dim_pixels: int = 64,
        latent_dim_state: int = 16,
        hidden_dims: Sequence[int] = (256, 256),
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        tau: float = 0.005,
        discount: float = 0.99,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
    ):
        """Feature-based DrQ / SAC learner"""

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # Actor Network
        action_dim = actions.shape[-1]
        if target_entropy is None:
            target_entropy = -action_dim / 2

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = FeatureMultiplexer(
            network_cls=actor_cls,
            latent_dim_pixels=latent_dim_pixels,
            latent_dim_state=latent_dim_state,
            stop_gradient=True,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(actor_lr),
        )

        # Critic Network
        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_core_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(Ensemble, net_cls=critic_core_cls, num=num_qs)

        critic_def = FeatureMultiplexer(
            network_cls=critic_cls,
            latent_dim_pixels=latent_dim_pixels,
            latent_dim_state=latent_dim_state,
        )
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(critic_lr),
        )

        # Target Critic
        critic_target_def = FeatureMultiplexer(
            network_cls=partial(Ensemble, net_cls=critic_core_cls, num=num_min_qs or num_qs),
            latent_dim_pixels=latent_dim_pixels,
            latent_dim_state=latent_dim_state,
        )
        target_critic = TrainState.create(
            apply_fn=critic_target_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # Temperature
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(temp_lr),
        )

        # No data augmentation needed for features
        def data_augmentation_fn(rng, obs):
            return obs

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            data_augmentation_fn=data_augmentation_fn,
        )

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        # Features sind schon batch-fähig, keine weitere Aug nötig
        new_agent = self
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        batch = batch.copy()
        batch["observations"] = observations
        batch["next_observations"] = next_observations
        return SACLearner.update(new_agent, batch, utd_ratio)