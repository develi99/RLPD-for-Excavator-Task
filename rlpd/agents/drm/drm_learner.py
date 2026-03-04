"""DrM implementation in JAX/Flax."""

from functools import partial
from typing import Sequence, Optional, Callable

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState

from rlpd.distributions import TanhNormal
from rlpd.data.dataset import DatasetDict


# ------------------------------------------------------------
# Networks
# ------------------------------------------------------------
class Encoder(nn.Module):
    features: Sequence[int] = (32,32,32,32)

    @nn.compact
    def __call__(self, x):
        x = x / 255.0 - 0.5
        for i, feat in enumerate(self.features):
            stride = 2 if i == 0 else 1
            x = nn.Conv(feat, (3,3), strides=(stride,stride))(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return x


class Actor(nn.Module):
    action_dim: int
    feature_dim: int = 50
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.feature_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        mu = nn.Dense(self.action_dim)(x)
        mu = nn.tanh(mu)

        return mu


class Critic(nn.Module):
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)

        def q_net(x):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            return nn.Dense(1)(x)

        q1 = q_net(x)
        q2 = q_net(x)

        return q1, q2


class ValueNet(nn.Module):
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x)


# ------------------------------------------------------------
# DrM Learner
# ------------------------------------------------------------

class DrMLearner(struct.PyTreeNode):

    rng: jax.random.PRNGKey
    encoder: TrainState
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    value_net: TrainState

    discount: float
    tau: float
    expectile: float
    target_lambda: float

    # --------------------------------------------------------

    @classmethod
    def create(
        cls,
        seed: int,
        observations,
        actions,
        actor_lr=3e-4,
        critic_lr=3e-4,
        value_lr=3e-4,
        discount=0.99,
        tau=0.005,
        expectile=0.7,
        target_lambda=0.5,
    ):

        rng = jax.random.PRNGKey(seed)
        rng, enc_key, act_key, cri_key, val_key = jax.random.split(rng, 5)

        encoder_def = Encoder()
        enc_params = encoder_def.init(enc_key, observations)["params"]
        encoder = TrainState.create(
            apply_fn=encoder_def.apply,
            params=enc_params,
            tx=optax.adam(actor_lr),
        )

        latent = encoder_def.apply({"params": enc_params}, observations)

        actor_def = Actor(action_dim=actions.shape[-1])
        act_params = actor_def.init(act_key, latent)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=act_params,
            tx=optax.adam(actor_lr),
        )

        critic_def = Critic()
        cri_params = critic_def.init(cri_key, latent, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=cri_params,
            tx=optax.adam(critic_lr),
        )

        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=cri_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_def = ValueNet()
        val_params = value_def.init(val_key, latent)["params"]
        value_net = TrainState.create(
            apply_fn=value_def.apply,
            params=val_params,
            tx=optax.adam(value_lr),
        )

        return cls(
            rng=rng,
            encoder=encoder,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            value_net=value_net,
            discount=discount,
            tau=tau,
            expectile=expectile,
            target_lambda=target_lambda,
        )

    # --------------------------------------------------------

    @partial(jax.jit, static_argnames=())
    def update(self, batch: DatasetDict):

        rng, key = jax.random.split(self.rng)

        obs = self.encoder.apply_fn(
            {"params": self.encoder.params},
            batch["observations"]
        )
        next_obs = self.encoder.apply_fn(
            {"params": self.encoder.params},
            batch["next_observations"]
        )

        # ---------------- Critic Update ----------------

        def critic_loss_fn(params):

            q1, q2 = self.critic.apply_fn(
                {"params": params}, obs, batch["actions"]
            )

            next_action = self.actor.apply_fn(
                {"params": self.actor.params}, next_obs
            )

            target_q1, target_q2 = self.target_critic.apply_fn(
                {"params": self.target_critic.params},
                next_obs,
                next_action
            )

            target_v = jnp.minimum(target_q1, target_q2)

            target = batch["rewards"] + self.discount * batch["discounts"] * target_v

            loss = ((q1 - target)**2 + (q2 - target)**2).mean()
            return loss

        grads = jax.grad(critic_loss_fn)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        # ---------------- Value Update (Expectile) ----------------

        def value_loss_fn(params):

            q1, q2 = critic.apply_fn(
                {"params": critic.params}, obs, batch["actions"]
            )
            q = jnp.minimum(q1, q2)

            v = self.value_net.apply_fn({"params": params}, obs)

            diff = v - q
            weight = jnp.where(diff > 0, 1 - self.expectile, self.expectile)

            return (weight * diff**2).mean()

        grads = jax.grad(value_loss_fn)(self.value_net.params)
        value_net = self.value_net.apply_gradients(grads=grads)

        # ---------------- Actor Update ----------------

        def actor_loss_fn(params):
            action = self.actor.apply_fn({"params": params}, obs)
            q1, q2 = critic.apply_fn(
                {"params": critic.params}, obs, action
            )
            return -jnp.minimum(q1, q2).mean()

        grads = jax.grad(actor_loss_fn)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        # ---------------- Soft Update ----------------

        new_target_params = optax.incremental_update(
            critic.params,
            self.target_critic.params,
            self.tau,
        )

        target_critic = self.target_critic.replace(
            params=new_target_params
        )

        return self.replace(
            rng=rng,
            critic=critic,
            actor=actor,
            value_net=value_net,
            target_critic=target_critic,
        )