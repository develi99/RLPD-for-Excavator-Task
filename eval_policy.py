import os
import glob
import numpy as np
import gymnasium as gym
from gym.spaces import Box
import jax
from flax.serialization import from_bytes
from rlpd.agents import SACLearner
from agxcave.agxenvs.utils.parse_cfg import parse_env_cfg
import agxcave.agxtasks  # registriert Tasks
import torch
from configs.rlpd_config import get_config


import jax
import jax.numpy as jnp

TASK_NAME = "AgxCave-Rock-Capturing-Vision-v0"

def flatten_field(x):
    if hasattr(x, "detach"):  # PyTorch Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).ravel()

def make_agx_env(headless=True, render_mode=None, device="cpu"):
    cfg = parse_env_cfg(
        TASK_NAME,
        device=device,
        headless=headless,
        render_mode=render_mode,
    )
    env = gym.make(
        TASK_NAME,
        cfg=cfg,
        agx_args=[],
    )
    return env

def convert_obs(obs):
    return np.concatenate([
        flatten_field(obs["policy"].flatten()[:3]),
        flatten_field(obs["stone"]),
        # flatten_field(obs["bucket"]),
        # flatten_field(obs["cabin_position"]),
        # flatten_field(obs["cabin_pitch"])
    ])

def load_checkpoint(agent_template, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        agent = from_bytes(agent_template, f.read())

    print(f"Loaded checkpoint: {checkpoint_path}")
    return agent

def run_policy(save_dir, episodes=5, headless=True):
    # 1️⃣ Environment erstellen
    env = make_agx_env(headless=headless, render_mode="human")
    
    # Dummy observation_space/action_space (nur für Agent.create)
    obs_sample, _ = env.reset()
    obs_flat = convert_obs(obs_sample)
    observation_space_flat = Box(low=-np.inf, high=np.inf, shape=obs_flat.shape, dtype=np.float32)
    # action_sample = env.action_space.sample()
    # action_space_flat = Box(low=env.action_space.low, high=env.action_space.high, dtype=np.float32)
    action_space_flat = Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)

    config = get_config()
    config.hidden_dims=(256, 256)
    config.num_min_qs=1
    config.backup_entropy=False
    kwargs = dict(config)
    model_cls = kwargs.pop("model_cls")

    agent = globals()[model_cls].create(
        seed=0,
        observation_space=observation_space_flat,
        action_space=action_space_flat,
        **kwargs
    )


    # 3️⃣ Checkpoint laden
    agent = load_checkpoint(agent, save_dir)
    
    # 4️⃣ Policy ausführen
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = convert_obs(obs)
        done = False
        ep_return = 0.0
                
        while not done:
            action = agent.eval_actions(obs)

            next_obs, reward, terminated, truncated, info = env.step(
                [action[0]*2, action[1]*2, action[2]*2, 0, 0]
            )

            obs = convert_obs(next_obs)
            done = terminated or truncated
            ep_return += reward

        print(f"Episode {ep+1}: Return = {ep_return:.2f}")

if __name__ == "__main__":
    import sys
    save_dir = sys.argv[1]
    run_policy(save_dir, episodes=5, headless=False)
