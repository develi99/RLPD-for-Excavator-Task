import os
import glob
import numpy as np
import gymnasium as gym
from gym.spaces import Box
import jax
from flax.serialization import from_bytes
from rlpd.agents import SACLearner
from rlpd.agents import DrQLearner
from agxcave.agxenvs.utils.parse_cfg import parse_env_cfg
import agxcave.agxtasks  # registriert Tasks
from rlpd.wrappers.frame_stack import FrameStack
from rlpd.wrappers.repeat_action import RepeatAction
import torch
from configs.rlpd_config import get_config
from configs.rlpd_pixels_config import get_config as get_config_pixels
from flax.training import checkpoints
from agx_utils import convert_obs, convert_obs_pixel, make_agx_env
from rlpd.evaluation import evaluate_policy, log_evaluation
import argparse
import jax
import jax.numpy as jnp

TASK_NAME = "AgxCave-Rock-Capturing-Vision-v0"

def flatten_field(x):
    if hasattr(x, "detach"):  # PyTorch Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).ravel()

def convert_obs(obs):
    return np.concatenate([
        flatten_field(obs["policy"].flatten()[:3]),
        flatten_field(obs["stone"]),
        # flatten_field(obs["bucket"]),
        # flatten_field(obs["cabin_position"]),
        # flatten_field(obs["cabin_pitch"])
    ])

def load_checkpoint(agent_template, checkpoint_path, jax=True, step=150000):
    if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if jax:
        agent = checkpoints.restore_checkpoint( 
            ckpt_dir=checkpoint_path,
            target=agent_template,          # Struktur-Vorlage
            step=step                       # oder spezifischen Step laden
        )

        print(f"Loaded checkpoint: {checkpoint_path} step: {step}")
        return agent
    else:
        with open(checkpoint_path, "rb") as f:
            agent = from_bytes(agent_template, f.read())

        print(f"Loaded checkpoint: {checkpoint_path}")
        return agent

def run_policy(save_dir, episodes=5, headless=False, jax=False, step=150000, pixel=False, action_repeat=2, num_stack=3, image_size=64, reward=1):
    # 1️⃣ Environment erstellen
    env = make_agx_env(headless=False, render_mode="human", reward=reward)
    if pixel:
        if action_repeat > 1:
            env = RepeatAction(env, action_repeat)
        if num_stack is not None:
            env = FrameStack(env, num_stack=num_stack, img_size=image_size)


    # Dummy observation_space/action_space (nur für Agent.create)
    obs_sample, _ = env.reset()
    if pixel:
        obs_sample = convert_obs_pixel(obs_sample)
    else:
        obs_sample = convert_obs(obs_sample)

    if pixel:
        action_space_flat = Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)
        
        config = get_config_pixels()
        kwargs = dict(config)
        model_cls = kwargs.pop("model_cls")
        agent = globals()[model_cls].create(
            seed=0,
            observations = obs_sample,
            actions = action_space_flat.sample(),
            **kwargs
        )

    else:
        observation_space_flat = Box(low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=np.float32)
        action_space_flat = Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)

        config = get_config()
        config.hidden_dims=(512, 512)
        # config.num_min_qs=4
        config.num_min_qs=2
        config.num_qs=2
        config.backup_entropy=False

        kwargs = dict(config)
        model_cls = kwargs.pop("model_cls")
        agent = globals()[model_cls].create(
            seed=0,
            observation_space=observation_space_flat,
            action_space=action_space_flat,
            **kwargs
        )

    agent = load_checkpoint(agent, save_dir, jax=jax, step=step)

    log_evaluation(agent, env, episodes, pixel, True)

    # Old Code
    # 4️⃣ Policy ausführen
    # for ep in range(episodes):
    #     obs, _ = env.reset()
    #     if pixel:
    #         obs = convert_obs_pixel(obs)
    #     else:
    #         obs = convert_obs(obs)
    #     done = False
    #     ep_return = 0.0
    #     i = 0
    #     while not done:
    #         i += 1
    #         print(i)
    #         action = agent.eval_actions(obs)
    #         print(action)

    #         next_obs, reward, terminated, truncated, info = env.step(
    #             [action[0]*2, action[1]*2, action[2]*2, 0, 0]
    #         )
    #         print(reward)

    #         if pixel:
    #             obs = convert_obs_pixel(next_obs)
    #         else:
    #             obs = convert_obs(next_obs)
            
    #         done = terminated or truncated
    #         ep_return += reward

    #     print(f"Episode {ep+1}: Return = {ep_return:.2f}")



import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run policy with configurable parameters."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where results or models are stored."
    )

    parser.add_argument(
        "--reward",
        type=int,
        default=1,
        help="Rewardconfig in agxcave"
    )

    # optional arguments with defaults matching your current code
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes (default: 5)"
    )

    parser.add_argument(
        "--step",
        type=int,
        default=150000,
        help="Checkpoint step to load (default: 150000)"
    )

    parser.add_argument(
        "--action-repeat",
        type=int,
        default=2,
        dest="action_repeat",
        help="Action repeat factor (default: 2)"
    )

    parser.add_argument(
        "--num-stack",
        type=int,
        default=3,
        dest="num_stack",
        help="Number of stacked frames (default: 3)"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        dest="image_size",
        help="Image resolution (default: 64)"
    )

    # boolean flags
    parser.add_argument(
        "--jax",
        action="store_true",
        default=True,
        help="Enable JAX (default: True)"
    )

    parser.add_argument(
        "--no-jax",
        action="store_false",
        dest="jax",
        help="Disable JAX"
    )

    parser.add_argument(
        "--pixel",
        action="store_true",
        default=False,
        help="Enable pixel observations (default: False)"
    )

    args = parser.parse_args()

    run_policy(
        args.save_dir,
        episodes=args.episodes,
        jax=args.jax,
        step=args.step,
        pixel=args.pixel,
        action_repeat=args.action_repeat,
        num_stack=args.num_stack,
        image_size=args.image_size,
        reward=args.reward,
        headless=False
    )


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     import sys
#     save_dir = sys.argv[1]
#     run_policy(save_dir, episodes=5, jax=True, step=150000, pixel=False, action_repeat=2, num_stack=3, image_size=64)
