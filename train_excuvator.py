import gym
import numpy as np

import os
# os.environ["WANDB_MODE"] = "disabled"
import wandb

from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer

import pickle
import glob
import os


import gymnasium as gym
from agxcave.agxenvs.utils.parse_cfg import parse_env_cfg
import agxcave.agxtasks  # registriert Tasks
from gym.spaces import Box
import jax
from flax.serialization import to_bytes
from datetime import datetime
from configs.rlpd_config import get_config
from rlpd.wrappers import wrap_gym # not working for our environment


TASK_NAME = "AgxCave-Rock-Capturing-Vision-v0"
path_to_demo = "../offline2online_praktikum_ws2526/demonstrations_no_images"

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
        # flatten_field(obs["cabin_position"])
    ])


def calc_reward(obs):
    stone_pos = obs["stone_pos"]
    z = stone_pos[2]
    target_z = 1.7

    # distance to target height
    dist = z - target_z
    reward = -abs(dist)

    # If proper height reached
    if z >= 1.5:
        reward += 10

    return reward




class OfflineDataset:
    def __init__(self, path):
        # Get all demo files
        files = sorted(glob.glob(os.path.join(path, "demonstration_*.pkl")))

        # Storage lists
        obs_list = []
        actions_list = []
        rewards_list = []
        next_obs_list = []
        dones_list = []

        for file_path in files:
            with open(file_path, "rb") as f:
                demo = pickle.load(f)  # load demo

            prev_obs = None
            prev_reward = 0.0
            prev_done = False

            print(len(demo))

            for i, frame in enumerate(demo):
                # Build observation (state + stone + bucket)
                obs = np.concatenate([
                    flatten_field(frame["state"])[:3],
                    flatten_field(frame["stone_pos"]),
                    # flatten_field(frame["bucket_pos"]),
                ])

                # Action (first 3 components)
                action = frame["action"][:3]

                reward = 0.0
                done = False

                if prev_obs is not None:
                    # store next_obs for previous step
                    next_obs_list.append(obs)
                    rewards_list.append(prev_reward)
                    dones_list.append(prev_done)

                # store current step
                obs_list.append(obs)
                actions_list.append(action)
                prev_obs = obs
                prev_reward = calc_reward(frame)
                prev_done = done

            # End of demo: final next_obs, reward, done
            next_obs_list.append(prev_obs)
            last_reward = 200.0 if demo[-1]["stone_pos"][2] >= 1.5 else 0.0
            rewards_list.append(prev_reward + last_reward)
            dones_list.append(True)

        # Convert to numpy arrays
        self.obs = np.array(obs_list, dtype=np.float32)
        self.actions = np.array(actions_list, dtype=np.float32)
        self.rewards = np.array(rewards_list, dtype=np.float32)
        self.next_obs = np.array(next_obs_list, dtype=np.float32)
        self.dones = np.array(dones_list, dtype=np.float32)

        self.size = self.obs.shape[0]

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_obs[idx],
            "dones": self.dones[idx],
            "masks": 1.0 - self.dones[idx],  # 0 if done, else 1
        }


def evaluate_policy(agent, env, episodes=3):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        obs = convert_obs(obs)
        done = False
        ep_return = 0.0

        while not done:
            action = agent.eval_actions(obs)
            next_obs, reward, terminated, truncated, info = env.step(
                [action[0], action[1], action[2], 0, 0]
            )
            next_obs = convert_obs(next_obs)
            done = terminated or truncated
            ep_return += reward
            obs = next_obs

        returns.append(ep_return)

    return np.mean(returns)

# own implementation but works also with different offline ratios
def combine_batches(offline_batch, online_batch, shuffle=True):
    combined = {}

    for k in offline_batch:
        offline_data = offline_batch[k]
        online_data = online_batch[k]

        if isinstance(offline_data, dict):
            # Recursion for nested dics
            combined[k] = combine_batches(offline_data, online_data, shuffle=shuffle)
        else:
            # combine
            combined_data = np.concatenate([offline_data, online_data], axis=0)

            if shuffle:
                perm = np.random.permutation(combined_data.shape[0])
                combined_data = combined_data[perm]

            combined[k] = combined_data

    return combined

# original, but only works with offline_ratio 0.5 else there is a error
def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined


def save_training_state(agent, replay_buffer, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)

    # Agent speichern
    agent_path = os.path.join(save_dir, f"agent_checkpoint_{step}.ckpt")
    with open(agent_path, "wb") as f:
        f.write(to_bytes(agent))
    print(f"[Checkpoint] Agent saved to {agent_path}")

    # Replay Buffer speichern
    buffer_path = os.path.join(save_dir, f"replay_buffer_{step}.pkl")
    with open(buffer_path, "wb") as f:
        pickle.dump(replay_buffer, f)
    print(f"[Checkpoint] Replay buffer saved to {buffer_path}")

# normalize actions, action space of env is [-2, 2]
def normalize_actions(a, low=-2.0, high=2.0):
    return 2 * (a - low) / (high - low) - 1


def main(data_path, save_dir="training_runs"):
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)


    # ===============================
    # Hyperparameter
    # ===============================
    batch_size = 256
    offline_ratio = 0.5
    start_training = 5_000
    max_steps = 1_500_000
    pretrain_steps = 0
    utd_ratio = 20
    eval_interval = 10_000

    config = get_config()
    config.hidden_dims=(256, 256)
    config.num_min_qs=1
    config.backup_entropy=False
    kwargs = dict(config)
    model_cls = kwargs.pop("model_cls")

    wandb.init(
        project="rlpd",
        name=f"SparseReward",
        config={**dict(config), 
            "batch_size": batch_size,
            "offline_ratio": offline_ratio,
            "start_training": start_training,
            "max_steps": max_steps,
            "pretrain_steps": pretrain_steps,
            "utd_ratio": utd_ratio,
            "eval_interval": eval_interval}
    )

    # ===============================
    # AGX Environments
    # ===============================
    env = make_agx_env(headless=True)

    print("loading data")
    dataset = OfflineDataset(data_path)
    
    print("create agent and replay buffer")
    # low_obs = np.min(dataset.obs, axis=0)
    # high_obs = np.max(dataset.obs, axis=0)
    # observation_space_flat = Box(low=low_obs, high=high_obs, dtype=np.float32)
    observation_space_flat = Box(low=-np.inf, high=np.inf, shape=dataset.obs[0].shape, dtype=np.float32)

    # low_act = np.min(dataset.actions, axis=0)
    # high_act = np.max(dataset.actions, axis=0)
    # action_space_flat = Box(low=low_act, high=high_act, dtype=np.float32)
    action_space_flat = Box(low=-2.0, high=2.0, shape=dataset.actions[0].shape, dtype=np.float32)

    # init Agent
    agent = globals()[model_cls].create(
        seed=0,
        observation_space=observation_space_flat,
        action_space=action_space_flat,
        **kwargs
    )

    # init Replay Buffer
    replay_buffer = ReplayBuffer(
        observation_space=observation_space_flat,
        action_space=action_space_flat,
        capacity=max_steps
    )


    # ===============================
    # Offline Pretraining
    # ===============================
    print("Starting offline pretraining...")
    for step in range(0,pretrain_steps):
        batch = dataset.sample(batch_size * utd_ratio)
        batch["actions"] = normalize_actions(batch["actions"], low=-2.0, high=2.0)
        agent, info = agent.update(batch, utd_ratio=utd_ratio)
        print(f"step {step} done")
        if step % 1000 == 0:
            wandb.log(
                {f"offline_pretrain/{k}": v for k, v in info.items()},
                step=step
            )

        if step > 0 and step % eval_interval == 0:
            avg_return = evaluate_policy(agent, env)
            wandb.log(
                {"offline_pretrain/eval_return": avg_return},
                step=step
            )


    # ===============================
    # Online + Mixed Training
    # ===============================
    print("Starting online trainig...")
    obs, _ = env.reset()
    obs = convert_obs(obs)
    for step in range(max_steps):

        # ---- interaction ----
        if step < start_training:
            action = env.action_space.sample()
            action = action[:3]
            action = normalize_actions(action, low=-2.0, high=2.0)
        else:
            action, agent = agent.sample_actions(obs)
            
        next_obs, reward, terminated, truncated, info = env.step([action[0]*2,action[1]*2,action[2]*2,0,0]) # upscaling because of normalization
        next_obs = convert_obs(next_obs)

        done = terminated or truncated

        if terminated:
            mask = 0.0
        else:
            mask = 1.0
        
        replay_buffer.insert(
            dict(
                observations=obs,
                actions=action,
                rewards=reward,
                dones=done,
                masks=mask,
                next_observations=next_obs,
            )
        )

        obs = next_obs
        if done:
            obs, _ = env.reset()
            obs = convert_obs(obs)

        # ---- training ----
        if step >= start_training:
            online = replay_buffer.sample(
                int(batch_size * utd_ratio * (1 - offline_ratio))
            )
            offline = dataset.sample(
                int(batch_size * utd_ratio * offline_ratio)
            )
            offline["actions"] = normalize_actions(offline["actions"], low=-2.0, high=2.0)

            batch = combine(offline, online)
            agent, info = agent.update(batch, utd_ratio=utd_ratio)

            if step % 1000 == 0:
                wandb.log(
                    {f"training/{k}": v for k, v in info.items()},
                    step=step + pretrain_steps
                )
                print(step)

        # ---- evaluation ----
        if step % eval_interval == 0 and step >= start_training:
            avg_return = evaluate_policy(agent, env)
            wandb.log(
                {"evaluation/avg_return": avg_return},
                step=step + pretrain_steps
            )
            save_training_state(agent=agent, replay_buffer=replay_buffer, save_dir=save_dir, step=step)

    print("Training done")
    save_training_state(agent=agent, replay_buffer=replay_buffer, save_dir=save_dir, step=step)
    
if __name__ == "__main__":
    main(path_to_demo)
