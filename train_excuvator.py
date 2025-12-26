# train_offline2online.py
import gym
import numpy as np
import wandb

from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer

from excavator_env import ExcavatorEnv

import pickle
import numpy as np

import pickle
import numpy as np
import glob
import os



class OfflineDataset:
    def __init__(self, path):
        # Alle .pkl Dateien mit dem Muster laden
        files = sorted(glob.glob(os.path.join(path, "demonstration_*.pkl")))
        all_frames = []

        for file_path in files:
            with open(file_path, "rb") as f:
                frames = pickle.load(f)
                all_frames.extend(frames)

        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []

        for i, frame in enumerate(all_frames):
            # Observation zusammenstellen
            obs = np.concatenate([
                frame["rgb_cabine"].flatten(),
                frame["depth_cabine"].flatten(),
                frame["state"],
                frame["target"],
                frame["stone_pos"],
                frame["bucket_pos"],
                frame["cabin_pos"],
                np.array([frame["cabin_pitch"]])
            ])
            self.obs.append(obs)
            self.actions.append(frame["action"])

            # Reward und done
            if i == len(all_frames) - 1:
                reward = 100.0 if frame["stone_pos"][2] >= 1.5 else 0.0
                done = True
            else:
                reward = 0.0
                done = False

            self.rewards.append(reward)
            self.dones.append(done)

            # next_obs
            if i < len(all_frames) - 1:
                next_frame = all_frames[i + 1]
                next_obs = np.concatenate([
                    next_frame["rgb_cabine"].flatten(),
                    next_frame["depth_cabine"].flatten(),
                    next_frame["state"],
                    next_frame["target"],
                    next_frame["stone_pos"],
                    next_frame["bucket_pos"],
                    next_frame["cabin_pos"],
                    np.array([next_frame["cabin_pitch"]])
                ])
            else:
                next_obs = obs
            self.next_obs.append(next_obs)

        # in numpy arrays umwandeln
        self.obs = np.array(self.obs)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.next_obs = np.array(self.next_obs)
        self.dones = np.array(self.dones)

        self.size = self.obs.shape[0]

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_obs[idx],
            "dones": self.dones[idx],
            "masks": 1.0 - self.dones[idx],
        }




def combine(offline, online):
    batch = {}
    for k in offline:
        batch[k] = np.concatenate([offline[k], online[k]], axis=0)
    return batch




def main(data_path):
    wandb.init(project="offline2online_excavator")

    env = ExcavatorEnv()
    eval_env = ExcavatorEnv()

    dataset = OfflineDataset(data_path)

    agent = SACLearner.create(
        seed=0,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, capacity=1_000_000
    )

    offline_ratio = 0.5
    batch_size = 256
    start_training = 10_000
    max_steps = 200_000

    obs = env.reset()

    for step in range(max_steps):

        # ---- interact ----
        if step < start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(obs)

        next_obs, reward, done, _ = env.step(action)

        replay_buffer.insert(
            dict(
                observations=obs,
                actions=action,
                rewards=reward,
                dones=done,
                masks=1.0,
                next_observations=next_obs,
            )
        )

        obs = next_obs if not done else env.reset()

        # ---- training ----
        if step >= start_training:
            online = replay_buffer.sample(int(batch_size * (1 - offline_ratio)))
            offline = dataset.sample(int(batch_size * offline_ratio))

            batch = combine(offline, online)

            agent, info = agent.update(batch, utd_ratio=1)

            if step % 1000 == 0:
                wandb.log(info, step=step)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
