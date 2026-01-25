from typing import Dict

import gym
import numpy as np

from rlpd.wrappers.wandb_video import WANDBVideo
from agx_utils import convert_obs, convert_obs_pixel
from tqdm import trange

def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}


# Own Implementation of evaluate function
def evaluate_policy(agent, env, episodes=25, pixel=False):
    returns = []
    successes = []
    over_boundarys = []
    falled_down = []
    end_positions = []

    for _ in trange(episodes, desc="Episodes"):
        obs, _ = env.reset()
        obs = convert_obs_pixel(obs)
        done = False
        ep_return = 0.0
        success = False
        over_boundary = False
        fall_down = False
        end_position = 0.0

        while not done:
            action = agent.eval_actions(obs)
            next_obs, reward, terminated, truncated, info = env.step(
                [action[0]*2, action[1]*2, action[2]*2, 0, 0]
            )

            stone_pos = next_obs["stone"]
            z = stone_pos[0][2]
            end_position = z

            if z >= 1.5:
                over_boundary = True

            if over_boundary and z <= 1.0:
                fall_down = True

            rock_stable = info.get('extras', None)["Step_Reward/rock_stable"]
            if rock_stable == 12000.0:
                success = True
            
            done = terminated or truncated
            ep_return += reward

            if pixel:
                next_obs = convert_obs_pixel(next_obs)
            else:
                next_obs = convert_obs(next_obs)
            
            obs = next_obs

        returns.append(ep_return)
        successes.append(success)
        over_boundarys.append(over_boundary)
        falled_down.append(fall_down)
        end_positions.append(end_position)


    mean_return = np.mean(returns)

    success_ratio = np.mean(successes)
    over_boundary_ratio = np.mean(over_boundarys)
    fall_down_ratio = np.mean(falled_down)
    ends = np.mean(end_positions)

    return {
        "mean_return": mean_return,
        "success_ratio": success_ratio,
        "over_boundary_ratio": over_boundary_ratio,
        "fall_down_ratio": fall_down_ratio,
        "mean_end_position": ends
    }
