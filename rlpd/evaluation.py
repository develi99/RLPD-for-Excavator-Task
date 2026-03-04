from typing import Dict
import gym
import numpy as np
from rlpd.wrappers.wandb_video import WANDBVideo
from agx_utils import convert_obs, convert_obs_pixel
from tqdm import trange
from collections import defaultdict


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


def get_rewards(info, show=False):
    extras = info.get("extras")
    if extras is None:
        return None

    rewards = {
        "energy_regularization": extras.get("Step_Reward/energy_regularization"),
        "rock_stable": extras.get("Step_Reward/rock_stable"),
        "rock_z_pos": extras.get("Step_Reward/rock_z_pos"),
        "rock_1_5": extras.get("Step_Reward/rock_1_5"),
        "rock_bucket_distance": extras.get("Step_Reward/rock_bucket_dis"),
        "rock_z_pos_clipped": extras.get("Step_Reward/rock_z_pos_clipped"),
        "energy_reg": extras.get("Step_Reward/energy_reg")
    }

    if show:
        for key, value in rewards.items():
            if value is not None:
                print(f"{key}: {value}")

    return rewards


def log_evaluation(agent, env, episodes=25, pixel=False, show_reward=True, seed=0, scaling=(2,2,2)):

    results = evaluate_policy(agent, env, episodes, pixel, show_reward, seed=seed, scaling=scaling)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    # Hauptmetriken (bekannte Keys)
    main_keys = [
        "mean_return",
        "success_ratio",
        "over_boundary_ratio",
        "fall_down_ratio",
        "mean_end_position",
    ]

    print("\nCore Metrics:")
    for key in main_keys:
        if key in results:
            print(f"  {key:<25} : {results[key]:>10.6f}")

    # Reward-Komponenten (alles andere)
    reward_keys = [k for k in results.keys() if k not in main_keys]

    if reward_keys:
        print("\nReward Breakdown:")
        for key in sorted(reward_keys):
            print(f"  {key:<25} : {results[key]:>10.6f}")

    print("="*50 + "\n")

    return results


def evaluate_policy(agent, env, episodes=25, pixel=False, show_rewards=False, seed=0, scaling=(2,2,2)):
    reward_storage = defaultdict(list)
    returns = []
    successes = []
    over_boundarys = []
    falled_down = []
    end_positions = []

    for _ in trange(episodes, desc="Episodes"):
        obs, _ = env.reset(seed=seed)
        if pixel:
            obs = convert_obs_pixel(obs)
        else:
            obs = convert_obs(obs)
        done = False
        ep_return = 0.0
        success = False
        over_boundary = False
        fall_down = False
        end_position = 0.0

        while not done:
            action = agent.eval_actions(obs)
            next_obs, reward, terminated, truncated, info = env.step(
                [action[0]*scaling[0], action[1]*scaling[1], action[2]*scaling[2], 0, 0]
                # [action[0]*2, action[1]*2, action[2]*4, 0, 0]
            )

            stone_pos = next_obs["stone"]
            z = stone_pos[0][2]
            end_position = z

            if z >= 1.5:
                over_boundary = True

            if over_boundary and z <= 1.0:
                fall_down = True

            rock_stable = info.get('extras', None)["Step_Reward/rock_stable"]
            if rock_stable != 0:
                success = True

            # Read rewards

            # all rewards, too much bullshit is logged
            # extras = info.get("extras", {})
            # for key, value in extras.items():
            #     if key.startswith("Step_Reward/"):
            #         clean_key = key.replace("Step_Reward/", "")
            #         reward_storage[clean_key].append(value)

            # specific rewards
            rewards = get_rewards(info, show_rewards)

            if rewards is not None:
                for key, value in rewards.items():
                    if value is not None:
                        reward_storage[key].append(value)

            
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
    mean_end_position = np.mean(end_positions)

    # return {
    #     "mean_return": mean_return,
    #     "success_ratio": success_ratio,
    #     "over_boundary_ratio": over_boundary_ratio,
    #     "fall_down_ratio": fall_down_ratio,
    #     "mean_end_position": mean_end_position
    # }

    mean_rewards = {
        key: float(np.mean(values))
        for key, values in reward_storage.items()
        if len(values) > 0
    }

    return {
        "mean_return": mean_return,
        "success_ratio": success_ratio,
        "over_boundary_ratio": over_boundary_ratio,
        "fall_down_ratio": fall_down_ratio,
        "mean_end_position": mean_end_position,
        **mean_rewards,
    }
