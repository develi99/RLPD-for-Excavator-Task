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


from collections import defaultdict
import numpy as np
from tqdm import trange


def log_evaluation(agent, env, episodes=25, pixel=False, show_reward=True, seed=0, scaling=(2,2,2)):

    results = evaluate_policy(agent, env, episodes, pixel, show_reward, seed=seed, scaling=scaling)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    main_keys = [
        "episode_reward",
        "episode_length",
        "success_ratio",
        "rock_lifted_ratio",
        "fall_down_ratio",
        "mean_end_position",
    ]

    print("\nCore Metrics:")
    for key in main_keys:
        print(f"  {key:<30} : {results[key]:>10.6f}")

    term_keys = [k for k in results.keys() if k.startswith("term/")]

    print("\nTermination Breakdown:")
    for key in sorted(term_keys):
        print(f"  {key:<30} : {results[key]:>10.6f}")

    reward_keys = [k for k in results.keys() if k.startswith("reward/")]

    print("\nReward Breakdown:")
    for key in sorted(reward_keys):
        print(f"  {key:<30} : {results[key]:>10.6f}")

    print("="*50 + "\n")

    return results


def evaluate_policy(agent, env, episodes=25, pixel=False, show_rewards=False, seed=0, scaling=(2,2,2), encoder=None):

    # fixed termination keys (same as original eval)
    termination_keys = {
        "max_steps": 0,
        "too_deep_termination": 0,
        "stone_x_distance_termination": 0,
        "stone_height_termination": 0,
        "cabin_pitch_termination": 0,
    }

    # fixed reward keys (same as original eval)
    reward_keys = {
        "rock_stable": [],
        "rock_z_pos": [],
        "rock_1_5": [],
        "rock_bucket_dis": [],
        "rock_z_pos_clipped": [],
        "energy_reg": [],
    }

    returns = []
    successes = []
    rock_lifted = []
    fall_downs = []
    end_positions = []
    episode_lengths = []

    for _ in trange(episodes, desc="Episodes"):

        obs, info = env.reset(seed=seed)

        if pixel:
            obs = convert_obs_pixel(obs, encoder)
        else:
            obs = convert_obs(obs)

        done = False
        ep_return = 0.0
        episode_step = 0

        episode_success = False
        episode_rock_lifted = False
        episode_fall_down = False
        end_position = 0.0

        while not done:

            action = agent.eval_actions(obs)

            next_obs, reward, terminated, truncated, info = env.step(
                [action[0]*scaling[0], action[1]*scaling[1], action[2]*scaling[2], 0, 0]
            )

            done = terminated or truncated
            episode_step += 1

            stone_pos = next_obs["stone"]
            z = stone_pos[0][2]
            end_position = z

            if z >= 1.5:
                episode_rock_lifted = True

            if episode_rock_lifted and z <= 1.0:
                episode_fall_down = True

            rock_stable = info.get("extras", {}).get("Step_Reward/rock_stable", 0)
            if rock_stable != 0:
                episode_success = True

            # reward logging
            log_info = info.get("extras", {}).get("log", {})
            step_rewards = info.get("extras", {})

            for key in reward_keys.keys():
                step_key = f"Step_Reward/{key}"
                if step_key in step_rewards:
                    reward_keys[key].append(step_rewards[step_key])

            # termination logging
            if done:
                for key in termination_keys.keys():
                    term_key = f"Episode_Termination/{key}"
                    if log_info.get(term_key, 0) == 1:
                        termination_keys[key] += 1

            ep_return += reward

            if pixel:
                next_obs = convert_obs_pixel(next_obs, encoder)
            else:
                next_obs = convert_obs(next_obs)

            obs = next_obs

        returns.append(ep_return)
        successes.append(episode_success)
        rock_lifted.append(episode_rock_lifted)
        fall_downs.append(episode_fall_down)
        end_positions.append(end_position)
        episode_lengths.append(episode_step)

    results = {
        "episode_reward": float(np.mean(returns)),
        "episode_length": float(np.mean(episode_lengths)),
        "success_ratio": float(np.mean(successes)),
        "rock_lifted_ratio": float(np.mean(rock_lifted)),
        "fall_down_ratio": float(np.mean(fall_downs)),
        "mean_end_position": float(np.mean(end_positions)),
    }

    # add termination stats (always present)
    for key, count in termination_keys.items():
        results[f"term/{key}"] = count / episodes

    # add reward stats (always present)
    for key, values in reward_keys.items():
        if len(values) > 0:
            results[f"reward/{key}"] = float(np.mean(values))
        else:
            results[f"reward/{key}"] = 0.0

    return results


    

# from collections import defaultdict
# import numpy as np
# from tqdm import trange


# def log_evaluation(agent, env, episodes=25, pixel=False, show_reward=True, seed=0, scaling=(2,2,2)):

#     results = evaluate_policy(agent, env, episodes, pixel, show_reward, seed=seed, scaling=scaling)

#     print("\n" + "="*50)
#     print("EVALUATION RESULTS")
#     print("="*50)

#     # Core metrics
#     main_keys = [
#         "episode_reward",
#         "episode_length",
#         "success_ratio",
#         "rock_lifted_ratio",
#         "fall_down_ratio",
#         "mean_end_position",
#     ]

#     print("\nCore Metrics:")
#     for key in main_keys:
#         if key in results:
#             print(f"  {key:<30} : {results[key]:>10.6f}")

#     # Termination metrics
#     term_keys = [k for k in results.keys() if k.startswith("term/")]

#     if term_keys:
#         print("\nTermination Breakdown:")
#         for key in sorted(term_keys):
#             print(f"  {key:<30} : {results[key]:>10.6f}")

#     # Reward metrics
#     reward_keys = [k for k in results.keys() if k.startswith("reward/")]

#     if reward_keys:
#         print("\nReward Breakdown:")
#         for key in sorted(reward_keys):
#             print(f"  {key:<30} : {results[key]:>10.6f}")

#     print("="*50 + "\n")

#     return results


# def evaluate_policy(agent, env, episodes=25, pixel=False, show_rewards=False, seed=0, scaling=(2,2,2)):
#     reward_storage = defaultdict(list)

#     returns = []
#     successes = []
#     rock_lifted = []
#     fall_downs = []
#     end_positions = []
#     episode_lengths = []

#     terminations_info = defaultdict(int)

#     for _ in trange(episodes, desc="Episodes"):
#         obs, info = env.reset(seed=seed)

#         if pixel:
#             obs = convert_obs_pixel(obs)
#         else:
#             obs = convert_obs(obs)

#         done = False
#         ep_return = 0.0
#         episode_step = 0

#         episode_success = False
#         episode_rock_lifted = False
#         episode_fall_down = False
#         end_position = 0.0

#         while not done:
#             action = agent.eval_actions(obs)

#             next_obs, reward, terminated, truncated, info = env.step(
#                 [action[0]*scaling[0], action[1]*scaling[1], action[2]*scaling[2], 0, 0]
#             )

#             done = terminated or truncated
#             episode_step += 1

#             stone_pos = next_obs["stone"]
#             z = stone_pos[0][2]
#             end_position = z

#             # rock lifted
#             if z >= 1.5:
#                 episode_rock_lifted = True

#             # fall down
#             if episode_rock_lifted and z <= 1.0:
#                 episode_fall_down = True

#             # success
#             rock_stable = info.get("extras", {}).get("Step_Reward/rock_stable", 0)
#             if rock_stable != 0:
#                 episode_success = True

#             # reward logging
#             rewards = get_rewards(info, show_rewards)
#             if rewards is not None:
#                 for key, value in rewards.items():
#                     if value is not None:
#                         reward_storage[key].append(value)

#             # termination logging
#             log_info = info.get("extras", {}).get("log", {})

#             if done:
#                 for key, value in log_info.items():
#                     if key.startswith("Episode_Termination/") and value == 1:
#                         terminations_info[key.replace("Episode_Termination/", "")] += 1

#             ep_return += reward

#             if pixel:
#                 next_obs = convert_obs_pixel(next_obs)
#             else:
#                 next_obs = convert_obs(next_obs)

#             obs = next_obs

#         returns.append(ep_return)
#         successes.append(episode_success)
#         rock_lifted.append(episode_rock_lifted)
#         fall_downs.append(episode_fall_down)
#         end_positions.append(end_position)
#         episode_lengths.append(episode_step)
#         print(episode_step)

#     mean_return = float(np.mean(returns))
#     mean_length = float(np.mean(episode_lengths))
#     success_ratio = float(np.mean(successes))
#     rock_lifted_ratio = float(np.mean(rock_lifted))
#     fall_down_ratio = float(np.mean(fall_downs))
#     mean_end_position = float(np.mean(end_positions))

#     mean_rewards = {
#         key: float(np.mean(values))
#         for key, values in reward_storage.items()
#         if len(values) > 0
#     }

#     terminations = {
#         f"term/{key}": value / episodes
#         for key, value in terminations_info.items()
#     }

#     return {
#         "episode_reward": mean_return,
#         "episode_length": mean_length,
#         "success_ratio": success_ratio,
#         "rock_lifted_ratio": rock_lifted_ratio,
#         "fall_down_ratio": fall_down_ratio,
#         "mean_end_position": mean_end_position,
#         **terminations,
#         **{f"reward/{k}": v for k, v in mean_rewards.items()},
#     }