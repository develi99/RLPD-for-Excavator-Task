#! /usr/bin/env python
import os

# os.environ["WANDB_MODE"] = "disabled"
import wandb

import pickle

# import d4rl
# import d4rl.gym_mujoco
# import d4rl.locomotion
# import dmcgym
# import gym
import numpy as np
import tqdm
from absl import app, flags

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer
# from rlpd.data.d4rl_datasets import D4RLDataset

try:
    from rlpd.data.binary_datasets import BinaryDataset
except:
    print("not importing binary dataset")
from rlpd.evaluation import evaluate, evaluate_policy
from rlpd.wrappers import wrap_gym
from agx_utils import convert_obs, make_agx_env_and_dataset, sample_batch, normalize_actions, make_agx_env, set_global_seed
from gym.spaces import Box


FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
flags.DEFINE_string("name", "rlpdrun", "wandb run name.")
flags.DEFINE_string("env_name", "AgxCave-Rock-Capturing-Vision-v0", "D4rl dataset name.")
flags.DEFINE_string("demo_dir", "../offline2online_praktikum_ws2526/demonstrations_no_images_reseted_env/", "Directory containing demonstration pickles.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 25, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 50000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(500000), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(5000), "Number of training steps to start training."
)
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", True, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
)
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("agxreward", 1, "Update to data ratio.")
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)

config_flags.DEFINE_config_file(
    "config",
    "configs/rlpd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

flags.DEFINE_boolean("video", False, "record video")
flags.DEFINE_string("video_dir", "./videos", "video dir")

import random
import numpy as np
import os

def build_agx_args(video: bool, video_dir: str, run_name: str):
    agx_args = []

    if video:
        os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(video_dir, run_name)

        agx_args.extend([
            "--captureVideo", "1",
            "--videoName", video_path,
        ])

    return agx_args

def set_global_seed(seed: int):
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # Python hash seed (wichtig für dict ordering etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

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

def main(_):
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0
    set_global_seed(FLAGS.seed)

    wandb.init(project=FLAGS.project_name, name=FLAGS.name)
    wandb.config.update(FLAGS)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(FLAGS.log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(FLAGS.log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    # env = gym.make(FLAGS.env_name)
    # env = wrap_gym(env, rescale_actions=True)
    # env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # env.seed(FLAGS.seed)
    # not ideal, but works for now:
    # if "binary" in FLAGS.env_name:
    #     ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    # else:
    #     ds = D4RLDataset(env)
    # eval_env = gym.make(FLAGS.env_name)
    # eval_env = wrap_gym(eval_env, rescale_actions=True)
    # eval_env.seed(FLAGS.seed + 42)
    agx_args = build_agx_args(FLAGS.video, FLAGS.video_dir, str(0))
    env, eval_env, ds, _ = make_agx_env_and_dataset(FLAGS.env_name, FLAGS.demo_dir, reward=FLAGS.agxreward, agx_args=agx_args)
    action_space_flat = Box(low=-2.0, high=2.0, shape=ds["actions"][0].shape, dtype=np.float32)
    action_space_flat.seed(FLAGS.seed)
    observation_space_flat = Box(low=-np.inf, high=np.inf, shape=ds["observations"][0].shape, dtype=np.float32)
    observation_space_flat.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    # agent = globals()[model_cls].create(
    #     FLAGS.seed, env.observation_space, env.action_space, **kwargs
    # )
    agent = globals()[model_cls].create(
        FLAGS.seed, observation_space_flat, action_space_flat, **kwargs
    )

    # replay_buffer = ReplayBuffer(
    #     env.observation_space, env.action_space, FLAGS.max_steps
    # )
    replay_buffer = ReplayBuffer(
        observation_space_flat, action_space_flat, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    for i in tqdm.tqdm(
        range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        batch = {}
        for k, v in offline_batch.items():
            batch[k] = v
            if "antmaze" in FLAGS.env_name and k == "rewards":
                batch[k] -= 1

        agent, update_info = agent.update(batch, FLAGS.utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)

    observation, _ = env.reset(seed=FLAGS.seed)
    done = False
    observation = convert_obs(observation)
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps+1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = action_space_flat.sample()
            action = normalize_actions(action, low=-2.0, high=2.0)
        else:
            action, agent = agent.sample_actions(observation)
            action = np.clip(action, -1.0, 1.0) # just to be sure

        next_obs, reward, terminated, truncated, info = env.step([action[0]*2, action[1]*2 ,action[2]*2 ,0 ,0]) # upscaling because of normalization
        next_obs = convert_obs(next_obs)

        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        
        # always log this at every step
        wandb.log({f'env/{k}': v for k, v in env_info.items()}, step=i)


        done = terminated or truncated

        if terminated:
            mask = 0.0
        else:
            mask = 1.0
        
        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_obs,
            )
        )
        observation = next_obs

        if done:
            observation, _ = env.reset(seed=FLAGS.seed)
            done = False
            observation = convert_obs(observation)
            # for k, v in info["episode"].items():
            #     decode = {"r": "return", "l": "length", "t": "time"}
            #     wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            offline_batch = sample_batch(
                ds,
                int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
            )

            batch = combine(offline_batch, online_batch)

            if "antmaze" in FLAGS.env_name:
                batch["rewards"] -= 1

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            # eval_info = evaluate(
            #     agent,
            #     eval_env,
            #     num_episodes=FLAGS.eval_episodes,
            #     save_video=FLAGS.save_video,
            # )
            # env.close()
            # agx_args = build_agx_args(video=FLAGS.video, video_dir=FLAGS.video_dir, run_name=str(i))
            # env = make_agx_env(env_name=FLAGS.env_name, reward=FLAGS.agxreward, agx_args=agx_args, headless=False, render_mode="human")
            # env.reset(seed=FLAGS.seed)
            eval_info = evaluate_policy(agent, env, episodes=FLAGS.eval_episodes, pixel=False, seed=FLAGS.seed)

            for k, v in eval_info.items():
                wandb.log({f"eval/{k}": v}, step=i + FLAGS.pretrain_steps)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print("Could not save agent buffer.")



if __name__ == "__main__":
    app.run(main)
