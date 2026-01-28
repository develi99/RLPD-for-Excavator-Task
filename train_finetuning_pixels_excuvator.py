#! /usr/bin/env python
# import dmcgym
import os
# os.environ["WANDB_MODE"] = "disabled"
import gym
import numpy as np
import tqdm
from absl import app, flags
from flax.core import FrozenDict
from ml_collections import config_flags
from flax.core import unfreeze
from rlpd.data.replay_buffer_sample import ReplayBufferSample

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.") 

import pickle
import wandb
from rlpd.agents import DrQLearner
from rlpd.data import MemoryEfficientReplayBuffer
from rlpd.data.vd4rl_datasets import VD4RLDataset
from rlpd.evaluation import evaluate, evaluate_policy
from rlpd.wrappers import WANDBVideo, wrap_pixels
from agx_utils import convert_obs, load_demo_pickles, demos_to_dataset, make_agx_env_and_dataset, sample_batch, normalize_actions, convert_obs_pixel, sample_pixel_batch
from gym.spaces import Box

# Env Wrapper
from rlpd.wrappers.repeat_action import RepeatAction
from rlpd.wrappers.frame_stack import FrameStack

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd_pixels", "wandb project name.")
flags.DEFINE_string("env_name", "AgxCave-Rock-Capturing-Vision-v0", "D4rl dataset name.")
flags.DEFINE_string("demo_dir", "../offline2online_praktikum_ws2526/demonstrations_downscaled_reseted_env/", "Directory containing demonstration pickles.")

flags.DEFINE_string(
    "dataset_level", "expert", "Dataset level (e.g., random, expert, etc.)."
)

# flags.DEFINE_string("log_dir", "", "Path to checkpoint")
flags.DEFINE_string("dataset_path", None, "Path to dataset. If None, uses '~/.vd4rl'.")
flags.DEFINE_integer("dataset_size", 500_000, "How many samples to load")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 25, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 50000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1000000), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", None, "Number of training steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean(
    "memory_efficient_replay_buffer", False, "Use a memory efficient replay buffer."
)
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_string("save_dir", None, "Directory to save checkpoints.")
flags.DEFINE_integer("utd_ratio", 5, "Update to data ratio.")
config_flags.DEFINE_config_file(
    "config",
    "configs/rlpd_pixels_config.py",
    # "configs/drq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_boolean("checkpoint_model", True, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
)


PLANET_ACTION_REPEAT = {
    "cartpole-swingup-v0": 8,
    "reacher-easy-v0": 4,
    "cheetah-run-v0": 4,
    "finger-spin-v0": 2,
    "ball_in_cup-catch-v0": 4,
    "walker-walk-v0": 2,
    "AgxCave-Rock-Capturing-Vision-v0": 2
}


# def combine(one_dict, other_dict):
#     combined = {}

#     for k, v in one_dict.items():
#         if isinstance(v, FrozenDict):
#             if len(v) == 0:
#                 combined[k] = v
#             else:
#                 combined[k] = combine(v, other_dict[k])
#         else:
#             tmp = np.empty(
#                 (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
#             )
#             tmp[0::2] = v
#             tmp[1::2] = other_dict[k]
#             combined[k] = tmp

#     return FrozenDict(combined)
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
    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(FLAGS.log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(FLAGS.log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)


    action_repeat = FLAGS.action_repeat or PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    def wrap(env):
        if "quadruped" in FLAGS.env_name:
            camera_id = 2
        else:
            camera_id = 0
        return wrap_pixels(
            env,
            action_repeat=action_repeat,
            image_size=FLAGS.image_size,
            num_stack=FLAGS.num_stack,
            camera_id=camera_id,
        )

    env, _, ds, _ = make_agx_env_and_dataset(FLAGS.env_name, FLAGS.demo_dir, image_size=FLAGS.image_size, num_stack=3, pixel=True)
    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)
    if FLAGS.num_stack is not None:
        env = FrameStack(env, num_stack=FLAGS.num_stack, img_size=FLAGS.image_size)
    
    action_space_flat = Box(low=-2.0, high=2.0, shape=ds["actions"][0].shape, dtype=np.float32)
    # env = gym.make(FLAGS.env_name)
    # env, pixel_keys = wrap(env)
    # env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # if FLAGS.save_video:
    #     env = WANDBVideo(env)
    # env.seed(FLAGS.seed)

    # ds = VD4RLDataset(
    #     env,
    #     FLAGS.dataset_level,
    #     pixel_keys=pixel_keys,
    #     capacity=FLAGS.dataset_size,
    #     dataset_path=FLAGS.dataset_path,
    # )
    # ds_iterator = ds.get_iterator(
    #     sample_args={
    #         "batch_size": int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio),
    #         "pack_obs_and_next_obs": True,
    #     }
    # )
    # eval_env = gym.make(FLAGS.env_name)
    # eval_env, _ = wrap(eval_env)
    # eval_env.seed(FLAGS.seed + 42)

    replay_buffer_size = FLAGS.replay_buffer_size or FLAGS.max_steps // action_repeat
    if FLAGS.memory_efficient_replay_buffer:
        replay_buffer = MemoryEfficientReplayBuffer(
            env.observation_space, env.action_space, replay_buffer_size
        )
        replay_buffer_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": int(
                    FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
                ),
                "pack_obs_and_next_obs": True,
            }
        )
    else:
        replay_buffer = ReplayBufferSample(
            observation_sample=ds["observations"][0], action_sample=ds["actions"][0], capacity=replay_buffer_size
        )
        # replay_buffer_iterator = replay_buffer.get_iterator(
        #     sample_args={
        #         "batch_size": int(
        #             FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
        #         ),
        #     }
        # )

    replay_buffer.seed(FLAGS.seed)

    # Crashes on some setups if agent is created before replay buffer.
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        seed=FLAGS.seed,
        observations=ds["observations"][0],
        actions=ds["actions"][0],
        pixel_keys=("pixels",),
        **kwargs,
    )

    observation, _ = env.reset()
    done = False
    observation = convert_obs_pixel(observation)
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        if i < FLAGS.start_training:
            action = action_space_flat.sample()
            action = normalize_actions(action, low=-2.0, high=2.0)
        else:
            action, agent = agent.sample_actions(observation)
            action = np.clip(action, -1.0, 1.0) # just to be sure

        next_obs, reward, terminated, truncated, info = env.step([action[0]*2, action[1]*2 ,action[2]*2 ,0 ,0]) # upscaling because of normalization
        next_obs = convert_obs_pixel(next_obs)

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
            observation, _ = env.reset()
            done = False
            observation = convert_obs_pixel(observation)
            # for k, v in info["episode"].items():
            #     decode = {"r": "return", "l": "length", "t": "time"}
            #     wandb.log({f"training/{decode[k]}": v}, step=i * action_repeat)

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            offline_batch = sample_pixel_batch(
                ds,
                int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
            )

            batch = combine(offline_batch, unfreeze(online_batch))
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i * action_repeat)

        if i % FLAGS.eval_interval == 0:
            # eval_info = evaluate(
            #     agent,
            #     eval_env,
            #     num_episodes=FLAGS.eval_episodes,
            #     save_video=FLAGS.save_video,
            # )
            eval_info = evaluate_policy(agent, env, episodes=FLAGS.eval_episodes, pixel=True)

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i * action_repeat)

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
