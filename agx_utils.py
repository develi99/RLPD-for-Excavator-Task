from ast import Dict, List
import os
import pickle
from typing import Optional
import jax
import jax
import numpy as np
import gymnasium
from agxcave.agxenvs.utils.parse_cfg import parse_env_cfg
import agx
import agxcave.agxtasks  # registers tasks
import cv2
from rewards import calc_reward
import agxcave.agxtasks.excavator.rock_capturing.config.rock_capturing_cfg as agxrewards
import random
import torch
from rlpd.networks.encoders.d4pg_encoder import ResNet18Encoder
from tqdm import tqdm
import jax.numpy as jnp

from stratified_sampling import n_samples_from_percent, sample_loaded_pkls

def set_global_seed(seed: int):
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# normalize actions, action space of env is [-2, 2]
def normalize_actions(a, low=-2.0, high=2.0):
    # Affine map to [-1, 1]
    action = 2.0 * (a - low) / (high - low) - 1.0
    return np.clip(action, -1.0, 1.0)

# For case -2,2 you can simply multiply 2
def denormalize_actions(a, low=-2.0, high=2.0):
    # Affine inverse map back to [low, high]
    action = low + (a + 1.0) * 0.5 * (high - low)
    return np.clip(action, low, high)





def load_demo_pickles(demo_dir):
    demos = []

    for fname in sorted(os.listdir(demo_dir)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(demo_dir, fname), "rb") as f:
            demos.append(pickle.load(f))

    return demos


def demos_to_dataset(demos, reward=0):
    obs, actions, rewards, terminals, next_obs = [], [], [], [], []

    for traj in demos:
        T = len(traj)
        print(T)
        for t in range(T):
            obs.append(np.concatenate(
                [traj[t]["state"][:3], traj[t]["stone_pos"]],
                axis=-1
            ))
            action = normalize_actions(50*-traj[t]["action"][:3])
            actions.append(action)

            rew = calc_reward(traj[t], t == T-1, reward=reward)
            rewards.append(rew)

            terminals.append(float(t == T - 1))
            next_obs.append(np.concatenate(
                [
                    traj[t + 1]["state"][:3] if t + 1 < T else traj[t]["state"][:3],
                    traj[t + 1]["stone_pos"] if t + 1 < T else traj[t]["stone_pos"],
                ],
                axis=-1
            ))

    dataset = dict(
        observations=np.asarray(obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(terminals, dtype=np.float32),
        next_observations=np.asarray(next_obs, dtype=np.float32),
        masks=1.0 - np.asarray(terminals, dtype=np.float32),
    )

    return dataset


def demos_to_pixel_dataset(demos, num_stack=3, img_size=64, reward=0):
    obs, actions, rewards, terminals, next_obs = [], [], [], [], []

    for traj in demos:
        T = len(traj)
        print(T)

        # Buffer für Frame-Stack
        frame_buffer = []

        for t in range(T):
            frame = traj[t]["rgb_cabine"]  # (H, W, 3)
            frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
            frame_buffer.append(frame)

            # fill at beginning to fullfill num_stacks
            if len(frame_buffer) < num_stack:
                while len(frame_buffer) < num_stack:
                    frame_buffer.insert(0, frame)
            else:
                frame_buffer = frame_buffer[-num_stack:]  # letzten num_stack Frames behalten

            # Stack entlang letzter Achse -> (H, W, 3, num_stack)
            imgs = np.stack(frame_buffer, axis=-1)

            obs.append(
                {
                    "state": np.asarray(traj[t]["state"][:3]),
                    "pixels": imgs.astype(np.uint8),
                }
            )
            
            action = normalize_actions(50*-traj[t]["action"][:3])
            actions.append(action)

            rew = calc_reward(traj[t], t == T-1, reward=reward)
            rewards.append(rew)
            terminals.append(float(t == T - 1))

            # --- NEXT OBSERVATION ---
            # Nächster Frame in Buffer für next_observation
            next_frame_buffer = frame_buffer.copy()

            if t + 1 < T:
                next_frame = traj[t + 1]["rgb_cabine"]
                next_frame = cv2.resize(next_frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
            else:
                next_frame = frame  # Letzter Frame wiederverwenden

            next_frame_buffer.append(next_frame)
            next_frame_buffer = next_frame_buffer[-num_stack:]  # letzten num_stack Frames behalten
            next_imgs = np.stack(next_frame_buffer, axis=-1)

            next_obs.append(
                {
                    "state": np.asarray(
                        traj[t + 1]["state"][:3] if t + 1 < T else traj[t]["state"][:3]
                    ),
                    "pixels": next_imgs.astype(np.uint8),
                }
            )

    dataset = dict(
        observations=np.asarray(obs),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(terminals, dtype=np.float32),
        next_observations=np.asarray(next_obs),
        masks=1.0 - np.asarray(terminals, dtype=np.float32),
    )

    return dataset


def demos_to_feature_dataset(
    demos,
    encoder_def,
    img_size,
    reward,
    encoder_batch_size=64,
):

    obs, actions, rewards, terminals, next_obs = [], [], [], [], []

    # --------------------------------------------------
    # Encoder einmal initialisieren
    # --------------------------------------------------

    rng = jax.random.PRNGKey(0)

    first_frame = demos[0][0]["rgb_cabine"]
    first_frame = cv2.resize(first_frame, (img_size, img_size))
    x_input = np.expand_dims(first_frame.astype(np.float32) / 255.0, axis=0)

    encoder_variables = encoder_def.init(rng, x_input)

    encoder_def.make_forward(encoder_variables)
    forward_fn = encoder_def.forward

    # --------------------------------------------------
    # Helper: GPU SAFE Encoding
    # --------------------------------------------------
    def encode_chunks(x_np):

        N = len(x_np)
        feats = []

        for i in range(0, N, encoder_batch_size):

            chunk = x_np[i : i + encoder_batch_size]

            # ---- PAD (verhindert JAX recompilation) ----
            if len(chunk) < encoder_batch_size:
                pad = encoder_batch_size - len(chunk)
                chunk = np.pad(
                    chunk,
                    ((0, pad), (0, 0), (0, 0), (0, 0)),
                    mode="edge",
                )
                padded = True
            else:
                padded = False

            chunk = jnp.asarray(chunk)

            f = forward_fn(chunk)
            f = np.array(f)

            if padded:
                f = f[: len(chunk) - pad]

            feats.append(f)

        return np.concatenate(feats, axis=0)

    # --------------------------------------------------
    # Trajectories
    # --------------------------------------------------
    for traj in tqdm(demos, desc="Processing demos"):

        T = len(traj)

        frames = []
        next_frames = []

        for t in range(T):

            frame = cv2.resize(
                traj[t]["rgb_cabine"], (img_size, img_size)
            )

            frames.append(frame.astype(np.float32) / 255.0)

            if t + 1 < T:
                next_frame = cv2.resize(
                    traj[t + 1]["rgb_cabine"], (img_size, img_size)
                )
            else:
                next_frame = frame

            next_frames.append(next_frame.astype(np.float32) / 255.0)

        x_batch = np.stack(frames, axis=0)
        x_next_batch = np.stack(next_frames, axis=0)

        # -------- GPU SAFE FEATURE EXTRACTION --------

        feats_batch = encode_chunks(x_batch)
        feats_next_batch = encode_chunks(x_next_batch)

        # -------- Dataset bauen --------

        for t in range(T):

            obs.append(
                dict(
                    state=np.asarray(
                        traj[t]["state"][:3], dtype=np.float32
                    ),
                    pixels=feats_batch[t],
                )
            )

            next_obs.append(
                dict(
                    state=np.asarray(
                        traj[t + 1]["state"][:3]
                        if t + 1 < T
                        else traj[t]["state"][:3],
                        dtype=np.float32,
                    ),
                    pixels=feats_next_batch[t],
                )
            )

            action = normalize_actions(
                50 * -traj[t]["action"][:3]
            )
            actions.append(action)

            rew = calc_reward(
                traj[t], t == T - 1, reward=reward
            )
            rewards.append(rew)

            terminals.append(float(t == T - 1))

    dataset = dict(
        observations=np.asarray(obs),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(terminals, dtype=np.float32),
        next_observations=np.asarray(next_obs),
        masks=1.0 - np.asarray(terminals, dtype=np.float32),
    )

    return dataset





def make_agx_env(headless=True, render_mode=None, device="cpu", reward=1, env_name="AgxCave-Rock-Capturing-Vision-v0", agx_args=[]):

    cfg = parse_env_cfg(
        env_name,
        device=device,
        headless=headless,
        render_mode=render_mode,
    )

    reward_map = {
        1: agxrewards.RockRewards1Cfg,
        2: agxrewards.RockRewards2Cfg,
        3: agxrewards.RockRewards3Cfg,
        4: agxrewards.RockRewards4Cfg,
        5: agxrewards.RockRewards5Cfg,
    }

    if reward not in reward_map:
        raise ValueError(f"Unknown reward config: {reward}")

    cfg.rewards = reward_map[reward]()

    env = gymnasium.make(
        env_name,
        cfg=cfg,
        agx_args=agx_args,
    )

    return env


def make_agx_env_and_dataset(env_name, demo_dir, image_size=64, num_stack=3, pixel=False, reward=1, agx_args=[], encoder=None, pretrain=False, percentage=100, seed=0):
    if pixel:
        env = make_agx_env(render_mode="rgb_array", reward=reward, env_name=env_name, agx_args=agx_args)
    else:
        env = make_agx_env(env_name=env_name, reward=reward, agx_args=agx_args)

    # demos = load_demo_pickles(demo_dir)
    n = n_samples_from_percent(percent=percentage, total_size=110)
    demos = sample_loaded_pkls(folder=demo_dir, n_samples=n, seed=seed)
    
    if pixel:
        if pretrain:
            train_dataset = demos_to_feature_dataset(demos=demos, encoder_def=encoder, img_size=image_size, reward=reward)
        else:
            train_dataset = demos_to_pixel_dataset(demos=demos, num_stack=num_stack, img_size=image_size, reward=reward)
    else:
        train_dataset = demos_to_dataset(demos, reward=reward)

    return env, env, train_dataset, None





def convert_obs(obs):
    return np.concatenate([
        flatten_field(obs["policy"].flatten()[:3]), 
        flatten_field(obs["stone"])
    ])


def convert(obs, forward_fn=None, pretrain=False):
    if pretrain:
        return convert_obs_feature(obs, forward_fn)
    else:
        return convert_obs_pixel(obs)


def convert_obs_pixel(obs, encoder):
    """
    Konvertiert eine Observation in das Dict-Format für PixelMultiplexer:
    - pixels: obs["rgb_cabine"], uint8
    - state: Low-Dim Vektor aus policy[:3] und stone
    """
    if encoder is not None:
        return convert_obs_feature(obs, encoder.forward)
    
    state = obs["policy"].flatten()[:3]
    pixel = obs["camera"]["rgb"]
    pixel = (pixel * 255).clip(0, 255).astype(np.uint8)

    return {
        "state": state.astype(np.float32),
        "pixels": pixel
    }


def convert_obs_feature(obs, forward_fn):
    """
    Konvertiert Observation in Feature-Dict für Encoder:
    - pixels: Feature-Vektor
    - state: obs["policy"][:3]
    """
    # Nur erster RGB-Kanal, um 4D Input zu gewährleisten
    pixel_img = obs["camera"]["rgb"][..., 0]  # Shape: (H, W, 3)
    pixel_img = pixel_img.astype(np.float32)
    x_input = np.expand_dims(pixel_img, axis=0)  # Batch-Dim

    pixels = forward_fn(x_input)
    pixels = np.array(pixels)

    state = np.array(obs["policy"].flatten()[:3], dtype=np.float32)
    return {"state": state, "pixels": pixels}


def flatten_field(x):
    if x is None:
        return np.array([], dtype=np.float32)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).ravel() 





def sample_batch(dataset, batch_size):
    num_samples = len(dataset["observations"])
    indices = np.random.choice(num_samples, size=batch_size, replace=False)

    batch = {
        "observations": dataset["observations"][indices],
        "actions": dataset["actions"][indices],
        "rewards": dataset["rewards"][indices],
        "dones": dataset["dones"][indices],
        "next_observations": dataset["next_observations"][indices],
        "masks": dataset["masks"][indices],
    }

    return batch


def sample_pixel_batch(dataset, batch_size):
    """
    Sample a batch from a dataset where observations are dicts with 'state' and 'pixels'.
    Returns a batch dict with arrays only (no nested dicts).
    """
    num_samples = len(dataset["observations"])
    indices = np.random.choice(num_samples, size=batch_size, replace=True)

    obs_states = np.stack([dataset["observations"][i]["state"] for i in indices], axis=0)
    obs_pixels = np.stack([dataset["observations"][i]["pixels"] for i in indices], axis=0)

    next_obs_states = np.stack([dataset["next_observations"][i]["state"] for i in indices], axis=0)
    next_obs_pixels = np.stack([dataset["next_observations"][i]["pixels"] for i in indices], axis=0)

    batch = {
        "observations": {
            "state": obs_states,
            "pixels": obs_pixels
        },
        "next_observations": {
            "state": next_obs_states,
            "pixels": next_obs_pixels
        },
        "actions": dataset["actions"][indices],
        "rewards": dataset["rewards"][indices],
        "dones": dataset["dones"][indices],
        "masks": dataset["masks"][indices],
    }

    return batch
