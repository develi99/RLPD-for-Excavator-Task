import numpy as np
from typing import Optional
from flax.core import frozen_dict
import copy

class FeatureReplayBufferSample:
    def __init__(
        self,
        observation_sample: dict,
        action_sample: np.ndarray,
        capacity: int,
        next_observation_sample: Optional[dict] = None,
        seed: int = 0,
    ):
        if next_observation_sample is None:
            next_observation_sample = observation_sample

        self.capacity = capacity
        self.insert_index = 0
        self.size = 0
        self.np_random = np.random.default_rng(seed)

        # --- Speicher für das Dataset im demos_to_feature_dataset-Format ---
        self.dataset = dict(
            observations=[None] * capacity,        # list of dicts: {"state":..., "pixels":...}
            next_observations=[None] * capacity,   # same structure
            actions=np.zeros((capacity, *action_sample.shape), dtype=action_sample.dtype),
            rewards=np.zeros((capacity,), dtype=np.float32),
            dones=np.zeros((capacity,), dtype=np.float32),
            masks=np.zeros((capacity,), dtype=np.float32),
        )

    def insert(self, data_dict: dict):
        idx = self.insert_index

        # direkte Speicherung der dicts
        self.dataset["observations"][idx] = copy.deepcopy(data_dict["observations"])
        self.dataset["next_observations"][idx] = copy.deepcopy(data_dict["next_observations"])
        self.dataset["actions"][idx] = data_dict["actions"]
        self.dataset["rewards"][idx] = data_dict["rewards"]
        self.dataset["dones"][idx] = data_dict["dones"]
        self.dataset["masks"][idx] = data_dict["masks"]

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.np_random.integers(self.size, size=batch_size)

        # ---------- OBS ----------
        obs_states = np.stack(
            [self.dataset["observations"][i]["state"] for i in idx],
            axis=0,
        )

        obs_pixels = np.stack(
            [self.dataset["observations"][i]["pixels"] for i in idx],
            axis=0,
        )

        # ---------- NEXT OBS ----------
        next_obs_states = np.stack(
            [self.dataset["next_observations"][i]["state"] for i in idx],
            axis=0,
        )

        next_obs_pixels = np.stack(
            [self.dataset["next_observations"][i]["pixels"] for i in idx],
            axis=0,
        )

        batch = dict(
            observations=dict(
                state=obs_states,
                pixels=obs_pixels,
            ),
            next_observations=dict(
                state=next_obs_states,
                pixels=next_obs_pixels,
            ),
            actions=self.dataset["actions"][idx],
            rewards=self.dataset["rewards"][idx],
            dones=self.dataset["dones"][idx],
            masks=self.dataset["masks"][idx],
        )

        return frozen_dict.freeze(batch)
