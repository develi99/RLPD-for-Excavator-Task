import copy
from typing import Iterable, Optional, Tuple

import numpy as np
from flax.core import frozen_dict

from rlpd.data.dataset import DatasetDict, _sample
from rlpd.data.replay_buffer_sample import ReplayBufferSample


class MemoryEfficientReplayBufferSample(ReplayBufferSample):
    def __init__(
        self,
        observation_sample,
        action_sample,
        capacity: int,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        next_observation_sample: Optional = None,
    ):
        """
        observation_sample: einzelnes Observation-Dict
        action_sample: einzelnes Action-Array
        """

        self.pixel_keys = pixel_keys

        if next_observation_sample is None:
            next_observation_sample = observation_sample

        observation_sample = copy.deepcopy(observation_sample)
        next_observation_sample = copy.deepcopy(next_observation_sample)

        # ---- Pixel-Unstack vorbereiten ----
        self._num_stack = None
        for pixel_key in self.pixel_keys:
            pixel_obs = observation_sample[pixel_key]

            if self._num_stack is None:
                self._num_stack = pixel_obs.shape[-1]
            else:
                assert self._num_stack == pixel_obs.shape[-1]

            # remove stack dimension
            observation_sample[pixel_key] = pixel_obs[..., 0]

        # next_obs speichert KEINE pixel keys
        next_obs_dict = next_observation_sample.copy()
        for pixel_key in self.pixel_keys:
            next_obs_dict.pop(pixel_key)

        self._first = True
        self._is_correct_index = np.full(capacity, False, dtype=bool)

        super().__init__(
            observation_sample=observation_sample,
            action_sample=action_sample,
            capacity=capacity,
            next_observation_sample=next_obs_dict,
        )

    # ---------------------------------------------------
    # Insert (identisch zur Original-Logik)
    # ---------------------------------------------------

    def insert(self, data_dict: DatasetDict):
        if (
            self._insert_index == 0
            and self._capacity == len(self)
            and not self._first
        ):
            indxs = np.arange(len(self) - self._num_stack, len(self))
            for indx in indxs:
                element = super().sample(1, indx=indx)
                self._is_correct_index[self._insert_index] = False
                super().insert(element)

        data_dict = data_dict.copy()
        data_dict["observations"] = data_dict["observations"].copy()
        data_dict["next_observations"] = data_dict["next_observations"].copy()

        obs_pixels = {}
        next_obs_pixels = {}

        for pixel_key in self.pixel_keys:
            obs_pixels[pixel_key] = data_dict["observations"].pop(pixel_key)
            next_obs_pixels[pixel_key] = data_dict["next_observations"].pop(pixel_key)

        # Initial Stack füllen
        if self._first:
            for i in range(self._num_stack):
                for pixel_key in self.pixel_keys:
                    data_dict["observations"][pixel_key] = obs_pixels[pixel_key][..., i]

                self._is_correct_index[self._insert_index] = False
                super().insert(data_dict)

        # nur letztes Frame speichern
        for pixel_key in self.pixel_keys:
            data_dict["observations"][pixel_key] = next_obs_pixels[pixel_key][..., -1]

        self._first = data_dict["dones"]

        self._is_correct_index[self._insert_index] = True
        super().insert(data_dict)

        for i in range(self._num_stack):
            indx = (self._insert_index + i) % len(self)
            self._is_correct_index[indx] = False

    # ---------------------------------------------------
    # Sample (unverändert außer Space-Abhängigkeit entfernt)
    # ---------------------------------------------------

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> frozen_dict.FrozenDict:

        if indx is None:
            indx = self.np_random.integers(len(self), size=batch_size)
            for i in range(batch_size):
                while not self._is_correct_index[indx[i]]:
                    indx[i] = self.np_random.integers(len(self))
        else:
            raise NotImplementedError()

        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            assert "observations" in keys

        keys = list(keys)
        keys.remove("observations")

        batch = super().sample(batch_size, keys, indx)
        batch = batch.unfreeze()

        obs_keys = list(self.dataset_dict["observations"].keys())
        for pixel_key in self.pixel_keys:
            obs_keys.remove(pixel_key)

        batch["observations"] = {}

        for k in obs_keys:
            batch["observations"][k] = _sample(
                self.dataset_dict["observations"][k], indx
            )

        for pixel_key in self.pixel_keys:
            obs_pixels = self.dataset_dict["observations"][pixel_key]

            obs_pixels = np.lib.stride_tricks.sliding_window_view(
                obs_pixels, self._num_stack + 1, axis=0
            )

            obs_pixels = obs_pixels[indx - self._num_stack]

            if pack_obs_and_next_obs:
                batch["observations"][pixel_key] = obs_pixels
            else:
                batch["observations"][pixel_key] = obs_pixels[..., :-1]
                if "next_observations" in keys:
                    batch["next_observations"][pixel_key] = obs_pixels[..., 1:]

        return frozen_dict.freeze(batch)
