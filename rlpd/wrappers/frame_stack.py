import collections

import gym
import numpy as np
from gym.spaces import Box,Dict
import cv2

# class FrameStack(gym.Wrapper):
#     def __init__(self, env, num_stack: int, stacking_key: str = "pixels", img_size=64):
#         super().__init__(env)
#         self._num_stack = num_stack
#         self._stacking_key = stacking_key

#         # Wrong for algoryx
#         # assert stacking_key in self.observation_space.spaces
#         # pixel_obs_spaces = self.observation_space.spaces[stacking_key]

#         # self._env_dim = pixel_obs_spaces.shape[-1]

#         # low = np.repeat(pixel_obs_spaces.low[..., np.newaxis], num_stack, axis=-1)
#         # high = np.repeat(pixel_obs_spaces.high[..., np.newaxis], num_stack, axis=-1)
#         # new_pixel_obs_spaces = Box(low=low, high=high, dtype=pixel_obs_spaces.dtype)
#         # self.observation_space.spaces[stacking_key] = new_pixel_obs_spaces

#         self._frames = collections.deque(maxlen=num_stack)
#         self.img_size = img_size

#     def reset(self):
#         obs = self.env.reset()
#         for i in range(self._num_stack):
#             img = obs[self._stacking_key]
#             img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
#             self._frames.append(img)
#         obs[self._stacking_key] = self.frames
#         return obs

#     @property
#     def frames(self):
#         return np.stack(self._frames, axis=-1)

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         img = obs[self._stacking_key]
#         img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
#         self._frames.append(img)
#         obs[self._stacking_key] = self.frames
#         return obs, reward, done, info



class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int, img_size=64):
        super().__init__(env)

        self.num_stack = num_stack
        self.img_size = img_size
        self.frames = collections.deque(maxlen=num_stack)

        obs_space = env.observation_space
        assert "camera" in obs_space.spaces
        assert "rgb" in obs_space.spaces["camera"].spaces

        rgb_space = obs_space.spaces["camera"].spaces["rgb"]

        # rgb_space shape: (3, H, W) oder (H, W, 3)
        if rgb_space.shape[0] == 3:
            c, h, w = rgb_space.shape
            new_shape = (3, img_size, img_size, num_stack)
        else:
            h, w, c = rgb_space.shape
            new_shape = (img_size, img_size, c, num_stack)

        low = np.repeat(rgb_space.low[..., None], num_stack, axis=-1)
        high = np.repeat(rgb_space.high[..., None], num_stack, axis=-1)

        # Observation space korrekt anpassen
        new_obs_space = obs_space
        new_obs_space.spaces["camera"].spaces["rgb"] = Box(
            low=low.min(),
            high=high.max(),
            shape=new_shape,
            dtype=rgb_space.dtype,
        )

        self.observation_space = new_obs_space

    def _process_frame(self, frame):
        frame = np.asarray(frame)

        # frame: (3, H, W) → (H, W, 3)
        if frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))

        frame = cv2.resize(frame, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_AREA)
        return frame
    
    def reset(self):
        obs, _ = self.env.reset()

        frame = self._process_frame(obs["camera"]["rgb"])
        for _ in range(self.num_stack):
            self.frames.append(frame)

        obs["camera"]["rgb"] = np.stack(self.frames, axis=-1)
        obs["policy"] = np.asarray(obs["policy"])
        return obs, _
    
    # @property
    # def frames(self):
    #     # (3, img_size, img_size, num_stack)
    #     return np.stack(self._frames, axis=-1)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        frame = self._process_frame(obs["camera"]["rgb"])
        self.frames.append(frame)

        obs["camera"]["rgb"] = np.stack(self.frames, axis=-1)
        obs["policy"] = np.asarray(obs["policy"])
        return obs, reward, terminated, truncated, info
