import gymnasium
import numpy as np


class NormalizeObsWrapper(gymnasium.Wrapper):
    def _normalize(self, obs: np.ndarray):
        if obs.ndim == 2:
            obs = obs[..., np.newaxis]
        if isinstance(obs, np.ndarray) and obs.dtype == np.uint8:
            return obs.astype(np.float32) / 255.0

        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info
