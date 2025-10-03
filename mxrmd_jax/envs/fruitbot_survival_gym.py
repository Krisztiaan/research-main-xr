from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any

try:
    from procgen import ProcgenEnv
    from gym3 import ToGymEnv
except Exception as e:
    ProcgenEnv = None

class FruitBotSurvivalGym(gym.Env):
    """Gymnasium outward wrapper for Procgen FruitBot with survival+reset chain."""
    metadata = {"render_modes": []}

    def __init__(self, R_reset: int = 50, distribution: str = "easy", seed: int = 0):
        assert ProcgenEnv is not None, "Install procgen to use FruitBotSurvivalGym."
        self.raw = ProcgenEnv(num_envs=1, env_name="fruitbot", render_mode="rgb_array",
                              center_agent=True, use_generated_assets=True, distribution_mode=distribution,
                              use_sequential_levels=False)
        self.env = ToGymEnv(self.raw)
        self.R_reset = R_reset
        self._resets_left = 0
        obs = self.env.reset()
        obs = obs[0]  # single env
        self.observation_space = spaces.Box(low=0, high=255, shape=obs.shape, dtype=np.uint8)
        self.action_space = self.env.action_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()[0]
        self._resets_left = 0
        return obs.astype(np.uint8), {}

    def step(self, action: int):
        if self._resets_left > 0:
            self._resets_left -= 1
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            reward = 0.0
            terminated = False; truncated = False
            info = {"life_ended": False, "reset_phase": True}
            if self._resets_left == 0:
                obs = self.env.reset()[0]
            return obs.astype(np.uint8), reward, terminated, truncated, info

        obs, rew, done, info_raw = self.env.step(np.array([action]))
        obs = obs[0]
        done = bool(done[0])
        reward = float(1.0)  # survival reward
        terminated = False; truncated = False
        info = {"life_ended": done, "reset_phase": False}
        if done:
            self._resets_left = self.R_reset
        return obs.astype(np.uint8), reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self.env.close()
