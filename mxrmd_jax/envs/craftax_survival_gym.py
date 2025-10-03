from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    import jax, jax.numpy as jnp
    from craftax import make as make_craftax
except Exception as e:
    make_craftax = None

class CraftaxSurvivalGym(gym.Env):
    """Gymnasium outward wrapper with internal Craftax (JAX) stepping.
    Survival objective: r=1 per alive step; after termination, enter a policy-independent reset chain (R_reset steps with r=0).
    Observations during reset chain are zeros; then respawn (Craftax reset).
    """
    metadata = {"render_modes": []}

    def __init__(self, env_id: str = "craftax-classic-v1", R_reset: int = 50, seed: int = 0):
        assert make_craftax is not None, "Install craftax to use CraftaxSurvivalGym."
        self.R_reset = R_reset
        self.key = jax.random.PRNGKey(seed)
        self.env, self.params = make_craftax(env_id)
        self.state = None
        # Do a dummy reset to infer spaces
        self.key, sub = jax.random.split(self.key)
        obs, self.state = self.env.reset(sub, self.params)
        obs = np.array(obs)
        self.observation_space = spaces.Box(low=0, high=1, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(self.env.num_actions(self.params))
        self._resets_left = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.key = jax.random.PRNGKey(int(seed))
        self.key, sub = jax.random.split(self.key)
        obs, self.state = self.env.reset(sub, self.params)
        self._resets_left = 0
        return np.array(obs, dtype=np.float32), {}

    def step(self, action: int):
        if self._resets_left > 0:
            self._resets_left -= 1
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = False; truncated = False
            info = {"life_ended": False, "reset_phase": True}
            # When resets reach zero, we respawn next call
            if self._resets_left == 0:
                self.key, sub = jax.random.split(self.key)
                obs_j, self.state = self.env.reset(sub, self.params)
                obs = np.array(obs_j, dtype=np.float32)
            return obs, reward, terminated, truncated, info

        # normal step
        self.key, sub = jax.random.split(self.key)
        obs_j, state_j, rew_j, done_j, info_j = self.env.step(sub, self.state, int(action), self.params)
        self.state = state_j
        # Survival reward and termination handling
        obs = np.array(obs_j, dtype=np.float32)
        reward = float(1.0)  # alive step
        terminated = False; truncated = False
        info = {"life_ended": bool(done_j), "reset_phase": False}
        if bool(done_j):
            # enter reset chain AFTER giving alive reward
            self._resets_left = self.R_reset
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass
