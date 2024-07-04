from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class IsaacSimWrapper(gym.Wrapper):
    def __init__(
        self, 
        map_size=(84, 84), 
        rgb_size=(84, 84),
        seed: Optional[int] = None,
     ):
        # Initialize the parent class and environment
        super().__init__(gym.Env())
        self.map_size = map_size
        self.rgb_size = rgb_size

        self._render_mode: str = "rgb_array"

        # Define observation and action spaces
        self._observation_space = spaces.Dict({
            "position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "local_map": spaces.Box(low=0, high=255, shape=(*map_size, 1), dtype=np.uint8),
            "rgb_image": spaces.Box(low=0, high=255, shape=(*rgb_size, 3), dtype=np.uint8)
        })

        self._action_space = spaces.Discrete(4)  # 0: forward, 1: rotate left, 2: rotate right, 3: no movement
        self._reward_range = (-np.inf, np.inf)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
    
    @property
    def reward_range(self) -> Tuple[float, float]:
        return self._reward_range

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def compute_obs(self):
        # Generate a random position
        position = np.random.rand(2) * 10  # Example position
        # Generate a random local map
        local_map = self._generate_random_map(self.map_size)
        # Generate a random RGB image
        rgb_image = self._generate_random_image(self.rgb_size)
        # Combine the observations into a dictionary
        obs = {
            "position": np.array(position),
            "local_map": local_map,
            "rgb_image": rgb_image
        }
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # Reset the environment
        if seed is not None:
            np.random.seed(seed)
        obs = self.compute_obs()
        return obs, {}

    def step(self, action):
        # Simulate the effect of the action on the position
        position = np.random.rand(2) * 10  # Example updated position after action
        reward = -np.linalg.norm(position)  # Example reward based on position
        done = False  # Example done condition
        truncated = False  # Example truncated condition
        obs = self.compute_obs()
        return obs, reward, done, truncated, {}

    def render(self, mode='rgb'):
        # Return a RGB image (to be comatible with RecordVideo)
        return self._generate_random_image(self.rgb_size)

    def close(self):
        pass

    def _generate_random_map(self, size):
        # Generate a random local map
        return np.random.randint(0, 256, (*size, 1), dtype=np.uint8)

    def _generate_random_image(self, size):
        # Generate a random RGB image
        return np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
