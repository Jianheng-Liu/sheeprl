from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from SyntheticToolkit.Sensors.Camera import DepthCamera

class IsaacSimWrapper(gym.Wrapper):
    def __init__(
        self, 
        map_size=(128, 128), 
        camera_rgb_size=(128, 128),
        camera_height=1.0,
        camera_linear_velocity=0.5,
        camera_angular_velocity=1.0,
        camera_parent="/World",
        camera_name="Camera",
        seed: Optional[int]=None,
     ):
        # Initialize the parent class and environment
        super().__init__(gym.Env())
        self.map_size = map_size
        self.camera_rgb_size = camera_rgb_size
        self.camera_height = camera_height
        self.camera_linear_velocity = camera_linear_velocity
        self.camera_angular_velocity = camera_angular_velocity
        self.camera_parent = camera_parent
        self.camera_name = camera_name
        
        self.map_pixel_values = {
            'Empty': int(0),
            'Static Obstacle': int(1.0/8.0*255),
            'Dynamic Obstacle': int(2.0/8.0*255),
            'Dynamic Obstacle Trajectory': int(3.0/8.0*255),
            'Position of Interest': int(4.0/8.0*255),
            'Agent': int(5.0/8.0*255),
            'Agent Trajectory': int(6.0/8.0*255),
            'A* Path': int(7.0/8.0*255),
            'Goal': int(255)
        }

        self.map_global = None
        self.empty_positions = []

        self._render_mode: str = "rgb_array"
            

        # Define observation and action spaces
        self._observation_space = spaces.Dict({
            "position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "local_map": spaces.Box(low=0, high=255, shape=(map_size[0], map_size[1]), dtype=np.uint8),
            "rgb_image": spaces.Box(low=0, high=255, shape=(3, camera_rgb_size[0], camera_rgb_size[1]), dtype=np.uint8)
        })

        self._action_space = spaces.Discrete(4)  # One-hot, 0: forward, 1: rotate left, 2: rotate right, 3: no movement
        # 2d array, 0: linear, 1: angular
        self._action_mapping = {
            0 : [[1, 0, 0], [0, 0, 0]],
            1 : [[0, 0, 0], [0, 0, 1]],
            2 : [[0, 0, 0], [0, 0, -1]],
            3 : [[0, 0, 0], [0, 0, 0]]
        }
        self._reward_range = (-np.inf, np.inf)

        self._agent_ref = None

        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

    """def _collision_callback(self, _, __):
        for _ in range():
            for __ in range():
                if __ == __:
                    self._collided = True"""
    
    def set_camera_name(self, env_idx):
        self.camera_name = self.camera_name + str(env_idx)
        

    def get_global_map(self, map_global):
        self.map_global = map_global


    def pre_step(self, action):
        # do whatver
        #forward - > [1,0,0]
        velocs = self._action_mapping[action]
        lin = velocs[0]
        ang = velocs[1]

        self._agent_ref.apply_velocity(linear_veloc = lin, angular_veloc = ang)

    def post_step(self, action):
        # compute observations and rewards.
        # {"rgb": ___, "depth": _____, "segmentation":___}
        obs_dict = self._agent_ref.compute_observations()

        # check for collisions.
        #if self._collided:
        #    done = True



        # calculate rewards
        #return 
        return obs, reward, done, truncated, {}


        
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

        # update the map
        # use the pos of agent.
        # and its previous positioons

        # calculate global path

        # Generate a random RGB image
        rgb_image = self._generate_random_image(self.camera_rgb_size)
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

        # camera_xy = np.random.choice(empty_positions)

        # self.camera = DepthCamera(
        #     position=(camera_xy[0], camera_xy[1], 0), 
        #     rotation=(0, 0, -1),
        #     parent="/World",
        #     name="Camera_1"
        # )
        self.camera = DepthCamera(
            position=(0, 0, 0), 
            rotation=(0, 0, -1),
            parent="/World",
            name="Camera_1"
        )
        self.camera.init_sensor(parent="/World")

        # Init at empty position
        # Get obs


        
        obs = self.compute_obs()
        # calculate where to spawn.
        # local_map
        # check ^ for an empty space
        #valid = False
        #while not valid:
        #    x = random.randint()
        
        #self._agent_ref.reset(translate, orientation)
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
        return self._generate_random_image(self.camera_rgb_size)

    def close(self):
        pass

    def _generate_random_map(self, size):
        # Generate a random local map
        return np.random.randint(0, 256, size, dtype=np.uint8)

    def _generate_random_image(self, size):
        # Generate a random RGB image
        return np.random.randint(0, 256, (3, *size), dtype=np.uint8)
