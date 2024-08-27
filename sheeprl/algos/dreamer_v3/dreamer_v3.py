"""Dreamer-V3 implementation from [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
Adapted from the original implementation from https://github.com/danijar/dreamerv3
"""

from __future__ import annotations

import copy
import os
import warnings
from functools import partial
from typing import Any, Dict, Sequence

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.distributions import Distribution, Independent, OneHotCategorical
from torch.optim import Optimizer
from torchmetrics import SumMetric

from sheeprl.algos.dreamer_v3.agent import WorldModel, build_agent
from sheeprl.algos.dreamer_v3.loss import reconstruction_loss
from sheeprl.algos.dreamer_v3.utils import Moments, compute_lambda_values, prepare_obs, test
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.envs.wrappers import RestartOnException
from sheeprl.utils.distribution import (
    BernoulliSafeMode,
    MSEDistribution,
    SymlogDistribution,
    TwoHotEncodingDistribution,
)
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import Ratio, save_configs

# Decomment the following two lines if you cannot start an experiment with DMC environments
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ["MUJOCO_GL"] = "osmesa"

from collections import OrderedDict
import math
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.physx import get_physx_simulation_interface
from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics, UsdGeom

from map_utils import map_generation, find_empty_points, get_poi
from map_utils import position_meter_to_pixel, position_pixel_to_meter, waypoints_2d_to_3d
from a_star import a_star
from gridmap import OccupancyGridMap
from SyntheticToolkit.core.dynamic_objects import DynamicObject
from SyntheticToolkit.utils.omni_utils import euler_to_quaternion

physics_dt = 0.1

# Keywords to check for map generation
map_keywords_to_check=[
    'RackPile', 
    'RackLong', 
    'Forklift', 
    'Table', 
    'IndustrialSteelShelving', 
    'WallA', 
    'RackShelf', 
    'RackFrame', 
    'RackShield',
    'PaletteA',
    'EmergencyBoardFull',
    'FuseBox',
    'FireExtinguisher'
]


map_size = (128, 128)
map_resolution = 0.25
map_offset = (26.5, 0.5, 0.0)
occupancy_map = np.zeros(map_size)
map_global = np.zeros(map_size)
min_clear_radius = 2
min_edge_distance = 2

success_rate = 0.0
ready_to_upgrade = False

curriculum = 1
max_curriculum = 6
curriculum_threshold = 0.75
worker_number_mapping = {
    1: 4,
    2: 4,
    3: 4,
    4: 4,
    5: 4,
    6: 4,
}

map_pixel_values = {
    'Empty': int(0),
    'Static Obstacle': int(1.0/9.0*255),
    'Dynamic Obstacle': int(2.0/9.0*255),
    'Dynamic Obstacle Trajectory': int(3.0/9.0*255),
    'Position of Interest': int(4.0/9.0*255),
    'Agent': int(5.0/9.0*255),
    'Agent Trajectory': int(6.0/9.0*255),
    'Agent Orientation': int(7.0/9.0*255),
    'A* Path': int(8.0/9.0*255),
    'Goal': int(255)
}

empty_points = [] 
empty_positions = []

poi_points = []
poi_positions = []
poi_available_idxs = []

worker_number = 8
worker_list = []
worker_min_distance = 0.5

class Worker(DynamicObject):
    def __init__(
        self,
        position,
        orientation,
        scale,
        prim_name,
        parent_path,
        stage,
        usd_path=None,
        semantic_class="None",
        instanceable=False,
        visibility="inherited",
        disable_gravity=True,
        scale_delta=0,
    ) -> None:
        super().__init__(
            position,
            orientation,
            scale,
            prim_name,
            parent_path,
            stage,
            usd_path,
            semantic_class,
            instanceable,
            visibility,
            disable_gravity,
            scale_delta
        )

        self.position_origin = position
        self.done = False
        # self.stuck = False
        self.offset = map_offset

        self.trajectory = []
        self.trajectory_length = 32

        self.position_target = [0.0, 0.0]
        self.position_target_idx = None
        self.worker_min_distance = worker_min_distance

        self.prim_name = prim_name
        self.collision = False
        self.contact_report_sub = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                self.on_contact_report_event
            )
        )

        # Add collider
        prim = stage.GetPrimAtPath(self._prim_path)
        collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
        cylinderGeom = UsdGeom.Cylinder.Define(stage, self._prim_path + "/Collider")
        cylinderGeom.GetHeightAttr().Set(1.5)  # Set Height (m)
        cylinderGeom.GetRadiusAttr().Set(0.25)  # Set Radius (m)
        cylinderGeom.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)  # Invisible

    def on_contact_report_event(self, contact_headers, contact_data):
        
        for contact_header in contact_headers:
            act0_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            act1_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
            # print(f"Collision detected between {act0_path} and {act1_path}")

            # if (self.prim_name in act0_path or self.prim_name in act1_path) and "GroundPlane" not in act0_path and "GroundPlane" not in act1_path:
            if self.prim_name in act0_path or self.prim_name in act1_path:
                if "Camera" not in act0_path and "Camera" not in act1_path:
                    if "RecRed" not in act0_path and "RecRed" not in act1_path:
                        self.collision = True
                        print(f"Collision detected between {act0_path} and {act1_path}")
                        return
            

    def set_worker_waypoints(self, waypoints_a_star, poi_idx):
        waypoints_worker = []
        for waypoint in waypoints_a_star:
            waypoints_worker.append([waypoint[0]-self.offset[0], waypoint[1]-self.offset[1], waypoint[2]-self.offset[2]])
        # print('Worker waypoints: ', waypoints_worker)
        self.position_target = [waypoints_worker[-1][0], waypoints_worker[-1][1]]
        self.position_target_idx = poi_idx
        self.init_waypoints(waypoints_worker)
        self.done = False
        # self.stuck = False
        self.collision = False

    def update_trajectory(self):
        position_current, _ = self.get_pos_rot()
        position_meter = [position_current[0], position_current[1]]
        position_pixel = position_meter_to_pixel(position_meter=position_meter, resolution=map_resolution, offset=map_offset)
        if len(self.trajectory) < self.trajectory_length:
            self.trajectory.append(position_pixel)
        else:
            self.trajectory.pop(0)
            self.trajectory.append(position_pixel)
    
    def is_done(self):
        translation = self.get_translate()
        position_current = [translation[0], translation[1]]
        distance = np.linalg.norm(np.array(self.position_target) - np.array(position_current))
        if distance < self.worker_min_distance:
            self.done = True
        return self.done
    
    # def is_stuck(self):
    #     self.stuck = False
    #     if len(self.trajectory) == self.trajectory_length:
    #         first_pos = self.trajectory[0]
    #         for pos in self.trajectory:
    #             if pos != first_pos:
    #                 self.stuck = False
    #                 return self.stuck
    #     self.stuck = True
    #     return self.stuck

        # translation = self.get_translate()
        # position_current = [translation[0], translation[1]]
        # distance = np.linalg.norm(np.array(self.position_target) - np.array(position_current))
        # if distance < self.worker_min_distance:
        #     self.done = True
        # return self.done
    

# ==================== Imagine ====================
import imageio
import numpy as np
from torch.utils.tensorboard import SummaryWriter

color_map = {
    'Empty': (255, 255, 255),  # White
    'Static Obstacle': (0, 0, 0),  # Black
    'Dynamic Obstacle': (255, 0, 0),  # Red
    'Dynamic Obstacle Trajectory': (255, 165, 0),  # Orange
    'Position of Interest': (0, 255, 0),  # Green
    'Agent': (0, 0, 255),  # Blue
    'Agent Trajectory': (75, 0, 130),  # Indigo
    'Agent Orientation': (255, 127, 80),  # Coral
    'A* Path': (238, 130, 238),  # Violet
    'Goal': (255, 255, 0)  # Yellow
}

def paint_neighbor(map, point, value):
    neighbors = [
        [point[0], point[1]],
        [point[0]+1, point[1]],
        [point[0]-1, point[1]],
        [point[0], point[1]+1],
        [point[0], point[1]-1],
        [point[0]+1, point[1]+1],
        [point[0]+1, point[1]-1],
        [point[0]-1, point[1]+1],
        [point[0]-1, point[1]-1],
        [point[0]+2, point[1]],
        [point[0]-2, point[1]],
        [point[0], point[1]+2],
        [point[0], point[1]-2]
    ]
    for neighbor in neighbors:
        if 0 <= neighbor[0] < map_size[0] and 0 <= neighbor[1] < map_size[1]:
            map[neighbor[0], neighbor[1]] = value

def get_closest_category(pixel_value):
    closest_category = 'Empty'
    min_diff = float('inf')
    for category, value in map_pixel_values.items():
        diff = abs(pixel_value - value)
        if diff < min_diff:
            min_diff = diff
            closest_category = category
    return closest_category

def convert_to_color_image(single_channel_obs):
    height, width = single_channel_obs.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            category = get_closest_category(single_channel_obs[i, j])
            color_image[i, j] = color_map[category]

    return color_image

def imagine(
    fabric: Fabric,
    world_model: WorldModel,
    actor: _FabricModule,
    stochastic_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    actions: torch.Tensor,
    horizon: int,
    action_space,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Imagine trajectories in the latent space and reconstruct observations.
    (For Now Only Support For env: num_envs: 1)

    Args:
        fabric (Fabric): the fabric instance.
        world_model (WorldModel): the world model instance.
        actor (_FabricModule): the actor model.
        stochastic_state (torch.Tensor): the initial stochastic state.
        recurrent_state (torch.Tensor): the initial recurrent state.
        actions (torch.Tensor): the action gonna take.
        horizon (int): the number of steps to imagine.
        action_space: the action space of the environment to get the action dimension.
        cfg (Dict[str, Any]): the configuration dictionary.
        
    Returns:
        Dict[str, np.ndarray]: the reconstructed observations.
        np.ndarray: the imagined actions.
    """
    device = fabric.device
    batch_size = stochastic_state.size(1)
    stoch_state_size = stochastic_state.size(-1)
    recurrent_state_size = recurrent_state.size(-1)

    # Determine action dimension based on action space
    if hasattr(action_space, 'n'):
        action_dim = action_space.n
    else:
        action_dim = np.prod(action_space.shape)

    imagined_latent_states = torch.cat((stochastic_state, recurrent_state), -1)

    # Initialize tensors for imagined trajectories and actions
    imagined_trajectories = torch.empty(
        horizon,
        batch_size,
        stoch_state_size + recurrent_state_size,
        device=device,
    )

    imagined_actions = torch.empty(
        horizon,
        batch_size,
        action_dim,
        device=device,
    )

    # Imagine trajectories in the latent space
    for i in range(horizon):
        imagined_prior, recurrent_state = world_model.rssm.imagination(stochastic_state, recurrent_state, actions)
        imagined_prior = imagined_prior.view(1, batch_size, stoch_state_size)
        imagined_latent_states = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_states
        actions = torch.cat(actor(imagined_latent_states.detach())[0], dim=-1)
        imagined_actions[i] = actions

    # Compute the final reconstructed observations
    imagined_trajectories = imagined_trajectories.view(-1, stoch_state_size + recurrent_state_size)
    reconstructed_obs = world_model.observation_model(imagined_trajectories)

    # Convert tensors to numpy arrays and reverse normalization
    reconstructed_obs_np = {}
    for k in cfg.algo.cnn_keys.decoder:
        reconstructed_obs_np[k] = (reconstructed_obs[k].detach().cpu().numpy() + 0.5) * 255.0
        reconstructed_obs_np[k] = np.clip(reconstructed_obs_np[k], 0, 255).astype(np.uint8)
    for k in cfg.algo.mlp_keys.decoder:
        reconstructed_obs_np[k] = reconstructed_obs[k].detach().cpu().numpy()

    imagined_actions_np = imagined_actions.detach().cpu().numpy()

    return reconstructed_obs_np, imagined_actions_np


def save_as_gif(
    reconstructed_obs_np: Dict[str, np.ndarray], 
    key: str, 
    output_path: str, 
    duration: float = 0.1
):
    """
    Save the specified observation from reconstructed_obs_np as a gif.

    Args:
        reconstructed_obs_np (Dict[str, np.ndarray]): The reconstructed observations as a dictionary of numpy arrays.
        key (str): The key for the observation to save as a gif.
        output_path (str): The output path for the gif.
        duration (float): Duration of each frame in the gif.
    """
    if key not in reconstructed_obs_np:
        print(f"Key '{key}' not found in the reconstructed observations.")
        return

    observation = reconstructed_obs_np[key]

    # Assuming observation shape is (horizon, channels, height, width)
    # We need to transpose it to (horizon, height, width, channels) for imageio
    observation = observation.transpose(0, 2, 3, 1)

    # Convert observation to list of images
    images = [observation[i] for i in range(observation.shape[0])]

    # Save as gif
    imageio.mimsave(output_path, images, duration=duration)
    print(f"Saved gif to {output_path}")


def log_imagination_as_gif(
    fabric,
    reconstructed_obs_np: Dict[str, np.ndarray],
    key: str,
    cfg: Dict[str, Any],
    gif_name: str,
    duration: float = 0.1,
):
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)

    if key not in reconstructed_obs_np:
        print(f"Key '{key}' not found in the reconstructed observations.")
        return

    observation = reconstructed_obs_np[key]

    if key == 'map':
        # print('Map Shape: ', observation.shape)
        for channel_idx in range(2):
            single_channel_obs = observation[:, channel_idx, :, :]  # (horizon, height, width)
            color_observation = np.stack([convert_to_color_image(frame) for frame in single_channel_obs])
            for i, frame in enumerate(color_observation):
                imageio.imwrite(f'./visualize/imagination/channel_{channel_idx}_frame_{i}.png', frame)
            color_observation = np.clip(color_observation, 0, 255).astype(np.uint8)
            # print('imagine map color observation shape: ', color_observation.shape)
            # magine map color observation shape:  (64, 128, 128, 3)
            # imagine map observation tensor shape:  torch.Size([1, 64, 3, 128, 128])

            # Save GIF locally
            # print('color_observation dtype: ', color_observation.dtype)
            # print(color_observation)
            color_observation = np.array(color_observation)

            # gif_path = os.path.join(f"{gif_name}_channel_{channel_idx}.gif")
            # imageio.mimsave(gif_path, color_observation, duration=duration)
            # print(f"Saved GIF to {gif_path}")

            observation_tensor = torch.tensor(color_observation, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0)  # (1, horizon, 3, height, width)
            # print('imagine map observation tensor shape: ', observation_tensor.shape)

            # [0, 1]
            observation_tensor /= 255.0

            # print('imagine map tensor max value: ', torch.max(observation_tensor))
            # print('imagine map tensor min value: ', torch.min(observation_tensor))

            # print(f'Map Single Channel {channel_idx} Shape: ', observation_tensor.shape)
            # print('Observation tensor dtype:', observation_tensor.dtype)

            writer = SummaryWriter(log_dir=log_dir)
            writer.add_video(f"{gif_name}_channel_{channel_idx}", observation_tensor, fps=int(1/duration))
            writer.close()

    # else:
    #     # Save GIF locally
    #     # gif_path = os.path.join(f"{gif_name}_rgb.gif")
    #     # imageio.mimsave(gif_path, observation, duration=duration)
    #     # print(f"Saved GIF to {gif_path}")
    #     # # print('imagine observation shape: ', observation.shape)
    #     # imagine observation shape:  (64, 3, 128, 128)
    #     # imagine observation tensor shape:  torch.Size([1, 64, 3, 128, 128])
        
    #     observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # (1, horizon, channels, height, width)
    #     # print('imagine observation tensor shape: ', observation_tensor.shape)
    #     observation_tensor /= 255.0

    #     # print('imagine tensor max value: ', torch.max(observation_tensor))
    #     # print('imagine tensor min value: ', torch.min(observation_tensor))

    #     writer = SummaryWriter(log_dir=log_dir)
    #     writer.add_video(gif_name, observation_tensor, fps=int(1/duration))
    #     writer.close()
    
    print(f"Logged gif to TensorBoard at {log_dir}")


def log_true_observation_as_gif(
    fabric,
    true_obs_np: Dict[str, np.ndarray],
    key: str,
    cfg: Dict[str, Any],
    gif_name: str,
    duration: float = 0.1,
):
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)

    if key not in true_obs_np:
        print(f"Key '{key}' not found in the reconstructed observations.")
        return

    observation = true_obs_np[key][0, :, 0, :, :, :]

    # print('True Observation Shape: ', observation.shape)

    if key == 'map':
        for channel_idx in range(2): 
            single_channel_obs = observation[:, channel_idx, :, :]  # (horizon, height, width)
            color_observation = np.stack([convert_to_color_image(frame) for frame in single_channel_obs])
            color_observation = np.clip(color_observation, 0, 255).astype(np.uint8)

            for i, frame in enumerate(color_observation):
                imageio.imwrite(f'./visualize/true/channel_{channel_idx}_frame_{i}.png', frame)

            # Save GIF locally
            # gif_path = os.path.join(f"{gif_name}_channel_{channel_idx}.gif")
            # imageio.mimsave(gif_path, color_observation, duration=duration)
            # print(f"Saved GIF to {gif_path}")
            
            observation_tensor = torch.tensor(color_observation, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0)  # (1, horizon, 3, height, width)

            observation_tensor /= 255.0

            # print('true map tensor max value: ', torch.max(observation_tensor))
            # print('true map tensor min value: ', torch.min(observation_tensor))

            # print(f'Map Single Channel {channel_idx} Shape: ', observation_tensor.shape)
            # print('Observation tensor dtype:', observation_tensor.dtype)

            writer = SummaryWriter(log_dir=log_dir)
            writer.add_video(f"{gif_name}_channel_{channel_idx}", observation_tensor, fps=int(1/duration))
            writer.close()
    # else:
    #     # Save GIF locally
    #     # gif_path = os.path.join(f"{gif_name}_rgb.gif")
    #     # imageio.mimsave(gif_path, color_observation, duration=duration)
    #     # print(f"Saved GIF to {gif_path}")

    #     observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # (1, horizon, channels, height, width)
    #     # print('observation tensor shape: ', observation_tensor.shape)
    #     observation_tensor /= 255.0
    #     # print('true tensor max value: ', torch.max(observation_tensor))
    #     # print('true tensor min value: ', torch.min(observation_tensor))

    #     writer = SummaryWriter(log_dir=log_dir)
    #     writer.add_video(gif_name, observation_tensor, fps=int(1/duration))
    #     writer.close()
    
    print(f"Logged gif to TensorBoard at {log_dir}")


# ==================== Imagine ====================
def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor: _FabricModule,
    critic: _FabricModule,
    target_critic: torch.nn.Module,
    world_optimizer: Optimizer,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    is_continuous: bool,
    actions_dim: Sequence[int],
    moments: Moments,
) -> None:
    """Runs one-step update of the agent.

    Args:
        fabric (Fabric): the fabric instance.
        world_model (_FabricModule): the world model wrapped with Fabric.
        actor (_FabricModule): the actor model wrapped with Fabric.
        critic (_FabricModule): the critic model wrapped with Fabric.
        target_critic (nn.Module): the target critic model.
        world_optimizer (Optimizer): the world optimizer.
        actor_optimizer (Optimizer): the actor optimizer.
        critic_optimizer (Optimizer): the critic optimizer.
        data (Dict[str, Tensor]): the batch of data to use for training.
        aggregator (MetricAggregator, optional): the aggregator to print the metrics.
        cfg (DictConfig): the configs.
        is_continuous (bool): whether or not the environment is continuous.
        actions_dim (Sequence[int]): the actions dimension.
        moments (Moments): the moments for normalizing the lambda values.
    """
    # The environment interaction goes like this:
    # Actions:           a0       a1       a2      a4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     v  /     v  /     v /
    # Observations:  o0       o1       o2       o3
    # Rewards:       0        r1       r2       r3
    # Dones:         0        d1       d2       d3
    # Is-first       1        i1       i2       i3

    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device = fabric.device
    batch_obs = {k: data[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])

    # Given how the environment interaction works, we remove the last actions
    # and add the first one as the zero action
    batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)

    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    priors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)

    # Embed observations from the environment
    embedded_obs = world_model.encoder(batch_obs)

    if cfg.algo.world_model.decoupled_rssm:
        posteriors_logits, posteriors = world_model.rssm._representation(embedded_obs)
        for i in range(0, sequence_length):
            if i == 0:
                posterior = torch.zeros_like(posteriors[:1])
            else:
                posterior = posteriors[i - 1 : i]
            recurrent_state, posterior_logits, prior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                data["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
    else:
        posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
        posteriors = torch.empty(sequence_length, batch_size, stochastic_size, discrete_size, device=device)
        posteriors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)
        for i in range(0, sequence_length):
            recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                embedded_obs[i : i + 1],
                data["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
            posteriors[i] = posterior
            posteriors_logits[i] = posterior_logits
    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

    # Compute predictions for the observations
    reconstructed_obs: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)

    # Compute the distribution over the reconstructed observations
    po = {
        k: MSEDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
        for k in cfg.algo.cnn_keys.decoder
    }
    po.update(
        {
            k: SymlogDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
            for k in cfg.algo.mlp_keys.decoder
        }
    )

    # Compute the distribution over the rewards
    pr = TwoHotEncodingDistribution(world_model.reward_model(latent_states), dims=1)

    # Compute the distribution over the terminal steps, if required
    pc = Independent(BernoulliSafeMode(logits=world_model.continue_model(latent_states)), 1)
    continues_targets = 1 - data["terminated"]

    # Reshape posterior and prior logits to shape [B, T, 32, 32]
    priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
    posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

    # World model optimization step. Eq. 4 in the paper
    world_optimizer.zero_grad(set_to_none=True)
    rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
        po,
        batch_obs,
        pr,
        data["rewards"],
        priors_logits,
        posteriors_logits,
        cfg.algo.world_model.kl_dynamic,
        cfg.algo.world_model.kl_representation,
        cfg.algo.world_model.kl_free_nats,
        cfg.algo.world_model.kl_regularizer,
        pc,
        continues_targets,
        cfg.algo.world_model.continue_scale_factor,
    )
    fabric.backward(rec_loss)
    world_model_grads = None
    if cfg.algo.world_model.clip_gradients is not None and cfg.algo.world_model.clip_gradients > 0:
        world_model_grads = fabric.clip_gradients(
            module=world_model,
            optimizer=world_optimizer,
            max_norm=cfg.algo.world_model.clip_gradients,
            error_if_nonfinite=False,
        )
    world_optimizer.step()

    # Behaviour Learning
    imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    imagined_trajectories = torch.empty(
        cfg.algo.horizon + 1,
        batch_size * sequence_length,
        stoch_state_size + recurrent_state_size,
        device=device,
    )
    imagined_trajectories[0] = imagined_latent_state
    imagined_actions = torch.empty(
        cfg.algo.horizon + 1,
        batch_size * sequence_length,
        data["actions"].shape[-1],
        device=device,
    )
    actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
    imagined_actions[0] = actions

    # The imagination goes like this, with H=3:
    # Actions:           a'0      a'1      a'2     a'4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     \  /     \  /     \ /
    # States:        z0 ---> z'1 ---> z'2 ---> z'3
    # Rewards:       r'0     r'1      r'2      r'3
    # Values:        v'0     v'1      v'2      v'3
    # Lambda-values:         l'1      l'2      l'3
    # Continues:     c0      c'1      c'2      c'3
    # where z0 comes from the posterior, while z'i is the imagined states (prior)

    # Imagine trajectories in the latent space
    for i in range(1, cfg.algo.horizon + 1):
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_state
        actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions[i] = actions

    # Predict values, rewards and continues
    predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean
    predicted_rewards = TwoHotEncodingDistribution(world_model.reward_model(imagined_trajectories), dims=1).mean
    continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode
    true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
    continues = torch.cat((true_continue, continues[1:]))

    # Estimate lambda-values
    lambda_values = compute_lambda_values(
        predicted_rewards[1:],
        predicted_values[1:],
        continues[1:] * cfg.algo.gamma,
        lmbda=cfg.algo.lmbda,
    )

    # Compute the discounts to multiply the lambda values to
    with torch.no_grad():
        discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

    # Actor optimization step. Eq. 11 from the paper
    # Given the following diagram, with H=3
    # Actions:          [a'0]    [a'1]    [a'2]    a'3
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     \  /     \  /     \ /
    # States:       [z0] -> [z'1] -> [z'2] ->  z'3
    # Values:       [v'0]   [v'1]    [v'2]     v'3
    # Lambda-values:        [l'1]    [l'2]    [l'3]
    # Entropies:    [e'0]   [e'1]    [e'2]
    actor_optimizer.zero_grad(set_to_none=True)
    policies: Sequence[Distribution] = actor(imagined_trajectories.detach())[1]

    baseline = predicted_values[:-1]
    offset, invscale = moments(lambda_values, fabric)
    normed_lambda_values = (lambda_values - offset) / invscale
    normed_baseline = (baseline - offset) / invscale
    advantage = normed_lambda_values - normed_baseline
    if is_continuous:
        objective = advantage
    else:
        objective = (
            torch.stack(
                [
                    p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                    for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
                ],
                dim=-1,
            ).sum(dim=-1)
            * advantage.detach()
        )
    try:
        entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
    except NotImplementedError:
        entropy = torch.zeros_like(objective)
    policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
    fabric.backward(policy_loss)
    actor_grads = None
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_grads = fabric.clip_gradients(
            module=actor, optimizer=actor_optimizer, max_norm=cfg.algo.actor.clip_gradients, error_if_nonfinite=False
        )
    actor_optimizer.step()

    # Predict the values
    qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
    predicted_target_values = TwoHotEncodingDistribution(
        target_critic(imagined_trajectories.detach()[:-1]), dims=1
    ).mean

    # Critic optimization. Eq. 10 in the paper
    critic_optimizer.zero_grad(set_to_none=True)
    value_loss = -qv.log_prob(lambda_values.detach())
    value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
    value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))

    fabric.backward(value_loss)
    critic_grads = None
    if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
        critic_grads = fabric.clip_gradients(
            module=critic, optimizer=critic_optimizer, max_norm=cfg.algo.critic.clip_gradients, error_if_nonfinite=False
        )
    critic_optimizer.step()

    # Log metrics
    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/world_model_loss", rec_loss.detach())
        aggregator.update("Loss/observation_loss", observation_loss.detach())
        aggregator.update("Loss/reward_loss", reward_loss.detach())
        aggregator.update("Loss/state_loss", state_loss.detach())
        aggregator.update("Loss/continue_loss", continue_loss.detach())
        aggregator.update("State/kl", kl.mean().detach())
        aggregator.update(
            "State/post_entropy",
            Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update(
            "State/prior_entropy",
            Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update("Loss/policy_loss", policy_loss.detach())
        aggregator.update("Loss/value_loss", value_loss.detach())
        if world_model_grads:
            aggregator.update("Grads/world_model", world_model_grads.mean().detach())
        if actor_grads:
            aggregator.update("Grads/actor", actor_grads.mean().detach())
        if critic_grads:
            aggregator.update("Grads/critic", critic_grads.mean().detach())

    # Reset everything
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    # These arguments cannot be changed
    cfg.env.frame_stack = -1
    if 2 ** int(np.log2(cfg.env.screen_size)) != cfg.env.screen_size:
        raise ValueError(f"The screen size must be a power of 2, got: {cfg.env.screen_size}")

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    # Environment setup
    # ==================== Isaac Sim Initializing ====================
    open_stage(usd_path='/home/jianheng/omniverse/assets/Warehouse_Collision_01.usd')
    world = World(physics_dt=physics_dt, rendering_dt=physics_dt,stage_units_in_meters=1.0)
    world.reset()

    occupancy_map = map_generation(
        world=world, 
        map_size=map_size, 
        resolution=map_resolution,
        offset=map_offset, 
        keywords_to_check=map_keywords_to_check, 
    )

    # Fill the edges
    for x in range(map_size[0]):
        for y in range(map_size[1]):
            if x < 2 or y < 3 or x > map_size[0] - 2 or y > map_size[1] - 5:
                occupancy_map[x][y] = 1

    map_global = np.array(occupancy_map.copy()) * map_pixel_values["Static Obstacle"]

    poi_points, poi_positions = get_poi(
        world=world,
        resolution=map_resolution, 
        offset=map_offset
    )

    for point in poi_points:
        # map_global[point[0]][point[1]] = map_pixel_values["Position of Interest"]
        paint_neighbor(map_global, point, map_pixel_values["Position of Interest"])

    print('POIs: ', poi_positions)
    for i in range(len(poi_positions)):
        poi_available_idxs.append(True)

    empty_points, empty_positions = find_empty_points(
        occupancy_map=map_global, 
        resolution=map_resolution,
        offset=map_offset,
        min_clear_radius=min_clear_radius, 
        min_edge_distance=min_edge_distance
    )

    stage = omni.usd.get_context().get_stage()

    map_global_with_worker = map_global.copy()

    global curriculum
    worker_number = worker_number_mapping[curriculum]
    
    for i in range(worker_number):

        name = "Worker_" + str(i+1)
        random_position_index = np.random.randint(len(empty_positions))
        position_xy = empty_positions[random_position_index]

        # print('position_xy: ', position_xy)

        worker_list.append(
            Worker(
                position=(position_xy[0], position_xy[1], 1.0),
                orientation=euler_to_quaternion(0, 0, 0),
                scale=(1.0, 1.0, 1.0),
                prim_name=name,
                parent_path="/World",
                stage=stage,
                usd_path='/home/jianheng/omniverse/assets/Cube.usd',
                semantic_class="worker",
                instanceable=False,
                visibility="inherited",
                disable_gravity=True,
                scale_delta=0,
            )
        )
        map_a_star = OccupancyGridMap(data_array=occupancy_map, cell_size=map_resolution)
        poi_available = [idx for idx, value in enumerate(poi_available_idxs) if value]
        target_poi_idx = np.random.choice(poi_available)
        poi_available_idxs[target_poi_idx] = False
        target_poi_point = poi_points[target_poi_idx]
        position_start_meter = [position_xy[0], position_xy[1]]
        position_start = position_meter_to_pixel(position_start_meter)
        # map_global_with_worker[position_start[0]][position_start[1]] = map_pixel_values["Dynamic Obstacle"]
        paint_neighbor(map_global_with_worker, position_start, map_pixel_values["Dynamic Obstacle"])
        target_poi_waypoints, _ = a_star((position_start[0] * map_resolution, position_start[1] * map_resolution), (target_poi_point[0] * map_resolution, target_poi_point[1] * map_resolution), map_a_star)
        worker_list[i].set_worker_waypoints(waypoints_2d_to_3d(target_poi_waypoints, 1.0), target_poi_idx)

    # Add collision API for every prim in stage
    # for prim in stage.Traverse():
    #     prim_path = prim.GetPath()

    #     collider_api = UsdPhysics.CollisionAPI.Get(stage, prim_path)
    #     if collider_api:
    #         PhysxSchema.PhysxContactReportAPI.Apply(prim)
            # collision_enabled_attr = collider_api.GetCollisionEnabledAttr()
            # collision_enabled = collision_enabled_attr.Get() if collision_enabled_attr else False
            
            # collision_type = "Unknown"
            # if collision_enabled:
                # if UsdPhysics.MeshCollisionAPI.CanApply(prim):
                #     collision_type = "Mesh"
                # elif UsdPhysics.SphereCollisionAPI.CanApply(prim):
                #     collision_type = "Sphere"
                # elif UsdPhysics.BoxCollisionAPI.CanApply(prim):
                #     collision_type = "Box"
                # elif UsdPhysics.CapsuleCollisionAPI.CanApply(prim):
                #     collision_type = "Capsule"
                # elif UsdPhysics.PlaneCollisionAPI.CanApply(prim):
                #     collision_type = "Plane"
    # import matplotlib.pyplot as plt
    # plt.imshow(map_global, cmap='gray', origin='lower')
    # plt.title("Occupancy Map")
    # plt.savefig("occupancy_map.png")

    # print('Empty Area: ', empty_points)
    # print('Empty positions: ', empty_positions)

    # world.reset()

    # Test workers and agents
    # obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder
    # vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    # envs = vectorized_env(
    #     [
    #         partial(
    #             RestartOnException,
    #             make_env(
    #                 cfg,
    #                 cfg.seed + rank * cfg.env.num_envs + i,
    #                 rank * cfg.env.num_envs,
    #                 log_dir if rank == 0 else None,
    #                 "train",
    #                 vector_env_idx=i,
    #             ),
    #         )
    #         for i in range(cfg.env.num_envs)
    #     ]
    # )

    # Get the first environment observation and start the optimization
    # for env_idx, env in enumerate(envs.envs):
    #     env.get_global_map(
    #         map_global=occupancy_map.copy() * map_pixel_values["Static Obstacle"],
    #         map_global_with_worker=map_global_with_worker.copy(),
    #         resolution=map_resolution,
    #         offset=map_offset
    #     )
    #     env.get_env_idx(idx=env_idx)
    #     env.init_camera(seed=None)

    # for prim in stage.Traverse():
    #     prim_path = prim.GetPath()

    #     collider_api = UsdPhysics.CollisionAPI.Get(stage, prim_path)
    #     if collider_api:
    #         PhysxSchema.PhysxContactReportAPI.Apply(prim)

    # world.reset()

    # # Wait for camera ready
    # i = 0
    # while i<=10:
    #     world.step()
    #     i+=1

    # step_data = {}
    # obs = envs.reset(seed=None)[0]


    # obs_dict = {key: [] for key in obs_keys}

    # for env in envs.envs:
    #     _obs = env.compute_obs()
    #     for key in obs_keys:
    #         obs_dict[key].append(_obs[key])

    # obs = OrderedDict()
    # for key, obs_list in obs_dict.items():
    #     obs[key] = np.stack(obs_list, axis=0)

    # step = 0
    # while True:
        # step += 1
        # print('step number: ', step)

        # for worker in worker_list:
        #     worker.move()

        # for env_idx, env in enumerate(envs.envs):
        #     action = [1, 0, 0, 0]
        #     env.pre_step(action)

        # world.step(render=True)

        # map_global_with_worker = map_global.copy()

        # for worker_idx, worker in enumerate(worker_list):
        #     position_current, rotation_current = worker.get_pos_rot()
        #     for point in worker.trajectory:
        #         if 0 <= point[0] < map_size[0] and 0 <= point[1] < map_size[1]:
        #             map_global_with_worker[point[0]][point[1]] = map_pixel_values["Dynamic Obstacle Trajectory"]
        #     position_pixel = position_meter_to_pixel(
        #         position_meter=position_current,
        #         resolution=map_resolution, 
        #         offset=map_offset
        #         )
        #     if 0 <= position_pixel[0] < map_size[0] and 0 <= position_pixel[1] < map_size[1]:
        #         # map_global_with_worker[position_pixel[0]][position_pixel[1]] = map_pixel_values["Dynamic Obstacle"]
        #         paint_neighbor(map_global_with_worker, position_pixel, map_pixel_values["Dynamic Obstacle"])
        #     else:
        #         print('Out of the world')

        # for env_idx, env in enumerate(envs.envs):
        #     env.get_global_map(
        #         map_global=occupancy_map.copy() * map_pixel_values["Static Obstacle"],
        #         map_global_with_worker=map_global_with_worker.copy(),
        #         resolution=map_resolution,
        #         offset=map_offset
        #     )

        # obs_dict = {key: [] for key in obs_keys}
        # rewards = []
        # terminated = []
        # truncated = []
        # infos = []


        # for env in envs.envs:
        #     _obs, _rewards, _terminated, _truncated, _infos = env.post_step()
        #     for key in obs_keys:
        #         obs_dict[key].append(_obs[key])
        #     rewards.append(_rewards)
        #     terminated.append(_terminated)
        #     truncated.append(_truncated)
        #     infos.append(_infos)

        # next_obs = OrderedDict()
        # for key, obs_list in obs_dict.items():
        #     next_obs[key] = np.stack(obs_list, axis=0) 

        # rewards = np.array(rewards)
        # terminated = np.array(terminated)
        # truncated = np.array(truncated)


        # for worker in worker_list:
        #     worker.update_trajectory()
        #     # print('Trajectory:', worker.trajectory)
        #     worker_done = worker.is_done()
        #     # worker_stuck = worker.is_stuck()
        #     if worker.collision:
        #         # Respawn
        #         # position_current = worker.get_translate()
        #         # nearest_position = None
        #         # nearest_distance = float('inf')
        #         # for pos in empty_positions:
        #         #     distance = math.sqrt((position_current[0] - pos[0])**2 + (position_current[1] - pos[1])**2)
        #         #     if distance < nearest_distance:
        #         #         nearest_distance = distance
        #         #         nearest_position = pos
        #         random_position_index = np.random.randint(len(empty_positions))
        #         position_xy = empty_positions[random_position_index]          
        #         worker.set_translate([position_xy[0], position_xy[1], 1])
        #         worker.set_orient(euler_to_quaternion(0, 0, 0))

        #         # world.step()

        #         map_a_star = OccupancyGridMap(data_array=occupancy_map, cell_size=map_resolution)
        #         target_position = worker.position_target
        #         target_poi_point = position_meter_to_pixel(target_position)
        #         position_current = worker.get_translate()
        #         position_start_meter = [position_current[0], position_current[1]]
        #         position_start = position_meter_to_pixel(position_start_meter)
        #         target_poi_waypoints, _ = a_star(
        #             (position_start[0] * map_resolution, position_start[1] * map_resolution), 
        #             (target_poi_point[0] * map_resolution, target_poi_point[1] * map_resolution), 
        #             map_a_star
        #         )
        #         worker.set_worker_waypoints(waypoints_2d_to_3d(target_poi_waypoints, 1.0))

        #     if worker_done:
        #         # Go to the next POI
        #         map_a_star = OccupancyGridMap(data_array=occupancy_map, cell_size=map_resolution)
        #         target_poi_idx = np.random.randint(len(poi_points))
        #         target_poi_point = poi_points[target_poi_idx]
        #         position_current = worker.get_translate()
        #         position_start_meter = [position_current[0], position_current[1]]
        #         position_start = position_meter_to_pixel(position_start_meter)
        #         target_poi_waypoints, _ = a_star(
        #             (position_start[0] * map_resolution, position_start[1] * map_resolution), 
        #             (target_poi_point[0] * map_resolution, target_poi_point[1] * map_resolution), 
        #             map_a_star
        #         )
        #         worker.set_worker_waypoints(waypoints_2d_to_3d(target_poi_waypoints, 1.0))

    # ==================== Isaac Sim Initializing ====================

    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            partial(
                RestartOnException,
                make_env(
                    cfg,
                    cfg.seed + rank * cfg.env.num_envs + i,
                    rank * cfg.env.num_envs,
                    log_dir if rank == 0 else None,
                    "train",
                    vector_env_idx=i,
                ),
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    clip_rewards_fn = lambda r: np.tanh(r) if cfg.env.clip_rewards else r
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

    if (
        len(set(cfg.algo.cnn_keys.encoder).intersection(set(cfg.algo.cnn_keys.decoder))) == 0
        and len(set(cfg.algo.mlp_keys.encoder).intersection(set(cfg.algo.mlp_keys.decoder))) == 0
    ):
        raise RuntimeError("The CNN keys or the MLP keys of the encoder and decoder must not be disjointed")
    if len(set(cfg.algo.cnn_keys.decoder) - set(cfg.algo.cnn_keys.encoder)) > 0:
        raise RuntimeError(
            "The CNN keys of the decoder must be contained in the encoder ones. "
            f"Those keys are decoded without being encoded: {list(set(cfg.algo.cnn_keys.decoder))}"
        )
    if len(set(cfg.algo.mlp_keys.decoder) - set(cfg.algo.mlp_keys.encoder)) > 0:
        raise RuntimeError(
            "The MLP keys of the decoder must be contained in the encoder ones. "
            f"Those keys are decoded without being encoded: {list(set(cfg.algo.mlp_keys.decoder))}"
        )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
        fabric.print("Decoder CNN keys:", cfg.algo.cnn_keys.decoder)
        fabric.print("Decoder MLP keys:", cfg.algo.mlp_keys.decoder)
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

    world_model, actor, critic, target_critic, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"] if cfg.checkpoint.resume_from else None,
        state["actor"] if cfg.checkpoint.resume_from else None,
        state["critic"] if cfg.checkpoint.resume_from else None,
        state["target_critic"] if cfg.checkpoint.resume_from else None,
    )

    # Optimizers
    world_optimizer = hydra.utils.instantiate(
        cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
    )
    actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=actor.parameters(), _convert_="all")
    critic_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=critic.parameters(), _convert_="all")
    if cfg.checkpoint.resume_from:
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    world_optimizer, actor_optimizer, critic_optimizer = fabric.setup_optimizers(
        world_optimizer, actor_optimizer, critic_optimizer
    )
    moments = Moments(
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    if cfg.checkpoint.resume_from:
        moments.load_state_dict(state["moments"])

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 2
    rb = EnvIndependentReplayBuffer(
        buffer_size,
        n_envs=cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        buffer_cls=SequentialReplayBuffer,
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], EnvIndependentReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")

    # Global variables
    train_step = 0
    last_train = 0
    start_step = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["update"] // fabric.world_size) + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["update"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs * fabric.world_size)
    num_updates = int(cfg.algo.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size
        if not cfg.buffer.checkpoint:
            learning_starts += start_step

    # Create Ratio class
    ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
    if cfg.checkpoint.resume_from:
        ratio.load_state_dict(state["ratio"])

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_update != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )
    if cfg.checkpoint.every % policy_steps_per_update != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )

    # Get the first environment observation and start the optimization
    for env_idx, env in enumerate(envs.envs):
        env.get_global_map(
            map_global=occupancy_map.copy() * map_pixel_values["Static Obstacle"],
            map_global_with_worker=map_global_with_worker.copy(),
            worker_list = worker_list,
            resolution=map_resolution,
            offset=map_offset
        )
        env.get_env_idx(idx=env_idx)
        env.init_camera(seed=None)

    # Add collider api
    for prim in stage.Traverse():
        prim_path = prim.GetPath()

        collider_api = UsdPhysics.CollisionAPI.Get(stage, prim_path)
        if collider_api:
            PhysxSchema.PhysxContactReportAPI.Apply(prim)

    world.reset()

    # Wait for camera ready
    i = 0
    while i<=60:
        world.step()
        i+=1

    step_data = {}
    obs = envs.reset(seed=None)[0]

    world.step()

    # print('Initialize map: ', obs['map'].shape)
    # print('Initialize rgb: ', obs['rgb'].shape)
    # print('Initialize goal: ', obs['goal'].shape)


    obs_dict = {key: [] for key in obs_keys}

    for env in envs.envs:
        _obs = env.compute_obs()
        for key in obs_keys:
            obs_dict[key].append(_obs[key])

    obs = OrderedDict()
    for key, obs_list in obs_dict.items():
        obs[key] = np.stack(obs_list, axis=0)
    # print('My obs map: ', obs['map'].shape)
    # print('My obs rgb: ', obs['rgb'].shape)
    # print('My obs goal: ', obs['goal'].shape)

    
    for k in obs_keys:
        step_data[k] = obs[k][np.newaxis]
    step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["is_first"] = np.ones_like(step_data["terminated"])
    player.init_states()

    
    cumulative_per_rank_gradient_steps = 0
    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs * world_size

        with torch.inference_mode():
            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                # Sample an action given the observation received by the environment
                if (
                    update <= learning_starts
                    and cfg.checkpoint.resume_from is None
                    and "minedojo" not in cfg.env.wrapper._target_.lower()
                ):
                    real_actions = actions = np.array(envs.action_space.sample())
                    if not is_continuous:
                        actions = np.concatenate(
                            [
                                F.one_hot(torch.as_tensor(act), act_dim).numpy()
                                for act, act_dim in zip(actions.reshape(len(actions_dim), -1), actions_dim)
                            ],
                            axis=-1,
                        )
                else:
                    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                    mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                    if len(mask) == 0:
                        mask = None

                    real_actions = actions = player.get_actions(torch_obs, mask=mask)
                    actions = torch.cat(actions, -1).cpu().numpy()
                    if is_continuous:
                        real_actions = torch.cat(real_actions, dim=-1).cpu().numpy()
                    else:
                        real_actions = (
                            torch.cat([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
                        )

                step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
                rb.add(step_data, validate_args=cfg.buffer.validate_args)


                # ================= Interact with the Isaac Sim Environment =======================
                map_global_with_worker = map_global.copy()

                work_list_length = len(worker_list)
                if work_list_length < worker_number:
                    print('Adding new workers')
                    for i in range(worker_number - work_list_length):
                        name = "Worker_" + str(work_list_length+i+1)
                        random_position_index = np.random.randint(len(empty_positions))
                        position_xy = empty_positions[random_position_index]
                        worker_list.append(
                            Worker(
                                position=(position_xy[0], position_xy[1], 1.0),
                                orientation=euler_to_quaternion(0, 0, 0),
                                scale=(1.0, 1.0, 1.0),
                                prim_name=name,
                                parent_path="/World",
                                stage=stage,
                                usd_path='/home/jianheng/omniverse/assets/Cube.usd',
                                semantic_class="worker",
                                instanceable=False,
                                visibility="inherited",
                                disable_gravity=True,
                                scale_delta=0,
                            )
                        )
                        world.step()
                        map_a_star = OccupancyGridMap(data_array=occupancy_map, cell_size=map_resolution)
                        target_poi_idx = np.random.randint(len(poi_points))
                        target_poi_point = poi_points[target_poi_idx]
                        position_start_meter = [position_xy[0], position_xy[1]]
                        position_start = position_meter_to_pixel(position_start_meter)
                        # map_global_with_worker[position_start[0]][position_start[1]] = map_pixel_values["Dynamic Obstacle"]
                        paint_neighbor(map_global_with_worker, position_start, map_pixel_values["Dynamic Obstacle"])
                        target_poi_waypoints, _ = a_star((position_start[0] * map_resolution, position_start[1] * map_resolution), (target_poi_point[0] * map_resolution, target_poi_point[1] * map_resolution), map_a_star)
                        worker_list[-1].set_worker_waypoints(waypoints_2d_to_3d(target_poi_waypoints, 1.0))
                        # Add collider api
                        prim_path = worker_list[-1]._full_prim_path
                        collider_api = UsdPhysics.CollisionAPI.Get(stage, prim_path)
                        if collider_api:
                            PhysxSchema.PhysxContactReportAPI.Apply(prim)

                for worker in worker_list:
                    worker.move()

                # print('Original Actions: ', actions)
                for env_idx, env in enumerate(envs.envs):
                    if len(actions.shape) == 2:
                        # print('Action taken: ', actions[env_idx])
                        env.pre_step(actions[env_idx])
                    elif len(actions.shape) == 3:
                        # print('Actions: ', actions[0][env_idx])
                        env.pre_step(actions[0][env_idx])

                world.step(render=True) # execute one physics step and one rendering step

                map_global_with_worker = map_global.copy()

                for worker_idx, worker in enumerate(worker_list):
                    position_current, rotation_current = worker.get_pos_rot()
                    for point in worker.trajectory:
                        if 0 <= point[0] < map_size[0] and 0 <= point[1] < map_size[1]:
                            map_global_with_worker[point[0]][point[1]] = map_pixel_values["Dynamic Obstacle Trajectory"]
                    position_pixel = position_meter_to_pixel(
                        position_meter=position_current,
                        resolution=map_resolution, 
                        offset=map_offset
                        )
                    if 0 <= position_pixel[0] < map_size[0] and 0 <= position_pixel[1] < map_size[1]:
                        # map_global_with_worker[position_pixel[0]][position_pixel[1]] = map_pixel_values["Dynamic Obstacle"]
                        paint_neighbor(map_global_with_worker, position_pixel, map_pixel_values["Dynamic Obstacle"])
                    else:
                        print('Out of the world')

                for env_idx, env in enumerate(envs.envs):
                    env.get_global_map(
                        map_global=occupancy_map.copy() * map_pixel_values["Static Obstacle"],
                        map_global_with_worker=map_global_with_worker.copy(),
                        worker_list = worker_list,
                        resolution=map_resolution,
                        offset=map_offset
                    )

                obs_dict = {key: [] for key in obs_keys}
                rewards = []
                terminated = []
                truncated = []
                infos = {
                    "final_info": []
                }
                
                for env in envs.envs:
                    _obs, _rewards, _terminated, _truncated, _infos = env.post_step()
                    for key in obs_keys:
                        obs_dict[key].append(_obs[key])
                    rewards.append(_rewards)
                    terminated.append(_terminated)
                    truncated.append(_truncated)
                    if "episode" in _infos:
                        infos["final_info"].append(_infos)

                next_obs = OrderedDict()
                for key, obs_list in obs_dict.items():
                    next_obs[key] = np.stack(obs_list, axis=0) 

                rewards = np.array(rewards)
                terminated = np.array(terminated)
                truncated = np.array(truncated)

                for worker in worker_list:
                    worker.update_trajectory()
                    # print('Trajectory:', worker.trajectory)
                    worker_done = worker.is_done()
                    # worker_stuck = worker.is_stuck()
                    if worker.collision:
                        # Respawn
                        # position_current = worker.get_translate()
                        # nearest_position = None
                        # nearest_distance = float('inf')
                        # for pos in empty_positions:
                        #     distance = math.sqrt((position_current[0] - pos[0])**2 + (position_current[1] - pos[1])**2)
                        #     if distance < nearest_distance:
                        #         nearest_distance = distance
                        #         nearest_position = pos
                        worker.trajectory = []

                        map_global_with_agent = map_global.copy()

                        for env in envs.envs:
                            position_agent_raw, _ = env.camera_rig.get_pos_rot()
                            position_agent_meter = [position_agent_raw[0], position_agent_raw[1]]
                            position_agent_pixel = position_meter_to_pixel(position_agent_meter)
                            paint_neighbor(map_global_with_agent, point, map_pixel_values["Agent"])
                             

                        _, empty_positions_reset = find_empty_points(
                            occupancy_map=map_global_with_agent, 
                            resolution=map_resolution,
                            offset=map_offset,
                            min_clear_radius=min_clear_radius, 
                            min_edge_distance=min_edge_distance
                        )

                        random_position_index = np.random.randint(len(empty_positions_reset))
                        position_xy = empty_positions_reset[random_position_index]          
                        worker.set_translate([position_xy[0], position_xy[1], 1])
                        worker.set_orient(euler_to_quaternion(0, 0, 0))

                        world.step()

                        map_a_star = OccupancyGridMap(data_array=occupancy_map, cell_size=map_resolution)
                        target_position = worker.position_target
                        target_poi_point = position_meter_to_pixel(target_position)
                        position_current = worker.get_translate()
                        position_start_meter = [position_current[0], position_current[1]]
                        position_start = position_meter_to_pixel(position_start_meter)
                        target_poi_waypoints, _ = a_star(
                            (position_start[0] * map_resolution, position_start[1] * map_resolution), 
                            (target_poi_point[0] * map_resolution, target_poi_point[1] * map_resolution), 
                            map_a_star
                        )
                        worker.set_worker_waypoints(waypoints_2d_to_3d(target_poi_waypoints, 1.0), worker.position_target_idx)

                    if worker_done:
                        # Go to the next POI
                        poi_available_idxs[worker.position_target_idx] = True
                        map_a_star = OccupancyGridMap(data_array=occupancy_map, cell_size=map_resolution)
                        poi_available = [idx for idx, value in enumerate(poi_available_idxs) if value]
                        target_poi_idx = np.random.choice(poi_available)
                        poi_available_idxs[target_poi_idx] = False
                        target_poi_point = poi_points[target_poi_idx]
                        position_current = worker.get_translate()
                        position_start_meter = [position_current[0], position_current[1]]
                        position_start = position_meter_to_pixel(position_start_meter)
                        target_poi_waypoints, _ = a_star(
                            (position_start[0] * map_resolution, position_start[1] * map_resolution), 
                            (target_poi_point[0] * map_resolution, target_poi_point[1] * map_resolution), 
                            map_a_star
                        )
                        worker.set_worker_waypoints(waypoints_2d_to_3d(target_poi_waypoints, 1.0), target_poi_idx)
                
                    
                # ================= Interact with the Isaac Sim Environment =======================
                dones = np.logical_or(terminated, truncated).astype(np.uint8)

            step_data["is_first"] = np.zeros_like(step_data["terminated"])
            if "restart_on_exception" in infos:
                for i, agent_roe in enumerate(infos["restart_on_exception"]):
                    if agent_roe and not dones[i]:
                        last_inserted_idx = (rb.buffer[i]._pos - 1) % rb.buffer[i].buffer_size
                        rb.buffer[i]["terminated"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["terminated"][last_inserted_idx]
                        )
                        rb.buffer[i]["truncated"][last_inserted_idx] = np.ones_like(
                            rb.buffer[i]["truncated"][last_inserted_idx]
                        )
                        rb.buffer[i]["is_first"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["is_first"][last_inserted_idx]
                        )
                        step_data["is_first"][i] = np.ones_like(step_data["is_first"][i])
            
            global success_rate
            global ready_to_upgrade
            if cfg.metric.log_level > 0 and "final_info" in infos:
                for i, agent_ep_info in enumerate(infos["final_info"]):
                    if agent_ep_info is not None:
                        ep_suc = agent_ep_info["episode"]["s"]
                        success_rate = ep_suc
                        ready_to_upgrade = agent_ep_info["episode"]["ready"]
                        if aggregator and not aggregator.disabled:
                            aggregator.update("Game/success_rate", ep_suc)

                        if "r" in agent_ep_info["episode"]:
                            ep_rew = agent_ep_info["episode"]["r"]
                            ep_len = agent_ep_info["episode"]["l"]
                            ep_col = agent_ep_info["episode"]["c"]
                            ep_goa = agent_ep_info["episode"]["g"]
                            ep_col_d = agent_ep_info["episode"]["c_d"]
                            ep_col_s = agent_ep_info["episode"]["c_s"]
                        
                            if aggregator and not aggregator.disabled:
                                aggregator.update("Rewards/rew_avg", ep_rew)
                                # print("log reward ave: ", ep_rew)
                                aggregator.update("Game/ep_len_avg", ep_len)
                                aggregator.update("Game/goal", ep_goa)
                                aggregator.update("Game/collision", ep_col)
                                # print("log len ave: ", ep_len)
                                aggregator.update("Game/curriculum_level", curriculum)
                                aggregator.update("Game/collision_dynamic", ep_col_d)
                                aggregator.update("Game/collision_static", ep_col_s)
                            
                            fabric.print(f"Rank-0: policy_step={policy_step}, average_reward_env_{i}={ep_rew}")

            if policy_step - last_log >= cfg.metric.log_every and ready_to_upgrade:
                print("Judging Increasing")
                if success_rate > curriculum_threshold and curriculum < max_curriculum:
                    print(">>>>>>>>>>>>>>>Increasing Difficulty>>>>>>>>>>>>>>>>>>>")
                    increase_difficulty = True
                    curriculum += 1
                    worker_number = worker_number_mapping[curriculum]
                    for env in envs.envs:
                        env.increase_difficulty(increase_difficulty)      

                    success_rate = 0.0 
                    ready_to_upgrade = False                               

                        # Increasing the difficulty
                        # if ep_cur > curriculum and curriculum < max_curriculum:
                        #     curriculum += 1
                        #     worker_number = worker_number_mapping[curriculum]

            # Save the real next observation
            real_next_obs = copy.deepcopy(next_obs)
            if "final_observation" in infos:
                for idx, final_obs in enumerate(infos["final_observation"]):
                    if final_obs is not None:
                        for k, v in final_obs.items():
                            real_next_obs[k][idx] = v

            for k in obs_keys:
                step_data[k] = next_obs[k][np.newaxis]

            # next_obs becomes the new obs
            obs = next_obs

            rewards = rewards.reshape((1, cfg.env.num_envs, -1))
            step_data["terminated"] = terminated.reshape((1, cfg.env.num_envs, -1))
            step_data["truncated"] = truncated.reshape((1, cfg.env.num_envs, -1))
            step_data["rewards"] = clip_rewards_fn(rewards)

            dones_idxes = dones.nonzero()[0].tolist()
            reset_envs = len(dones_idxes)
            if reset_envs > 0:
                reset_data = {}
                for k in obs_keys:
                    reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
                reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
                reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
                reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
                reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
                reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
                rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)

                # Reset already inserted step data
                step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
                step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
                step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
                step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
                player.init_states(dones_idxes)

            # ==================== imagine ====================
            # Example Usage
            # check_iteration = 10000
            # check_iteration = [1000, 5000, 10000]
            # imagination_horizon = 32

            # if (policy_step + imagination_horizon) in check_iteration:
            #     print('Reach policy_step: ', policy_step)
            #     imagination_stochastic_state = player.stochastic_state.clone()
            #     imagination_recurrent_state = player.recurrent_state.clone()

            #     # actions to take
            #     imagined_latent_states = torch.cat((imagination_stochastic_state, imagination_recurrent_state), -1)
            #     actions_to_take = torch.cat(actor(imagined_latent_states.detach())[0], dim=-1) # can take any specific action
                
            #     # Run imagine function
            #     imagined_observations, imagined_actions = imagine(
            #         fabric,
            #         world_model,
            #         actor,
            #         imagination_stochastic_state,
            #         imagination_recurrent_state,
            #         actions_to_take,
            #         imagination_horizon,
            #         envs.single_action_space,
            #         cfg,
            #     )

            #     # Save RGB observations as gif
            #     print('Saved as gif')
            #     output_path_imagined = "imagined_observations.gif"
            #     save_as_gif(imagined_observations, "rgb", output_path_imagined, duration=0.1)
            #     log_imagination_as_gif(
            #         reconstructed_obs_np=imagined_observations,
            #         key="rgb",
            #         cfg=cfg,
            #         gif_name="Imagined Observation",
            #         duration=0.1
            #     )

            # elif policy_step in check_iteration:
            #     # Get the latest observations from the replay buffer using sample_latest_sequence
            #     latest_sequence = rb.sample_latest_sequence(imagination_horizon)

            #     # Process latest observations
            #     latest_observations = {}
            #     for k in cfg.algo.cnn_keys.encoder:
            #         latest_observations[k] = latest_sequence[k]

            #     log_true_observation_as_gif(
            #         true_obs_np=latest_observations,
            #         key="rgb",
            #         cfg=cfg,
            #         gif_name="True Observation",
            #         duration=0.1,
            #     )

            #     latest_observations['rgb'] = latest_observations['rgb'][0, :, 0, :, :, :]
            #     latest_observations['rgb'] = np.clip(latest_observations['rgb'], 0, 255).astype(np.uint8)

            #     # Save as gif
            #     print('Saved as gif')
            #     output_path_true = "true_observations.gif"
            #     save_as_gif(latest_observations, "rgb", output_path_true, duration=0.1)
            # ==================== imagine ====================

        # Train the agent
        if update >= learning_starts:
            per_rank_gradient_steps = ratio(policy_step / world_size)
            if per_rank_gradient_steps > 0:
                local_data = rb.sample_tensors(
                    cfg.algo.per_rank_batch_size,
                    sequence_length=cfg.algo.per_rank_sequence_length,
                    n_samples=per_rank_gradient_steps,
                    dtype=None,
                    device=fabric.device,
                    from_numpy=cfg.buffer.from_numpy,
                )
                with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                    for i in range(per_rank_gradient_steps):
                        if (
                            cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq
                            == 0
                        ):
                            tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
                            for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
                                tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                        batch = {k: v[i].float() for k, v in local_data.items()}
                        train(
                            fabric,
                            world_model,
                            actor,
                            critic,
                            target_critic,
                            world_optimizer,
                            actor_optimizer,
                            critic_optimizer,
                            batch,
                            aggregator,
                            cfg,
                            is_continuous,
                            actions_dim,
                            moments,
                        )
                        cumulative_per_rank_gradient_steps += 1
                    train_step += world_size

        # Log imagination gifs
        if policy_step - last_log + cfg.algo.per_rank_sequence_length == cfg.metric.log_every:

            # Latent states
            imagination_stochastic_state = player.stochastic_state.clone()
            # print('imagined sto shape: ', imagination_stochastic_state.shape)
            imagination_stochastic_state_single = imagination_stochastic_state[:, 0:1, :]
            imagination_recurrent_state = player.recurrent_state.clone()
            # print('imagined recu shape: ', imagination_recurrent_state.shape)
            imagination_recurrent_state_sigle = imagination_recurrent_state[:, 0:1, :]
            imagined_latent_states_single = torch.cat((imagination_stochastic_state_single.detach(), imagination_recurrent_state_sigle.detach()), -1)
            # Actions to take
            actions_to_take = torch.cat(actor(imagined_latent_states_single.detach())[0], dim=-1) # can take any specific action

            # Imagine
            imagined_observations, imagined_actions = imagine(
                fabric=fabric,
                world_model=world_model,
                actor=actor,
                stochastic_state=imagination_stochastic_state_single,
                recurrent_state=imagination_recurrent_state_sigle,
                actions=actions_to_take,
                horizon=cfg.algo.per_rank_sequence_length,
                action_space=envs.single_action_space,
                cfg=cfg,
            )

            # Log gif
            for k in cfg.algo.cnn_keys.decoder:
                log_imagination_as_gif(
                    fabric=fabric,
                    reconstructed_obs_np=imagined_observations,
                    key=k,
                    cfg=cfg,
                    gif_name="Imagined Observation: " + k,
                    duration=0.1,
                )

        # Log true observation gifs
        if policy_step - last_log == cfg.metric.log_every:
            # Get the latest observations from the replay buffer using sample_latest_sequence
                latest_sequence = rb.sample_latest_sequence(cfg.algo.per_rank_sequence_length)

                # Process latest observations
                latest_observations = {}
                for k in cfg.algo.cnn_keys.encoder:
                    latest_observations[k] = latest_sequence[k]

                # Log true observation gif
                for k in cfg.algo.cnn_keys.decoder:
                    log_true_observation_as_gif(
                        fabric=fabric,
                        true_obs_np=latest_observations,
                        key=k,
                        cfg=cfg,
                        gif_name="True Observation: " + k,
                        duration=0.1,
                    )

        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or update == num_updates):
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Log replay ratio
            fabric.log(
                "Params/replay_ratio", cumulative_per_rank_gradient_steps * world_size / policy_step, policy_step
            )

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_log) / world_size * cfg.env.action_repeat)
                        / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()

            # Reset counters
            last_log = policy_step
            last_train = train_step

        # Checkpoint Model
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            update == num_updates and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "world_model": world_model.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "moments": moments.state_dict(),
                "ratio": ratio.state_dict(),
                "update": update * fabric.world_size,
                "batch_size": cfg.algo.per_rank_batch_size * fabric.world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    envs.close()
    if fabric.is_global_zero and cfg.algo.run_test:
        test(player, fabric, cfg, log_dir, greedy=False)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": world_model,
            "actor": actor,
            "critic": critic,
            "target_critic": target_critic,
            "moments": moments,
        }
        register_model(fabric, log_models, cfg, models_to_log)
