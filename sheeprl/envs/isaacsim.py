from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import omni
from omni.physx import get_physx_simulation_interface
from pxr import Gf, UsdGeom, UsdPhysics, PhysicsSchemaTools
import math
import cv2
from PIL import Image, ImageDraw

from map_utils import find_empty_points, position_meter_to_pixel, find_nearby_empty_points_by_path
from gridmap import OccupancyGridMap
from a_star import a_star
from SyntheticToolkit.Sensors.rig import Rig
from SyntheticToolkit.utils.omni_utils import euler_to_quaternion, euler_from_quaternion

# TODO: Remove Global Information; Do not Update A* Waypoints Every Step
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

        self.camera_rig = None

        # self.goal_distance = 1.0
        self.curriculum = 1
        self.max_curriculum = 8
        # self.curriculum_threshold = 1.0
        self.env_idx = 0
        self.step_num = 0
        self.episode_reward = 0
        self.achieve_goal = 0
        self.return_goal = 0
        self.return_collision = 0
        self.step_num_log_min = 4
        # self.max_step_num = 512
        self.log_episode_number = 25
        self.log_rew = []
        self.log_len = []
        self.log_goa = []
        self.log_col = []
        self.log_suc = []
        
        self.map_pixel_values = {
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

        # 1-7
        # 8
        self.curriculum_mapping = {
            "heading": [np.pi, np.pi,  np.pi, np.pi, np.pi, np.pi, np.pi, np.pi],
            "distance": [[2.0, 5.0], [5.0, 8.0], [8.0, 12.0], [12.0, 16.0], [16.0, 20.0], [20.0, 24.0], [24.0, 28.0], [0.0, np.inf]],
            "step_num": [200, 250, 300, 350, 400, 400, 400, 400],
            "goal_distance": [1.4, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0]
        }
        # "distance": [[2.0, 5.0], [3.0, 7.0], [4.0, 9.0], [5.0, 12.0], [6.0, 15.0], [7.0, 18.0], [8.0, 21.0], [0.0, np.inf]],
        

        self.map_global = None
        self.map_global_with_worker = None
        self.worker_list = []
        self.map_resolution = None
        self.map_offset = None
        self.min_clear_radius = 2
        self.min_edge_distance = 2

        self.collision = False
        self.collision_with_agent = False
        self.done = False
        self.goal_position = []
        self.trajectory = []
        self.trajectory_length = 32
        self.subgoal_idx = 2
        

        self.min_depth_value = 0.5

        self._render_mode: str = "rgb_array"
            
        # Define observation and action spaces
        self._observation_space = spaces.Dict({
            "goal": spaces.Box(low=-2, high=2, shape=(4,), dtype=np.float32),
            "heading": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "orientation": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "distance": spaces.Box(low=0, high=64, shape=(1,), dtype=np.float32),
            "dynamic": spaces.Box(low=0, high=64, shape=(16,), dtype=np.float32),
            "map": spaces.Box(low=0, high=255, shape=(2, map_size[0], map_size[1]), dtype=np.uint8),
            "depth": spaces.Box(low=0.5, high=32.0, shape=(1, camera_rgb_size[0], camera_rgb_size[1]), dtype=np.float32)
        })

        self._action_space = spaces.Discrete(4)  # One-hot, 0: forward, 1: rotate left, 2: rotate right, 3: no movement
        # 2d array, 0: linear, 1: angular
        self._action_mapping = {
            (1, 0, 0, 0) : [[self.camera_linear_velocity, 0, 0], [0, 0, 0]],
            (0, 1, 0, 0) : [[0, 0, 0], [0, 0, self.camera_angular_velocity]],
            (0, 0, 1, 0) : [[0, 0, 0], [0, 0, -self.camera_angular_velocity]],
            (0, 0, 0, 1) : [[0, 0, 0], [0, 0, 0]]
        }
        self._reward_range = (-np.inf, np.inf)

        self._agent_ref = None

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        self.contact_report_sub = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                self.on_contact_report_event
            )
        )

    def on_contact_report_event(self, contact_headers, contact_data):
        
        for contact_header in contact_headers:
            act0_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            act1_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
            # print(f"Collision detected between {act0_path} and {act1_path}")

            if "Camera" in act0_path and "Camera" in act1_path:
                self.collision_with_agent = True
                # print(f"Agent Collision detected between {act0_path} and {act1_path}")
                return
            
            if self.camera_name in act0_path or self.camera_name in act1_path:
                self.collision = True
                # print(f"Collision detected between {act0_path} and {act1_path}")
                return
            
    def get_global_map(self, map_global, map_global_with_worker, worker_list, resolution, offset):
        self.map_global = map_global.copy()
        self.map_global_with_worker = map_global_with_worker.copy()
        self.worker_list = worker_list
        self.map_resolution = resolution
        self.map_offset = offset
    # def get_global_map(self, map_global, map_global_with_worker, resolution, offset):
    #     self.map_global = map_global.copy()
    #     self.map_global_with_worker = map_global_with_worker.copy()
    #     self.map_resolution = resolution
    #     self.map_offset = offset



    def get_env_idx(self, idx):
        self.env_idx = idx + 1
        self.camera_name = "Camera_" + str(self.env_idx)

    # Increasing the difficulty
    def increase_difficulty(self, increase):
        if increase and self.curriculum < self.max_curriculum:
            self.curriculum += 1
            self.log_rew = []
            self.log_len = []
            self.log_goa = []
            self.log_col = []
            self.log_suc = []


    def update_trajectory(self):
        position_current, _ = self.camera_rig.get_pos_rot()
        position_meter = [position_current[0], position_current[1]]
        position_pixel = position_meter_to_pixel(position_meter=position_meter, resolution=self.map_resolution, offset=self.map_offset)
        if len(self.trajectory) < self.trajectory_length:
            self.trajectory.append(position_pixel)
        else:
            self.trajectory.pop(0)
            self.trajectory.append(position_pixel)

    def standardize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle <= -np.pi:
            angle += 2 * np.pi

        return angle
    
    def get_distance(self, pos_1, pos_2):
        return math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)
    
    def paint_neighbor(self, map, point, value):
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
            if 0 <= neighbor[0] < self.map_size[0] and 0 <= neighbor[1] < self.map_size[1]:
                if map[neighbor[0], neighbor[1]] != self.map_pixel_values['Dynamic Obstacle']:
                    map[neighbor[0], neighbor[1]] = value


    def pre_step(self, action):
        # do whatver
        #forward - > [1,0,0]
        # print('Action type: ', type(action))
        # print('Action: ', action)
        if np.array_equal(action, np.array([1, 0, 0, 0])):
            _, orient_quat = self.camera_rig.get_pos_rot()
        # Orientation
            orientation_agent = euler_from_quaternion(orient_quat[0], orient_quat[1], orient_quat[2], orient_quat[3])
            orient = np.array(orientation_agent[2])

            # print('Agent: ', self.env_idx, 'Orient: ', orient)

            velocs = self._action_mapping[tuple(action)]
            vel_1 = velocs[0][0] * math.cos(orient)
            vel_2 = velocs[0][0] * math.sin(orient)
            lin = [vel_1, vel_2, 0]
            ang = velocs[1]
            self.camera_rig.apply_veloc(lin, ang)
            
        else:
            velocs = self._action_mapping[tuple(action)]
            lin = velocs[0]
            ang = velocs[1]
            self.camera_rig.apply_veloc(lin, ang)


    def post_step(self):
        # compute observations and rewards.
        # {"rgb": ___, "depth": _____, "segmentation":___}
        # check for collisions.
        #if self._collided:
        #    done = True
        self.update_trajectory()

        done = False
        truncated = False
        goal = False
        over_time = False

        info = {}

        # calculate obs
        obs = self.compute_obs()

        # calculate rewards
        # reward_time = -1

        reward_heading = math.cos(obs["heading"][0])

        reward_moving = 0

        if len(self.trajectory) > 4:
            travel_distance = 0
            for i in range(4):
                travel_distance += self.get_distance(self.trajectory[-(i+1)], self.trajectory[-(i+2)])
            if travel_distance < self.map_resolution:
                reward_moving = -1
        goal_distance = math.sqrt((obs['goal'][0] * self.map_resolution * self.map_size[0] / 2) ** 2 + (obs['goal'][1] * self.map_resolution * self.map_size[1] / 2) ** 2)
        # reward_distance = - (goal_distance - 0.5) * 0.5
        # print('obs post step: ', obs["goal"])

        reward_overlay = 0

        position_agent, _ = self.camera_rig.get_pos_rot()
        position_agent_pixel = position_meter_to_pixel(
            position_meter=position_agent,
            resolution=self.map_resolution,
            offset=self.map_offset
        )
        point = position_agent_pixel
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
            if 0 <= neighbor[0] < self.map_size[0] and 0 <= neighbor[1] < self.map_size[1]:
                if self.map_global_with_worker[neighbor[0], neighbor[1]] == self.map_pixel_values['Dynamic Obstacle'] or self.map_global_with_worker[neighbor[0], neighbor[1]] == self.map_pixel_values['Static Obstacle']:
                    reward_overlay -= 1.0
        
        self.step_num += 1

        reward_goal = 0
        if goal_distance < self.curriculum_mapping["goal_distance"][self.curriculum-1]:
            # Arrive at the goal
            print('Arrive at the goal! - agent: ', self.env_idx, ' - stage: ', self.curriculum)
            reward_goal = 100
            self.achieve_goal = 1
            self.return_goal += reward_goal
            done = False
            goal = True

        reward_collision = 0
        if self.collision:
            print('Collide! - agent: ', self.env_idx, ' - stage: ', self.curriculum)
            reward_collision = -50
            self.return_collision += reward_collision
            done = True
        if self.collision_with_agent:
            # print('Collide with agent - agent: ', self.env_idx, ' - stage: ', self.curriculum)
            self.collision_with_agent = False
            # truncated = True
            # self.reset()

        if self.step_num >= self.curriculum_mapping["step_num"][self.curriculum-1]:
            print('Max step number - agent: ', self.env_idx)
            # truncated = True
            done = True
            over_time = True

        # print('Reward heading: ', reward_heading)
        # print('Reward distance: ', reward_distance)
        # print('Reward goal: ', reward_goal)
        # print('Reward collision: ', reward_collision)

        reward = reward_goal + reward_collision + reward_heading + reward_moving + reward_overlay
        self.episode_reward += reward

        if done or truncated or goal:
            if done and self.step_num >= self.step_num_log_min:
                # log
                if len(self.log_rew) < self.log_episode_number:
                    self.log_rew.append(self.episode_reward)
                    self.log_len.append(self.step_num)
                    rew_avg = sum(self.log_rew) / len(self.log_rew)
                    len_avg = sum(self.log_len) / len(self.log_len)

                    self.log_goa.append(self.return_goal)
                    self.log_col.append(self.return_collision)
                    goa = sum(self.log_goa) / len(self.log_goa)
                    col = sum(self.log_col) / len(self.log_col)

                    self.log_suc.append(self.achieve_goal)
                    suc = sum(self.log_suc) / len(self.log_suc)

                    info["episode"] = {
                        "r": rew_avg,
                        "l": len_avg,
                        "c": col,
                        "g": goa,
                        "s": suc,
                        "ready": False
                    }
                    

                else:
                    self.log_rew.pop(0)
                    self.log_rew.append(self.episode_reward)
                    self.log_len.pop(0)
                    self.log_len.append(self.step_num)
                    rew_avg = sum(self.log_rew) / len(self.log_rew)
                    len_avg = sum(self.log_len) / len(self.log_len)
                    
                    self.log_goa.pop(0)
                    self.log_goa.append(self.return_goal)
                    self.log_col.pop(0)
                    self.log_col.append(self.return_collision)
                    goa = sum(self.log_goa) / len(self.log_goa)
                    col = sum(self.log_col) / len(self.log_col)

                    self.log_suc.pop(0)
                    self.log_suc.append(self.achieve_goal)
                    suc = sum(self.log_suc) / len(self.log_suc)
                    
                    # # Increasing the difficulty
                    # if rew_avg >= self.curriculum_threshold and self.curriculum < self.max_curriculum:
                    #     print('>>>>>>>>>>>>>>>>>Increasing Difficulty!>>>>>>>>>>>>')
                    #     self.curriculum += 1
                    #     self.log_rew = []
                    #     self.log_len = []

                    info["episode"] = {
                        "r": rew_avg,
                        "l": len_avg,
                        "c": col,
                        "g": goa,
                        "s": suc,
                        "ready": True
                    }

            if goal and not done:
                if len(self.log_rew) < self.log_episode_number:
                    self.log_suc.append(self.achieve_goal)
                    suc = sum(self.log_suc) / len(self.log_suc)

                    info["episode"] = {
                        "s": suc,
                        "ready": False
                    }
                else:
                    self.log_suc.pop(0)
                    self.log_suc.append(self.achieve_goal)
                    suc = sum(self.log_suc) / len(self.log_suc)

                    info["episode"] = {
                        "s": suc,
                        "ready": True
                    }
                    
            
            if not goal:
                self.reset(spawn=True)

            elif goal:
                self.reset(spawn=False)

            
        #return 
        return obs, reward, done, truncated, info


        
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

    def compute_obs(self) -> Dict[str, np.ndarray]:
        obs = {}

        position_agent_raw, orient_quat = self.camera_rig.get_pos_rot()
        position_agent = np.array([position_agent_raw[0], position_agent_raw[1]])
        position_agent_pixel = position_meter_to_pixel(
            position_meter=position_agent,
            resolution=self.map_resolution,
            offset=self.map_offset
        )
        position_goal_pixel = position_meter_to_pixel(
            position_meter=self.goal_position,
            resolution=self.map_resolution,
            offset=self.map_offset
        )

        # Goal relative position to the agent
        relative_goal_position_raw = [self.goal_position[0] - position_agent[0], self.goal_position[1] - position_agent[1]]
        relative_goal_position = [relative_goal_position_raw[0] / self.map_resolution / self.map_size[0] * 2, relative_goal_position_raw[1] / self.map_resolution / self.map_size[1] * 2]

        # Orientation
        orientation_agent = euler_from_quaternion(orient_quat[0], orient_quat[1], orient_quat[2], orient_quat[3])
        orient = self.standardize_angle(orientation_agent[2])

        map_global_gridmap = OccupancyGridMap(
            data_array=self.map_global.copy(),
            cell_size=self.map_resolution
        )

        waypoints_meter, waypoints_pixel = a_star(
            start_m=(position_agent_pixel[0]*self.map_resolution, position_agent_pixel[1]*self.map_resolution),
            goal_m=(position_goal_pixel[0]*self.map_resolution, position_goal_pixel[1]*self.map_resolution),
            gmap=map_global_gridmap
        )

        if len(waypoints_meter) > self.subgoal_idx:
            subgoal_position = [waypoints_meter[self.subgoal_idx][0]-self.map_offset[0], waypoints_meter[self.subgoal_idx][1]-self.map_offset[1]]
        else:
            subgoal_position = [waypoints_meter[-1][0]-self.map_offset[0], waypoints_meter[-1][1]-self.map_offset[1]]


        goal_angle = self.standardize_angle(math.atan2(subgoal_position[1] - position_agent[1], subgoal_position[0] - position_agent[0]))
        heading = self.standardize_angle(goal_angle - orient)
        distance_obs = np.array([len(waypoints_meter) * self.map_resolution])

        # if self.step_num == 32:
        # print("Env Idx: ", self.env_idx)
        # print('Current Pos: ', position_agent_raw)
        # print('Goal Pos: ', self.goal_position)
        # print('Current Orient: ', orient)
        # print("Goal Position Relative: ", relative_goal_position)
        # print("Subgoal Position: ", subgoal_position)
        # print("Heading: ", heading)
        orient_pixel = [0, 0]
        if -np.pi/8 < orient <= np.pi/8:
            orient_pixel = [2, 0]
        elif np.pi/8 < orient <= 3*np.pi/8:
            orient_pixel = [1, 1]
        elif 3*np.pi/8 < orient <= 5*np.pi/8:
            orient_pixel = [0, 2]
        elif 5*np.pi/8 < orient <= 7*np.pi/8:
            orient_pixel = [-1, 1]
        elif -3*np.pi/8 < orient <= -np.pi/8:
            orient_pixel = [1, -1]
        elif -5*np.pi/8 < orient <= -3*np.pi/8:
            orient_pixel = [0, -2]
        elif -7*np.pi/8 < orient <= -5*np.pi/8:
            orient_pixel = [-1, -1]
        else:
            orient_pixel = [-2, 0]

        orient_map = [position_agent_pixel[0] + orient_pixel[0], position_agent_pixel[1] + orient_pixel[1]]
        
        map_layer_1 = self.map_global_with_worker.copy()
        for point in self.trajectory:
            map_layer_1[point[0]][point[1]] = self.map_pixel_values["Agent Trajectory"]
        # map_layer_1[position_goal_pixel[0]][position_goal_pixel[1]] = self.map_pixel_values["Goal"]
        self.paint_neighbor(map_layer_1, position_goal_pixel, self.map_pixel_values["Goal"])
        # map_layer_1[position_agent_pixel[0]][position_agent_pixel[1]] = self.map_pixel_values["Agent"]
        self.paint_neighbor(map_layer_1, position_agent_pixel, self.map_pixel_values["Agent"])
        
        map_layer_2 = self.map_global.copy()
        for point in self.trajectory:
            map_layer_2[point[0]][point[1]] = self.map_pixel_values["Agent Trajectory"]
        for point in waypoints_pixel:
            map_layer_2[point[0]][point[1]] = self.map_pixel_values["A* Path"]
        # map_layer_2[position_goal_pixel[0]][position_goal_pixel[1]] = self.map_pixel_values["Goal"]
        self.paint_neighbor(map_layer_2, position_goal_pixel, self.map_pixel_values["Goal"])
        # map_layer_2[position_agent_pixel[0]][position_agent_pixel[1]] = self.map_pixel_values["Agent"]
        self.paint_neighbor(map_layer_2, position_agent_pixel, self.map_pixel_values["Agent"])

        if 0 <= orient_map[0] < self.map_size[0] and 0 <= orient_map[1] < self.map_size[1]:
            map_layer_1[orient_map[0]][orient_map[1]] = self.map_pixel_values["Agent Orientation"]
            map_layer_2[orient_map[0]][orient_map[1]] = self.map_pixel_values["Agent Orientation"]
        # import matplotlib.pyplot as plt
        # # # # if self.step_num == 32:
        # plt.imshow(map_layer_1, cmap='gray')
        # plt.title("map_layer_1")
        # plt.savefig("map_layer_1_" + str(self.env_idx) + ".png")

        # plt.imshow(map_layer_2, cmap='gray')
        # plt.title("map_layer_2")
        # plt.savefig("map_layer_2_" + str(self.env_idx) + ".png")

        map_observation = np.stack((map_layer_1, map_layer_2), axis=0)

        camera_output = self.camera_rig.compute_observations()[0]

        # rgb_image_raw = camera_output['rgb'][:, :, :3]
        depth_image_raw = camera_output['depth']
        depth_image_resized = cv2.resize(depth_image_raw, self.camera_rgb_size, interpolation=cv2.INTER_LINEAR)
        depth_image_resized = np.nan_to_num(depth_image_resized, nan=self.min_depth_value)
        depth_image_obs = np.expand_dims(depth_image_resized, axis=0)
        

        # worker_bboxes = []
        # bounding_boxes = camera_output["bounding_boxes"]['data']
        # labels = camera_output["bounding_boxes"]['info']['idToLabels']
        
        # for bbox in bounding_boxes:
        #     label = labels[str(bbox['semanticId'])]['class']
        #     # print(label)
        #     if label == "worker":
        #         # print('Worker insight')
        #         worker_bboxes.append(bbox)

        # Convert the image from NumPy array to PIL Image
        # pil_image = Image.fromarray(rgb_image_raw)
        # draw = ImageDraw.Draw(pil_image)
        
        # Draw each bounding box
        # color = (255, 0, 0)
        # for bbox in worker_bboxes:
        #     draw.rectangle([(bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max'])], outline=color, width=12)

        # rgb_image_np = np.array(pil_image)

        # if self.step_num == 32:
        # plt.imshow(rgb_image_np)
        # plt.title("rgb_image_raw")
        # plt.savefig("rgb_image_raw_" + str(self.env_idx) + ".png")

        # rgb_image = Image.fromarray(rgb_image_np)
        # rgb_image = rgb_image.resize((128, 128), Image.Resampling.LANCZOS)
        # rgb_image = np.array(rgb_image)
        # rgb_image = np.transpose(rgb_image.copy(), (2, 0, 1))

        # if self.step_num == 32:
        # rgb_imshow = np.transpose(rgb_image, (1, 2, 0))

        # plt.imshow(rgb_imshow)
        # plt.title("rgb_image")
        # plt.savefig("rgb_image_" + str(self.env_idx) + ".png")

        # segmentation_image = np.zeros((rgb_image_raw.shape[0], rgb_image_raw.shape[1]), dtype=np.uint8)
        # for bbox in worker_bboxes:
        #     segmentation_image[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']] = 255
        # # add noise
        # noise = np.random.normal(0, 20, segmentation_image.shape)
        # segmentation_image_noisy = segmentation_image + noise
        # segmentation_image_noisy_clipped = np.clip(segmentation_image_noisy, 0, 255).astype(np.uint8)
        # segmentation_image_resized_img = Image.fromarray(segmentation_image_noisy_clipped).resize((128, 128), Image.Resampling.LANCZOS)

        # segmentation_image_resized = np.array(segmentation_image_resized_img)
        # segmentation_image_resized = np.expand_dims(segmentation_image_resized, axis=0) # 1， 128， 128

        # segmentation_pil_image_resized = Image.fromarray(segmentation_image_resized)
        # segmentation_pil_image_resized.save("segmentation_output_" + str(self.env_idx) + ".png")

        # update the map
        # use the pos of agent.
        # and its previous positioons

        # calculate global path

        # Generate a random RGB image

        # Combine the observations into a dictionary
        map_observation = map_observation.astype(np.uint8)
        # rgb_image = rgb_image.astype(np.uint8)
        # segmentation_image_resized.astype(np.uint8)

        orient_obs = np.array([np.cos(orient), np.sin(orient)])
        heading_obs = np.array([np.cos(heading), np.sin(heading)])

        worker_positions = []
        for worker in self.worker_list:
            worker_position_current, _ = worker.get_pos_rot()
            worker_position = [(worker_position_current[0]-position_agent[0]) / self.map_resolution / self.map_size[0] * 2, (worker_position_current[1]-position_agent[1]) / self.map_resolution / self.map_size[0] * 2]
            worker_angle = self.standardize_angle(math.atan2(worker_position[1], worker_position[0]))
            worker_positions.append(worker_position[0])
            worker_positions.append(worker_position[1])
            worker_positions.append(np.cos(worker_angle))
            worker_positions.append(np.sin(worker_angle))
        worker_positions_obs = np.array(worker_positions)

        global_goal_angle = self.standardize_angle(math.atan2(relative_goal_position[1], relative_goal_position[0]))
        goal_obs = np.array([relative_goal_position[0], relative_goal_position[1], np.cos(global_goal_angle), np.sin(global_goal_angle)])
        obs = {
            "goal": goal_obs,
            "heading": heading_obs,
            "orientation": orient_obs,
            "distance": distance_obs,
            "dynamic": worker_positions_obs,
            "map": map_observation,
            "depth": depth_image_obs
        }
        return obs


    def init_camera(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        stage = omni.usd.get_context().get_stage()

        self.camera_rig = Rig(
            [0, 0, 1],
            euler_to_quaternion(0, 0, 0),
            [
                0.01,
                0.01,
                0.01,
            ],
            self.camera_name,
            "/World",
            stage,
        )
        self.camera_rig.create_rig_from_file("/home/jianheng/SyntheticToolkit/sensors.json")

        # Add Collider
        prim_path = self.camera_rig._full_prim_path
        stage = self.camera_rig._stage

        prim = stage.GetPrimAtPath(prim_path)
        UsdPhysics.CollisionAPI.Apply(prim)
        cylinderGeom = UsdGeom.Cylinder.Define(stage, prim_path + "/Collider")
        cylinderGeom.GetHeightAttr().Set(1)  # Set Height (cm)
        cylinderGeom.GetRadiusAttr().Set(25)  # Set Radius (cm)
        xform = UsdGeom.XformCommonAPI(cylinderGeom)
        offset = Gf.Vec3d(0.0, 0.0, (self.env_idx-4) * 2.5)  # Offset
        xform.SetTranslate(offset)

        cylinderGeom.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)  # Invisible


    def reset(self, spawn = True, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # Reset the environment
        if seed is not None:
            np.random.seed(seed)

        _, empty_positions = find_empty_points(
                occupancy_map=self.map_global.copy(), 
                resolution=self.map_resolution,
                offset=self.map_offset,
                min_clear_radius=self.min_clear_radius, 
                min_edge_distance=self.min_edge_distance
            )
        
        if spawn:
            self.step_num = 0
            self.episode_reward = 0
            self.achieve_goal = 0
            self.return_goal = 0
            self.return_collision = 0
            self.trajectory = []

            random_position_index = np.random.randint(len(empty_positions))
            position_reset = empty_positions[random_position_index]
            orient_reset = np.random.uniform(-np.pi, np.pi)
            self.camera_rig.set_translate([position_reset[0], position_reset[1], self.camera_height])
            self.camera_rig.set_orient(euler_to_quaternion(0, 0, orient_reset))
        # print('Reseted position: ', position_reset)
        # print('Reseted Orient: ', orient_reset)
        # print('Empty points: ', empty_positions)

        if not spawn:
            position_current, orient_quat = self.camera_rig.get_pos_rot()
            position_reset = [position_current[0], position_current[1]]
            orientation_eular = euler_from_quaternion(orient_quat[0], orient_quat[1], orient_quat[2], orient_quat[3])
            orient_reset = np.array(orientation_eular[2])

        # Spawn the goal
        if self.curriculum < self.max_curriculum:
            curriculum_empty_positions = []
            curriculum_empty_positions_backup = []
            _, empty_positions_by_path = find_nearby_empty_points_by_path(
                occupancy_map=self.map_global.copy(), 
                point = position_meter_to_pixel(position_reset),
                min_distance = self.curriculum_mapping["distance"][self.curriculum-1][0],
                max_distance = self.curriculum_mapping["distance"][self.curriculum-1][1],
                resolution=self.map_resolution,
                offset=self.map_offset,
                min_clear_radius=self.min_clear_radius
            )
            # print(empty_positions_by_path)
            for empty_position in empty_positions_by_path:
                # print('empty position: ', empty_position)
                relative_goal_position_raw = [empty_position[0]-position_reset[0], empty_position[1]-position_reset[1]]
                distance = math.sqrt(relative_goal_position_raw[0] ** 2 + relative_goal_position_raw[1] **2)
                goal_angle = self.standardize_angle(math.atan2(relative_goal_position_raw[1], relative_goal_position_raw[0]))
                heading = self.standardize_angle(goal_angle - orient_reset)
                # print('Heading: ', heading)

                # if abs(heading) <= self.f["heading"][self.curriculum-1] and self.curriculum_mapping["distance"][self.curriculum-1][0] < distance < self.curriculum_mapping["distance"][self.curriculum-1][1]:
                if abs(heading) <= self.curriculum_mapping["heading"][self.curriculum-1] and self.curriculum_mapping["distance"][self.curriculum-1][0] < distance < self.curriculum_mapping["distance"][self.curriculum-1][1]:
                    curriculum_empty_positions.append(empty_position)

                else:
                    curriculum_empty_positions_backup.append(empty_position)

            if len(curriculum_empty_positions) > 0:
                print('Resetting Agent: ', self.env_idx)
                # print('Empty goal positions: ', curriculum_empty_positions)
                goal_random_position_index = np.random.randint(len(curriculum_empty_positions))
                self.goal_position = curriculum_empty_positions[goal_random_position_index]

            elif len(curriculum_empty_positions_backup) > 0:
                print('Resetting Agent: ', self.env_idx, ' Other Direction')
                # print('Empty goal positions: ', curriculum_empty_positions)
                goal_random_position_index = np.random.randint(len(curriculum_empty_positions_backup))
                self.goal_position = curriculum_empty_positions_backup[goal_random_position_index]

            else:
                print('Resetting Agent: ', self.env_idx, 'Goal Position Not Found, Randomly Setting Goal')
                goal_random_position_index = np.random.randint(len(empty_positions))
                self.goal_position = empty_positions[goal_random_position_index]

        elif self.curriculum == self.max_curriculum:
            goal_random_position_index = np.random.randint(len(empty_positions))
            self.goal_position = empty_positions[goal_random_position_index]

        
        self.collision = False
        self.collision_with_agent = False

        obs = self.compute_obs()
        # calculate where to spawn.
        # local_map
        # check ^ for an empty space
        #valid = False
        #while not valid:
        #    x = random.randint()
        
        #self._agent_ref.reset(translate, orientation)
        return obs, {}


    # def step(self, action):
    #     print("Env Step Action: ", action)
    #     self.camera_rig.apply_veloc(self._action_mapping[action][0], self._action_mapping[action][1])
    #     # Simulate the effect of the action on the position
    #     position = np.random.rand(2) * 10  # Example updated position after action
    #     reward = -np.linalg.norm(position)  # Example reward based on position
    #     done = False  # Example done condition
    #     truncated = False  # Example truncated condition
    #     obs = self.compute_obs()
    #     return obs, reward, done, truncated, {}

    def render(self, mode='rgb'):
        # Return a RGB image (to be comatible with RecordVideo)
        return self._generate_random_image(self.camera_rgb_size)

    def close(self):
        pass

    def _generate_random_map(self, size):
        # Generate a random local map
        return np.random.randint(0, 256, (2, *size), dtype=np.uint8)

    def _generate_random_image(self, size):
        # Generate a random RGB image
        return np.random.randint(0, 256, (3, *size), dtype=np.uint8)
