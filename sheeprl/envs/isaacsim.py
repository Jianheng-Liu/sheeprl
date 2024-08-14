from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import omni
from omni.physx import get_physx_simulation_interface
from pxr import Gf, UsdGeom, UsdPhysics, PhysicsSchemaTools
import math
from PIL import Image, ImageDraw

from map_utils import find_empty_points, position_meter_to_pixel
from gridmap import OccupancyGridMap
from a_star import a_star
from SyntheticToolkit.Sensors.rig import Rig
from SyntheticToolkit.utils.omni_utils import euler_to_quaternion, euler_from_quaternion

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
        self.max_curriculum = 5
        # self.curriculum_threshold = 1.0
        self.env_idx = 0
        self.step_num = 0
        self.episode_reward = 0
        # self.max_step_num = 512
        self.log_episode_number = 10
        self.log_rew = []
        self.log_len = []
        self.log_goa = []
        self.log_col = []
        
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

        # 1-4
        self.curriculum_mapping = {
            "heading": [np.pi/3, np.pi/2, 2*np.pi/3, np.pi, np.inf],
            "distance": [[2.0, 5.0], [3.0, 8.0], [6.0, 15.0], [10.0, 32.0], [-np.inf, np.inf]],
            "step_num": [200, 300, 400, 500, 600],
            "goal_distance": [1.4, 1.3, 1.2, 1.1, 1.0]
        }
        # 5: random

        self.map_global = None
        self.map_global_with_worker = None
        self.map_resolution = None
        self.map_offset = None
        self.min_clear_radius = 4
        self.min_edge_distance = 10

        self.collision = False
        self.collision_with_agent = False
        self.done = False
        self.goal_position = []
        self.trajectory = []
        self.trajectory_length = 32
        self.subgoal_idx = 2

        self._render_mode: str = "rgb_array"
            
        # Define observation and action spaces
        self._observation_space = spaces.Dict({
            "goal": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "heading": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "map": spaces.Box(low=0, high=255, shape=(2, map_size[0], map_size[1]), dtype=np.uint8),
            "segmentation": spaces.Box(low=0, high=255, shape=(1, camera_rgb_size[0], camera_rgb_size[1]), dtype=np.uint8)
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
            
    def get_global_map(self, map_global, map_global_with_worker, resolution, offset):
        self.map_global = map_global.copy()
        self.map_global_with_worker = map_global_with_worker.copy()
        self.map_resolution = resolution
        self.map_offset = offset


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

        info = {}

        # calculate obs
        obs = self.compute_obs()

        # calculate rewards
        # reward_time = -1

        # reward_heading = math.cos(obs["heading"]) * 5

        goal_distance = math.sqrt(obs['goal'][0] ** 2 + obs['goal'][1] ** 2)
        # reward_distance = - (goal_distance - 0.5) * 0.5
        
        self.step_num += 1

        reward_goal = 0
        if goal_distance < self.curriculum_mapping["goal_distance"][self.curriculum-1]:
            # Arrive at the goal
            print('Arrive at the goal! - agent: ', self.env_idx, ' - stage: ', self.curriculum)
            reward_goal = 1
            done = True
            goal = True

        reward_collision = 0
        if self.collision:
            print('Collide! - agent: ', self.env_idx, ' - stage: ', self.curriculum)
            reward_collision = -1
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

        # print('Reward heading: ', reward_heading)
        # print('Reward distance: ', reward_distance)
        # print('Reward goal: ', reward_goal)
        # print('Reward collision: ', reward_collision)

        reward = reward_goal + reward_collision
        self.episode_reward += reward

        if done or truncated:
            if len(self.log_rew) < self.log_episode_number:
                self.log_rew.append(self.episode_reward)
                self.log_len.append(self.step_num)
                self.log_goa.append(reward_goal)
                self.log_col.append(reward_collision)

            else:
                self.log_rew.pop(0)
                self.log_rew.append(self.episode_reward)
                self.log_len.pop(0)
                self.log_len.append(self.step_num)
                rew_avg = sum(self.log_rew) / len(self.log_rew)
                len_avg = sum(self.log_len) / len(self.log_len)
                goa = sum(self.log_goa)
                col = sum(self.log_col)
                
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
                    "g": goa
                }

            self.reset(spawn=True)
            
            # if not goal:
            #     self.reset(spawn=True)

            # elif goal:
            #     self.reset(spawn=False)
            
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
        relative_goal_position = np.array(relative_goal_position_raw)

        # Orientation
        orientation_agent = euler_from_quaternion(orient_quat[0], orient_quat[1], orient_quat[2], orient_quat[3])
        orient = np.array(orientation_agent[2])

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
        heading = [self.standardize_angle(goal_angle - orient)]
        heading = np.array(heading)

        # if self.step_num == 32:
        # print("Env Idx: ", self.env_idx)
        # print('Current Pos: ', position_agent_raw)
        # print('GOal Pos: ', self.goal_position)
        # print('Current Orient: ', orient)
        # print("Goal Position Relative: ", relative_goal_position)
        # print("Subgoal Position: ", subgoal_position)
        # print("Heading: ", heading)
        

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

        rgb_image_raw = camera_output['rgb'][:, :, :3]

        worker_bboxes = []
        bounding_boxes = camera_output["bounding_boxes"]['data']
        labels = camera_output["bounding_boxes"]['info']['idToLabels']
        
        for bbox in bounding_boxes:
            label = labels[str(bbox['semanticId'])]['class']
            # print(label)
            if label == "worker":
                # print('Worker insight')
                worker_bboxes.append(bbox)

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

        segmentation_image = np.zeros((rgb_image_raw.shape[0], rgb_image_raw.shape[1]), dtype=np.uint8)
        for bbox in worker_bboxes:
            segmentation_image[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']] = 255
        # add noise
        noise = np.random.normal(0, 20, segmentation_image.shape)
        segmentation_image_noisy = segmentation_image + noise
        segmentation_image_noisy_clipped = np.clip(segmentation_image_noisy, 0, 255).astype(np.uint8)
        segmentation_image_resized_img = Image.fromarray(segmentation_image_noisy_clipped).resize((128, 128), Image.Resampling.LANCZOS)

        segmentation_image_resized = np.array(segmentation_image_resized_img)
        segmentation_image_resized = np.expand_dims(segmentation_image_resized, axis=0) # 1， 128， 128

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
        segmentation_image_resized.astype(np.uint8)
        
        obs = {
            "goal": relative_goal_position,
            "heading": heading,
            "map": map_observation,
            "segmentation": segmentation_image_resized
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
        cylinderGeom.GetRadiusAttr().Set(40)  # Set Radius (cm)
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
        if self.curriculum < 5:
            curriculum_empty_positions = []
            curriculum_empty_positions_backup = []
            for empty_position in empty_positions:
                # print('empty position: ', empty_position)
                relative_goal_position_raw = [empty_position[0] - position_reset[0], empty_position[1] - position_reset[1]]
                distance = math.sqrt(relative_goal_position_raw[0] ** 2 + relative_goal_position_raw[1] **2)
                # print('Distance: ', distance)
                goal_angle = self.standardize_angle(math.atan2(relative_goal_position_raw[1], relative_goal_position_raw[0]))
                heading = self.standardize_angle(goal_angle - orient_reset)
                # print('Heading: ', heading)

                # if abs(heading) <= self.f["heading"][self.curriculum-1] and self.curriculum_mapping["distance"][self.curriculum-1][0] < distance < self.curriculum_mapping["distance"][self.curriculum-1][1]:
                if abs(heading) <= self.curriculum_mapping["heading"][self.curriculum-1] and self.curriculum_mapping["distance"][self.curriculum-1][0] < distance < self.curriculum_mapping["distance"][self.curriculum-1][1]:
                    curriculum_empty_positions.append(empty_position)

                if self.curriculum_mapping["distance"][self.curriculum-1][0] < distance < self.curriculum_mapping["distance"][self.curriculum-1][1]:
                    curriculum_empty_positions_backup.append(empty_position)

            if len(curriculum_empty_positions) > 0:
                print('Resetting Agent: ', self.env_idx)
                # print('Empty goal positions: ', curriculum_empty_positions)
                goal_random_position_index = np.random.randint(len(curriculum_empty_positions))
                self.goal_position = curriculum_empty_positions[goal_random_position_index]

            elif len(curriculum_empty_positions_backup) > 0:
                print('Resetting Agent: ', self.env_idx, ' Backup')
                # print('Empty goal positions: ', curriculum_empty_positions)
                goal_random_position_index = np.random.randint(len(curriculum_empty_positions_backup))
                self.goal_position = curriculum_empty_positions_backup[goal_random_position_index]

            else:
                print('Resetting Agent: ', self.env_idx, 'Goal Position Not Found, Randomly Setting Goal')
                goal_random_position_index = np.random.randint(len(empty_positions))
                self.goal_position = empty_positions[goal_random_position_index]

        elif self.curriculum == 5:
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
