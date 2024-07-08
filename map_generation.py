from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import matplotlib.pyplot as plt
from pxr import Usd, UsdGeom

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage

open_stage(usd_path='Warehouse_01.usd')
world = World()

world.reset()

map_size = (128, 128)  # size of map (pixels)
resolution = 0.25  # meter per pixel
offset = np.array([26.5, 0.5, 0.0])  # offset in meters
height_range = np.array([0.1, 7.0])  # check if the height is between 0.1m and 7.0m

occupancy_map = np.zeros(map_size)

def get_bounding_box(prim):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    min_corner = bbox.GetMin()
    max_corner = bbox.GetMax()
    return min_corner, max_corner

def update_occupancy_map(prim):
    bbox_min, bbox_max = get_bounding_box(prim)
    
    # Check if the height is between 0.1m and 7.0m and located at the ground (not hang on the roof)
    if height_range[0] <= bbox_max[2] - bbox_min[2] <= height_range[1] and bbox_min[2] + offset[2] <= height_range[0]:
        x_min = int((bbox_min[0] + offset[0]) / resolution)
        x_max = int((bbox_max[0] + offset[0]) / resolution)
        y_min = int((bbox_min[1] + offset[1]) / resolution)
        y_max = int((bbox_max[1] + offset[1]) / resolution)
        
        # Ensure the indices are within map boundaries
        x_min = max(0, min(map_size[0] - 1, x_min))
        x_max = max(0, min(map_size[0] - 1, x_max))
        y_min = max(0, min(map_size[1] - 1, y_min))
        y_max = max(0, min(map_size[1] - 1, y_max))
        
        occupancy_map[x_min:x_max, y_min:y_max] = 1

# Iterate over all objects in the world
stage = world.stage
for prim in stage.Traverse():
    if prim.GetTypeName() == "Xform":  # Only take Xforms (or check object name)
        update_occupancy_map(prim)

plt.imshow(occupancy_map, cmap='gray')
plt.title("Occupancy Map")
plt.show()

while True:
    world.step()

simulation_app.close()