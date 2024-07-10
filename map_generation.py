from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import matplotlib.pyplot as plt
from pxr import Usd, UsdGeom

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage

def generate_occupancy_map(world, map_size=(128, 128), resolution=0.25, offset=(26.5, 0.5, 0.0), keywords_to_check=["RackPile"]):
    """
    Generate an occupancy map from the given world.

    Args:
        world (World): The simulation world containing objects.
        map_size (tuple): The size of the map in pixels (width, height).
        resolution (float): The resolution of the map in meters per pixel.
        offset (tuple): The offset to apply to the coordinates (x_offset, y_offset, z_offset).
        keywords_to_check (list): List of keywords to check in object names.

    Returns:
        np.ndarray: The generated occupancy map.
    """
    
    map_size = map_size  # size of map (pixels)
    resolution = resolution  # meter per pixel
    offset = np.array(offset)  # offset in meters

    occupancy_map = np.zeros(map_size)

    def get_bounding_box(prim):
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        min_corner = bbox.GetMin()
        max_corner = bbox.GetMax()
        return min_corner, max_corner

    def update_occupancy_map(prim):
        bbox_min, bbox_max = get_bounding_box(prim)

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

    def check_name_for_keywords(name):
        # Split the name by '_' and check if any part contains a keyword
        parts = name.split('_')
        for part in parts:
            for keyword in keywords_to_check:
                if keyword in part:
                    return True
        return False

    # Iterate over all objects in the world
    stage = world.stage
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Xform":  # Only take Xforms
            # Check if the object's name contains any of the keywords
            if check_name_for_keywords(prim.GetName()):
                update_occupancy_map(prim)

    return occupancy_map

# Initialize and open the USD stage
open_stage(usd_path='Warehouse_02.usd')
world = World()

world.reset()

# Generate the occupancy map
keywords_to_check=[
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
    ]
occupancy_map = generate_occupancy_map(world=world, keywords_to_check=keywords_to_check)

# Display the occupancy map using matplotlib
plt.imshow(occupancy_map, cmap='gray')
plt.title("Occupancy Map")
plt.show()

# Keep the simulation running
while True:
    world.step()

simulation_app.close()
