import numpy as np
from pxr import Usd, UsdGeom, Gf

def map_generation(
        world, 
        map_size=(128, 128), 
        resolution=0.25, 
        offset=(26.5, 0.5, 0.0), 
        keywords_to_check=["RackPile"], 
    ):
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

        x_min = int((bbox_min[0] + offset[0]) / resolution) - 1
        x_max = int((bbox_max[0] + offset[0]) / resolution) + 1
        y_min = int((bbox_min[1] + offset[1]) / resolution) - 1
        y_max = int((bbox_max[1] + offset[1]) / resolution) + 1
        
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


def find_empty_points(occupancy_map, resolution=0.25, offset=(26.5, 0.5, 0.0), min_clear_radius=4, min_edge_distance=10):
    """
    Find a random point on the map that satisfies the conditions:
    1. The point and all points within the Euclidean distance of `min_clear_radius` are 0.
    2. The point is at least `min_edge_distance` away from any edge of the map.
    
    Args:
        occupancy_map (np.ndarray): The occupancy map.
        resolution (float): The resolution of the map in meters per pixel.
        offset (tuple): The offset to apply to the coordinates (x_offset, y_offset, z_offset).
        min_clear_radius (int): The minimum Euclidean distance radius around the point that must be clear.
        min_edge_distance (int): The minimum distance from any edge of the map.
        
    Returns:
        valid_points (list): A list of points in the occupancy map of clear area.
        valid_positions (list): A list of points in the world of clear area.
    """
    
    map_height, map_width = occupancy_map.shape
    valid_points = []
    valid_positions = []
    
    for x in range(min_edge_distance, map_width - min_edge_distance):
        for y in range(min_edge_distance, map_height - min_edge_distance):
            if occupancy_map[y, x] == 0:
                clear = True
                for dx in range(-min_clear_radius, min_clear_radius + 1):
                    for dy in range(-min_clear_radius, min_clear_radius + 1):
                        if (dx**2 + dy**2 <= min_clear_radius**2):
                            if (0 <= x + dx < map_width and 0 <= y + dy < map_height):
                                if occupancy_map[y + dy, x + dx] != 0:
                                    clear = False
                                    break
                    if not clear:
                        break
                if clear:
                    valid_points.append((y, x))
                    valid_positions.append((y * resolution - offset[0], x * resolution - offset[1]))
    
    return valid_points, valid_positions


def get_poi(
    world, 
    resolution=0.25, 
    offset=(26.5, 0.5, 0.0)
):
    """
    Get POI postions from the given world.

    Args:
        world (World): The simulation world containing objects.
        map_size (tuple): The size of the map in pixels (width, height).
        resolution (float): The resolution of the map in meters per pixel.
        offset (tuple): The offset to apply to the coordinates (x_offset, y_offset, z_offset).

    Returns:
        poi_points (list): A list of points in the occupancy map of POIs.
        poi_positions (list): A list of points in the world of POIs.
    """

    keywords_to_check = ["POI"]

    offset = np.array(offset)  # offset in meters

    poi_points = []
    poi_positions = []

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
                xformable = UsdGeom.Xformable(prim)
                local_transformation: Gf.Matrix4d = xformable.GetLocalTransformation()
                translation: Gf.Vec3d = local_transformation.ExtractTranslation()
                poi_positions.append((translation[0], translation[1]))
                poi_points.append((int((translation[0]+offset[0])/resolution), int((translation[1]+offset[1])/resolution)))
                                      
    return poi_points, poi_positions


def position_meter_to_pixel(
    position_meter=[0, 0],
    resolution=0.25, 
    offset=(26.5, 0.5, 0.0)
):
    position_pixel = [0, 0]
    position_pixel[0] = int((position_meter[0] + offset[0]) / resolution)
    position_pixel[1] = int((position_meter[1] + offset[1]) / resolution)
    return position_pixel
    

def position_pixel_to_meter(
    position_pixel=[0, 0],
    resolution=0.25, 
    offset=(26.5, 0.5, 0.0)
):
    position_meter = [0.0, 0.0]
    position_meter[0] = resolution * position_pixel[0] - offset[0]
    position_meter[1] = resolution * position_pixel[1] - offset[1]
    return position_meter


def waypoints_2d_to_3d(waypoints_2d, height):
    waypoints_3d = []
    for waypoint in waypoints_2d:
        waypoints_3d.append([waypoint[0], waypoint[1], height])
    return waypoints_3d

# ======================= Example Usage ==================================
# from omni.isaac.kit import SimulationApp
# simulation_app = SimulationApp({"headless": False})

# from map_generation import map_generation
# import matplotlib.pyplot as plt

# from omni.isaac.core import World
# from omni.isaac.core.utils.stage import open_stage

# # Initialize and open the USD stage
# open_stage(usd_path='/home/jianheng/omniverse/assets/Warehouse_01.usd')
# world = World()

# world.reset()

# # Generate the occupancy map
# keywords_to_check=[
#     'RackPile', 
#     'RackLong', 
#     'Forklift', 
#     'Table', 
#     'IndustrialSteelShelving', 
#     'WallA', 
#     'RackShelf', 
#     'RackFrame', 
#     'RackShield',
#     'PaletteA',
#     'EmergencyBoardFull',
#     'FuseBox',
#     ]
# occupancy_map = map_generation(world=world, keywords_to_check=keywords_to_check)

# # Display the occupancy map using matplotlib
# plt.imshow(occupancy_map, cmap='gray', origin='lower')
# plt.title("Occupancy Map")
# plt.show()

# # Keep the simulation running
# while True:
#     world.step()

# simulation_app.close()

# ======================= Example Usage ==================================
