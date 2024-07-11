"""
Adapted from https://github.com/richardos/occupancy-grid-a-star/tree/master
"""
import math
from heapq import heappush, heappop
import math

def dist2d(point1, point2):
    """
    Euclidean distance between two points
    :param point1:
    :param point2:
    :return:
    """

    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]

    dist2 = (x1 - x2)**2 + (y1 - y2)**2

    return math.sqrt(dist2)


def _get_movements_4n():
    """
    Get all possible 4-connectivity movements.
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0)]


def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]


def a_star(start_m, goal_m, gmap, movement='8N', occupancy_cost_factor=3):
    """
    A* for 2D occupancy grid.

    :param start_m: start node (x, y) in meters
    :param goal_m: goal node (x, y) in meters
    :param gmap: the grid map
    :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :param occupancy_cost_factor: a number the will be multiplied by the occupancy probability
        of a grid map cell to give the additional movement cost to this cell (default: 3).

    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """

    # get array indices of start and goal
    start = gmap.get_index_from_coordinates(start_m[0], start_m[1])
    goal = gmap.get_index_from_coordinates(goal_m[0], goal_m[1])

    # check if start and goal nodes correspond to free spaces
    if gmap.is_occupied_idx(start):
        raise Exception('Start node is not traversable')

    if gmap.is_occupied_idx(goal):
        raise Exception('Goal node is not traversable')

    # add start node to front
    # front is a list of (total estimated cost to goal, total cost from start to node, node, previous node)
    start_node_cost = 0
    start_node_estimated_cost_to_goal = dist2d(start, goal) + start_node_cost
    front = [(start_node_estimated_cost_to_goal, start_node_cost, start, None)]

    # use a dictionary to remember where we came from in order to reconstruct the path later on
    came_from = {}

    # get possible movements
    if movement == '4N':
        movements = _get_movements_4n()
    elif movement == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')

    # while there are elements to investigate in our front.
    while front:
        # get smallest item and remove from front.
        element = heappop(front)

        # if this has been visited already, skip it
        total_cost, cost, pos, previous = element
        if gmap.is_visited_idx(pos):
            continue

        # now it has been visited, mark with cost
        gmap.mark_visited_idx(pos)

        # set its previous node
        came_from[pos] = previous

        # if the goal has been reached, we are done!
        if pos == goal:
            break

        # check all neighbors
        for dx, dy, deltacost in movements:
            # determine new position
            new_x = pos[0] + dx
            new_y = pos[1] + dy
            new_pos = (new_x, new_y)

            # check whether new position is inside the map
            # if not, skip node
            if not gmap.is_inside_idx(new_pos):
                continue

            # add node to front if it was not visited before and is not an obstacle
            if (not gmap.is_visited_idx(new_pos)) and (not gmap.is_occupied_idx(new_pos)):
                potential_function_cost = gmap.get_data_idx(new_pos)*occupancy_cost_factor
                new_cost = cost + deltacost + potential_function_cost
                new_total_cost_to_goal = new_cost + dist2d(new_pos, goal) + potential_function_cost

                heappush(front, (new_total_cost_to_goal, new_cost, new_pos, pos))

    # reconstruct path backwards (only if we reached the goal)
    path = []
    path_idx = []
    if pos == goal:
        while pos:
            path_idx.append(pos)
            # transform array indices to meters
            pos_m_x, pos_m_y = gmap.get_coordinates_from_index(pos[0], pos[1])
            path.append((pos_m_x, pos_m_y))
            pos = came_from[pos]

        # reverse so that path is from start to goal.
        path.reverse()
        path_idx.reverse()

    return path, path_idx


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

# # Display the new image with the path
# plt.imshow(occupancy_map, cmap='gray', origin='lower')
# plt.title("occupancy_map")
# plt.show()


# from a_star import a_star
# from gridmap import OccupancyGridMap

# cell_size = 0.25

# gmap = OccupancyGridMap(data_array=occupancy_map, cell_size=cell_size)

# start_point = (6, 6) # meters (pixel number * cell_size)
# goal_point = (25, 19)

# # A*
# path, path_idx = a_star(start_point, goal_point, gmap)

# print('Waypoints in meters',  path)
# print('Waypoints in pixels', path_idx)
# if path:  # Draw the path is find the path
#     gmap.plot() # Plot the Map
#     path_x, path_y = zip(*path_idx)
#     plt.plot(path_x, path_y, marker='o', color='r') # Plot the path
# plt.show()

# # Keep the simulation running
# while True:
#     world.step()

# simulation_app.close()

# ======================= Example Usage ==================================