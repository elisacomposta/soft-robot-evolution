import numpy as np
from copy import deepcopy
import math
import warnings

### FITNESS COMPUTATION FUNCTIONS ###
def compute_length(structure):
    coordinates = []
    for i in range(structure.shape[0]):
        for j in range(structure.shape[1]):
            if structure.body[i][j] != 0:
                coordinates.append([i, j])
    
    minL, maxL = range_y(coordinates)
    length = maxL - minL + 1
    return length


def compute_height(structure):
    coordinates = []
    for i in range(structure.shape[0]):
        for j in range(structure.shape[1]):
            if structure.body[i][j] != 0:
                coordinates.append([i, j])
    minH, maxH = range_x(coordinates)
    height = maxH - minH + 1
    return height


def compute_base_length(structure):
    baseLength = len(np.nonzero(structure.body[len(structure.body)-1])[0])
    return baseLength


def compute_emptiness(structure):
    emptiness = (structure.body == 0).sum() / (structure.shape[0] * structure.shape[1])
    return emptiness


# VOXEL_TYPES = { 'EMPTY': 0, 'RIGID': 1, 'SOFT': 2, 'H_ACT': 3, 'V_ACT': 4, 'FIXED': 5} (see evogym/utils.py)
def compute_actuation(structure):
    body = np.asarray(structure.body)

    v_total = (body > 0).sum()
    v_actuation = (body == 3).sum() / v_total
    h_actuation = (body == 4).sum() / v_total
    actuation = v_actuation + h_actuation

    return actuation, v_actuation, h_actuation


def compute_compactness(structure):
    # approximate convex hull
    convexHull = deepcopy(structure.body)
    shape = structure.shape

    # loop as long as there are empty cells have at least five of the eight Moore neighbors as true
    none = False
    while not none:
        none = True
        for i in range(shape[0]):
            for j in range(shape[1]):
                if convexHull[i][j]==0:     # empty voxel found
                    adjacentCount = 0
                    # count not empty Moore neighbors
                    for a in [-1, 0, 1]:
                        for b in [-1, 0, 1]:
                            i_neigh = i+a if (i+a > 0 and i+a < shape[0]) else -1
                            j_neigh = j+b if (j+b > 0 and j+b < shape[1]) else -1
                            if not (a == 0 and b == 0) and i_neigh>=0 and j_neigh>=0 and convexHull[i_neigh][j_neigh]>0:
                                adjacentCount += 1
                                
                    # if at least five, fill the cell (with -1, nonzero value)
                    if adjacentCount >= 5:
                        convexHull[i][j] = -1
                        none = False

    nVoxels = len(np.nonzero(structure.body)[0])            # non empty voxels in body
    nConvexHull = len(np.matrix.nonzero(convexHull)[0])     # non empty voxels in convex hull
    return nVoxels / nConvexHull    # -> 0.0 for less compact shapes, -> 1.0 for more compact shapes


def compute_elongation(structure, n_dir):
    if n_dir < 0:
        warnings.warn(UserWarning("n_dir shoud be a non negative number"))

    diameters = []
    coordinates = []
    for i in range(structure.shape[0]):
        for j in range(structure.shape[1]):
            if structure.body[i][j] != 0:
                coordinates.append([i, j])

    for i in range(n_dir):
        theta = (2 * i * math.pi) / n_dir
        rotated_coordinates = []
        
        for p in coordinates:
            x = p[0]
            y = p[1]
            new_x = round( x * math.cos(theta) - y * math.sin(theta) )
            new_y = round( x * math.sin(theta) + y * math.cos(theta) )
            rotated_coordinates.append([new_x, new_y])

        minX, maxX = range_x(rotated_coordinates)
        minY, maxY = range_y(rotated_coordinates)
        sideX = maxX - minX +1
        sideY = maxY - minY +1
        diameters.append( min(sideX, sideY) / max(sideX, sideY) )

    return 1 - min(diameters)

def range_x(coordinates):
    coordinates = np.array(coordinates)
    sorted = np.argsort(coordinates[:,0])
    min_x = coordinates[sorted[0],:][0]
    max_x = coordinates[sorted[-1],:][0]
    return min_x, max_x

def range_y(coordinates):
    coordinates = np.array(coordinates)
    sorted = np.argsort(coordinates[:,1])
    min_y = coordinates[sorted[0],:][1]
    max_y = coordinates[sorted[-1],:][1]
    return min_y, max_y

