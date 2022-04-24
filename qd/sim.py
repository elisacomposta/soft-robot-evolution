#!/usr/bin/env python3

import os
import numpy as np
import random
import time
import warnings
import math
from copy import deepcopy

from evogym import sample_robot
from evogym.envs.walk import WalkingFlat, SoftBridge, Duck
from evogym.envs.climb import Climb0, Climb1, Climb2
from evogym.envs.flip import Flipping
from evogym.envs.balance import Balance, BalanceJump
from evogym.envs.change_shape import MaximizeShape, MinimizeShape
from evogym.envs.traverse import StepsUp, StepsDown, WalkingBumpy, WalkingBumpy2, VerticalBarrier, FloatingPlatform, Gaps, BlockSoup
from evogym.envs.manipulate import CarrySmallRect, CarrySmallRectToTable, PushSmallRect, PushSmallRectOnOppositeSide, ThrowSmallRect, CatchSmallRect
from utils.algo_utils import Structure, TerminationCondition, mutate
import utils.mp_group as mp
from evogym import EvoWorld, EvoSim, EvoViewer

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
from ppo import run_ppo

###### SIMULATION FUNCTIONS ######

def simulate(env_name, inds, experiment_name, num_episode=5, num_cores=4):

    ## DEFINE TERMINATION CONDITION
    tc = TerminationCondition(num_episode)

    group = mp.Group()
    for ind in inds:
        ## RESULT DIR
        save_path = os.path.join(root_dir, "results", experiment_name, "generation_" + str(ind.structure.generation), "ind" + str(ind.structure.label))    # evaluated ind dir
        try:
            os.makedirs(save_path)
        except:
            pass
        
        file_path = os.path.join(save_path, "structure")
        np.savez(file_path, ind.structure.body, ind.structure.connections)

        ## COMPUTE FITNESS: RUN PPO OR GROUP JOBS
        #ind.structure.reward = run_ppo(structure=(ind.structure.body, ind.structure.connections), termination_condition=tc, saving_convention=(save_path, ind.structure.label), verbose=False)
        ppo_args = (ind, tc, (save_path, ind.structure.label), env_name, False)
        group.add_job(run_ppo, ppo_args, callback=ind.structure.set_reward)

    group.run_jobs(num_cores)
    
    for ind in inds:
        #ind.structure.reward = np.random.uniform(10.)
        #print("Reward: ", ind.structure.reward)
        ind.structure.compute_fitness()
        ind.fitness.values = [ind.structure.fitness]


def make_env(env_name, shape, label, seed=-1, ind=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    if ind.structure is None:
        body, conn = sample_robot(shape)
    else:
        body, conn = mutate(ind.structure.body, shape)
        if body is None or conn is None:    # mutation failed after num_attempts
            body, conn = sample_robot(shape)

    structure = Structure(body, conn, label, ind.generation, shape)
    ind.structure = structure

    env = get_env(ind, env_name)
        
    if (seed >= 0):
        env.seed(seed)

    return env


def compute_features(ind, features_list):
    length = compute_length(ind.structure)
    height = compute_height(ind.structure)
    baseLength = compute_base_length(ind.structure)
    emptiness = compute_emptiness(ind.structure)
    compactness = compute_compactness(ind.structure)
    elongation = compute_elongation(ind.structure, 2)
    actuation, v_actuation, h_actuation = compute_actuation(ind.structure)
    scores = {
            "reward": ind.structure.fitness,
            "length": length,     
            "height": height,
            "baseLength": baseLength,
            "emptiness": emptiness,
            "compactness": compactness,
            "elongation": elongation,
            "actuation": actuation,
            "verticalActuation": v_actuation,
            "horizontalActuation": h_actuation
    } 
    ind.features.values = [scores[x] for x in features_list]


### GET ENVIRONMENT ###
def get_env(ind, env_name):

    # walk
    if env_name == "Walker-v0":
        env = WalkingFlat(ind.structure.body, ind.structure.connections)        # build env with 'set()' method
    elif env_name == "BridgeWalker-v0":
        env = SoftBridge(ind.structure.body, ind.structure.connections)
    elif env_name == "CaveCrawler-v0":          #non va
        env = Duck(ind.structure.body, ind.structure.connections)
    
    # climb
    elif env_name == "Climber-v0":
        env = Climb0(ind.structure.body, ind.structure.connections)
    elif env_name == "Climber-v1":
        env = Climb1(ind.structure.body, ind.structure.connections)
    elif env_name == "Climber-v2":
        env = Climb2(ind.structure.body, ind.structure.connections) #non va
    
    # flip
    elif env_name == "Flipper-v0":
        env = Flipping(ind.structure.body, ind.structure.connections)
    
    #balance
    elif env_name == "Balancer-v0":
        env = Balance(ind.structure.body, ind.structure.connections)
    elif env_name == "Balancer-v1":
        env = BalanceJump(ind.structure.body, ind.structure.connections)
    
    # change_shape
    elif env_name == "ShapeChange":
        env = MaximizeShape(ind.structure.body, ind.structure.connections)  #non va
    elif env_name == "ShapeChange":
        env = MinimizeShape(ind.structure.body, ind.structure.connections) #non va
    # ne mancano altri

    # traverse
    elif env_name == "UpStepper-v0":
        env = StepsUp(ind.structure.body, ind.structure.connections)
    elif env_name == "DownStepper-v0":
        env = StepsDown(ind.structure.body, ind.structure.connections) #non va
    elif env_name == "ObstacleTraverser-v0":
        env = WalkingBumpy(ind.structure.body, ind.structure.connections) #non va
    elif env_name == "ObstacleTraverser-v1":
        env = WalkingBumpy2(ind.structure.body, ind.structure.connections) #non va
    elif env_name == "Hurdler-v0":
        env = VerticalBarrier(ind.structure.body, ind.structure.connections) #non va
    elif env_name == "PlatformJumper-v0":
        env = FloatingPlatform(ind.structure.body, ind.structure.connections) #non va
    elif env_name == "GapJumper-v0":
        env = Gaps(ind.structure.body, ind.structure.connections) #non va
    elif env_name == "Traverser-v0":
        env = BlockSoup(ind.structure.body, ind.structure.connections)
    
    # manipulate
    elif env_name == "Carrier-v0":
        env = CarrySmallRect(ind.structure.body, ind.structure.connections) #no
    elif env_name == "Carrier-v1":
        env = CarrySmallRectToTable(ind.structure.body, ind.structure.connections) #no
    elif env_name == "Pusher-v0":
        env = PushSmallRect(ind.structure.body, ind.structure.connections)
    elif env_name == "Pusher-v1":
        env = PushSmallRectOnOppositeSide(ind.structure.body, ind.structure.connections)
    elif env_name == "Thrower-v0":
        env = ThrowSmallRect(ind.structure.body, ind.structure.connections) #no
    elif env_name == "Walker-v0":   # non entra mai
        env = CatchSmallRect(ind.structure.body, ind.structure.connections)
    # ne mancano altri

    else:
        print("ERROR: invalid env name")
        env = None
    return env


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
