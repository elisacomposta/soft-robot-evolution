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
#import utils.mp_group as mp
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

def simulate(env_name, inds, experiment_name, num_episode=5):

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

    num_cores = 12
    group.run_jobs(num_cores)

    for ind in inds:
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
    emptyRate = compute_empty_rate(ind.structure)
    compactness = compute_compactness(ind.structure)
    #elongation = compute_elongation(ind.structure, 2)

    scores = {
            "reward": ind.structure.fitness,
            "length": length,     
            "height": height,
            "baseLength": baseLength,
            "emptyRate": emptyRate,
            "compactness": compactness,
            #"elongation": elongation    #define domains
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
    start = -1
    end = -1
    for i in range (len((structure.body[0]))):
        for j in range (len((structure.body))):
            if structure.body[j][i] > 0:
                start = i
                break
        if(start >= 0):
            break
    for i in range (len((structure.body))-1, 0, -1):
        for j in range (len((structure.body[i]))):
            if structure.body[i][j] > 0:
                end = i
                break
        if(end >= 0):
            break
    length = end - start + 1
    return length


def compute_height(structure):
    start = -1
    end = -1
    for i in range (len((structure.body[0]))):
        for j in range (len((structure.body))):
            if structure.body[i][j] > 0:
                start = i
                break
        if(start >= 0):
            break
    
    for i in range (len((structure.body[0]))-1, 0, -1):
        for j in range (len((structure.body))):
            if structure.body[i][j] > 0:
                end = i
                break
        if(end >= 0):
            break
    height = end - start + 1
    return height


def compute_base_length(structure):
    baseLength = len(np.nonzero(structure.body[len(structure.body)-1])[0])
    return baseLength


def compute_empty_rate(structure):
    totEmpty=0
    for i in structure.body:
        for j in i:
            totEmpty += 1 if j==0 else 0
    emptyRate = totEmpty/(len(structure.body) * len(structure.body[0]) ) * 100
    return emptyRate


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

    # -> 0.0 for less compact shapes, -> 1.0 for more compact shapes
    return nVoxels / nConvexHull


def compute_elongation(structure, n_dir):
    
    print("\ncomputing elongation in", n_dir, " directions")
    

    """if (posture.values().stream().noneMatch(e -> e)) {
      throw new IllegalArgumentException("Grid is empty");
    } else if (n <= 0) {
      throw new IllegalArgumentException(String.format("Non-positive number of directions provided: %d", n));
    }"""

    """List<Grid.Key> coordinates = posture.stream()
        .filter(Grid.Entry::value)
        .map(Grid.Entry::key)"""

    coordinates = structure.body #?
    shape = structure.shape

    from scipy.ndimage.interpolation import rotate
    #List<Double> diameters = new ArrayList<>();
    diameters = []
    for i in range(n_dir):
        theta = (2 * i * math.pi) / n_dir
        """rotatedCoordinates = coordinates.stream() #list
            .map(p -> new Grid.Key(
                (int) Math.round(p.x() * Math.cos(theta) - p.y() * Math.sin(theta)),
                (int) Math.round(p.x() * Math.sin(theta) + p.y() * Math.cos(theta))
            ))
            .toList()

        #rotated = rotate(structure.body, angle=45, reshape=False)

        #print("body:\n", structure.body)
        #print("\nRotated structure ", i, ":")
        #print(rotated)



        #rotatedCoordinates = deepcopy(structure.body)
        rotatedCoordinates = np.zeros((shape[0], shape[1]))
        for x in range(shape[0]):
            for y in range(shape[1]):
                rotX = round( x * math.cos(theta) - y * math.sin(theta) ) + shape[0]-1
                rotY = round( x * math.sin(theta) + y * math.cos(theta) ) + shape[1]-1
                print(f"mapping {x},{y} to {rotX},{rotY}")
                voxel =[rotX, rotY]
                print("voxel: ", voxel)
                #structure.body[voxel[0]][voxel[1]]
            rotatedCoordinates[x][y] = rotatedCoordinates[rotX][rotY]
            
                
        minX = rotatedCoordinates.stream().min(Comparator.comparingInt(Grid.Key::x)).get().x()
        maxX = rotatedCoordinates.stream().max(Comparator.comparingDouble(Grid.Key::x)).get().x()
        minY = rotatedCoordinates.stream().min(Comparator.comparingDouble(Grid.Key::y)).get().y()
        maxY = rotatedCoordinates.stream().max(Comparator.comparingDouble(Grid.Key::y)).get().y()
        sideX = maxX - minX + 1
        sideY = maxY - minY + 1
        diameters.add(min(sideX, sideY) / max(sideX, sideY))
    
    #ret = 1.0 - min(diameters)
    return ret"""
