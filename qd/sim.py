#!/usr/bin/env python3

import os
import numpy as np
import warnings

from evogym.envs.walk import WalkingFlat, SoftBridge, Duck
from evogym.envs.climb import Climb0, Climb1, Climb2
from evogym.envs.flip import Flipping
from evogym.envs.jump import StationaryJump
from evogym.envs.balance import Balance, BalanceJump
from evogym.envs.traverse import StepsUp, StepsDown, WalkingBumpy, WalkingBumpy2, VerticalBarrier, FloatingPlatform, Gaps, BlockSoup
from evogym.envs.manipulate import CarrySmallRect, CarrySmallRectToTable, PushSmallRect, PushSmallRectOnOppositeSide, ThrowSmallRect, CatchSmallRect, ToppleBeam, SlideBeam, LiftSmallRect
from utils.algo_utils import TerminationCondition
import utils.mp_group as mp

from qd.features import *

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
        group.add_job(run_ppo, ppo_args, callback=ind.set_fitness)

    group.run_jobs(num_cores)
    return inds


def make_env(env_name, seed=-1, ind=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
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
    if env_name == "Walker-v0":                 # easy
        env = WalkingFlat(ind.structure.body, ind.structure.connections)        # build env with 'set()' method
    elif env_name == "BridgeWalker-v0":         # easy
        env = SoftBridge(ind.structure.body, ind.structure.connections)
    elif env_name == "CaveCrawler-v0":          # medium
        env = Duck(ind.structure.body, ind.structure.connections)
    
    # climb
    elif env_name == "Climber-v0":              # medium
        env = Climb0(ind.structure.body, ind.structure.connections)
    elif env_name == "Climber-v1":              # medium
        env = Climb1(ind.structure.body, ind.structure.connections)
    elif env_name == "Climber-v2":              # hard
        env = Climb2(ind.structure.body, ind.structure.connections)
    
    # flip
    elif env_name == "Flipper-v0":              # easy
        env = Flipping(ind.structure.body, ind.structure.connections)
    
    # jump
    elif env_name == "Jumper-v0":               # easy
        env = StationaryJump(ind.structure.body, ind.structure.connections)

    # balance
    elif env_name == "Balancer-v0":             # easy
        env = Balance(ind.structure.body, ind.structure.connections)
    elif env_name == "Balancer-v1":             # medium
        env = BalanceJump(ind.structure.body, ind.structure.connections)

    # traverse
    elif env_name == "UpStepper-v0":            # medium
        env = StepsUp(ind.structure.body, ind.structure.connections)
    elif env_name == "DownStepper-v0":          # easy
        env = StepsDown(ind.structure.body, ind.structure.connections)
    elif env_name == "ObstacleTraverser-v0":    # medium
        env = WalkingBumpy(ind.structure.body, ind.structure.connections)
    elif env_name == "ObstacleTraverser-v1":    # hard
        env = WalkingBumpy2(ind.structure.body, ind.structure.connections)
    elif env_name == "Hurdler-v0":              # hard
        env = VerticalBarrier(ind.structure.body, ind.structure.connections)
    elif env_name == "PlatformJumper-v0":       # hard
        env = FloatingPlatform(ind.structure.body, ind.structure.connections)
    elif env_name == "GapJumper-v0":            # hard
        env = Gaps(ind.structure.body, ind.structure.connections)
    elif env_name == "Traverser-v0":            # hard
        env = BlockSoup(ind.structure.body, ind.structure.connections)
    
    # manipulate
    elif env_name == "Carrier-v0":              # easy
        env = CarrySmallRect(ind.structure.body, ind.structure.connections)
    elif env_name == "Carrier-v1":              # hard
        env = CarrySmallRectToTable(ind.structure.body, ind.structure.connections)
    elif env_name == "Pusher-v0":               # easy
        env = PushSmallRect(ind.structure.body, ind.structure.connections)
    elif env_name == "Pusher-v1":               # medium
        env = PushSmallRectOnOppositeSide(ind.structure.body, ind.structure.connections)
    elif env_name == "Thrower-v0":              # medium
        env = ThrowSmallRect(ind.structure.body, ind.structure.connections)
    elif env_name == "BeamToppler-v0":          # easy
        env = ToppleBeam(ind.structure.body, ind.structure.connections)
    elif env_name == "BeamSlider-v0":           # hard
        env = SlideBeam(ind.structure.body, ind.structure.connections)
    elif env_name == "Lifter-v0":               # hard
        env = LiftSmallRect(ind.structure.body, ind.structure.connections)
    elif env_name == "Catcher-v0":              # hard
        env = CatchSmallRect(ind.structure.body, ind.structure.connections)

    else:
        print("ERROR: invalid env name")
        env = None
    return env

