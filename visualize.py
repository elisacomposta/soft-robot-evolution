# Adapted from https://github.com/EvolutionGym/evogym/tree/main/examples

import os, sys
root_dir = os.path.dirname(os.path.abspath(__file__))
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'PyTorch-NEAT'))
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

import json
import argparse
import sys
import numpy as np
import torch
import gym

from utils.algo_utils import get_ind_path, pretty_print, get_stored_structure
from ppo.envs import make_vec_envs
from ppo.utils import get_vec_normalize

def visualize_codesign(args, exp_name):
    global EXPERIMENT_PARENT_DIR
    exp_path = os.path.join(EXPERIMENT_PARENT_DIR, exp_name)

    if args.env_name != None:
        env_name = args.env_name
    else:
        ## READ ENVIRONMENT NAME FROM METADATA
        f_path = os.path.join(exp_path, "metadata.txt")
        f = open(f_path, "r")
        line = f.readline().rstrip().split(": ")
        while line[0]!= "ENVIRONMENT":
            line = f.readline().rstrip().split(": ")
        env_name = line[1]
        f.close()

    print()
    print("EXPERIMENT ENVIRONMENT:", env_name)

    while(True):
        print()
        print("Enter ind number: ", end="")
        ind_number = int(input())

        print("Enter num iters: ", end="")
        num_iters = int(input())

        try:
            save_path_structure = get_ind_path(ind_number, exp_path)
            structure = get_stored_structure(save_path_structure)
            print(f'\nRobot body:\n{structure[0]}\n')
        except:
            print(f'\nCould not load robot strucure data at {save_path_structure}/structure.npz.\n')
            continue

        if num_iters == 0:
            continue

        env = make_vec_envs(
            env_name,
            structure,
            1000,
            1,
            None,
            None,
            device='cpu',
            allow_early_resets=False)

        # We need to use the same statistics for normalization as used in training
        try:
            save_path_controller = os.path.join(get_ind_path(ind_number, exp_path), "controller.pt")
            actor_critic, obs_rms = \
                        torch.load(save_path_controller,
                                    map_location='cpu')
        except:
            print(f'\nCould not load robot controller data at {save_path_controller}.\n')
            continue

        vec_norm = get_vec_normalize(env)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.obs_rms = obs_rms

        recurrent_hidden_states = torch.zeros(1,
                                            actor_critic.recurrent_hidden_state_size)
        masks = torch.zeros(1, 1)

        obs = env.reset()
        env.render('screen')

        total_steps = 0
        while total_steps < num_iters:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=args.det)

            # Obser reward and next obs
            obs, reward, done, _ = env.step(action)
            masks.fill_(0.0 if (done) else 1.0)
            
            if done == True:
                env.reset()

            env.render('screen')

            total_steps += 1
        
        env.venv.close()


EXPERIMENT_PARENT_DIR = os.path.join(root_dir, 'results')
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--env-name',
        help='environment to train on')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    args = parser.parse_args()
    args.det = not args.non_det

    exp_list = os.listdir(EXPERIMENT_PARENT_DIR)
    pretty_print(exp_list)

    print("\nEnter experiment name: ", end="")
    exp_name = input()
    while exp_name not in exp_list:
        print("Invalid name. Try again:")
        exp_name = input()

    visualize_codesign(args, exp_name)