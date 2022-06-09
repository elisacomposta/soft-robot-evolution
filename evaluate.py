# Inspired by https://github.com/EvolutionGym/evogym/tree/main/examples (visualize.py)

import os, sys
root_dir = os.path.dirname(os.path.abspath(__file__))
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'PyTorch-NEAT'))
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

import torch
import gym

from utils.algo_utils import get_ind_path, pretty_print, get_stored_structure
from ppo.utils import get_vec_normalize
from ppo.envs import make_vec_envs


def evaluate(env_name, structure, path_controller, num_iters=500):
    print('\nEvaluating structure:\n', structure[0], '\n')

    path_controller = os.path.join(root_dir, 'results', path_controller)
    
    env = make_vec_envs(env_name, structure, 1000, 1, None, None, device='cpu', allow_early_resets=False)

    # We need to use the same statistics for normalization as used in training
    try:
        save_path_controller = path_controller
        actor_critic, obs_rms = torch.load(save_path_controller, map_location='cpu')
    except:
        print(f'\nCould not load robot controller data at {save_path_controller}.\n')
        return

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    obs = env.reset()

    total_steps = 0
    reward_sum = 0
    while total_steps < num_iters:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=True)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)
        masks.fill_(0.0 if (done) else 1.0)
        reward_sum += reward
        
        if done == True:
            env.reset()
            reward_sum = float(reward_sum.numpy().flatten()[0])
            env.venv.close()
            return reward_sum

        total_steps += 1
