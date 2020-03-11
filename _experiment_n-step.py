import gym
import numpy as np
import matplotlib.pyplot as plt
import os, tqdm, time
from pathlib import Path

from environments.grid_world import grid_world
from value_based.Semi_Gradient_SARSA import SG_SARSA
from value_based.Differential_Semi_Gradient_SARSA import SG_SARSA_Differential
from policy_based.REINFORCE import REINFORCE

from tile_coding import tile_coding

params = {
    'num_of_runs': 100,
    'num_of_episodes' : 1000,
    'max_steps' : 1000,
    'alpha' : 2 ** (-13),
    'gamma' : 0.98,
    'beta' : 2 ** (-12),
    'n' : 8,
    # Creating the tilings
    'grid_size' : 5,
    'tile_size' : 4,
    'num_of_tiles' : 5
}

# Render the board on terminal or not
play_it_per = 100
play_it = False

# environment
env = grid_world(portal=True)
action_space = env.action_space.shape[0]

# tile coding
tilings = tile_coding(env.grid_size[0], params['num_of_tiles'], params['tile_size'], action_space)
state_space = tilings.num_of_tilings

# Keep stats for final print and data
episode_rewards = np.zeros(params['num_of_episodes'])

# Agent created
# agent = SG_SARSA(state_space, action_space, n, params['alpha'], params['gamma'])
agent = SG_SARSA_Differential(state_space, action_space, params['n'], params['alpha'], params['beta'])
# agent = REINFORCE(state_space, action_space, params['alpha'], params['gamma'])

for r in tqdm.tqdm(range(params['num_of_runs'])):
    np.random.seed(r+1)
    agent.reset_weights()
    for ep in range(params['num_of_episodes']):

        rewards = []
        observations = []
        actions = []

        obs = tilings.active_tiles(env.reset()) # a x d
        
        score = 0

        # while True:
        for t in range(params['max_steps']):
            action = agent.step(obs)
            observations.append(obs)

            obs, reward, done = env.step(action)
            obs = tilings.active_tiles(obs)

            rewards.append(reward)
            actions.append(action)

            score += reward

            if done:
                agent.end(observations, actions, rewards)
                break
            else:
                agent.update(observations, actions, rewards)


        episode_rewards[ep] += score
        print("EP: {} -------- Return: {}".format(ep, score), end="\r", flush=True)


## Saving the data
dir_path = os.path.dirname(os.path.realpath(__file__))
Path(dir_path + "/saves/").mkdir(parents=True, exist_ok=True)

name_addition = input('Enter the file name you want to find the rewards under:')
np.save(dir_path + "/saves/rewards_" + name_addition, episode_rewards)

plt.plot(episode_rewards/params['num_of_runs'])
plt.show()