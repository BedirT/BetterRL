import gym
import numpy as np
import matplotlib.pyplot as plt
import os, tqdm, time

from environments.grid_world import grid_world
from value_based.Semi_Gradient_SARSA import SG_SARSA_MonteCarlo
from policy_based.REINFORCE import REINFORCE

from tile_coding import tile_coding

params = {
    'num_of_episodes' : 1000,
    'max_steps' : 1000,
    'alpha' : 2 ** (-13),
    'gamma' : 0.98,
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
# agent = REINFORCE(state_space, action_space, params['alpha'], params['gamma'])
agent = SG_SARSA_MonteCarlo(state_space, action_space, params['alpha'], params['gamma'])

np.random.seed(1)

for ep in range(params['num_of_episodes']):

    obs = tilings.active_tiles(env.reset()) # a x d

    grads = []
    rewards = []
    observations = []
    actions = []

    score = 0

    # while True:
    for _ in range(params['max_steps']):
        action = agent.step(obs)
        next_obs, reward, done = env.step(action)
        next_obs = tilings.active_tiles(next_obs)
        
        rewards.append(reward)
        observations.append(obs)
        actions.append(action)
        
        score += reward
        obs = next_obs

        if play_it and ep % play_it_per == 0:
            env.print_board(erase_all=True)

        if done:
            break

    agent.update(observations, actions, rewards)
    episode_rewards[ep] = score
    print("EP: {} -------- Return: {}      ".format(ep, score), end="\r", flush=True)

plt.plot(episode_rewards)
plt.show()