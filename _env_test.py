'''
Test platform for the environments
'''

from environments.grid_world import grid_world

env = grid_world(portal=True)
obs = env.reset()
done = False

while(not done):
    action = int(input())
    obs, reward, done = env.step(action)

    env.print_board(erase_all=True)