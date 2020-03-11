'''
. . . D1| . . .P1
. . . . | . . . .
. . . . | . . . .
G . . . | . . . .
________    _____
S . . . | . . . .
. . . . | . . . .
. . . . | . . . .
D2. . .   . . .P2

It's a gridworld environment with stochasticity implemented.
S is the starting point, and agent can move in cardinal directions,
the goal is to reach the point G. There is no direct way to get into
the room of G, so the agent needs to use one of the portals to teleport
inside the room. Points P are the portal locations and D's are the 
destinations. Stochasticity is here, that everytime when a portal is used
the destinations of the portals are renewed. One portal is matched with
the destination in the same room as G and the other is the one in the room
of S. The probabilities are almost random (or will be random depending on
the results), starting with %60 D1 for the P1 and %40 D1 for the P2.
(Trying to see the effect of the distance)

- Edits are made after this version, the main story is the same, but there are
couple other features. Plus you don't have to use the portals obviously
'''
import numpy as np
import os

class grid_world:

    def __init__(self, portal=False, portal_prob=.6, random_wind=False, wind_chance=.2):
        
        # if there is a random wind in the environment or not
        # basically agent will move randomly by wind_chance if there is a wind.
        # so if wind_chance is .3 then 30% of the time agents action is random.
        # - adds more stochasticity
        self.windy = random_wind
        if self.windy:
            self.wind_chance = wind_chance

        # if the portals are activated, the chance of P1 sending to D1
        # will be portal_prob. 
        self.portal = portal
        if self.portal:
            self.portal_prob = portal_prob
        
        # Everything else is initialized here
        self.start()

    def reset(self):
        '''
        Function to call for initial agent position.
        '''
        self.start() # resets the environment to initial position
        return self.agent_pos

    def start(self):
        '''
        The environment is created here, to have a custom environment just use
        the determined values for each item. The grid must be a 2D list. There are
        several items you can have use of:
            - P1, P2 - Portals, as explained before, to have more portals you need to modify
            the step function as well as constructor for probs.
            - D1, D2 - Destionation points, as portals you need to modify a bit for having more
            destination points.
            - '|', '-' - walls, you can use either one for an obstacle.
            - G - Goal point, ends the episode when reached.
            - A - Action starting point.
        '''
        self.the_world = [
            ['.', '.', '.', 'D1', '|', '.', '.', '.', 'P1'],
            ['.', '.', '.', '.' , '|', '.', '.', '.', '.'],
            ['.', '.', '.', '.' , '|', '.', '.', '.', '.'],
            ['G', '.', '.', '.' , '|', '.', '.', '.', '.'],
            ['-', '-', '-', '-' , '|', '.', '-', '-', '-'],
            ['A', '.', '.', '.' , '|', '.', '.', '.', '.'],
            ['.', '.', '.', '.' , '|', '.', '.', '.', '.'],
            ['.', '.', '.', '.' , '|', '.', '.', '.', '.'],
            ['D2','.', '.', '.' , '.', '.', '.', '.', 'P2']
        ]
        self.the_world = [
            ['.',  'D1','|', '.', 'P1'],
            ['G',  '.', '|', '.', '.'],
            ['-',  '-', '|', '.', '-'],
            ['A',  '.', '|', '.', '.'],
            ['D2', '.', '.', '.', 'P2']
        ]

        # Parsing the grid and locating the elements and getting the locations
        # - We are using locations instead of actual grid to make operations faster
        for row in range(len(self.the_world)):
            for col in range(len(self.the_world[row])):
                if self.the_world[row][col] == 'A':
                    self.agent_pos = [row, col]
                elif self.the_world[row][col] == 'G':
                    self.goal_pos = [row, col]
                if self.portal:
                    if self.the_world[row][col] == 'P1':
                        self.p1_location = [row, col]
                    elif self.the_world[row][col] == 'P2':
                        self.p2_location = [row, col]     
                    elif self.the_world[row][col] == 'D1':
                        self.d1_location = [row, col]
                    elif self.the_world[row][col] == 'D2':
                        self.d2_location = [row, col]

        ## Defining the rewards for type of consequences
        self.reward_b = 0 # reward when hitting the walls or boundries 
        self.reward_t = -1 # reward per time step
        self.reward_g = 10 # reward if the agent gets to the goal
 
        # Deciding on the portals if the portals are activated
        if self.portal:
            self._choose_portals()
        
        # size of the grid 
        self.grid_size = [len(self.the_world),len(self.the_world[0])]

        # Initializing the action possibilities, for example for chain env.
        # we would change the actions to 0, 1 instead of 0-3.
        self.action_space = np.array([0, 1, 2, 3])

    def step(self, action):
        '''
        Taking a step in the environment, towards the action given.
        - action: self.action_space is the possible actions otherwise there will be no
        change in the environment

        -> Implemented version has 4 actions;
            - 0 -> West, going left on the grid
            - 1 -> East, going right on the grid
            - 2 -> South, going down
            - 3 -> North, going up
        
        -> If the agent hits a wall, or out of boundries the location stands still
        
        -> If there is a wind then agent moves randomly for a chance

        -> If agent comes to a portal location is changed to the assigned destination
        '''

        # initializing the reward for the step
        reward = 0

        # If there is a wind, the agent will move randomly with the wind_chance
        if self.windy and np.random.rand() < self.wind_chance:
            action = np.random.choice(self.action_space)

        # Move according to the action
        if action == self.action_space[0]: # w: 0, col-1
            reward = self._move([self.agent_pos[0], self.agent_pos[1] - 1])
        elif action == self.action_space[1]: # e: 1, x+1
            reward = self._move([self.agent_pos[0], self.agent_pos[1] + 1])
        elif action == self.action_space[2]: # s: 2, y+1
            reward = self._move([self.agent_pos[0] + 1 , self.agent_pos[1]])
        elif action == self.action_space[3]: # n: 3, y-1
            reward = self._move([self.agent_pos[0] - 1, self.agent_pos[1]])
            
        # reached the goal            
        if self.agent_pos == self.goal_pos: 
            return self.agent_pos, reward + self.reward_g, True

        # If there are portals
        if self.portal: 
            if self.agent_pos == self.p1_location: # portal 1 
                self._update_agents_pos(self.p1_des)
                self.the_world[self.p1_location[0]][self.p1_location[1]] = 'P1'  # just for visuals
                # After using the portals updating the portal locations    
                self._choose_portals()
                 
            if self.agent_pos == self.p2_location: # portal 2
                self._update_agents_pos(self.p2_des)
                self.the_world[self.p2_location[0]][self.p2_location[1]] = 'P2'  # just for visuals
                self._choose_portals()
        
        # returns obs, reward, done
        return self.agent_pos, reward, False 

    ##### HELPER FUNCTIONS ######
    def _move(self, new_pos):
        '''
        Checks if the given position is a wall or out of boundries and updates the
        agents location, returns the reward.
        -> If hit the wall reward_t + reward_b otherwise only reward_t
        '''
        # Agent hits the wall or boundry
        if  new_pos[0] >= self.grid_size[0] or new_pos[1] >= self.grid_size[1] or \
            new_pos[0] < 0 or new_pos[1] < 0 or \
            self.the_world[new_pos[0]][new_pos[1]] in ['-', '|']:
            return self.reward_b + self.reward_t
        
        # Valid move
        self._update_agents_pos(new_pos)
        return self.reward_t

    def _choose_portals(self):
        '''
        Assigning a destination to portals, using portal_prob.
        '''
        if np.random.rand() < self.portal_prob: 
            self.p1_des, self.p2_des = self.d1_location, self.d2_location
        else:
            self.p1_des, self.p2_des = self.d2_location, self.d1_location
        
    def _update_agents_pos(self, new_pos):
        '''
        Updating the grid as well as the agents location
        '''
        self.the_world[self.agent_pos[0]][self.agent_pos[1]] = '.'
        self.the_world[new_pos[0]][new_pos[1]] = 'A'
        self.agent_pos = new_pos

    def print_board(self, erase_all=False):
        '''
        Printing the boar.
        If erase_all is active then output before the print will be erased;
        The board will look dynamic, though you should avoid if you need the rest
        of the output.
        '''
        if erase_all:
            os.system('cls' if os.name == 'nt' else 'clear')
        for i in range(len(self.the_world)):
            for j in range(len(self.the_world[i])):
                if [i, j] in [self.d1_location, self.d2_location]:
                    if self.the_world[i][j] == 'A':
                        print('A', end='  ')    
                    else:
                        print('D', end='  ')    
                else:
                    print(self.the_world[i][j], end='  ')
            print()   

    