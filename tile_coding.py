import numpy as np

class tile_coding:
    def __init__(self, grid_size, num_of_tiles, tile_size, action_space):
        self.num_of_tiles = num_of_tiles
        self.grid_size = grid_size
        self.tile_size = tile_size
        
        self.action_space = action_space

        self.tiles = [[[] for a in range(grid_size)] for b in range(grid_size)]
        self.num_of_tilings = 0

        self._create_tiles()

    def _get_the_tile(self, x, y):
        for i in range(x, x+self.tile_size):
            for j in range(y, y+self.tile_size):
                if i < self.grid_size and j < self.grid_size and i >= 0 and j >= 0:
                    self.tiles[i][j].append(self.num_of_tilings)
        

    def _tile_coding(self, start_point):
        for i in range(2):
            if start_point[i] % self.tile_size == 0:
                start_point[i] = 0
            else:
                start_point[i] = start_point[i] % self.tile_size - self.tile_size
        
        for x in range(start_point[0], self.grid_size, self.tile_size):
            for y in range(start_point[1], self.grid_size, self.tile_size):
                self._get_the_tile(x, y)
                self.num_of_tilings += 1

    def _create_tiles(self):
        '''
        Creates the tiles; needs to be called to create the tiles
        so that we can use one_hot_tiling.

        CHANGE: Called w constructor
        '''
        for i in range(self.num_of_tiles):
            start_pos = [i, i]
            self._tile_coding(start_pos)

    def active_tiles(self, s):
        '''
        Main functionality, call it after creating the tiles
        to get the one hot tiling of s
        s: [x, y]
        '''
        one_hot = np.zeros(self.num_of_tilings)
        one_hot[self.tiles[s[0]][s[1]]] = 1 # activate the tile for the action and state
        
        return one_hot
