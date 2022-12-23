"""
BLOCK MAZE ENVIRONMENT, 2D

A randomly generated maze, with pymunk physics. some blocks will be created, which are dynamic but cannot move between maze walls

you control the RED block, which can move freely between walls. Colliding with a dynamic block will cause it to move!

Controls:
i: move up
k: move down
j: move left
l: move right

g: grab objects within some distance of you
"""

import numpy as np

from sbrl.envs.block2d.block_env_2d import BlockEnv2D
from sbrl.utils.python_utils import get_with_default


class SimpleBlockEnv2D(BlockEnv2D):

    def _init_params_to_attrs(self, params):
        params.num_maze_cells = get_with_default(params, "num_maze_cells", 5, map_fn=int)
        params.fixed_np_maze = get_with_default(params, "fixed_np_maze", np.array([
            # bottom left ---> bottom right
            [0, 2, 0, 2, 0],
            [1, 13, 5, 13, 4],
            [1, 5, 7, 5, 4],
            [1, 6, 10, 3, 4],
            [0, 8, 8, 8, 0],
        ]))
        params.valid_start_idxs = get_with_default(params, "valid_start_idxs", np.array([
            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 0],                         [1, 4],
            [2, 0], [2, 1], [2, 2],         [2, 4],
            [3, 0],                         [3, 4],
            [4, 0], [4, 1], [4, 2], [4, 3], [4, 4],
        ]))

        super(SimpleBlockEnv2D, self)._init_params_to_attrs(params)

