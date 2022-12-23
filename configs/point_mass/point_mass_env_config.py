from argparse import ArgumentParser

import numpy as np

# declares this group's parser, and defines any sub groups we need
from sbrl.envs.point_mass_env import PointMassEnv
from sbrl.models.gpt.bet import logger
from sbrl.utils.python_utils import AttrDict as d


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument("--noise_std", type=float, default=0, help='noise of environment')
    parser.add_argument("--theta_noise_std", type=float, default=0, help='noise of target running away')
    parser.add_argument("--num_steps", type=int, default=np.inf)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--random_obs", action='store_true')
    parser.add_argument("--random_target", action='store_true')
    parser.add_argument("--stationary_target", action='store_true')

    # static obstacle stuff
    parser.add_argument("--num_obstacles", type=int, default=0)
    parser.add_argument("--random_obstacles", action='store_true',
                        help="true = rejection sampling. false = use init_obstacle_xy.")
    parser.add_argument("--init_obstacle_xy", nargs=2, type=float, default=[0.5, 0.5])
    parser.add_argument("--obstacle_radius", type=float, default=0.05)

    # parser.add_argument("--img_width", type=int, default=256)
    # parser.add_argument("--img_height", type=int, default=256)
    # parser.add_argument("--img_channels", type=int, default=3)
    return parser


def process_params(group_name, common_params):
    assert "env" in group_name, f"Group name must have \'env\': {group_name}"
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    prms = common_params >> group_name

    init_obs, init_targ = None, None
    if not prms >> "random_obs":
        init_obs = np.array([0.1, 0.1])
    if not prms >> "random_target":
        init_targ = np.array([0.9, 0.9])

    target_vel = 0.0125
    ego_vel = 0.025
    if prms >> "stationary_target":
        target_vel = 0

    # obstacle stuff
    N = prms >> "num_obstacles"
    obs_prms = d(num_obstacles=N)
    if N > 0:
        logger.debug(f"Using {N} obstacles")
        init_obstacle = None
        if not prms >> "random_obstacles":
            xy = prms >> "init_obstacle_xy"
            init_obstacle = np.array([list(xy) for _ in range(N)])  # default in the middle

        obs_prms['obstacle_radii'] = [prms >> "obstacle_radius" for _ in range(N)]
        obs_prms['initial_obstacle_positions'] = init_obstacle

    env_params = d(
        cls=PointMassEnv,
        params=d(
            noise_std=prms >> "noise_std",
            theta_noise_std=prms >> "theta_noise_std",
            num_steps=prms >> "num_steps",
            render=prms >> "render",
            initial_obs=init_obs,
            initial_target=init_targ,
            target_speed=target_vel,
            ego_speed=ego_vel,
        ) & obs_prms
    )

    common_params[group_name] = common_params[group_name] & env_params
    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
