from argparse import ArgumentParser

import numpy as np

from sbrl.envs.block2d.maze_navigation_envs import BottleneckNavigationBlockEnv3D
from sbrl.envs.param_spec import ParamEnvSpec
# declares this group's parser, and defines any sub groups we need
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument("--grid_size", type=float, nargs=2, default=[600, 600])
    parser.add_argument("--block_size", type=float, nargs=2, default=[20, 20])
    parser.add_argument("--block_lower", type=float, nargs=2, default=[20, 20])
    parser.add_argument("--block_upper", type=float, nargs=2, default=[40, 40])
    parser.add_argument("--block_mass", type=float, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--stochastic_dt", action='store_true')
    parser.add_argument("--action_noise_theta", type=float, default=10)
    parser.add_argument("--action_noise_prob", type=float, default=0)
    parser.add_argument("--max_vel", type=float, default=np.inf)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--imgs", action='store_true')
    parser.add_argument("--extra", action='store_true')
    parser.add_argument("--no_block_sizes", action='store_true')
    parser.add_argument("--no_block_contact", action='store_true')
    parser.add_argument("--done_on_success", action='store_true')

    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--num_maze_cells", type=int, default=8)
    parser.add_argument("--num_blocks", type=int, default=0)

    parser.add_argument("--randomize_block_sizes", action='store_true')
    parser.add_argument("--randomize_maze", action='store_true')
    parser.add_argument("--skip_half_dir", type=int, default=None, choices=[0, 1], help="0 for left, 1 for right")
    parser.add_argument("--skip_fixed_bn", action='store_true')
    # parser.add_argument("--start_near_bottom", action='store_true')
    # parser.add_argument("--maximize_agent_block_distance", action='store_true')
    # parser.add_argument("--num_envs", type=int, default=1)
    # parser.add_argument("--horizon", type=float, default=np.float("inf"))
    return parser


def process_params(group_name, common_params):
    assert "env" in group_name, f"Group name must have \'env\': {group_name}"
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()

    NUM_BLOCKS, DT, STOCH_DT, GRID_SIZE, IMG_HEIGHT, IMG_WIDTH, BLOCK_SIZE, BLOCK_MASS, NUM_MAZE_CELLS, RENDER, \
    EXTRA, IMGS, NO_BLOCK_SIZES, NO_BLOCK_CONTACT, RAND_MAZE, done_on_success, SKIPF, SKIPDIR, TH_NOISE, TH_PROB = \
        (common_params >> group_name).get_keys_required(
            ['num_blocks', 'dt', 'stochastic_dt', 'grid_size', 'img_height', 'img_width', 'block_size', 'block_mass', 'num_maze_cells',
             'render', 'extra', 'imgs', 'no_block_sizes', 'no_block_contact', 'randomize_maze', 'done_on_success', 'skip_fixed_bn', 'skip_half_dir',
             'action_noise_theta', 'action_noise_prob']
        )
    REALTIME = RENDER

    # BLK_PER_AX = np.array(GRID_SIZE) // np.array(BLOCK_SIZE)

    bn_indices = np.array([1, 3, 2]) * NUM_MAZE_CELLS // 4

    # 10% noise, underlying dt will go from [0.9 -> 1.1] * dt
    dt_scale = 0.1 if STOCH_DT else 0
    skip_fn = None

    assert SKIPDIR is None or not SKIPF, "Only skip-fixed or skip-dir, not both."

    if SKIPDIR or SKIPF:
        assert RAND_MAZE, "maze must be random to enable skipping certain bn configs..."

    if SKIPF:
        skip_bn_indices_ls = [bn_indices.copy()]  # avoid this bn configuration
        bn_indices = np.array([0, 0, 0])  # not conflicting
        skip_fn = lambda idxs: any(all(idxs == s) for s in skip_bn_indices_ls)
    elif SKIPDIR is not None:
        if SKIPDIR == 0:
            skip_fn = lambda idxs: any(idxs >= NUM_MAZE_CELLS/2)  # only left half
        else:
            skip_fn = lambda idxs: any(idxs < NUM_MAZE_CELLS/2)  # only right half
        bn_indices = np.array([0, 0, 0])  # not conflicting

    env_params = AttrDict(
        cls=BottleneckNavigationBlockEnv3D,
        params=AttrDict(
            num_blocks=NUM_BLOCKS,
            dt=DT,
            dt_scale=dt_scale,
            action_noise_theta=np.deg2rad(TH_NOISE),
            action_noise_prob=TH_PROB,
            grid_size=GRID_SIZE,
            image_size=np.array([IMG_HEIGHT, IMG_WIDTH]),
            block_size=BLOCK_SIZE,
            block_mass=BLOCK_MASS,
            num_maze_cells=NUM_MAZE_CELLS,
            render=RENDER,
            keep_in_bounds=True,
            disable_images=not IMGS and not RENDER,
            realtime=REALTIME,
            grab_action_binary=True,
            grab_action_max_force=10000,
            # specific ones to bottleneck
            # bottleneck_indices=missing_idxs,
            # horizontal=horizontal,
            ego_block_size=40,
            initialization_steps=5,
            do_wall_collisions=True,
            valid_start_idxs=np.array([[0, NUM_MAZE_CELLS - 1]]),
            valid_goal_idxs=np.array([[NUM_MAZE_CELLS - 1, 0]]),
            randomize_maze=RAND_MAZE,
            done_on_success=done_on_success,
            bottleneck_indices=bn_indices,
            skip_bottleneck_fn=skip_fn,
        ),
    )
    block_max_size = BLOCK_SIZE
    if common_params >> f"{group_name}/randomize_block_sizes":
        env_params.params.block_size_lower = common_params >> f"{group_name}/block_lower"
        env_params.params.block_size_upper = common_params >> f"{group_name}/block_upper"
        block_max_size = common_params >> f"{group_name}/block_upper"
    num_blocks = (np.asarray(GRID_SIZE) / np.asarray(block_max_size)).astype(int)

    # if common_params >> f"{group_name}/start_near_bottom":
    #     vs_default = np.meshgrid(range(num_blocks[0]), range(num_blocks[1] // 2))
    #     vs_default = list(zip(*[vs.reshape(-1) for vs in vs_default]))
    #     env_params.params.valid_start_idxs = vs_default

    # if common_params >> f"{group_name}/maximize_agent_block_distance":
    #     env_params.params.maximize_agent_block_distance = True

    common_params[group_name] = common_params[group_name] & env_params

    ## spec changes
    assert common_params >> "env_spec/cls" == ParamEnvSpec, "Only supports param env spec, processed first"
    new_obs_names = []
    if EXTRA:
        (common_params >> "env_spec/params/param_names").append('block_colors')
        extra_obs_names = ['angle', 'angular_velocity', 'force', 'torque',
                           'block_angles', 'block_angular_velocities', 'block_forces', 'block_torques']
        (common_params >> "env_spec/params/observation_names").extend(extra_obs_names)
        new_obs_names.extend(extra_obs_names)
    if IMGS:
        (common_params >> "env_spec/params/observation_names").append('image')
        new_obs_names.append('image')

    if not NO_BLOCK_CONTACT:
        (common_params >> "env_spec/params/observation_names").append('block_contact')
        (common_params >> "env_spec/params/observation_names").append('block_contact_points')
        (common_params >> "env_spec/params/observation_names").append('block_contact_normal')
        new_obs_names.extend(['block_contact', 'block_contact_points', 'block_contact_normal'])

    if not NO_BLOCK_SIZES:
        (common_params >> "env_spec/params/param_names").append('block_sizes')

    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
