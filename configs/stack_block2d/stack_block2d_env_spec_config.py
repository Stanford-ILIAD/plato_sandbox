from argparse import ArgumentParser

import numpy as np

# declares this group's parser, and defines any sub groups we need
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument("--act_low", type=float, nargs=3, default=[-130, -130, 0])
    parser.add_argument("--act_high", type=float, nargs=3, default=[130, 130, 1000.])
    parser.add_argument("--include_policy_name", action='store_true')
    parser.add_argument("--include_policy_switch", action='store_true')
    parser.add_argument("--include_goal_states", action='store_true')
    parser.add_argument("--goal_state_names", nargs='+', default=['block_positions', 'block_velocities', 'block_angles', 'block_angular_velocities', 'block_sizes'])
    parser.add_argument('--use_prefix', action='store_true')
    return parser


# strictly ordered processing order
def process_params(group_name, common_params):
    assert group_name == "env_spec"
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()

    # access to all the params for the current experiment here.
    env_prms = common_params >> "env_train"
    prms = common_params >> group_name
    NUM_BLOCKS = env_prms >> "num_blocks"
    NUM_MAZE_CELLS = env_prms >> "num_maze_cells"
    ACT_LOW = common_params >> "env_spec/act_low"
    ACT_HIGH = common_params >> "env_spec/act_high"
    INCLUDE_GOAL_STATES = common_params >> "env_spec/include_goal_states"
    GOAL_STATE_NAMES = common_params >> "env_spec/goal_state_names" if INCLUDE_GOAL_STATES else []
    GRID_SIZE = env_prms >> "grid_size"

    base_nsld = [
        # param
        ("image", (env_prms >> "img_height", env_prms >> "img_width", env_prms >> "img_channels"), (0, 255), np.uint8),
        ("position", (2,), (-np.inf, np.inf), np.float32),
        ("velocity", (2,), (-np.inf, np.inf), np.float32),
        ("bounding_box", (4,), (-np.inf, np.inf), np.float32),
        ("angle", (1,), (-np.inf, np.inf), np.float32),
        ("angular_velocity", (1,), (-np.inf, np.inf), np.float32),
        ("force", (2,), (-np.inf, np.inf), np.float32),
        ("torque", (1,), (-np.inf, np.inf), np.float32),
        ("block_positions", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
        ("block_bounding_boxes", (NUM_BLOCKS, 4), (-np.inf, np.inf), np.float32),
        ("block_velocities", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
        ("block_angles", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
        ("block_angular_velocities", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
        ("block_forces", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
        ("block_torques", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
        ("active_constraints", (NUM_BLOCKS,), (0, np.inf), np.float32),
        ("grab_binary", (NUM_BLOCKS,), (0, 1.), np.float32),
        ("grab_binary_single", (1,), (0, 1.), np.float32),
        ("grab_force", (NUM_BLOCKS,), (0, np.inf), np.float32),
        ("grab_distance", (NUM_BLOCKS,), (0, np.inf), np.float32),
        ("grab_vector", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),

        ("block_sizes", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
        ("block_masses", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
        ("block_colors", (NUM_BLOCKS, 4), (-np.inf, np.inf), np.float32),
        ("block_contact", (NUM_BLOCKS,), (False, True), np.bool),
        ("block_contact_points", (NUM_BLOCKS, 4), (-np.inf, np.inf), np.float32),
        ("block_contact_normal", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
        ("maze", (NUM_MAZE_CELLS, NUM_MAZE_CELLS), (0, 32), np.uint8),

        # raw action
        ("action", (3,), (ACT_LOW, ACT_HIGH), np.float32),
        ("policy_type", (1,), (0, 255), np.uint8),
        ("policy_name", (1,), (0, 1), np.object),
        ("policy_switch", (1,), (False, True), np.bool),  # marks the beginning of a policy

        # positional action
        ("target/position", (2,), (np.array([0.0, 0.0]), np.array(GRID_SIZE)), np.float32),
        ("target/grab_binary", (NUM_BLOCKS,), (0, 1.), np.float32),
        ("target/grab_binary_single", (1,), (0, 1.), np.float32),
        # object positional action (goal state)
        ("target/block_positions", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
        ("target/block_velocities", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),

    ]
    # we record these names in dataset  TOD0 'image'
    obs_names = [
        # 'image',
        'position', 'velocity', 'block_positions', 'block_velocities', 'block_angles', 'block_angular_velocities',
        'active_constraints', 'grab_binary', 'grab_force', 'grab_distance', 'grab_vector', 'block_bounding_boxes',
        'bounding_box',
    ]
    # if DEBUG:
    #     obs_names.extend(['angle', 'angular_velocity', 'force', 'torque',
    #                       'block_angles', 'block_angular_velocities', 'block_forces', 'block_torques'])
    # if args.add_imgs:
    #     obs_names.append('image')

    common_params["use_prefix"] = prms >> "use_prefix"  # this might get used by downstream configs.

    if prms >> "use_prefix":
        goal_prefix = "goal_"
    else:
        goal_prefix = "goal/"
    goal_names = [goal_prefix + goal_name for goal_name in GOAL_STATE_NAMES]
    param_names = []

    goal_state_nslds = []
    for nsld in base_nsld:
        if goal_prefix + nsld[0] in goal_names:
            goal_state_nsld = (goal_prefix + nsld[0], nsld[1], nsld[2], nsld[3])
            goal_state_nslds.append(goal_state_nsld)

    action_names = ['action',
                    'policy_type',
                    'target/position',
                    'target/grab_binary'
                    ]
    if common_params >> f'{group_name}/include_policy_name':
        action_names.append('policy_name')
    if common_params >> f'{group_name}/include_policy_switch':
        action_names.append('policy_switch')

    if common_params << "is_online_exp":
        # need next obs for RL experiments
        if prms >> "use_prefix":
            next_prefix = "next_"
        else:
            next_prefix = "next/"
        output_obs_names = [next_prefix + obs_name for obs_name in obs_names]

        # add the nsld info for the output observations
        obs_names_set = set(obs_names)
        output_obs_nsld = []
        for nsld in base_nsld:
            if nsld[0] in obs_names_set:
                output_obs_nsld.append((next_prefix + nsld[0], nsld[1], nsld[2], nsld[3]))

        # add reward to output obs and the nsld
        output_obs_names.append("reward")
        reward_nsld = ("reward", (1,), (-float("inf"), float("inf")), np.float32)
        output_obs_nsld.append(reward_nsld)

        # need next goals and nslds as well
        output_goal_names = [next_prefix + goal_name for goal_name in goal_names]
        output_goal_nslds = [(next_prefix + nsld[0], nsld[1], nsld[2], nsld[3]) for nsld in goal_state_nslds]
    else:
        output_obs_names = []
        output_obs_nsld = []
        output_goal_names = []
        output_goal_nslds = []

    # if DEBUG:
    #     param_names.extend(['block_colors'])
    # if INCLUDE_BLOCK_SIZES:
    #     param_names.append("block_sizes")
    final_names = []
    env_spec_params = AttrDict(
        cls=ParamEnvSpec,
        params=AttrDict(
            names_shapes_limits_dtypes=base_nsld + goal_state_nslds + output_obs_nsld + output_goal_nslds,
            output_observation_names=output_obs_names,
            observation_names=obs_names,
            action_names=action_names,
            goal_names=goal_names,
            output_goal_names=output_goal_names,
            param_names=param_names,
            final_names=final_names,
        )
    )

    common_params.env_spec = common_params.env_spec & env_spec_params
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
