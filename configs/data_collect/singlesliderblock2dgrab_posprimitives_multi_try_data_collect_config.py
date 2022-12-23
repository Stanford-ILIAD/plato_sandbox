from argparse import ArgumentParser

import numpy as np

from sbrl.datasets.np_dataset import NpDataset
from sbrl.envs.block2d.slider_env_2d import SliderBlockEnv2D
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.policies.block2d_policies import PushPrimitive, PullPrimitive
from sbrl.policies.meta_policy import MetaPolicy
from sbrl.utils.config_utils import get_config_args
from sbrl.utils.python_utils import AttrDict as d

parser = ArgumentParser()
parser.add_argument('--render', action="store_true")
parser.add_argument('--min_max_policies', type=int, nargs=2, default=[1, 5])
parser.add_argument('--min_max_retreat', type=int, nargs=2, default=[3, 10])
extras = get_config_args()
logger.debug("Config extras: " + str(extras))
args, _ = parser.parse_known_args(args=extras)

# BASE EXPERIMENT PARAMS
EXPERIMENT_NAME = "singleSliderBlock2D/primitives"  # sac_collect_for_classify_updated"
DEVICE = "cpu"

DATA_INPUT_TRAIN = "null"
DATA_OUTPUT_TRAIN = "null.npz"
DATA_INPUT_HOLDOUT = "null"
DATA_OUTPUT_HOLDOUT = "null.npz"

# env stuff

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_C = 3

GRID_SIZE = np.array([600, 600])
BLOCK_SIZE = GRID_SIZE[0] / 8
BLOCK_MASS = 10.
NUM_MAZE_CELLS = 5  # all zeros, doesnt matter
NUM_BLOCKS = 1
DT = 0.1

RENDER = args.render
REALTIME = RENDER
DISABLE_IMAGES = not RENDER  # this means don't even compute surface images
HORIZON = 20

MIN_NUM_COLLECTIONS_PER_ENV_RESET = args.min_max_policies[0]
MAX_NUM_COLLECTIONS_PER_ENV_RESET = args.min_max_policies[1]  # TODO modulate this
KEEP_IN_BOUNDS = True

ACT_DIM = 3
ACT_LOW = np.array((-GRID_SIZE / 3.).tolist() + [0.])
ACT_HIGH = np.array((GRID_SIZE / 3.).tolist() + [1000.])

# POLICY
ALL_POLICIES = [
    d(cls=PushPrimitive, params=d(vel_noise=0.)),
    d(cls=PullPrimitive, params=d(vel_noise=0.)),
]

nsld = [
    # param
    ("image", (IMG_HEIGHT, IMG_WIDTH, IMG_C), (0, 255), np.uint8),
    ("position", (2,), (-np.inf, np.inf), np.float32),
    ("velocity", (2,), (-np.inf, np.inf), np.float32),
    ("angle", (1,), (-np.inf, np.inf), np.float32),
    ("angular_velocity", (1,), (-np.inf, np.inf), np.float32),
    ("force", (2,), (-np.inf, np.inf), np.float32),
    ("torque", (1,), (-np.inf, np.inf), np.float32),

    ("block_positions", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
    ("block_linear_positions", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("block_velocities", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
    ("block_angles", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("block_angular_velocities", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("block_forces", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
    ("block_torques", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("active_constraints", (NUM_BLOCKS,), (0, np.inf), np.float32),
    ("grab_binary", (NUM_BLOCKS,), (0, 1.), np.float32),
    ("grab_force", (NUM_BLOCKS,), (0, np.inf), np.float32),
    ("grab_distance", (NUM_BLOCKS,), (0, np.inf), np.float32),
    ("grab_vector", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),

    ("block_masses", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("block_colors", (NUM_BLOCKS, 4), (-np.inf, np.inf), np.float32),
    ("maze", (NUM_MAZE_CELLS, NUM_MAZE_CELLS), (0, 32), np.uint8),

    ("action", (3,), (ACT_LOW, ACT_HIGH), np.float32),
    ("policy_type", (1,), (0, 255), np.uint8),

    # goal positions, if available
    ("target/position", (2,), (-np.inf, np.inf), np.float32),
    ("target/grab_binary", (NUM_BLOCKS,), (0, 1.), np.float32),
]

# we record these names in dataset  TOD0 'image'
obs_names = [
    # 'image',
     'position', 'velocity', 'angle', 'angular_velocity', 'force', 'torque',
     'block_positions', 'block_linear_positions', 'block_velocities', 'block_angles', 'block_angular_velocities', 'block_forces', 'block_torques',
     'active_constraints',
     'grab_binary', 'grab_force', 'grab_distance', 'grab_vector'
]
output_obs_names = []
action_names = ['action', 'policy_type', 'target/position', 'target/grab_binary']
goal_names = []
param_names = ['block_colors', 'block_masses']
final_names = []


###############  FUNCTIONS  ##############


def policy_next_params_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), **kwargs):
    # pick side that is close
    p, block_ps, lin_ps = obs.leaf_apply(lambda arr: arr[0, 0]).get_keys_required(['position', 'block_positions', 'block_linear_positions'])

    dists = np.linalg.norm(p[None] - block_ps, axis=-1)
    # print(p, block_ps, dists)
    block_idx = np.argmin(dists)

    offsets = np.asarray([[-BLOCK_SIZE * 1.08, 0],
                          [BLOCK_SIZE * 1.08, 0],
                          [0, -BLOCK_SIZE * 1.08],
                          [0, BLOCK_SIZE * 1.08]])

    min_offset = np.argmin(np.linalg.norm(p[None] - (block_ps[None, block_idx] + offsets), axis=-1))

    val_idxs = []
    if (block_idx == 1 and offsets[min_offset][1] != 0) or (block_idx == 0 and offsets[min_offset][0] != 0):
        val_idxs.extend([0, 2])  # approaching parallel
    if (block_idx == 1 and offsets[min_offset][0] != 0) or (block_idx == 0 and offsets[min_offset][1] != 0):
        val_idxs.extend([1, 3])  # approaching tangent

    assert len(val_idxs) != 0

    if not memory.has_leaf_key("reset_count"):
        memory.reset_count = 0
        memory.max_iters = np.random.randint(MIN_NUM_COLLECTIONS_PER_ENV_RESET, MAX_NUM_COLLECTIONS_PER_ENV_RESET)

    if memory.reset_count < memory.max_iters:
        policy_idx = np.random.choice(val_idxs)
        pps = d(
            block_idx=block_idx,
            offset=offsets[min_offset],
            kp_vel=1.5,
            speed_scale=np.random.uniform(1, 1.3),
            retreat_steps=np.random.randint(args.min_max_retreat[0], args.min_max_retreat[1]),
            timeout=50,
        )
        if policy_idx == 0:
            pps.push_steps = np.random.randint(5, 15)
        else:
            ptan = policy_idx - 2
            policy_idx = 1
            pps.pull_steps = np.random.randint(5, 15)
            pps.grab_force = np.random.uniform(750, 950)  # high (ignored)
            pps.grab_binary = True  # 0/1 actions, simpler
            pps.tolerance = np.random.uniform(0.03, 0.05)
            # direction to pull (0 is back)
            pps.pull_tangent = ptan

        memory.reset_count += 1
    else:
        # over
        policy_idx = None
        pps = d()
    return policy_idx, pps


############### START PARAMS ##############


def get_env_spec_params():
    return d(
        cls=ParamEnvSpec,
        params=d(
            names_shapes_limits_dtypes=nsld,
            output_observation_names=output_obs_names,
            observation_names=obs_names,
            action_names=action_names,
            goal_names=goal_names,
            param_names=param_names,
            final_names=final_names,
        )
    )


def get_env_params():
    return d(
        cls=SliderBlockEnv2D,
        params=d(
            num_blocks=NUM_BLOCKS,
            dt=DT,
            grid_size=GRID_SIZE,
            image_size=np.array([IMG_HEIGHT, IMG_WIDTH]),
            block_size=BLOCK_SIZE,
            block_mass=BLOCK_MASS,
            num_maze_cells=NUM_MAZE_CELLS,
            render=RENDER,
            realtime=REALTIME,
            disable_images=DISABLE_IMAGES,
            keep_in_bounds=KEEP_IN_BOUNDS,
            grab_action_binary=True,
            grab_action_max_force=10000,
        ),
    )


def get_dataset_params(input_file, output_file):
    return d(
        cls=NpDataset,
        params=d(
            file=input_file,
            output_file=output_file,
            save_every_n_steps=0,  # not used
            horizon=HORIZON,
            capacity=1e7 + 1 if DISABLE_IMAGES else 2e5,  # 2e5
            batch_size=10,
        )
    )


# TODO fix this
def get_model_params():
    return d(
        cls=Model,
        params=d(ignore_inputs=True)
    )


def get_trainer_params():
    return d()


def get_policy_params():
    policy_params = d(
        cls=MetaPolicy,
        params=d(
            all_policies=ALL_POLICIES,
            next_param_fn=policy_next_params_fn,
        )
    )
    return policy_params


params = d(
    exp_name=EXPERIMENT_NAME,

    env_spec=get_env_spec_params(),
    env=get_env_params(),
    dataset_train=get_dataset_params(DATA_INPUT_TRAIN, DATA_OUTPUT_TRAIN),
    dataset_holdout=get_dataset_params(DATA_INPUT_HOLDOUT, DATA_OUTPUT_HOLDOUT),
    model=get_model_params(),
    trainer=get_trainer_params(),
    policy=get_policy_params(),

)
