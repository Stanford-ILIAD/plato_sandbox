import os
from argparse import ArgumentParser

import numpy as np
from sbrl.models.lmp.lmp import LMPModel

from sbrl.datasets.np_dataset import NpDataset
from sbrl.envs.block2d.stack_block_env_2d import StackBlockEnv2D
from sbrl.envs.block2d.teleop_functions import pygame_mouse_teleop_fn
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.experiments.file_manager import FileManager
from sbrl.models.model import Model
from sbrl.policies.block2d_policies import PushPrimitive, PullPrimitive
from sbrl.policies.blocks.stack_block2d_policies import get_push_pull_lift_memory_meta_policy_params_fn, \
    TipBlockPrimitive, RotateBlockPrimitive, get_push_pull_lift_rotate_memory_meta_policy_params_fn, \
    SideRotateBlockPrimitive, get_rotate_only_memory_meta_policy_params_fn
from sbrl.policies.meta_policy import MetaPolicy
from sbrl.trainers.trainer import Trainer
from sbrl.utils.config_utils import get_config_args
from sbrl.utils.python_utils import AttrDict as d

# BASE EXPERIMENT PARAMS
parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--render', action='store_true')
parser.add_argument('--start_near_bottom', action='store_true')
parser.add_argument('--use_primitives', action='store_true')
parser.add_argument('--randomize_block_sizes', action='store_true')
parser.add_argument('--use_rotate', action='store_true')
parser.add_argument('--no_push_pull', action='store_true')
parser.add_argument('--oversample_rot', action='store_true', help="specific to push_pull_rot")
parser.add_argument('--oversample_tip', action='store_true', help="specific to rotate only case")
parser.add_argument('--prefer_idxs', type=int, nargs="*", default=[])
parser.add_argument('--save_names', action='store_true')
parser.add_argument('--save_block_contact', action='store_true')
parser.add_argument('--imgs', action='store_true')
# only relevant if using primitives
parser.add_argument('--min_max_retreat', type=int, nargs=2, default=[5, 12])
parser.add_argument('--min_max_policies', type=int, nargs=2, default=[3, 6])


# globally stored
extras = get_config_args()
logger.debug("Config extras: " + str(extras))
args, _ = parser.parse_known_args(args=extras)

EXPERIMENT_NAME = f"stackBlock2D/{args.exp_name}"
DEVICE = "cpu"

DATA_INPUT_TRAIN = os.path.join(FileManager.data_dir, "stackBlock2D/play.npz")
DATA_OUTPUT_TRAIN = "null.npz"
DATA_INPUT_HOLDOUT = os.path.join(FileManager.data_dir, "stackBlock2D/play_val.npz")
DATA_OUTPUT_HOLDOUT = "null.npz"

# env stuff

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_C = 3

GRID_SIZE = np.array([400, 400])
# BLOCK_SIZE = 30
BLOCK_MASS = 10.
NUM_MAZE_CELLS = 8
NUM_BLOCKS = 1
DT = 0.1
ROTATE = args.use_rotate
PUSH_PULL = not args.no_push_pull
assert ROTATE or PUSH_PULL, "one must be specified at least"

RENDER = args.render
HORIZON = 20

# TODO parameterize these?
BLOCK_LOWER = (30, 30)
BLOCK_UPPER = (60, 80)

ACT_DIM = 3
ACT_NUM_MIX = 8  # gmm action mixture model
ACT_LOW = np.array((-GRID_SIZE / 4.).tolist() + [0.])
ACT_HIGH = np.array((GRID_SIZE / 4.).tolist() + [1000.])

# OBS_DIM = 3 + 4 + 1 + 3 + 3 + 4
# ACT_DIM = 3 + 3  # using AxisAngle (theta, phi, omega(rot)) + offsets
# ACT_LOW = np.array([-np.pi / 7, -np.pi / 21, -np.pi / 21])  # we use small pose deltas
# ACT_LOW = np.append(ACT_LOW, np.array([-0.02, -0.005, -0.02])).astype(np.float32)  # final pose deformations (1cm)
# ACT_HIGH = -ACT_LOW
# ACTION_LOG_STD_BOUNDS = [np.array(-5.).astype(np.float32), np.array(2).astype(np.float32)]  # in -1 to 1 space

ALL_POLICIES = [
    d(cls=PushPrimitive, params=d(vel_noise=0.)),
    d(cls=PullPrimitive, params=d(vel_noise=0.)),
]

if ROTATE:
    ALL_POLICIES.extend([
        d(cls=TipBlockPrimitive, params=d(vel_noise=0.)),
        d(cls=RotateBlockPrimitive, params=d(vel_noise=0.)),
        d(cls=SideRotateBlockPrimitive, params=d(vel_noise=0.)),
    ])

nsld = [
    # param
    ("image", (IMG_HEIGHT, IMG_WIDTH, IMG_C), (0, 255), np.uint8),
    ("position", (2,), (-np.inf, np.inf), np.float32),
    ("velocity", (2,), (-np.inf, np.inf), np.float32),
    ("bounding_box", (4,), (-np.inf, np.inf), np.float32),
    ("angle", (1,), (-np.inf, np.inf), np.float32),
    ("angular_velocity", (1,), (-np.inf, np.inf), np.float32),
    ("force", (2,), (-np.inf, np.inf), np.float32),
    ("torque", (1,), (-np.inf, np.inf), np.float32),
    ("block_positions", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
    # ("block_linear_positions", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("block_bounding_boxes", (NUM_BLOCKS, 4), (-np.inf, np.inf), np.float32),
    ("block_velocities", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
    ("block_angles", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("block_sizes", (NUM_BLOCKS, 2), (0, np.inf), np.float32),
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
    ("block_contact", (NUM_BLOCKS,), (False, True), np.bool),
    ("block_contact_points", (NUM_BLOCKS, 4), (-np.inf, np.inf), np.float32),
    ("block_contact_normal", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),

    ("action", (3,), (ACT_LOW, ACT_HIGH), np.float32),
    ("policy_type", (1,), (0, 255), np.uint8),
    ("policy_name", (1,), (0, 1), np.object),
    ("policy_switch", (1,), (False, True), np.bool),  # marks the beginning of a policy

    # goal positions, if available
    ("target/position", (2,), (-np.inf, np.inf), np.float32),
    ("target/grab_binary", (NUM_BLOCKS,), (0, 1.), np.float32),

]

# we record these names in dataset  TOD0 'image'
obs_names = [
    # 'image',
    'position', 'velocity', 'angle', 'angular_velocity', 'force', 'torque',
    'block_positions', 'block_velocities', 'block_angles', 'block_angular_velocities', 'block_forces', 'block_torques',
    'active_constraints', 'block_bounding_boxes', 'bounding_box',
    'grab_binary', 'grab_force', 'grab_distance', 'grab_vector',
]
if args.imgs:
    obs_names.append('image')
if args.save_block_contact:
    obs_names.append('block_contact')
    obs_names.append('block_contact_points')
    obs_names.append('block_contact_normal')

output_obs_names = []
action_names = ['action', 'policy_type', 'target/position', 'target/grab_binary']  # , 'policy_type']
if args.save_names:
    action_names.append('policy_name')
    action_names.append('policy_switch')
goal_names = []
param_names = ['block_colors', 'block_masses']
if args.randomize_block_sizes:
    param_names.append('block_sizes')
final_names = []


###############  FUNCTIONS  ##############

if ROTATE and PUSH_PULL:
    policy_next_params_fn = get_push_pull_lift_rotate_memory_meta_policy_params_fn(*args.min_max_policies,
                                                                            *args.min_max_retreat, random_side=True, randomize_offset=True, oversample_rot=args.oversample_rot, prefer_idxs=args.prefer_idxs)
elif ROTATE:
    policy_next_params_fn = get_rotate_only_memory_meta_policy_params_fn(*args.min_max_policies,
                                                                            *args.min_max_retreat, random_side=True, randomize_offset=True, oversample_tip=args.oversample_tip)
else:
    policy_next_params_fn = get_push_pull_lift_memory_meta_policy_params_fn(*args.min_max_policies, *args.min_max_retreat, random_side=True)

def policy_model_forward_fn(model: LMPModel, obs: d, goal: d, **kwargs):
    raise NotImplementedError


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
    assert NUM_BLOCKS == 1
    env_params = d(
        cls=StackBlockEnv2D,
        params=d(
            dt=DT,
            image_size=np.array([IMG_HEIGHT, IMG_WIDTH]),
            # block_size=BLOCK_SIZE,
            block_mass=BLOCK_MASS,
            grab_action_binary=True,
            keep_in_bounds=True,
            render=RENDER,
            teleop_fn=pygame_mouse_teleop_fn,
        ),
    )
    block_max_size = (40, 80)
    if args.randomize_block_sizes:
        env_params.params.block_size_lower = BLOCK_LOWER
        env_params.params.block_size_upper = BLOCK_UPPER
        block_max_size = BLOCK_UPPER
    num_blocks = (np.asarray(GRID_SIZE) / np.asarray(block_max_size)).astype(int)

    if args.start_near_bottom:
        vs_default = np.meshgrid(range(num_blocks[0]), range(num_blocks[1] // 2))
        vs_default = list(zip(*[vs.reshape(-1) for vs in vs_default]))
        env_params.params.valid_start_idxs = vs_default
    return env_params


def get_dataset_params(input_file, output_file):
    return d(
        cls=NpDataset,
        params=d(
            file=input_file,
            output_file=output_file,
            save_every_n_steps=0,  # not used
            horizon=HORIZON,
            capacity=1e7,
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
    return d(
        cls=Trainer,
        params=d(
            step_train_env_every_n_steps=1,
        )
    )


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
