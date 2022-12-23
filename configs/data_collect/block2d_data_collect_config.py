import numpy as np
from sbrl.models.lmp.lmp import LMPModel

from sbrl.datasets.np_dataset import NpDataset
from sbrl.envs.block2d.block_env_2d import BlockEnv2D
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.models.model import Model
from sbrl.policies.basic_policy import BasicPolicy
from sbrl.utils.python_utils import AttrDict as d

# BASE EXPERIMENT PARAMS

EXPERIMENT_NAME = "block2D/lmp_daug_main"  # sac_collect_for_classify_updated"
DEVICE = "cuda:0"

DATA_INPUT_TRAIN = "/home/suneelbelkhale/stanford_sandbox/data/block2D/full_play_train.npz"
DATA_OUTPUT_TRAIN = "null.npz"
DATA_INPUT_HOLDOUT = "/home/suneelbelkhale/stanford_sandbox/data/block2D/full_play_val.npz"
DATA_OUTPUT_HOLDOUT = "null.npz"

# env stuff

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_C = 3

GRID_SIZE = np.array([600, 600])
BLOCK_SIZE = 30
BLOCK_MASS = 10.
NUM_MAZE_CELLS = 5
NUM_BLOCKS = 10
DT = 0.1

RENDER = True
HORIZON = 15

ACT_DIM = 3
ACT_NUM_MIX = 8  # gmm action mixture model
ACT_LOW = np.array((-GRID_SIZE / 10.).tolist() + [0.])
ACT_HIGH = np.array((GRID_SIZE / 10.).tolist() + [1000.])
QUANT_DIM = 256  # discretized
NUM_MIX = 8
ACT_MID = (ACT_LOW + ACT_HIGH) * 0.5
ACT_HALFRANGE = (ACT_HIGH - ACT_LOW) * 0.5

# OBS_DIM = 3 + 4 + 1 + 3 + 3 + 4
# ACT_DIM = 3 + 3  # using AxisAngle (theta, phi, omega(rot)) + offsets
# ACT_LOW = np.array([-np.pi / 7, -np.pi / 21, -np.pi / 21])  # we use small pose deltas
# ACT_LOW = np.append(ACT_LOW, np.array([-0.02, -0.005, -0.02])).astype(np.float32)  # final pose deformations (1cm)
# ACT_HIGH = -ACT_LOW
# ACTION_LOG_STD_BOUNDS = [np.array(-5.).astype(np.float32), np.array(2).astype(np.float32)]  # in -1 to 1 space

nsld = [
    # param
    ("image", (IMG_HEIGHT, IMG_WIDTH, IMG_C), (0, 255), np.uint8),
    ("position", (2,), (-np.inf, np.inf), np.float32),
    ("velocity", (2,), (-np.inf, np.inf), np.float32),
    ("block_positions", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
    ("block_velocities", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
    ("active_constraints", (NUM_BLOCKS,), (0, np.inf), np.float32),

    ("maze", (NUM_MAZE_CELLS, NUM_MAZE_CELLS), (0, 32), np.uint8),

    ("action", (3,), (ACT_LOW, ACT_HIGH), np.float32),
]

# we record these names in dataset
obs_names = ['image', 'position', 'velocity', 'block_positions', 'block_velocities', 'active_constraints']
output_obs_names = []
action_names = ['action']
goal_names = []
param_names = ['maze']
final_names = []

###############  FUNCTIONS  ##############


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
    return d(
        cls=BlockEnv2D,
        params=d(
            num_blocks=NUM_BLOCKS,
            dt=DT,
            grid_size=GRID_SIZE,
            image_size=np.array([IMG_HEIGHT, IMG_WIDTH]),
            block_size=BLOCK_SIZE,
            block_mass=BLOCK_MASS,
            num_maze_cells=NUM_MAZE_CELLS,
            render=RENDER,
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
            capacity=1e5,
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
        cls=BasicPolicy,
        params=d(
            policy_model_forward_fn=policy_model_forward_fn,
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
