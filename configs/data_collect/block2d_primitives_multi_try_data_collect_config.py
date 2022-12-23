import numpy as np

from sbrl.datasets.np_dataset import NpDataset
from sbrl.envs.block2d.block_env_2d import BlockEnv2D
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.models.model import Model
from sbrl.policies.block2d_policies import PushPrimitive, PullPrimitive
from sbrl.policies.meta_policy import MetaPolicy
from sbrl.utils.python_utils import AttrDict as d

# BASE EXPERIMENT PARAMS
EXPERIMENT_NAME = "block2D/primitives"  # sac_collect_for_classify_updated"
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
BLOCK_SIZE = 30
BLOCK_MASS = 10.
NUM_MAZE_CELLS = 5
NUM_BLOCKS = 1
DT = 0.1

RENDER = False
REALTIME = RENDER
DISABLE_IMAGES = not RENDER  # this means don't even compute surface images
HORIZON = 20

MAX_NUM_COLLECTIONS_PER_ENV_RESET = 5  # TODO modulate this
KEEP_IN_BOUNDS = True

ACT_DIM = 3
ACT_LOW = np.array((-GRID_SIZE / 3.).tolist() + [0.])
ACT_HIGH = np.array((GRID_SIZE / 3.).tolist() + [1000.])

# POLICY
ALL_POLICIES = [
    d(cls=PushPrimitive, params=d()),
    d(cls=PullPrimitive, params=d()),
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
    ("block_velocities", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
    ("block_angles", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("block_angular_velocities", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("block_forces", (NUM_BLOCKS, 2), (-np.inf, np.inf), np.float32),
    ("block_torques", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("active_constraints", (NUM_BLOCKS,), (0, np.inf), np.float32),

    ("block_masses", (NUM_BLOCKS,), (-np.inf, np.inf), np.float32),
    ("block_colors", (NUM_BLOCKS, 4), (-np.inf, np.inf), np.float32),
    ("maze", (NUM_MAZE_CELLS, NUM_MAZE_CELLS), (0, 32), np.uint8),

    ("action", (3,), (ACT_LOW, ACT_HIGH), np.float32),
    ("policy_type", (1,), (0, 255), np.uint8),
]

# we record these names in dataset  TOD0 'image'
obs_names = [
    # 'image',
             'position', 'velocity', 'angle', 'angular_velocity', 'force', 'torque',
             'block_positions', 'block_velocities', 'block_angles', 'block_angular_velocities', 'block_forces', 'block_torques',
             'active_constraints']
output_obs_names = []
action_names = ['action', 'policy_type']
goal_names = []
param_names = ['maze', 'block_colors', 'block_masses']
final_names = []


###############  FUNCTIONS  ##############


def policy_next_params_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), **kwargs):
    # pick side that is close
    p, block_ps = obs.leaf_apply(lambda arr: arr[0, 0]).get_keys_required(['position', 'block_positions'])

    dists = np.linalg.norm(p[None] - block_ps, axis=-1)
    # print(p, block_ps, dists)
    block_idx = np.argmin(dists)

    offsets = np.asarray([[-BLOCK_SIZE * 1.08, 0],
                          [BLOCK_SIZE * 1.08, 0],
                          [0, -BLOCK_SIZE * 1.08],
                          [0, BLOCK_SIZE * 1.08]])

    min_offset = np.argmin(np.linalg.norm(p[None] - (block_ps[None, block_idx] + offsets), axis=-1))

    if not memory.has_leaf_key("reset_count"):
        memory.reset_count = 0
        memory.max_iters = np.random.randint(1, MAX_NUM_COLLECTIONS_PER_ENV_RESET)

    if memory.reset_count < memory.max_iters:
        random_policy_idx = np.random.choice([0, 1], p=[0.25, 0.75])
        # PULL
        policy_idx = random_policy_idx
        pps = d(
            block_idx=block_idx,
            offset=offsets[min_offset],
            kp_vel=1.5,
            speed_scale=np.random.uniform(1., 5.),
            retreat_steps=np.random.randint(10, 15)
        )
        if policy_idx == 0:
            pps.push_steps = np.random.randint(5, 20)
        elif policy_idx == 1:
            pps.pull_steps = np.random.randint(5, 20)
            pps.tolerance = 0.05
            # direction to pull (0 is back)
            pps.pull_tangent = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])

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
            realtime=REALTIME,
            disable_images=DISABLE_IMAGES,
            keep_in_bounds=KEEP_IN_BOUNDS,
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
