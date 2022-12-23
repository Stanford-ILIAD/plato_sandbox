# PLATO code

Code for learning from play data, PLATO: Predicting Latent Affordances Through Object-Centric Play (CoRL 2022).

Website: https://sites.google.com/view/plato-corl22/home \
PDF: https://arxiv.org/pdf/2203.05630.pdf


## Installation

The code in this repo is mostly python, and is easily pip installable.

1. Create conda environment.
2. Install dependencies (a partial list is found under `requirements.txt`)
3. Install `torch` and `torchvision` with the corresponding GPU or CPU version (see torch conda docs)
4. `pip install -e <SBRL_ROOT>` to setup the package, which installs two packages
   1. `sbrl`: includes all source code
   2. `configs`: Configuration files for various experiments and sub-modules (`configs/README.md`).
5. Under `sbrl/envs`, download and unzip assets using: `gdown https://drive.google.com/uc?id=<FILE ON DRIVE>`
   1. `assets`: `FILE_ID = 1TR0ph1uUmtWFYJr8B0yo1hASjj0cP8fc` (latest: Jun 22)
6. Create several additional directories in the root:
   1. `data/`: This will store all the data
   2. `experiments/`: This is where runs will log and write models to.
   3. `plots/`: (optional)
   4. `videos/`: (optional)
7. Under `data/`, download and unzip data folders by environment:
   1. `stackBlock2D`: `FILE_ID = 1djnfM59b8DSMALS3TTy-p4IQD86I2sG8` (latest: Dec 22)
   2. `block3D`: `FILE_ID = 1nsoLKXC0PCdi_h3F1pkjyOhaYdS3Etei` (latest: Dec 22)

All set up!

## Design Overview
Here we give an overview of the design philosophy for this repository. [TODO] Submodules will have further explanations that go into greater detail.

The components are:
- **Datasets**: read / write access to data, and data statistics.
- **Models**: These hold all _parameters_, e.g, neural network weights. They also define `forward()` and `loss()` functions for training.
- **Policies**: These operate on a model and inputs to produce an "action" to be applied in the environment.
- **Environments**: Step-able, reset-able, gym-like format, where step(action) uses the output of the policy.
- **Metrics**: Things that will take inputs/outputs, and compute some tensor, e.g. representing a "loss", supporting group_by operations.
- **Rewards**: Similar to metrics, but compute a single value representing a reward potentially using the model too, and this can object be reset. 
- **Trainers**: Compose datasets, models, policies, environments, metrics, and rewards (optional) together into a training algorithm. Native tensorboard writing.

### AttrDict
We define a generic dictionary under `sbrl.utils.python_utils.AttrDict`, which implements nested dictionaries that are easy to access, filter, combine, and write to. Most classes accept AttrDict's rather than individual parameters for initialization, and often for methods as well.


**Creation**: Let's say we want to store the following nested structure of arrays:
```angular2html
- food
    - carrot: [1,2,3]
    - apple: [4,5,6]
    - broccoli: [7,8,9,10]
- utensils
    - fork
        - three_prong: [11,12]
        - four_prong: [13,14,15]
    - spoon: [16,17,18]
```

AttrDicts use '/' to separate keys, and this is built in to read/write operations. Here's two examples of how to instantiate the above structure.
```angular2html
d = AttrDict()
d['food/carrot'] = [1,2,3]
d.food.apple = [4,5,6]
d['food/broccoli'] = [7,8,9,10]
d['utensils/fork/three_prong'] = [11,12]
d['utensils/fork/four_prong'] = [13,14,15]
d['utensils/spoon'] = [16,17,18]
```
Note that both indexing and dot access work, since AttrDicts inherit from the DotMap class. Here's a slightly less effort way:
```angular2html
d = AttrDict(
    food=AttrDict(carrot=[1,2,3], apple=[4,5,6], broccoli=[7,8,9,10]),
    utensils=AttrDict(
        fork=AttrDict(three_prong=[11,12], four_prong=[13,14,15]), 
        spoon=[16,17,18]
    )
)
```
**Access**: There are several ways to access an AttrDict. The first three access strategies will create an AttrDict for the key if not present.
The second four ways are explicit and will not create new keys.
1. `d['utensils/fork/three_prong']`: standard dictionary access, but using '/' to implicitly sub-index. 
2. `d.utensils.fork.three_prong`: dotmap access
3. `d.utensils['fork/three_prong']`: mixed indexing + dotmap
4. `d >> 'utensils/fork/three_prong`: required key access, will error if not present.
5. `d << 'utensils/fork/three_prong`: optional key access, will return None if not present
6. `d > ['utensils/fork/three_prong,'utensils/spoon']`: required key filtering, returns sub-dict. errors if a key in the arg list is not present.
7. `d < ['utensils/fork/three_prong,'utensils/spoon']`: optional key access, returns sub-dict, ignores keys that aren't present.

**Node/Leaf operations**: Leaf nodes are any access pattern that returns something that isn't an AttrDict. In the above example, 'food' is a node key, while 'food/carrot' is a leaf key.
We can operate on all leaf nodes at once, here are some example methods:
1. `d.leaf_keys()`: Generator that yields leaf keys under a depth first traverse.
2. `d.list_leaf_keys()`: Outputs a list instead of generator.
3. `d.leaf_values()`: Generator that yields leaf values under a depth first traverse.
4. `applied_d = d.leaf_apply(lambda v: <new_v>)`: Apply a function(value) on all leaf values, and create a new AttrDict.
5. `filtered_d = d.leaf_filter(lambda k,v: <condition>)`: Only keep leaf keys where `condition` is true in new AttrDict.

Similarly, there are functions that operate on both nodes and leaves. 

**Combining**: Combining AttrDicts can be done in several ways:
1. `new_d = d1 & d2`: Standard join, returns a new AttrDict, which will favor keys from d2 if there are duplicates.
2. `d1.combine(d2)`: Mutates d1 to join the arrays.
3. `new_d = AttrDict.leaf_combine_and_apply([d1, d2, ...], lambda vs: <return one value>)`: Given a list of AttrDicts with the same keys, will create one AttrDict where the value for a given key `k` is some function of `vs = [d1[k], d2[k], ...]`.


### sbrl.datasets
Datasets implement various storage and reading mechanisms. 
`sbrl.datasets.NpDataset` is the one used for most things, and other types of datasets are built on top of this (e.g., see `sbrl.datasets.TorchDataset`).
Some datasets haven't been implemented (like `sbrl.datasets.Hdf5Dataset`).

Some methods of note:
- `get_batch(indices, ...)`: Gets a batch of data as two AttrDicts: inputs, outputs.
- `add_episode(inputs, outputs, ...)`: Adds data as an episode of inputs and outputs (both are AttrDicts).
- `add(input, output, ...)`: Adds a single input / output to the dataset (still AttrDicts).
- `__len__`: Size of the dataset.

### sbrl.envs
Environments are very similar to that in OpenAI's `gym` format. They use a shared asset folder `sbrl/envs/assets/` which should have been downloaded in installation.
These environments implement, for example:
- `step(action: AttrDict, ...) -> obs, goal, done`: Similar to gym, but everything is an AttrDict, except done which is a 1 element bool array.
- `reset(presets: AttrDict) -> obs, goal`: Like gym, but enables presets to constrain the resetting.

Environments used for PLATO are described under the Play-Specific Environments section below.

### sbrl.models
Models are an extension of `torch.nn.Module`, but with native support for AttrDicts, input / output normalization, pre/postprocessing, and much more.

We adopt the practice of first "declaring" architectures before instantiating them. We do this using `LayerParams` and `SequentialParams`, which accept individual layer arguments.
See `build_mlp_param_list()` in `sbrl.utils.param_utils` for an example of what this looks like. Model configurations will usually adopt a similar syntax.

See `RnnModel` for a recurrent structure model, which we use for most experiments.

### sbrl.grouped_models
GroupedModels enable multiple sub-modules (inheriting from `Model`) to be declared, checked, and debugged more easily. `LMPGroupedModel` and `PLATOGroupedModel` are two examples we use in PLATO.

### sbrl.policies
Policies _use_ models, but do not contain them. Think of policies as a functional abstraction that takes in `(model, obs, goal)` and outputs `(action)`. 
See `MemoryPolicy` for an example of how to keep track of a running memory (e.g., for `RNNModels`). 
The policy config will be responsible for providing the right `policy_model_forward_fn` for these more generic policies.

### sbrl.trainers
Trainers compose all the above modules into a training algorithm, involving optimizers, model losses, saving checkpoints, and optionally also involving stepping some environment.
For the experiments in PLATO we evaluate separately from training (purely offline training setup). The relevant class is `sbrl.trainers.Trainer`

---

## Play-Specific Environments & Data


### Environments

Running all the environments below will launch a basic teleop keyboard interface.

**BlockEnv3D**: Standard 3D table-top block environment with pybullet.

```
python sbrl/envs/bullet_envs/block3d/block_env_3d.py
```

**MugEnv3D**: Same tabletop environment, but with a mug instead of a block.

```
python sbrl/envs/bullet_envs/block3d/block_env_3d.py --use_mug
```
**BlockEnv3D-Platforms**: 3D tabletop with raised platform edges.

```
python sbrl/envs/bullet_envs/block3d/platform_block_env_3d.py
```

**MugEnv3D-Platforms**: Tabletop + platform, with a mug

```
python sbrl/envs/bullet_envs/block3d/platform_block_env_3d.py --use_mug
```

**Playroom3D**: Full cabinet + drawer + block desk environment (+ buttons with `--use_buttons` flag)

```
python sbrl/envs/bullet_envs/block3d/playroom.py [--use_buttons]
```

**StackBlockEnv2D**: 2D environment with pymunk, with tethering action (press `g` to grab onto an object when nearby)

```
python sbrl/envs/block2d/stack_block_env_2d.py
```

### Data

All scripted data is found under `data/`:

**BlockEnv3D-Platform**: `data/block3D/scripted_multiplay_multishape_uv_yawlim_randstart_diverse_push_lift_ud.npz`\
**MugEnv3D-Platform**: `data/block3D/scripted_multiplay_multishape_mug_fo_aw_ml_uv_yawlim_randstart_diverse_toprot_push_lift_ud_2mil.npz`\
**Playroom3D**: `data/block3D/scripted_multiplay_multishape_uv_randstart_diverse_push_drawer_cabinet_sequential_2mil.npz`\
**Playroom3D+Buttons**: `data/block3D/scripted_multiplay_multishape_uv_randstart_diverse_push_drawer_cabinet_button_sequential_2mil.npz`\
**StackBlockEnv2D**: `data/stackBlock2D/scripted_multiplay_multishape_named_push_pull_tipsrot_balanced_success_c.npz`

Each of the above has a corresponding validation dataset, suffixed by `_val`.

---

## Running Play-GCBC, Play-LMP, and PLATO

Now we will overview the training code and evaluation code. Make sure you read about the config structure in the readme under `configs/` first.

### Training

Here, we utilize `scripts/train.py` with the relevant base config, env, model parameters, dataset, etc. We will use the example of the playroom+buttons environment here.

#### Play-GCBC
```
python scripts/train.py configs/exp_lfp/base_3d_config.py --batch_size 1024 --horizon 40 --dataset scripted_multiplay_multishape_uv_randstart_diverse_push_drawer_cabinet_button_sequential_2mil \
%env_train --use_drawer --randomize_robot_start --use_buttons \
%model configs/exp_lfp/gcbc_model_config.py --include_block_sizes --do_grab_norm --hidden_size 256 --no_encode_actions \
%policy --replan_horizon 20 %dataset_train --capacity 3e6 --index_all_keys
```

#### Play-LMP
```
python scripts/train.py configs/exp_lfp/base_3d_config.py --batch_size 1024 --horizon 40 --dataset scripted_multiplay_multishape_uv_randstart_diverse_push_drawer_cabinet_button_sequential_2mil \
%env_train --use_drawer --randomize_robot_start --use_buttons \
%model configs/exp_lfp/lmp_model_config.py --include_block_sizes --do_grab_norm --hidden_size 256 --proposal_width 256 --beta 1e-4 --plan_size 64 --no_encode_actions \
%policy --replan_horizon 20 %dataset_train --capacity 3e6 --index_all_keys
```

#### PLATO
```
python scripts/train.py configs/exp_lfp/base_3d_config.py --device cpu --contact --batch_size 1024 --horizon 40 --dataset scripted_multiplay_multishape_uv_randstart_diverse_push_drawer_cabinet_button_sequential_2mil \
%env_train --use_drawer --randomize_robot_start --use_buttons \
%model configs/exp_lfp/plato_model_config.py --include_block_sizes --do_grab_norm --hidden_size 256 --proposal_width 256 --beta 1e-4 --plan_size 64 --no_encode_actions --prior_objects_only --do_initiation --sample_goal_start --soft_boundary_length 20 \
%policy --replan_horizon 20 %dataset_train --capacity 3e6 --index_all_keys
```

The above configs all will save under a certain experiment folder (printed in the output of the command) under `experiments/`. If you run, terminate, and wish to continue, you can specify the `--continue` flag immediately after `scripts/train.py` before the config.

A high level overview of this long command (sequence of different groups and their arguments):
- base config <args>: e.g., `configs/exp_lfp/base_3d_config.py`, specifies a lot of defaults for the module, including which base env/spec/policy/dataset to use. Model must be provided
  - `device`: which device to use, default is cuda, so you don't have to explicitly specify this in that case.
  - `horizon`: sampling window length from play, default is 20.
  - `batch_size`: keep this large, default is 1024 
  - `contact`: This flag tells the base config to use contact-based data parsing for the dataset(s), e.g. for PLATO.
- %env_train <args>: additional arguments needed for the environment for this specific dataset (e.g., here, buttons, drawers, and random robot starting pos).
- %model <file> <args>: specific to the model you are training. Probably don't change these arguments too much.
  - `plan_size`: Size of latent variable
  - `beta`: Regularization with KL divergence penalty
  - `hidden_size`: hidden size of RNN's and MLP's
  - `proposal_width`: hidden size of MLP for prior network (proposal) specifically.
  - `soft_boundary_length`: for PLATO, the soft boundary between pre-contact and contact, used for sampling flexibly (set to half the horizon length)
- %policy
  - `replan_horizon`: set to be 20 for stability, how often to replan the latent variable.
- %dataset_train:
  - `capacity`: set to be larger than the dataset size (3mil > 2mil in this case). 
  - `index_all_keys`: for training, this merges all keys into one big dataset for faster sampling. Do not remove this.

### Evaluation

During training, checkpoints will save under `experiments/<EXP_NAME>/models/chkpt_<>.pt`. The latest model will be under `model.pt` in that directory. Default saves every 20k steps of training.
Other things that save in the experiment folder:
- `config_args.txt`: the same config args after the base config that were specified during training. useful for running new commands
- `config.py`: copied base config that was used
- `git/`: git diff for recreating exact code
- `events...`: tensorboard log file for visualization
- `log_train.txt`: the output of training.
- `loss.csv`: training losses in csv format.
- `models/`: where all models are saved.

To evaluate, we use `scripts/eval_play.py`. This script loops the following:
1. Run a hard coded "play" policy to generate the multi-task goal
2. Run the model to reach that goal
3. Record success across tasks (split by policy name)

After finishing the given number of episodes (optionally in parallel), it aggregates the statistics and prints them. 

**NOTE**: This script will NOT save anything, only prints to console.

Here's an example eval command for the playroom+button env, running 500 episodes across 10 processes (recommend 2 CPUS per process):

```
python scripts/eval_play.py --num_eps 500 --num_proc 10 configs/exp_lfp/base_3d_config.py \
`cat experiments/block3D/posact_contact_b1024_lr0_0004_h40-40_scripted_multiplay_multishape_uv_randstart_diverse_push_drawer_cabinet_button_sequential_2mil_fast-clfp_softbnd20_initpol_noactenc_probjonly_normgrabil_bsz_p64_prop256_hs256_beta0_0001/config_args.txt` \ 
%policy_hardcoded configs/data_collect/drawer_block3d_policy_config.py --do_drawer --do_push --sequential --do_buttons --random_motion --single_policy
```

**NOTE**: NUM_EPS must be divisible by NUM_PROC.

Here, policy hardcoded is told to run the playroom environment with the drawer primitive, pushing, button pressing, and randomized motion for each to generate diverse goals.

For 2D environment, use: `%policy_hardcoded configs/data_collect/stackblock2d_policy_config.py --use_rotate --oversample_rot --single_policy`

For the 3D block environment, use: 
`%policy_hardcoded configs/data_collect/block3d_policy_config.py --do_lift --no_pull --random_motion --uniform_velocity --lift_sample --single_policy`

With mugs, same config as above, but args are: `--no_push --do_lift --lift_sample --use_rotate --random_motion --uniform_velocity --more_lift --mug_rot_allow_wall`

For regular playroom, just remove `--use_buttons` flag from the template command.