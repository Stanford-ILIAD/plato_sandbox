from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.envs.polymetis.polymetis_panda_env import get_polymetis_panda_example_spec_params
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and prefixes by goal/
    parser.add_argument('--include_click_state', action='store_true')
    parser.add_argument('--include_mode', action='store_true')
    return parser


# strictly ordered processing order
def process_params(group_name, common_params):
    assert "env_spec" in group_name
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    env_args = common_params >> 'env_train'
    prms = common_params >> group_name

    env_spec_params = get_polymetis_panda_example_spec_params(action_space=env_args >> 'action_space', 
                            use_imgs=env_args >> 'imgs', img_height=env_args >> 'img_height', img_width=env_args >> 'img_width')

    env_spec_params = AttrDict(
        cls=ParamEnvSpec,
        params=env_spec_params
    )

    if prms >> 'include_click_state':
        (env_spec_params >> 'params/observation_names').append('click_state')

    if prms >> 'include_mode':
        (env_spec_params >> 'params/observation_names').append('mode')

    common_params.env_spec = common_params.env_spec & env_spec_params
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
