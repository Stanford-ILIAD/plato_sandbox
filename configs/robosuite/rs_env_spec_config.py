import enum
from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.envs.robosuite.robosuite_env import get_rs_example_spec_params
from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict


class FNPreset(enum.Enum):
    f_all_object = 0


fnops = [e.name for e in FNPreset]

preset2names = {
    FNPreset.f_all_object: ["object"],
}


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and prefixes by goal/
    parser.add_argument("--add_final_names", nargs="*", type=str, default=[])
    parser.add_argument("--add_param_names", nargs="*", type=str, default=[])
    parser.add_argument("--no_names", action='store_true')
    parser.add_argument("--minimal", action='store_true')
    parser.add_argument("--include_mode", action='store_true')
    parser.add_argument("--include_real", action='store_true')
    parser.add_argument("--include_click_state", action='store_true')
    parser.add_argument("--include_target_names", action='store_true')
    parser.add_argument("--include_target_gripper", action='store_true')
    return parser


# strictly ordered processing order
def process_params(group_name, common_params):
    assert "env_spec" in group_name
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()

    # access to all the params for the current experiment here.
    prms = common_params >> group_name
    env_prms = common_params >> "env_train"

    env_spec_params = get_rs_example_spec_params(env_prms >> "env_name", img_width=env_prms >> "img_width",
                                                 img_height=env_prms >> "img_height", no_names=prms >> 'no_names',
                                                 minimal=prms >> "minimal")

    # fill in presets
    add_final_names = prms >> "add_final_names"
    if len(add_final_names) == 1:
        if add_final_names[0] in fnops:
            en = FNPreset[add_final_names[0]]
            logger.debug(f"Using preset final name: {add_final_names[0]}: names = {preset2names[en]}")
            add_final_names = preset2names[en]

    add_param_names = prms >> "add_param_names"
    if len(add_param_names) == 1:
        if add_param_names[0] in fnops:
            en = FNPreset[add_param_names[0]]
            logger.debug(f"Using preset param name: {add_param_names[0]}: names = {preset2names[en]}")
            add_param_names = preset2names[en]

    # names to add to final_names in env_spec (dataset will parse these too)
    if len(add_final_names + add_param_names) > 0:
        nsld = env_spec_params >> "names_shapes_limits_dtypes"
        nsld_names = [n[0] for n in nsld]
        for name in add_final_names:
            assert name in nsld_names and f"goal/{name}" not in env_spec_params.final_names, [name, nsld_names]
            tup = nsld[nsld_names.index(name)]
            env_spec_params.final_names.append(f"goal/{name}")
            # add new final names with same shapes, limits, dtypes
            nsld.append((f"goal/{name}", *tup[1:]))
        for name in add_param_names:
            assert name in nsld_names and f"goal/{name}" not in env_spec_params.param_names, [name, nsld_names]
            tup = nsld[nsld_names.index(name)]
            env_spec_params.param_names.append(f"goal/{name}")
            # add new final names with same shapes, limits, dtypes
            nsld.append((f"goal/{name}", *tup[1:]))


    if prms >> 'include_click_state':
        env_spec_params.action_names.append('click_state')

    if prms >> 'include_mode':
        env_spec_params.observation_names.append('mode')

    if prms >> 'include_real':
        env_spec_params.observation_names.append('real')

    if prms >> 'include_target_names':
        env_spec_params.action_names.extend(['target/position', 'target/orientation', 'target/orientation_eul'])

    if prms >> 'include_target_gripper':
        env_spec_params.action_names.extend(['target/gripper'])

    env_spec_params = AttrDict(
        cls=ParamEnvSpec,
        params=env_spec_params
    )

    common_params[group_name] = common_params[group_name] & env_spec_params
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
