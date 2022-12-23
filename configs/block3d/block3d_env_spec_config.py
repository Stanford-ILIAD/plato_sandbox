import enum
from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
from sbrl.envs.bullet_envs.block3d.block_env_3d import get_block3d_example_spec_params
from sbrl.envs.bullet_envs.block3d.playroom import get_playroom3d_example_spec_params
from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict


class FNPreset(enum.Enum):
    f_all_object = 0
    f_reduced_object = 1


fnops = [e.name for e in FNPreset]

preset2names = {
    FNPreset.f_all_object: ["objects/position", "objects/velocity", "objects/orientation_eul",
                            "objects/angular_velocity", "objects/size"],
    FNPreset.f_reduced_object: ["objects/position", "objects/orientation_eul", "objects/size"],
}


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and prefixes by goal/
    parser.add_argument("--add_final_names", nargs="*", type=str, default=[])
    parser.add_argument("--add_param_names", nargs="*", type=str, default=[])
    parser.add_argument("--no_names", action='store_true')
    return parser


# strictly ordered processing order
def process_params(group_name, common_params):
    assert group_name == "env_spec"
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()

    # access to all the params for the current experiment here.
    prms = common_params >> group_name
    env_prms = common_params >> "env_train"
    NUM_BLOCKS = env_prms >> "num_blocks"

    if env_prms >> "use_drawer":
        env_spec_params = get_playroom3d_example_spec_params(NB=NUM_BLOCKS, img_width=env_prms >> "img_width",
                                                             img_height=env_prms >> "img_height",
                                                             img_channels=env_prms >> "img_channels",
                                                             no_names=prms >> "no_names", use_buttons=env_prms >> "use_buttons")
    else:
        env_spec_params = get_block3d_example_spec_params(NB=NUM_BLOCKS, img_width=env_prms >> "img_width",
                                                          img_height=env_prms >> "img_height",
                                                          img_channels=env_prms >> "img_channels",
                                                          no_names=prms >> "no_names")

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

    env_spec_params = AttrDict(
        cls=ParamEnvSpec,
        params=env_spec_params
    )

    common_params.env_spec = common_params.env_spec & env_spec_params
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
