from argparse import ArgumentParser

from robosuite.controllers import ALL_CONTROLLERS

from sbrl.envs.param_spec import ParamEnvSpec
# declares this group's parser, and defines any sub groups we need
from sbrl.envs.robosuite.robosuite_env import RobosuiteEnv
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    parser.add_argument("--env_name", type=str, required=True, help="environment name")
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--imgs", action='store_true')
    parser.add_argument("--reward_stages", action='store_true')
    parser.add_argument("--done_on_success", action='store_true')
    parser.add_argument("--enable_preset_sweep_n", type=int, default=0,
                        help="presets will be generated from list instead of uniformly.")
    parser.add_argument("--controller", type=str, default="OSC_POSE", choices=ALL_CONTROLLERS)

    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--img_height", type=int, default=256)

    parser.add_argument("--pos_noise_std", type=float, default=0)
    parser.add_argument("--ori_noise_std", type=float, default=0)

    # TODO below
    # parser.add_argument("--dt", type=float, default=0.1)
    # parser.add_argument("--time_step", type=float, default=0.02)
    # parser.add_argument("--clip_ee_ori", action='store_true', help="clip ee to a conical region around downward axis (-z)")
    # parser.add_argument("--use_infinite_joint7", action='store_true', help="Robot joint 7 (revolute end effector) has no limits.")
    # parser.add_argument("--max_vel", type=float, default=np.inf)
    # parser.add_argument("--randomize_robot_start", action='store_true')
    # parser.add_argument("--fixed_object_pose", action='store_true')
    # parser.add_argument("--better_view", action='store_true')
    # parser.add_argument("--analog", action='store_true', help="Use analog of real world")
    return parser


nut_preset_sweep_n_map = {
    4: (2, 2),
    10: (2, 5),
    20: (5, 4),  # 5 pos, 4 ori
    50: (5, 10),
    100: (10, 10)
}

th_preset_sweep_n_map = {
    16: (2, 2),  # 2 pos, 2 ori, 2^2 * 2^2
    36: (2, 3),
    81: (3, 3)
}


def process_params(group_name, common_params):
    assert "env" in group_name, f"Group name must have \'env\': {group_name}"
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()

    prms = common_params >> group_name

    env_cls = RobosuiteEnv

    ENV_NAME = prms >> "env_name"
    IMGS = prms >> "imgs"

    enable_preset, sp, so = (prms >> "enable_preset_sweep_n") > 0, 8, 8
    if enable_preset:
        if ENV_NAME == "NutAssemblySquare":
            sp, so = nut_preset_sweep_n_map[prms.enable_preset_sweep_n]
        elif ENV_NAME == "ToolHang":
            sp, so = th_preset_sweep_n_map[prms.enable_preset_sweep_n]
        else:
            raise NotImplementedError(f"implement eval sweeping for robosuite env name: {ENV_NAME}")

    env_params = AttrDict(
        cls=env_cls,
        params=AttrDict(
            env_name=ENV_NAME,
            img_width=prms >> "img_width",
            img_height=prms >> "img_height",
            imgs=IMGS,
            render=prms >> "render",
            controller=prms >> "controller",
            done_on_success=prms >> "done_on_success",
            use_reward_stages=prms >> "reward_stages",

            enable_preset_sweep=enable_preset,
            preset_sweep_pos=sp,
            preset_sweep_ori=so,

            pos_noise_std=prms >> "pos_noise_std",
            ori_noise_std=prms >> "ori_noise_std",
            # realtime=REALTIME,
        )
    )

    common_params[group_name] = common_params[group_name] & env_params

    ## spec changes
    assert common_params >> "env_spec/cls" == ParamEnvSpec, "Only supports param env spec, processed first"

    if IMGS:
        (common_params >> "env_spec/params/observation_names").append('image')

    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
