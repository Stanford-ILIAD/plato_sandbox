from argparse import ArgumentParser

import numpy as np

from sbrl.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D
from sbrl.envs.bullet_envs.block3d.platform_block_env_3d import PlatformBlockEnv3D
from sbrl.envs.bullet_envs.block3d.playroom import DrawerPlayroomEnv3D
from sbrl.envs.param_spec import ParamEnvSpec
# declares this group's parser, and defines any sub groups we need
from sbrl.policies.controllers.robot_config import os_torque_control_panda_cfg, os_torque_control_zstiff_panda_cfg
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    # add arguments unique & local to this level, and
    # parser.add_argument("--block_size", type=float, nargs=2, default=[0.05, 0.04, 0.03])
    parser.add_argument("--block_lower", type=float, nargs=3, default=[0.025, 0.025, 0.025])
    parser.add_argument("--block_upper", type=float, nargs=3, default=[0.055, 0.055, 0.055])
    parser.add_argument("--block_rotation_range", type=float, nargs=2, default=[-np.pi / 4, np.pi / 4])
    parser.add_argument("--block_mass_range", type=float, nargs=2, default=[0.05, 0.15])
    parser.add_argument("--mug_scale_range", type=float, nargs=2, default=[0.05, 0.09])
    parser.add_argument("--mug_mass_range", type=float, nargs=2, default=[0.01, 0.05])
    parser.add_argument("--mug_rotation_range", type=float, nargs=2, default=[-np.pi / 4, np.pi / 4])
    parser.add_argument("--mug_full_ori", action='store_true')
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--time_step", type=float, default=0.02)
    parser.add_argument("--use_platform", action='store_true')
    parser.add_argument("--use_drawer", action='store_true')
    parser.add_argument("--use_buttons", action='store_true')
    parser.add_argument("--use_mug", action='store_true')
    parser.add_argument("--no_drawer_snap", action='store_true')
    parser.add_argument("--object_start_in_dcab", action='store_true')
    parser.add_argument("--use_zstiff", action='store_true')
    parser.add_argument("--clip_ee_ori", action='store_true', help="clip ee to a conical region around downward axis (-z)")
    # parser.add_argument("--max_vel", type=float, default=np.inf)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--imgs", action='store_true')
    parser.add_argument("--analog", action='store_true', help="Use analog of real world")
    # parser.add_argument("--extra", action='store_true')
    # parser.add_argument("--no_block_sizes", action='store_true')
    parser.add_argument("--no_block_contact", action='store_true')
    parser.add_argument("--no_drawer_contact", action='store_true')
    parser.add_argument("--short_cab", action='store_true', help="short cabinet for playroom")
    parser.add_argument("--use_infinite_joint7", action='store_true', help="Robot joint 7 (revolute end effector) has no limits.")

    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--num_maze_cells", type=int, default=8)
    parser.add_argument("--num_blocks", type=int, default=1)

    parser.add_argument("--randomize_robot_start", action='store_true')
    parser.add_argument("--fixed_object_pose", action='store_true')
    # parser.add_argument("--start_near_bottom", action='store_true')
    parser.add_argument("--better_view", action='store_true')
    return parser


def process_params(group_name, common_params):
    assert "env" in group_name, f"Group name must have \'env\': {group_name}"
    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()

    NUM_BLOCKS, BLOCK_LOW, BLOCK_HIGH, MUG_SCALES, MUG_MASSES, MUG_ROTS, DT, IMG_HEIGHT, IMG_WIDTH, BLOCK_MASS_RANGE, BLOCK_ROTATION_BOUNDS, RENDER, IMGS, NO_BC, NO_DC, PLAT, DRAWER, NO_DSNAP, use_mug, use_zstiff, fop, OSDC = \
        (common_params >> group_name).get_keys_required(
            ['num_blocks', 'block_lower', 'block_upper', 'mug_scale_range', 'mug_mass_range', 'mug_rotation_range',
             'dt', 'img_height', 'img_width', 'block_mass_range', 'block_rotation_range',
             'render', 'imgs', 'no_block_contact', "no_drawer_contact", "use_platform", "use_drawer", "no_drawer_snap",
             "use_mug", "use_zstiff", "fixed_object_pose", "object_start_in_dcab"]
        )
    REALTIME = RENDER

    prms = common_params >> group_name

    assert DT % 0.02 < 1e-11, f"DT ({DT}) must be divisible by 0.02"

    assert not (PLAT and DRAWER), "Only one!"

    assert not OSDC or DRAWER, "Object start in drawer-cab only makes sense for Playroom"

    if DRAWER:
        env_cls = DrawerPlayroomEnv3D
    elif PLAT:
        env_cls = PlatformBlockEnv3D
    else:
        env_cls = BlockEnv3D

    if prms >> "mug_full_ori":
        MUG_ROTS = [-np.pi, np.pi]

    env_params = AttrDict(
        cls=env_cls,
        params=AttrDict(
            num_blocks=NUM_BLOCKS,
            time_step=common_params[group_name] >> "time_step",
            skip_n_frames_every_step=int(DT // (common_params[group_name] >> "time_step")),
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            compute_images=IMGS,
            render=RENDER,
            realtime=REALTIME,
            reinit_objects_on_reset=True,
            object_shape_bounds={'block': (np.broadcast_to(BLOCK_LOW, (3,)), np.broadcast_to(BLOCK_HIGH, (3,))),
                                 'mug': (np.array([MUG_SCALES[0]]), np.array([MUG_SCALES[1]]))},
            object_mass_bounds={'block': tuple(BLOCK_MASS_RANGE), 'mug': tuple(MUG_MASSES)},
            object_rotation_bounds={'block': tuple(BLOCK_ROTATION_BOUNDS), 'mug': tuple(MUG_ROTS)},
            do_random_ee_position=(common_params >> group_name) >> "randomize_robot_start",
            fixed_object_pose=fop,
            object_start_in_dcab=OSDC,
            clip_ee_ori=prms >> "clip_ee_ori",
            use_buttons=prms >> "use_buttons",

            robot_params=AttrDict(use_infinite_joint7=prms >> "use_infinite_joint7"),

        ) & (os_torque_control_zstiff_panda_cfg if use_zstiff else os_torque_control_panda_cfg),  # torque control
    )

    # BETTER VIEWPOINT
    if prms >> "better_view":
        if DRAWER:
            env_params.params.debug_cam_dist = 0.5
            env_params.params.debug_cam_p = -30
            env_params.params.debug_cam_y = 120
            env_params.params.debug_cam_target_pos = [0.4, 0, 0.45]
        else:
            env_params.params.debug_cam_dist = 0.25
            env_params.params.debug_cam_p = -45
            env_params.params.debug_cam_y = 0
            env_params.params.debug_cam_target_pos = [0.4, 0, 0.45]

    if prms >> "analog":
        env_params.params.table_z = -0.28
        env_params.params.table_xy_offset = np.array([0., -0.15])  # offset for bound center
        env_params.params.robot_params.reset_q = [-1.43579845e+00, -3.55341683e-01, -1.40604696e-01, -2.69942909e+00, -6.86523834e-02, 2.34899834e+00, 5.10147958e-02]
        # TODO do we need to support different grippers here?
        # env_params.params.robot_params.urdf_file = "franka_panda/panda_regular.urdf"

    if not NO_DSNAP:
        env_params.params.random_init_snap_drawer = True
        env_params.params.random_init_snap_cabinet = True

    if DRAWER:
        env_params.params.use_short_cabinet = prms >> "short_cab"

    if use_mug:
        env_params.params.object_spec = ['mug']  # mug will be the only object.

    common_params[group_name] = common_params[group_name] & env_params

    ## spec changes
    assert common_params >> "env_spec/cls" == ParamEnvSpec, "Only supports param env spec, processed first"
    # if EXTRA:
    #     (common_params >> "env_spec/params/observation_names").extend([])
    #     (common_params >> "env_spec/params/param_names").append('block_colors')

    if IMGS:
        (common_params >> "env_spec/params/observation_names").append('image')
    if NO_BC:
        (common_params >> "env_spec/params/observation_names").remove('objects/contact')
    if NO_DC:
        (common_params >> "env_spec/params/observation_names").remove('drawer/contact')

    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
