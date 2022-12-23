import numpy as np

from sbrl.envs.bullet_envs.block3d.block_env_3d import BlockEnv3D, get_block3d_example_spec_params, \
    get_block3d_example_params
from sbrl.envs.bullet_envs.block3d.platform_block_env_3d import PlatformBlockEnv3D
from sbrl.envs.bullet_envs.block3d.playroom import DrawerPlayroomEnv3D
from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.policies.blocks.block3d_extra_policies import drawer_move_params_fn, DrawerMovePrimitive, \
    mug_grasp_move_policy_params_fn, mug_rot_directional_policy_params_fn, mug_lift_platform_policy_params_fn, \
    cabinet_rot_directional_policy_params_fn, \
    move_cabinet_block_to_freespace_policy_params_fn, put_block_into_cabinet_or_drawer_policy_params_fn, \
    move_drawer_block_to_freespace_policy_params_fn, button_press_params_fn
from sbrl.policies.blocks.block3d_policies import push_policy_params_fn, pull_policy_params_fn, PushPrimitive, \
    PullPrimitive, push_directional_policy_params_fn, pull_directional_policy_params_fn, \
    top_rot_directional_policy_params_fn, TopRotatePrimitive, lift_platform_policy_params_fn, LiftPrimitive
from sbrl.policies.meta_policy import MetaPolicy
from sbrl.utils.python_utils import AttrDict as d


def get_push_pull_meta_policy_params_fn(min_num_collections_per_reset, max_num_collections_per_reset,
                                        min_retreat, max_retreat, do_pull=True, do_push=True, one_dim_only=False,
                                        random_motion=False, directional=False, retreat_first=False, retreat_xy=False,
                                        randomize_offset=False, uniform_velocity=False,
                                        prefer_idxs=(), smooth_noise=0, random_slow_prob=0, stop_prob=0.33):
    axis = 0 if one_dim_only else None
    assert do_push or do_pull

    if directional:
        push_policy_params = push_directional_policy_params_fn
        pull_policy_params = pull_directional_policy_params_fn
    else:
        push_policy_params = push_policy_params_fn
        pull_policy_params = pull_policy_params_fn

    def next_param_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):

        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(min_num_collections_per_reset, max_num_collections_per_reset)

        options = []
        if do_push:
            options.append(0)
        if do_pull:
            options.append(1)
        next_policy_idx = np.random.choice(options)
        if memory.reset_count < memory.max_iters:
            retreat_vel = 0.2
            retreat_steps = np.random.randint(min_retreat, max_retreat)
            if retreat_first and memory.reset_count == 0:
                # disable retreat on the first go
                retreat_steps = 0

            if random_motion:
                # varies the motion a lot more (time and speed)
                move_steps = np.random.randint(10, 25)
                move_vel = np.random.uniform(0.04, 0.14)
            else:
                move_steps = np.random.randint(20, 25)
                move_vel = 0.1

            if next_policy_idx == 0:
                params = push_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                            goal.leaf_apply(lambda arr: arr[:, 0]),
                                            env=env, push_steps=move_steps, push_velocity=move_vel,
                                            retreat_first=retreat_first and memory.reset_count > 0,
                                            retreat_velocity=retreat_vel,
                                            retreat_steps=retreat_steps, retreat_xy=retreat_xy,
                                            uniform_velocity=uniform_velocity, axis=axis, smooth_noise=smooth_noise,
                                            random_slow_prob=random_slow_prob, stop_prob=stop_prob)
            elif next_policy_idx == 1:
                assert do_pull, "Pull was chosen but pull is not allowed."
                params = pull_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                            goal.leaf_apply(lambda arr: arr[:, 0]),
                                            env=env, pull_steps=move_steps, pull_velocity=move_vel,
                                            retreat_first=retreat_first and memory.reset_count > 0,
                                            retreat_velocity=retreat_vel,
                                            retreat_steps=retreat_steps, retreat_xy=retreat_xy,
                                            uniform_velocity=uniform_velocity, axis=axis)
            else:
                raise NotImplementedError
            params.timeout = 120  # 12 seconds max per primitive
            memory.reset_count += 1
        else:
            return None, d()

        return next_policy_idx, params

    return next_param_fn


def get_push_pull_toprot_meta_policy_params_fn(min_num_collections_per_reset, max_num_collections_per_reset,
                                               min_retreat, max_retreat, do_pull=True, do_push=True, do_toprot=True,
                                               one_dim_only=False, retreat_first=False, retreat_xy=False,
                                               random_motion=False, directional=False,
                                               randomize_offset=False,
                                               prefer_idxs=(), smooth_noise=0):
    axis = 0 if one_dim_only else None
    assert do_push or do_pull or do_toprot

    if directional:
        push_policy_params = push_directional_policy_params_fn
        pull_policy_params = pull_directional_policy_params_fn
    else:
        push_policy_params = push_policy_params_fn
        pull_policy_params = pull_policy_params_fn

    top_rot_policy_params = top_rot_directional_policy_params_fn

    def next_param_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):

        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(min_num_collections_per_reset, max_num_collections_per_reset)

        options = []
        if do_push:
            options.append(0)
        if do_pull:
            options.append(1)
        if do_toprot:
            options.append(2)
        next_policy_idx = np.random.choice(options)
        if memory.reset_count < memory.max_iters:
            retreat_steps = np.random.randint(min_retreat, max_retreat)
            if retreat_first and memory.reset_count == 0:
                # disable retreat on the first go
                retreat_steps = 0

            if random_motion:
                # varies the motion a lot more (time and speed)
                move_steps = np.random.randint(10, 25)
                move_vel = np.random.uniform(0.04, 0.14)
            else:
                move_steps = np.random.randint(20, 25)
                move_vel = 0.1

            if next_policy_idx == 0:
                params = push_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                            goal.leaf_apply(lambda arr: arr[:, 0]),
                                            env=env, push_steps=move_steps, push_velocity=move_vel,
                                            retreat_first=retreat_first and memory.reset_count > 0,
                                            retreat_xy=retreat_xy,
                                            retreat_steps=retreat_steps, axis=axis, smooth_noise=smooth_noise)
            elif next_policy_idx == 1:
                assert do_pull, "Pull was chosen but pull is not allowed."
                params = pull_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                            goal.leaf_apply(lambda arr: arr[:, 0]),
                                            env=env, pull_steps=move_steps, pull_velocity=move_vel,
                                            retreat_first=retreat_first and memory.reset_count > 0,
                                            retreat_xy=retreat_xy,
                                            retreat_steps=retreat_steps, axis=axis)
            elif next_policy_idx == 2:
                assert do_toprot, "Top rot was chosen but pull is not allowed."
                rot_dir = np.random.choice([-1, 1])  # 1 means right, -1 is left.
                params = top_rot_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                               goal.leaf_apply(lambda arr: arr[:, 0]),
                                               env=env, rotation_steps=move_steps,
                                               rotation_velocity=rot_dir * move_vel * 7.5,
                                               retreat_first=retreat_first and memory.reset_count > 0,
                                               retreat_xy=retreat_xy, retreat_steps=retreat_steps, axis=axis,
                                               smooth_noise=smooth_noise)
            else:
                raise NotImplementedError
            params.timeout = 120  # 12 seconds max per primitive
            memory.reset_count += 1
        else:
            return None, d()

        return next_policy_idx, params

    return next_param_fn


def get_push_pull_toprot_lift_meta_policy_params_fn(min_num_collections_per_reset, max_num_collections_per_reset,
                                                    min_retreat, max_retreat, do_pull=True, do_push=True,
                                                    do_toprot=True, do_lift=True,
                                                    one_dim_only=False, retreat_first=False, retreat_xy=False,
                                                    random_motion=False, shorter_motion=False, directional=False,
                                                    randomize_offset=False, uniform_velocity=False,
                                                    lift_sample_directions=False, use_mug=False,
                                                    mug_diverse_grasp=False, more_lift=False,
                                                    mug_rot_check_wall=True, less_rot=False,
                                                    prefer_idxs=(), smooth_noise=0):
    assert do_push or do_pull or do_toprot or do_lift
    axis = 0 if one_dim_only else None

    if use_mug:
        assert not do_push, "push isn't implemented"

    if directional:
        push_policy_params = push_directional_policy_params_fn
        pull_policy_params = pull_directional_policy_params_fn
    else:
        push_policy_params = push_policy_params_fn
        pull_policy_params = pull_policy_params_fn

    if use_mug:
        assert not directional, "not implemented..."
        pull_policy_params = mug_grasp_move_policy_params_fn
        top_rot_policy_params = mug_rot_directional_policy_params_fn
        lift_policy_params = mug_lift_platform_policy_params_fn
        extra_kwargs = {'diverse_grasp': mug_diverse_grasp}
    else:
        top_rot_policy_params = top_rot_directional_policy_params_fn
        lift_policy_params = lift_platform_policy_params_fn
        extra_kwargs = {}

    def next_param_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):

        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(min_num_collections_per_reset, max_num_collections_per_reset)

        pos = (obs >> "objects/position")[0, 0, 0]
        size = (obs >> "objects/size")[0, 0, 0]

        if use_mug:
            mug_is_on_platform = pos[2] >= env.platform_z - 0.005
            delta = env.surface_bounds[:2] / 2 - env._platform_extent
            # mug is on edge in one or both of [x,y]. but never none if on platform
            mug_is_on_edge = (pos[:2] > env.surface_center[:2] + delta + size[2] * 3 / 8) | (
                    pos[:2] < env.surface_center[:2] - delta - size[2] * 3 / 8)
            mug_is_on_edge = mug_is_on_edge[::-1]  # if past x_max, its on the y edge
            if not mug_is_on_edge.any() and mug_is_on_platform:
                logger.warn("Mug is not on platform, but is past or on an edge!!!")
                return None, d()

        options = []
        weights = []
        if do_push:
            options.append(0)
            weights.append(1.)
        if do_pull:
            options.append(1)
            weights.append(2. if use_mug else 1.)
        if do_toprot and (mug_rot_check_wall or not env.is_obj_contacting_walls(env.objects[0].id)):
            options.append(2)
            weights.append(0.5 if less_rot else 1.0)
        if do_lift:
            options.append(3)
            mw = 1.5 if more_lift else 0.75
            weights.append(mw if use_mug else 0.5)  # half as likely to lift than others

        p = np.asarray(weights)
        p /= p.sum()
        next_policy_idx = np.random.choice(options, p=p)

        is_upright = env.is_obj_upright(0)  # idx

        if memory.reset_count < memory.max_iters and is_upright:
            retreat_steps = np.random.randint(min_retreat, max_retreat)
            if retreat_first and memory.reset_count == 0:
                # disable retreat on the first go
                retreat_steps = 0

            if random_motion:
                # varies the motion a lot more (time and speed)
                move_steps = np.random.randint(10, 25)
                move_vel = np.random.uniform(0.075, 0.15)
            else:
                move_steps = np.random.randint(20, 25)
                move_vel = 0.1

            if shorter_motion and next_policy_idx != 2:
                move_steps = move_steps // 1.5  # cut the range in half, for things that aren't top rot

            # logger.debug(f"NEXT POLICY IDX: {next_policy_idx} from choices: {options}, weights: {weights}")

            if next_policy_idx == 0:
                assert do_push
                params = push_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                            goal.leaf_apply(lambda arr: arr[:, 0]),
                                            env=env, push_steps=move_steps, push_velocity=move_vel,
                                            retreat_first=retreat_first and memory.reset_count > 0,
                                            retreat_xy=retreat_xy, with_platform=do_lift,
                                            uniform_velocity=uniform_velocity,
                                            retreat_steps=retreat_steps, axis=axis, smooth_noise=smooth_noise,
                                            **extra_kwargs)
            elif next_policy_idx == 1:
                assert do_pull, "Pull was chosen but pull is not allowed."
                new_axis = axis
                if use_mug and mug_is_on_platform:
                    allowed_axes = mug_is_on_edge.nonzero()[0]
                    if new_axis is None or not allowed_axes[new_axis]:
                        new_axis = np.random.choice(allowed_axes)
                params = pull_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                            goal.leaf_apply(lambda arr: arr[:, 0]),
                                            env=env, pull_steps=move_steps, pull_velocity=move_vel,
                                            retreat_first=retreat_first and memory.reset_count > 0,
                                            retreat_xy=retreat_xy, uniform_velocity=uniform_velocity,
                                            retreat_steps=retreat_steps, axis=new_axis,
                                            **extra_kwargs)  # TODO smoothing noise
            elif next_policy_idx == 2:
                assert do_toprot, "Top rot was chosen but is not allowed."
                params = d(rotation_velocity=0.5, rotation_steps=0)
                i = 0
                while abs((params >> "rotation_velocity") * (params >> "rotation_steps")) * env.dt < np.deg2rad(
                        15) and i <= 5:
                    rot_dir = np.random.choice([-1, 1])  # 1 means right, -1 is left.
                    params = top_rot_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                                   goal.leaf_apply(lambda arr: arr[:, 0]),
                                                   env=env, rotation_steps=move_steps,
                                                   rotation_velocity=rot_dir * move_vel * 7.5,
                                                   retreat_first=retreat_first and memory.reset_count > 0,
                                                   uniform_velocity=uniform_velocity, stop_at_wall=mug_rot_check_wall,
                                                   retreat_xy=retreat_xy, retreat_steps=retreat_steps, axis=axis,
                                                   smooth_noise=smooth_noise, **extra_kwargs)  # TODO test smooth noise
                    i += 1
            elif next_policy_idx == 3:
                assert do_lift, "Lift was chosen but lift is not allowed."
                params = lift_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                            goal.leaf_apply(lambda arr: arr[:, 0]),
                                            env=env, lift_velocity=move_vel * 1.25,
                                            retreat_first=retreat_first and memory.reset_count > 0,
                                            retreat_xy=retreat_xy, uniform_velocity=uniform_velocity,
                                            retreat_steps=retreat_steps, axis=axis,
                                            sample_directions=lift_sample_directions, smooth_noise=smooth_noise,
                                            **extra_kwargs)
            else:
                raise NotImplementedError
            params.timeout = 120  # 12 seconds max per primitive
            memory.reset_count += 1
        else:
            return None, d()

        return next_policy_idx, params

    return next_param_fn


def get_push_toprot_drawer_cab_meta_policy_params_fn(min_num_collections_per_reset, max_num_collections_per_reset,
                                                     min_retreat, max_retreat, do_push=True,
                                                     do_toprot=True, do_drawer=True, do_cab=False,
                                                     one_dim_only=False, retreat_first=False, retreat_xy=False,
                                                     random_motion=False, directional=False, safe_open=False,
                                                     diverse=False,
                                                     prefer_idxs=(), smooth_noise=0):
    axis = 0 if one_dim_only else None
    assert do_push or do_toprot or do_drawer or do_cab

    if directional:
        push_policy_params = push_directional_policy_params_fn
    else:
        push_policy_params = push_policy_params_fn

    top_rot_policy_params = top_rot_directional_policy_params_fn
    drawer_policy_params = drawer_move_params_fn

    cab_rot_policy_params = cabinet_rot_directional_policy_params_fn

    def next_param_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):

        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(min_num_collections_per_reset, max_num_collections_per_reset)

        options = []
        weights = []
        if do_push:
            options.append(0)
            weights.append(1.)
        if do_toprot:
            options.append(2)
            weights.append(0.5)
        if do_cab:
            options.append(4)
            weights.append(1.)
        if do_drawer:
            options.append(5)
            weights.append(1.)

        p = np.asarray(weights)
        p /= p.sum()
        next_policy_idx = np.random.choice(options, p=p)

        if memory.reset_count < memory.max_iters:
            retreat_steps = np.random.randint(min_retreat, max_retreat)
            if retreat_first and memory.reset_count == 0:
                # disable retreat on the first go
                retreat_steps = 0

            if random_motion:
                # varies the motion a lot more (time and speed)
                move_steps = np.random.randint(10, 25)
                move_vel = np.random.uniform(0.075, 0.15)
            else:
                move_steps = np.random.randint(20, 25)
                move_vel = 0.1

            if next_policy_idx == 0:
                params = push_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                            goal.leaf_apply(lambda arr: arr[:, 0]),
                                            env=env, push_steps=move_steps, push_velocity=move_vel,
                                            retreat_first=retreat_first and memory.reset_count > 0,
                                            retreat_xy=retreat_xy,
                                            uniform_velocity=True, soft_bound=True, pitch=-np.pi / 7,
                                            retreat_steps=retreat_steps, axis=axis, smooth_noise=smooth_noise)
            elif next_policy_idx == 2:
                assert do_toprot, "Top rot was chosen but is not allowed."
                rot_dir = np.random.choice([-1, 1])  # 1 means right, -1 is left.
                params = top_rot_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                               goal.leaf_apply(lambda arr: arr[:, 0]),
                                               env=env, rotation_steps=move_steps,
                                               rotation_velocity=rot_dir * move_vel * 7.5,
                                               retreat_first=retreat_first and memory.reset_count > 0,
                                               uniform_velocity=True,
                                               retreat_xy=retreat_xy, retreat_steps=retreat_steps, axis=axis,
                                               smooth_noise=smooth_noise)
            elif next_policy_idx == 4:
                assert do_cab, "Cabinet rot was chosen but is not allowed."
                # will open if closed, and vv
                params = cab_rot_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                               goal.leaf_apply(lambda arr: arr[:, 0]),
                                               env=env, rotation_steps=move_steps,
                                               rotation_velocity=move_vel * 7.5,
                                               retreat_first=retreat_first and memory.reset_count > 0,
                                               uniform_velocity=True, do_partial=False, safe_open=safe_open,
                                               diverse=diverse,
                                               retreat_xy=retreat_xy, retreat_steps=retreat_steps, axis=axis,
                                               smooth_noise=smooth_noise)
            elif next_policy_idx == 5:
                assert do_drawer, "Drawer move was chosen but Drawer is not allowed."
                params = drawer_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                              goal.leaf_apply(lambda arr: arr[:, 0]),
                                              env=env)
            else:
                raise NotImplementedError
            params.timeout = 120  # 12 seconds max per primitive
            memory.reset_count += 1
        else:
            return None, d()

        return next_policy_idx, params

    return next_param_fn


def get_sequential_push_toprot_drawer_cab_meta_policy_params_fn(min_num_collections_per_reset,
                                                                max_num_collections_per_reset,
                                                                min_retreat, max_retreat, do_push=True,
                                                                do_toprot=True, do_object_move=True, do_drawer=True,
                                                                do_cabinet=True, retreat_first=False, retreat_xy=False,
                                                                random_motion=False, directional=False, safe_open=False,
                                                                diverse=False, do_buttons=False,
                                                                use_cab_waypoints=False, pull_xy_tol=False,
                                                                prefer_idxs=(), smooth_noise=0):
    """
    Sequences behaviors in the DrawerPlayroom3D environment.

    1) Open Drawer: when drawer is closed
    2) Close Drawer: when drawer is open
    3) Open Cabinet: when cabinet is closed
    4) Close Cabinet: when cabinet is open
    5) Move Block free space -> drawer: when block is in free space and drawer is open
    6) Move Block free space -> cabinet: when block is in free space and cabinet is open
    7) Move Block cabinet -> free space: when block is in cabinet and cabinet is open
    8) Move Block drawer -> free space: when block is in drawer and drawer is open
    9) (optional) Sweep Block (along table x/y): when block is in free space
    10) (optional) Rotate Block: when block is in free space TODO
    """

    if do_toprot:
        raise NotImplementedError

    if directional:
        push_policy_params = push_directional_policy_params_fn
    else:
        push_policy_params = push_policy_params_fn

    top_rot_policy_params = top_rot_directional_policy_params_fn
    drawer_policy_params = drawer_move_params_fn

    cab_rot_policy_params = cabinet_rot_directional_policy_params_fn

    def next_param_fn(policy_idx: int, model: Model, obs: d, goal: d, memory: d = d(), env=None, **kwargs):

        # computing preconditions
        block_pos = (obs >> "objects/position")[0, 0, 0]
        drawer_open = (obs >> "drawer/joint_position_normalized")[0, 0, 0] > 0.5
        cabinet_open = (obs >> "cabinet/joint_position_normalized")[0, 0, 0] > 0.75
        block_in_free_space = env.is_in_soft_bounds((obs >> "objects/position")[0, 0, 0])
        block_in_cabinet = env.is_in_cabinet(block_pos)
        block_in_drawer = env.is_in_drawer(block_pos)
        block_stuck_under_table = block_in_drawer and block_pos[1] <= env.table_aabb[
            1, 1] + 0.02  # max_y of table + margin

        if not memory.has_leaf_key("reset_count"):
            memory.reset_count = 0
            memory.max_iters = np.random.randint(min_num_collections_per_reset, max_num_collections_per_reset)

        options = []
        weights = []
        if block_in_free_space:
            if do_push:
                options.append("push")
                weights.append(0.5)
            if do_toprot:
                options.append("toprot")
                weights.append(0.5)

            if do_object_move and drawer_open:
                options.append("free2drawer")
                weights.append(2.)
            if do_object_move and cabinet_open:
                options.append("free2cabinet")
                weights.append(2.)

        elif do_object_move and block_in_drawer and drawer_open and not block_stuck_under_table:
            options.append("drawer2free")
            weights.append(2.)

        elif do_object_move and block_in_cabinet and cabinet_open:
            options.append("cabinet2free")
            weights.append(2.)

        if do_drawer:
            options.append("drawer_move")
            weights.append(1. - 0.7 * int(policy_idx == 5))  # less likely if you just did it
        if do_cabinet:
            options.append("cabinet_move")
            weights.append(1. - 0.7 * int(policy_idx == 4))

        if do_buttons and cabinet_open and not block_in_cabinet:
            options.append("button_press")
            weights.append(1.25)

        # print(options, block_stuck_under_table, block_pos, env.table_aabb[1])

        # center = 0.5 * (env.table_aabb[0] + env.table_aabb[1])
        # lens = (env.table_aabb[1] - env.table_aabb[0])

        # import pybullet as p
        # p.addUserDebugLine(center + np.array([lens[0] / 2, lens[1] / 2, 0]), center + np.array([lens[0] / 2, -lens[1] / 2, 0]))
        # p.addUserDebugLine(center + np.array([lens[0] / 2, -lens[1] / 2, 0]), center + np.array([-lens[0] / 2, -lens[1] / 2, 0]))
        # p.addUserDebugLine(center + np.array([-lens[0] / 2, -lens[1] / 2, 0]), center + np.array([-lens[0] / 2, lens[1] / 2, 0]))
        # p.addUserDebugLine(center + np.array([-lens[0] / 2, lens[1] / 2, 0]), center + np.array([lens[0] / 2, lens[1] / 2, 0]))

        if memory.reset_count >= memory.max_iters or len(options) == 0:
            return None, d()  # terminate on no options left.

        p = np.asarray(weights)
        p /= p.sum()
        next_policy_name = np.random.choice(options, p=p)
        logger.debug(f"Next policy: {next_policy_name}")

        retreat_steps = np.random.randint(min_retreat, max_retreat)
        if retreat_first and memory.reset_count == 0:
            # disable retreat on the first go
            retreat_steps = 0

        if random_motion:
            # varies the motion a lot more (time and speed)
            move_steps = np.random.randint(10, 25)
            move_vel = np.random.uniform(0.075, 0.15)
        else:
            move_steps = np.random.randint(20, 25)
            move_vel = 0.1

        if next_policy_name == "push":  # freespace push
            next_policy_idx = 0
            params = push_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                        goal.leaf_apply(lambda arr: arr[:, 0]),
                                        env=env, push_steps=move_steps, push_velocity=move_vel,
                                        retreat_first=retreat_first and memory.reset_count > 0,
                                        retreat_xy=retreat_xy,
                                        uniform_velocity=True, soft_bound=True, pitch=-np.pi / 20,
                                        retreat_steps=int(retreat_steps / 1.5), axis=0,
                                        smooth_noise=smooth_noise)  # push only along x to avoid hitting table.
        elif next_policy_name == "toprot":  # freespace rotate
            rot_dir = np.random.choice([-1, 1])  # 1 means right, -1 is left.
            next_policy_idx = 2
            params = top_rot_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                           goal.leaf_apply(lambda arr: arr[:, 0]),
                                           env=env, rotation_steps=move_steps,
                                           rotation_velocity=rot_dir * move_vel * 7.5,
                                           retreat_first=retreat_first and memory.reset_count > 0,
                                           uniform_velocity=True,
                                           retreat_xy=retreat_xy, retreat_steps=int(retreat_steps / 1.5),
                                           smooth_noise=smooth_noise)
        elif next_policy_name == "drawer_move":
            next_policy_idx = 5
            params = drawer_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                          goal.leaf_apply(lambda arr: arr[:, 0]),
                                          do_partial=False,  # either open or close, depending on current state.
                                          env=env)
        elif next_policy_name == "cabinet_move":
            next_policy_idx = 4
            params = cab_rot_policy_params(obs.leaf_apply(lambda arr: arr[:, 0]),
                                           goal.leaf_apply(lambda arr: arr[:, 0]),
                                           env=env, rotation_steps=move_steps,
                                           rotation_velocity=move_vel * 7.5,
                                           retreat_first=retreat_first and memory.reset_count > 0,
                                           uniform_velocity=True, do_partial=False, safe_open=safe_open,
                                           diverse=diverse,
                                           retreat_xy=retreat_xy, retreat_steps=retreat_steps,
                                           smooth_noise=smooth_noise)

        ## freespace motions
        elif next_policy_name in ["free2cabinet", "free2drawer"]:
            next_policy_idx = 1
            params = put_block_into_cabinet_or_drawer_policy_params_fn(obs.leaf_apply(lambda arr: arr[:, 0]),
                                                                       goal.leaf_apply(lambda arr: arr[:, 0]),
                                                                       drawer="drawer" in next_policy_name,
                                                                       env=env, pull_steps=move_steps,
                                                                       pull_velocity=move_vel,
                                                                       retreat_steps=retreat_steps,
                                                                       retreat_first=retreat_first,
                                                                       use_cab_waypoints=use_cab_waypoints,
                                                                       xy_tolerance=pull_xy_tol, uniform_velocity=True)

        elif next_policy_name == "cabinet2free":
            next_policy_idx = 1
            params = move_cabinet_block_to_freespace_policy_params_fn(obs.leaf_apply(lambda arr: arr[:, 0]),
                                                                      goal.leaf_apply(lambda arr: arr[:, 0]),
                                                                      env=env, pull_steps=move_steps,
                                                                      pull_velocity=move_vel,
                                                                      retreat_steps=retreat_steps,
                                                                      retreat_first=retreat_first,
                                                                      use_cab_waypoints=use_cab_waypoints,
                                                                      xy_tolerance=pull_xy_tol, uniform_velocity=True)

        elif next_policy_name == "drawer2free":
            next_policy_idx = 3
            params = move_drawer_block_to_freespace_policy_params_fn(obs.leaf_apply(lambda arr: arr[:, 0]),
                                                                     goal.leaf_apply(lambda arr: arr[:, 0]),
                                                                     env=env, lift_steps=move_steps,
                                                                     lift_velocity=move_vel,
                                                                     retreat_steps=retreat_steps,
                                                                     retreat_first=retreat_first, uniform_velocity=True)
        elif next_policy_name == "button_press":  # press a button
            next_policy_idx = 7
            params = button_press_params_fn(obs.leaf_apply(lambda arr: arr[:, 0]),
                                            goal.leaf_apply(lambda arr: arr[:, 0]),
                                            env=env, uniform_velocity=True)

        else:
            raise NotImplementedError

        params.timeout = 120  # 12 seconds max per primitive
        memory.reset_count += 1

        return next_policy_idx, params

    return next_param_fn


def test_meta():
    from sbrl.envs.param_spec import ParamEnvSpec

    platform = True
    drawer = False
    mug = True

    do_drawer = False
    do_cab = False
    rotate = True
    lift = True
    directional = False
    no_pull = False
    no_push = True
    retreat_first = False
    uniform_vel = True
    lift_sample_directions = True
    smooth_noise = 0

    mmr = [5, 12]
    # mmp = [1, 2]  # [1, 2]
    mmp = [3, 6]  # [1, 2]
    env_spec_params = get_block3d_example_spec_params()
    env_params = get_block3d_example_params()
    env_params.render = True
    env_params.do_random_ee_position = True

    if mug:
        env_params.object_spec = ['mug']
    # env_params.block_size = (30, 30)

    # ROTATE starts horizontal
    # if rotate and not use_meta:
    #     # flip the block
    #     env_params = get_stack_block2d_example_params(block_max_size=(80, 40))

    env_spec = ParamEnvSpec(env_spec_params)
    assert not (drawer and platform)
    if drawer:
        block_env = DrawerPlayroomEnv3D(env_params, env_spec)
    elif platform:
        block_env = PlatformBlockEnv3D(env_params, env_spec)
    else:
        block_env = BlockEnv3D(env_params, env_spec)

    # env presets
    presets = d()

    model = Model(d(ignore_inputs=True), env_spec, None)

    # cv2.namedWindow("image_test", cv2.WINDOW_AUTOSIZE)

    obs, goal = block_env.user_input_reset(1)  # trolling with a fake UI
    print(obs >> "ee_position")

    prms = d()
    if uniform_vel:
        prms.max_pos_vel = 0.4

    if drawer:
        assert not lift and no_pull
        assert uniform_vel
        assert not retreat_first
        get_next_policy_params = get_push_toprot_drawer_cab_meta_policy_params_fn(*mmp, *mmr, do_push=not no_push,
                                                                                  do_toprot=rotate, do_drawer=do_drawer,
                                                                                  do_cab=do_cab,
                                                                                  retreat_first=retreat_first,
                                                                                  retreat_xy=False,
                                                                                  directional=directional)

        all_policies = [d(cls=PushPrimitive, params=prms), d(cls=PullPrimitive, params=prms),
                        d(cls=TopRotatePrimitive, params=prms), d(cls=LiftPrimitive, params=prms),
                        d(cls=TopRotatePrimitive, params=prms), d(cls=DrawerMovePrimitive, params=prms)]

    elif mug or (rotate or lift):
        get_next_policy_params = get_push_pull_toprot_lift_meta_policy_params_fn(*mmp, *mmr, randomize_offset=True,
                                                                                 retreat_first=retreat_first,
                                                                                 do_pull=not no_pull,
                                                                                 do_push=not no_push,
                                                                                 do_toprot=rotate,
                                                                                 do_lift=lift,
                                                                                 uniform_velocity=uniform_vel,
                                                                                 directional=directional,
                                                                                 lift_sample_directions=lift_sample_directions,
                                                                                 smooth_noise=smooth_noise,
                                                                                 use_mug=mug)
        all_policies = [d(cls=PushPrimitive, params=prms), d(cls=PullPrimitive, params=prms),
                        d(cls=TopRotatePrimitive, params=prms), d(cls=LiftPrimitive, params=prms)]
    else:
        get_next_policy_params = get_push_pull_meta_policy_params_fn(*mmp, *mmr, randomize_offset=True,
                                                                     retreat_first=retreat_first,
                                                                     uniform_velocity=uniform_vel,
                                                                     do_pull=not no_pull,
                                                                     do_push=not no_push, directional=directional,
                                                                     smooth_noise=smooth_noise)
        all_policies = [d(cls=PushPrimitive, params=prms), d(cls=PullPrimitive, params=prms)]

    policy = MetaPolicy(d(all_policies=all_policies,
                          next_param_fn=get_next_policy_params), env_spec, env=block_env)

    policy.reset_policy(next_obs=obs, next_goal=goal)

    iters = 30
    gamma = 0.5
    last_act = np.zeros(3)
    act = np.zeros(3)
    i = 0
    while i < iters:
        act = policy.get_action(model, obs.leaf_apply(lambda arr: arr[:, None]),
                                goal.leaf_apply(lambda arr: arr[:, None]))

        obs, goal, done = block_env.step(act)
        # logger.debug(f"pos: {obs.ee_position} | ori: {obs.ee_orientation_eul}")
        logger.debug(f"pos: {act.target.ee_position} | ori: {act.target.ee_orientation_eul}")
        logger.debug(f"contact: {obs.objects.contact}")
        if np.any(done) or policy.is_terminated(model, obs, goal):
            logger.debug("Policy terminated, resetting")
            obs, goal = block_env.reset(presets)
            policy.reset_policy(next_obs=obs, next_goal=goal)
            i += 1


def test_playroom():
    from sbrl.envs.param_spec import ParamEnvSpec

    rotate = False
    retreat_first = False
    uniform_vel = True
    lift_sample_directions = True
    smooth_noise = 0

    mmr = [5, 12]
    # mmp = [1, 2]  # [1, 2]
    # mmp = [3, 6]  # [1, 2]
    mmp = [4, 10]  # [1, 2]
    env_spec_params = get_block3d_example_spec_params()
    env_params = get_block3d_example_params()
    env_params.render = True
    env_params.do_random_ee_position = True
    env_params.random_init_snap_drawer = False
    env_params.random_init_snap_cabinet = False

    env_params.debug_cam_dist = 0.7
    env_params.debug_cam_p = -45
    env_params.debug_cam_y = 120
    env_params.debug_cam_target_pos = [0.4, 0, 0.45]

    # env_params.block_size = (30, 30)

    # ROTATE starts horizontal
    # if rotate and not use_meta:
    #     # flip the block
    #     env_params = get_stack_block2d_example_params(block_max_size=(80, 40))

    env_spec = ParamEnvSpec(env_spec_params)
    block_env = DrawerPlayroomEnv3D(env_params, env_spec)
    # env presets
    presets = d()

    model = Model(d(ignore_inputs=True), env_spec, None)

    # cv2.namedWindow("image_test", cv2.WINDOW_AUTOSIZE)

    obs, goal = block_env.user_input_reset(1)  # trolling with a fake UI
    print(obs >> "ee_position")

    prms = d()
    if uniform_vel:
        prms.max_pos_vel = 0.4

    get_next_policy_params = get_sequential_push_toprot_drawer_cab_meta_policy_params_fn(*mmp, *mmr, do_push=True,
                                                                                         do_toprot=rotate,
                                                                                         retreat_first=retreat_first,
                                                                                         retreat_xy=False)

    all_policies = [d(cls=PushPrimitive, params=prms), d(cls=PullPrimitive, params=prms),
                    d(cls=TopRotatePrimitive, params=prms), d(cls=LiftPrimitive, params=prms),
                    d(cls=TopRotatePrimitive, params=prms), d(cls=DrawerMovePrimitive, params=prms)]

    policy = MetaPolicy(d(all_policies=all_policies,
                          next_param_fn=get_next_policy_params), env_spec, env=block_env)

    policy.reset_policy(next_obs=obs, next_goal=goal)

    iters = 30
    gamma = 0.5
    last_act = np.zeros(3)
    act = np.zeros(3)
    i = 0
    while i < iters:
        act = policy.get_action(model, obs.leaf_apply(lambda arr: arr[:, None]),
                                goal.leaf_apply(lambda arr: arr[:, None]))

        obs, goal, done = block_env.step(act)
        # logger.debug(f"pos: {obs.ee_position} | ori: {obs.ee_orientation_eul}")
        logger.debug(f"pos: {act.target.ee_position} | ori: {act.target.ee_orientation_eul}")
        # logger.debug(f"contact: {obs.objects.contact}")
        if np.any(done) or policy.is_terminated(model, obs, goal):
            logger.debug("Policy terminated, resetting")
            obs, goal = block_env.reset(presets)
            policy.reset_policy(next_obs=obs, next_goal=goal)
            i += 1


if __name__ == '__main__':
    test_meta()
    # test_playroom()
