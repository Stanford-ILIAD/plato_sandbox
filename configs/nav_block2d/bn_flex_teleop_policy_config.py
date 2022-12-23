"""
"""

from argparse import ArgumentParser

# declares this group's parser, and defines any sub groups we need
import torch

from sbrl.envs.block2d.block_env_2d import BlockEnv2D
from sbrl.envs.block2d.teleop_functions import get_pygame_mouse_teleop_fn
from sbrl.policies.teleop_policy import TeleopPolicy
from sbrl.utils import plt_utils
from sbrl.utils.input_utils import UserInput, KeyInput as KI
from sbrl.utils.python_utils import AttrDict


def declare_arguments(parser=ArgumentParser()):
    parser.add_argument('--timeout', type=int, default=0, help="max number of steps per episode")
    parser.add_argument('--num_buttons', type=int, default=3, choices=[3, 5], help="num buttons on mouse.")
    return parser


def process_params(group_name, common_params):
    assert 'policy' in group_name

    NB = common_params >> f"{group_name}/num_buttons"



    def get_teleop_model_forward_fn(env, ui: UserInput):

        assert isinstance(env, BlockEnv2D), "Env must be 2D block!"

        base_get_model_forward_fn = get_pygame_mouse_teleop_fn(posact_to_vel_fn=env.posact_to_vel_fn if hasattr(env, "posact_to_vel_fn") else None)
        base_model_forward_fn = base_get_model_forward_fn(env, ui)

        # draws the click state with text in upper left.
        def draw_click_state(ops):
            if env.extra_memory >> "click_state" >= 1:
                tcolor = (plt_utils.light_orange[:3] * 255).astype(int).tolist()
                if env.extra_memory.click_state == 1:
                    env.screen.blit(env.font.render(f"CLICK", False, tuple(tcolor)), (25, 25))
                else:
                    env.screen.blit(env.font.render(f"PRESS", False, tuple(tcolor)), (25, 25))

        extra_draw_actions = [draw_click_state]

        ui.register_callback(KI('space', KI.ON.pressed), lambda ui, ki: None)

        # register
        def model_forward_fn(model, obs: AttrDict, goal: AttrDict, user_input_state=None):
            if not env.extra_memory.has_leaf_key('last_key_down'):
                # detecting resets
                env.extra_memory.last_key_down = False
                env.extra_memory.continuous_clicks = 0
                env.extra_memory.click_state = 0
                env.set_draw_shapes(extra_draw_actions)  # has to be on reset.

            base_ac = base_model_forward_fn(model, obs, goal, user_input_state)
            click_state = 0  # no press

            if user_input_state is None:
                user_input_state = ui.read_input()

            if 'space' in user_input_state.keys() and KI.ON.pressed in user_input_state['space']:
                if env.extra_memory.last_key_down:
                    click_state = 2  # done pressing
                else:
                    click_state = 1  #
                env.extra_memory.last_key_down = True
                env.extra_memory.continuous_clicks += 1
            else:
                env.extra_memory.last_key_down = False
                env.extra_memory.continuous_clicks = 0

            # (B,1) int, B = 1 always for teleop
            base_ac.last_click_state = torch.tensor([[env.extra_memory >> "click_state"]], dtype=torch.int, device=model.device)
            base_ac.click_state = torch.tensor([[click_state]], dtype=torch.uint8, device=model.device)
            base_ac.num_clicks = torch.tensor([[env.extra_memory >> "continuous_clicks"]], dtype=torch.int, device=model.device)

            env.extra_memory.click_state = click_state
            return base_ac

        return model_forward_fn

    # check for all relevant params first (might be defined by other files)
    # then "compile" these into the params that this file defines
    common_params = common_params.leaf_copy()
    common_params[group_name] = common_params[group_name] & AttrDict(
        cls=TeleopPolicy,
        params=AttrDict(get_teleop_model_forward_fn=get_teleop_model_forward_fn,
                        timeout=common_params >> f"{group_name}/timeout")
    )
    return common_params


params = AttrDict(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
