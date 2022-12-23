"""
This is where actions actually get executed on the robot, and observations are received.

NOTE: Why is there no reward? Are we even doing reinforcement learning???
      The reward is just another observation! Viewing it this way is much more flexible,
      especially with model-based RL

"""
from typing import Callable

from sbrl.utils.input_utils import UserInput
from sbrl.utils.python_utils import AttrDict


class Env(object):
    # is this environment capable of handling ui input on its own (e.g. reset and quit triggers)
    step_ui_compatible = False

    def __init__(self, params, env_spec):
        self._env_spec = env_spec
        self._display = None  # might get filled later
        self._params = params

        self.user_input = None  # might get filled later

    def step(self, action):
        """
        :param action: AttrDict (B x ...)

        :return obs, goal, done: AttrDict (B x ...)
        """
        raise NotImplementedError

    def reset(self, presets: AttrDict = AttrDict()):
        """
        presets: some episodes support this
         - disable_user: Bool, if you are using

         returns the next episode obs, goal.
        """
        raise NotImplementedError

    def user_input_reset(self, user_input: UserInput, reset_action_fn=None, presets: AttrDict = AttrDict()):
        """
        :param user_input: UserInput object to query state from
        :param reset_action_fn: Optional external function to run at some point in user input reset
            default behavior is to call it before calling reset
        :param presets: presets for reset
        Used when we want user input in the loop during the reset. default is just reset
        """
        self.user_input = user_input
        if isinstance(reset_action_fn, Callable):
            reset_action_fn()
        return self.reset(presets)

    @property
    def env_spec(self):
        return self._env_spec

    @property
    def display(self):
        return self._display

    @property
    def params(self):
        return self._params.leaf_copy()

    def is_success(self) -> bool:
        return False