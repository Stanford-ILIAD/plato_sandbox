import numpy as np

from sbrl.utils.python_utils import AttrDict as d


class Controller(object):

    def __init__(self, params: d = d()):
        # # all the robots you want to control
        # self.memory = d()

        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params):
        # action class
        # self.action_type: ActionType = get_required(params, "action_type")
        pass

    # def get_state(self, name: str, required=False):
    #     if self.memory.has_leaf_key(name):
    #         return self.memory[name]
    #     elif required:
    #         raise ValueError("Controller<%s> memory does not contain key %s" % (type(self), name))
    #     else:
    #         return None

    def reset(self, presets: d = d()):
        """
        Should be called when controller state should be reset or cleared

        Parameters
        ----------
        presets: things you might care about resetting between controller runs

        Returns
        -------

        """
        # self.memory.clear()
        pass

    def forward(self, action: np.ndarray, **kwargs) -> np.ndarray:
        """
        takes high level "action" array and computes low level "control" array

        Parameters
        ----------
        action: comes from some high level, action for the robot

        Returns the low level actions specified by the type of controller
        -------

        """
        raise NotImplementedError

    def get_obs(self) -> d:
        return d()
