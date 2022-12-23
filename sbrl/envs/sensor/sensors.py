"""
This class holds sensor interfaces that might be used in real world environments
"""

from sbrl.utils.python_utils import AttrDict


class Sensor(object):

    def __init__(self, params: AttrDict):
        self._init_params_to_attrs(params)
        self.has_setup = False

    # this should be called externally, only once
    def open(self):
        assert not self.has_setup
        self._init_setup()
        self.has_setup = True

    def __del__(self):
        self.close()

    """ OVERRIDES """

    # parameters
    def _init_params_to_attrs(self, params: AttrDict):
        pass

    # sensor loading, override
    def _init_setup(self, **kwargs):
        pass

    def read_state(self, **kwargs) -> AttrDict:
        pass

    def reset(self, **kwargs):
        pass

    def close(self):
        pass
