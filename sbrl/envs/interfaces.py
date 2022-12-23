"""
Different interfaces that an environment can support, outside of the basic Env.

Classes will sub-class Env and <interfaces...>, so make sure things don't conflict.
"""
import abc


class VRInterface(abc.ABC):
    @abc.abstractmethod
    def get_safenet(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def change_view(self, **kwargs):
        raise NotImplementedError
