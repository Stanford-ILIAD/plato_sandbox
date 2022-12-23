from enum import Enum

import numpy as np

from sbrl.experiments import logger
from sbrl.policies.controllers.controller import Controller
from sbrl.utils.python_utils import AttrDict as d, get_required, get_with_default


class PIDController(Controller):

    def _init_params_to_attrs(self, params):
        self._type: ControlType = get_with_default(params, "type", ControlType.PID)
        # must have at least K_p & dt

        self._dim = get_required(params, "dim")
        self._dt = get_required(params, "dt")
        self._k_p = np.broadcast_to(get_with_default(params, "k_p", 1.0), self._dim)
        self._k_i = np.broadcast_to(get_with_default(params, "k_i", 0), self._dim)
        self._k_d = np.broadcast_to(get_with_default(params, "k_d", 0), self._dim)

        # integral window 2s default
        self._int_span = get_with_default(params, "int_span", 2., map_fn=float)
        self._int_steps = int(np.round(self._int_span / self._dt))

        # D checks
        if np.count_nonzero(self._k_d) > 0 and self._type in [ControlType.PI or ControlType.P]:
            logger.warn("non-zero Kd specified for non <PI>D mode")
        # I checks
        if np.count_nonzero(self._k_i) > 0 and self._type in [ControlType.PD or ControlType.P]:
            logger.warn("non-zero Ki specified for non <P>I<D> mode")

        # might for example, want a circular difference (e.g. orientation control), key dependent
        self._difference_fn = get_with_default(params, "difference_fn", lambda a, b: a - b)

        # for chaining ops
        self._norm_tolerance = get_with_default(params, "norm_tolerance", 1e-3)
        self.raw_int_err = np.zeros(self._dim)
        self.int_err = np.zeros(self._dim)
        self.old_int_err = [np.zeros(self._dim) for _ in range(self._int_steps)]
        self.d_err = np.zeros(self._dim)
        self.last_err = np.zeros(self._dim)
        self.last_target = np.zeros(self._dim)

    def reset(self, presets: d = d()):
        super(PIDController, self).reset(presets)
        self.raw_int_err[:] = 0
        self.int_err[:] = 0
        for i in range(self._int_steps):
            self.old_int_err[i][:] = 0
        self.last_err[:] = 0
        self.last_target[:] = 0

    def forward(self, inputs: d, err=None, **kwargs) -> np.ndarray:
        des, curr = inputs.get_keys_required(['desired', 'current'])
        assert len(des) == self._dim
        assert len(curr) == self._dim

        if err is None:
            err = self._difference_fn(des, curr)

        # time window integral
        self.raw_int_err += err * self._dt
        # buffered
        self.old_int_err.append(self.raw_int_err.copy())
        # integral from (t - int_span) -> (t)
        self.int_err[:] = self.raw_int_err - self.old_int_err.pop(0)  # baseline by the old integral error

        self.d_err = (err - self.last_err) / self._dt

        self.last_err[:] = err
        self.last_target[:] = des

        return self._k_p * err + self._k_i * self.int_err + self._k_d * self.d_err

    def get_obs(self):
        # err, err_dot, err_int
        return d(
            target=self.last_target.copy(),
            err=self.last_err.copy(),
            err_int=self.int_err.copy(),
            err_int_raw=self.raw_int_err.copy(),
            err_dot=self.d_err.copy(),
        )

class ControlType(Enum):
    # controller types
    P = 0
    PD = 1
    PI = 2
    PID = 3
