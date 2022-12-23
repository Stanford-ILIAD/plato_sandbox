"""
The policy may or may not use a model. Basic behavior follows a setpoint with a PID controller
"""
from numbers import Number

import numpy as np

from sbrl.policies.policy import Policy
from sbrl.utils.python_utils import get_required, get_with_default, AttrDict


class PIDPolicy(Policy):

    def _init_params_to_attrs(self, params):
        # self.sample_action = params.sample_action
        self._control_names = list(get_required(params, "control_names"))
        self._control_shapes = [self._env_spec.names_to_shapes[n] for n in self._control_names]
        # must have at least K_p & dt
        self._dt = get_required(params, "dt")
        self._k_p = get_required(params, "k_p")
        self._k_i = get_with_default(params, "k_i", 0)
        self._k_d = get_with_default(params, "k_d", 0)
        self._norm_tolerances = get_with_default(params, "norm_tolerances", 1e-3)

        # might for example, want a circular difference (e.g. orientation control), key dependent
        self._difference_fn = get_with_default(params, "difference_fn", lambda key, a, b: a - b)

        # map all to lists w/ same length as control_names
        for param in ['_k_p', '_k_i', '_k_d', '_norm_tolerances']:
            val = getattr(self, param)
            if isinstance(val, Number):
                val = [val] * len(self._control_names)
            assert len(val) == len(self._control_names), [val, self._control_names]
            # list of floats
            setattr(self, param, [float(f) for f in val])

        # for chaining ops
        self._norm_tolerances = AttrDict.from_dict({name: tol
                                                    for name, tol in zip(self._control_names, self._norm_tolerances)})

    def _init_setup(self):
        pass

    def warm_start(self, model, observation, goal):
        pass

    def reset_policy(self, **kwargs):
        self._last_err = AttrDict.from_dict({name: None for name in self._control_names})
        self._der_err = AttrDict.from_dict({name: np.zeros(sh, dtype=np.float32)
                                            for name, sh in zip(self._control_names, self._control_shapes)})
        self._int_err = AttrDict.from_dict({name: np.zeros(sh, dtype=np.float32)
                                            for name, sh in zip(self._control_names, self._control_shapes)})

        self.reset_targets(**kwargs)

    # override this
    def reset_targets(self, target: AttrDict = AttrDict(), **kwargs):
        assert target.has_leaf_keys(self._control_names)
        self._target = target.copy()

    # override this
    def get_targets(self, model, curr, goal, **kwargs):
        return self._target

    def get_action(self, model, observation, goal, **kwargs):
        """
        :param model: (Model)
        :param observation: (AttrDict)  (B x H x ...)
        :param goal: (AttrDict) (B x H x ...)
        note: kwargs are for model forward fn

        :return action: AttrDict (B x ...)
        """

        # make sure types are correct for each array
        observation = self._env_spec.map_to_types(observation)

        # errors
        observation.has_leaf_keys(self._control_names)

        # horizon selection
        curr = observation.leaf_filter_keys(self._control_names).leaf_apply(lambda arr: arr[:, 0])

        # getting target positions
        targ = self.get_targets(model, curr, goal, **kwargs)

        err = AttrDict()
        output = AttrDict()
        for name in self._control_names:
            # Proportional
            err[name] = self._difference_fn(name, targ[name], curr[name])
            # Integral
            self._int_err[name] += err[name] * self._dt
            # Derivative
            if self._last_err[name] is not None:
                self._der_err[name] = (err[name] - self._last_err[name]) / self._dt
            # posterity update
            self._last_err[name] = err[name].copy()

            # PID aggregation
            output[name] = self._k_p * err[name] + self._k_i * self._int_err[name] + self._k_d * self._der_err[name]

        # clipping
        self._env_spec.clip(output, self._control_names)

        # adding on some details about target positions nested below
        output.target = targ.copy()
        output.err = err.copy()
        output.int_err = self._int_err.copy()
        output.der_err = self._der_err.copy()
        return output

    def is_terminated(self, model, observation, goal, **kwargs) -> bool:
        curr = observation.leaf_filter_keys(self._control_names).leaf_apply(lambda arr: arr[:, 0])
        targ = self.get_targets(model, curr, goal, **kwargs)

        # return if current is within tolerance *dist* of target along all dimensions (norm computed per key)
        return AttrDict.leaf_combine_and_apply([curr, targ, self._norm_tolerances],
                                               lambda key, vs: (self._difference_fn(key, *vs[:2]), vs[2]),
                                               pass_in_key_to_func=True) \
            .leaf_reduce(lambda red, vs: red and np.linalg.norm(vs[0]) <= vs[1], seed=True)
