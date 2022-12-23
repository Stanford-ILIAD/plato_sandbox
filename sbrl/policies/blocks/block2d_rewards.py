import numpy as np

from sbrl.envs.block2d.block_env_2d import BlockEnv2D
from sbrl.policies.blocks.stack_block2d_success_metrics import get_l2_then_mean, circular_difference_fn
from sbrl.sandbox.new_trainer.reward import Reward
from sbrl.utils.python_utils import get_with_default


class Block2DReward(Reward):
    def _init_params_to_attrs(self, params):
        self._l2_tol = get_with_default(params, "position_tolerance", 15.)  # 15 grid squares
        self._l2_angle_tol = get_with_default(params, "angle_tolerance", np.deg2rad(15.))  # 15 degrees

    def _init_setup(self):
        pass

    def get_reward(self, env: BlockEnv2D, model, observation, goal, action, next_observation, next_goal, env_memory, done, policy_done, goal_policy_done=None,  **kwargs):
        """
        :param env
        :param model:
        :param observation:
        :param goal:
        :param action:
        :param next_observation:
        :param next_goal:
        :param env_memory:
        :param done:
        :param policy_done:
        :param goal_policy_done:
        """
        # shape = list(self._env_spec.get_front_size(observation)) + [1]
        # return torch.zeros(shape, device=model.device)
        assert goal.has_leaf_key("policy_type"), "Need to know the policy type to compute reward."

        gp = int((goal >> "policy_type").item())
        rew = 0.
        if gp != -1:
            l2_dist = get_l2_then_mean(observation, goal, 'block_positions').item()
            l2_angle = get_l2_then_mean(observation, goal, 'block_angles', diff_fn=circular_difference_fn).item()

            pos_weight = 0.75

            if gp >= 8:
                # rotation primitives
                pos_weight = 0.25

            rew = pos_weight * int(l2_dist <= self._l2_tol) + (1 - pos_weight) * int(l2_angle <= self._l2_angle_tol)

        shape = list(self._env_spec.get_front_size(observation)) + [1]

        return np.array(rew).reshape(shape)
        # if gp == 0:  # push
        #     reward_fn = sm.get_move_object_to_position_and_angle_reward(pos_weight=0.75)
        # elif gp == 1:  # pull
        #     reward_fn = sm.get_move_object_to_position_and_angle_reward(pos_weight=0.75)
        # elif gp == 2:  # liftrot
        #     reward_fn = sm.get_move_object_to_position_and_angle_reward(pos_weight=0.25)
        # elif gp == 3:  # tip
        #     reward_fn = sm.get_move_object_to_position_and_angle_reward(pos_weight=0.25)
        # elif gp == 4:  # side-rot
        #     reward_fn = sm.get_move_object_to_position_and_angle_reward(pos_weight=0.25)
        # else:
        #     raise NotImplementedError
        #
        # reward_fn()
        #
        # return reward_fn()

