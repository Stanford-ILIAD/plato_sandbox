from .env_spec import EnvSpec

"""
The EnvSpec defines the "hyperparameters" of the environment: the shapes/limits/dtypes of the
observations, goals, and actions.

Why is the EnvSpec separate from the Env? One way to think about it is that EnvSpec should probably be named
RobotSpec, since it defines what the robot's observations, goals, and actions. So with this separation, you can
have different robots (i.e., different EnvSpec) for the same Env.

The EnvSpec is needed for the dataset---so it knows what to store---and the model---so it knows what inputs/outputs
to expect.

Another advantage of separating the EnvSpec from the Env is if you do offline training (i.e., no on-policy data
gathering), you don't need the Env (which may have ugly robot-specific code like ROS)!
"""


class GymEnvSpec(EnvSpec):

    @property
    def output_observation_names(self):
        """
        Returns:
            list(str)
        """
        return ['next_obs', 'reward']

    @property
    def observation_names(self):
        """
        Returns:
            list(str)
        """
        return ['obs']

    @property
    def goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        return []

    @property
    def action_names(self):
        """
        Returns:
            list(str)
        """
        return ['action']
