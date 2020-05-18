import numpy as np
from gym.spaces import Box
from gym.spaces import Discrete

from rlkit.envs.wrappers import ProxyEnv

class ParticleEnv(ProxyEnv):
    """
    Convert discrete action space to continuous prob space
    """

    def __init__(
            self,
            env,
    ):
        ProxyEnv.__init__(self, env)
        self.num_agent = self._wrapped_env.n
        self.action_space = self._wrapped_env.action_space[0]
        self.observation_space = self._wrapped_env.observation_space[0]

    def step(self, action_n):
        obs_n, reward_n, done_n, info_n = self._wrapped_env.step(action_n)
        return np.array(obs_n), np.array(reward_n), np.array(done_n), {}

    def reset(self):
        obs_n = self._wrapped_env.reset()
        return np.array(obs_n)

    def __str__(self):
        return "ParticleEnv: %s" % self._wrapped_env