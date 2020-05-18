import numpy as np
from gym.spaces import Box
from gym.spaces import Discrete

from rlkit.envs.wrappers import ProxyEnv

class MAProbDiscreteEnv(ProxyEnv):
    """
    Convert discrete action space to continuous prob space
    """

    def __init__(
            self,
            env,
    ):
        ProxyEnv.__init__(self, env)
        self.n = self._wrapped_env.action_space.n
        self.action_space = Box(np.zeros(self.n), np.ones(self.n))

    def step(self, action_n):
        true_action_n = np.array([np.random.choice(np.arange(self.n),p=action) for action in action_n])
        return self._wrapped_env.step(true_action_n)

    def __str__(self):
        return "MAProbDiscrete: %s" % self._wrapped_env