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
        obs_dim = self._wrapped_env.observation_space[0].low.size
        for observation_space_i in self._wrapped_env.observation_space:
            if observation_space_i.low.size > obs_dim:
                obs_dim = observation_space_i.low.size
        self.observation_space = Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        self.obs_dim = obs_dim

    def step(self, action_n):
        obs_n_list, reward_n, done_n, info_n = self._wrapped_env.step(action_n)
        obs_n = self.convert_obs(obs_n_list)
        return obs_n, np.array(reward_n), np.array(done_n), {}

    def reset(self):
        obs_n_list = self._wrapped_env.reset()
        obs_n = self.convert_obs(obs_n_list)
        return obs_n

    def convert_obs(self, obs_n_list):
        obs_n = np.zeros((self.num_agent,self.obs_dim))
        for i,j in enumerate(obs_n_list):
            obs_n[i][0:len(j)] = j
        return obs_n

    def __str__(self):
        return "ParticleEnv: %s" % self._wrapped_env