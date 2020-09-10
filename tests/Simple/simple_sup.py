import random
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class SimpleSupEnv(gym.Env):
    def __init__(self, num_interval=10, num_obs=1):
        self.seed()
        self._state = None
        self._num_interval = num_interval
        self._num_obs = num_obs
        self._intervals=np.linspace(-1,1,num_interval+1)
        self.label_num = 1
        self.label_dim = self._num_interval

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    @property
    def observation_space(self):
        return spaces.Box(low=-np.ones(self._num_obs),high=np.ones(self._num_obs))

    @property
    def action_space(self):
        return spaces.Discrete(self._num_interval)

    def reset(self):
        self._state = self.np_random.rand()*2. - 1.
        return self.observe()

    def step(self, action):
        labels = self.get_sup_labels()
        info = dict(sup_labels=labels)
        reward = -np.abs(labels[0]-action)

        self._state = self.np_random.rand()*2. - 1.
        obs = self.observe()
        done = True
        return obs, reward, done, info

    def get_sup_labels(self):
        labels = np.array([np.sum(self._state > self._intervals) - 1])
        return labels

    def observe(self):
        obs = np.zeros(self._num_obs)
        obs[:-1] = self.np_random.rand(self._num_obs-1)*2. - 1.
        obs[-1] = self._state - np.sum(obs[:-1])
        return obs

    def render(self, mode='human', screen_size=800, extra_input=None):
        pass

    def close(self):
        pass
