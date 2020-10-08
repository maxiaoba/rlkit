import random
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class SimpleSupLSTMEnv(gym.Env):
    def __init__(self, node_num=5, node_dim=2, num_interval=10):
        self.seed()
        self.node_num = node_num
        self.node_dim = node_dim
        self.reset()

        self._num_interval = num_interval
        self._intervals = np.linspace(-1,1,num_interval+1)
        self._step = 2./float(num_interval)*3.
        self.label_num = self.node_num - 1
        self.label_dim = self._num_interval

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    @property
    def observation_space(self):
        return spaces.Box(low=-np.ones(self.node_num*self.node_dim),
                        high=np.ones(self.node_num*self.node_dim))

    @property
    def action_space(self):
        return spaces.Discrete(self._num_interval)

    def reset(self):
        self._old_state = -np.ones(self.node_num*self.node_dim)+1e-3 #np.zeros(node_num*node_dim)
        self._state = -np.ones(self.node_num*self.node_dim)+1e-3 #np.zeros(node_num*node_dim)
        return self.observe()

    def step(self, action):
        sup_labels = self.get_sup_labels()
        info = dict(sup_labels=sup_labels) # don't regress on first label
        labels = self.get_labels()
        reward = -np.abs(np.mean(labels)-action)

        self._old_state = self._state
        self._state = self._state \
            + self.np_random.rand(self.node_num*self.node_dim)*self._step
            # + (self.np_random.rand(self.node_num*self.node_dim)*2.-1.)*self._step
        self._state = np.clip(self._state, -(1-1e-3), 1.-1e-3)
        obs = self.observe()
        done = False
        return obs, reward, done, info

    def get_labels(self):
        labels = np.array([np.sum(np.mean(self._state[self.node_dim*i:self.node_dim*(i+1)])\
                                     > self._intervals) - 1\
                             for i in range(self.node_num)])
        return labels

    def get_sup_labels(self):
        return self.get_labels()[1:]

    def observe(self):
        return self._state - self._old_state

    def render(self, mode='human', screen_size=800, extra_input=None):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    env = SimpleSupLSTMEnv()
    print(env.reset())
    print(env.step(0))
    print(env._state)
    print(env.step(1))
    print(env._state)
    print(env.step(2))
    print(env._state)
    print(env.step(3))
    print(env._state)
    print(env.step(4))
    print(env._state)
