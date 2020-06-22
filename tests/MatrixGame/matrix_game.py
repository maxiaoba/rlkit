import numpy as np
from gym.spaces import Box,Discrete
from rlkit.core.serializable import Serializable

class MatrixGame(Serializable):
    def __init__(self, game_name):
        Serializable.quick_init(self, locals())
        self.game = game_name
        self.payoff = {}

        if self.game == 'deadlock':
            self.agent_num = 2
            self.action_num = 2
            self.payoff[0] = np.array([
                                    [-5.,0.],
                                    [5.,-10]
                                    ])
            self.payoff[1] = np.array([
                                    [-5.,5.],
                                    [0.,-10.]
                                    ])
        elif self.game == 'deadlock_coop':
            self.agent_num = 2
            self.action_num = 2
            self.payoff[0] = np.array([
                                    [-5.,5.],
                                    [5.,-10]
                                    ])
            self.payoff[1] = self.payoff[0]
        elif self.game == 'deadlock_coop_unsym':
            self.agent_num = 2
            self.action_num = 2
            self.payoff[0] = np.array([
                                    [-5.,10.],
                                    [5.,-10]
                                    ])
            self.payoff[1] = self.payoff[0]
        elif self.game == 'zero_sum':
            self.agent_num = 2
            self.action_num = 2
            self.payoff[0] = np.array([
                                    [-1.,1.],
                                    [1.,-1.]
                                    ])
            self.payoff[1] = -self.payoff[0]
        else:
            raise NotImplementedError

    @property
    def observation_space(self):
        return Box(np.zeros(1), np.ones(1), dtype=np.float32)

    @property
    def action_space(self):
        return Discrete(self.action_num)

    def step(self, actions):
        assert len(actions) == self.agent_num
        reward_n = np.zeros((self.agent_num,))
        for i in range(self.agent_num):
            reward_n[i] = self.payoff[i][tuple(actions)]
        state_n = np.array([[1] for i in range(self.agent_num)])
        info = {}
        done_n = np.array([True] * self.agent_num)
        return state_n, reward_n, done_n, info

    def reset(self):
        return np.array([[1] for i in range(self.agent_num)])

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Action Range {}\n'.format(self.game, self.agent_num, self.action_range)
        return content
