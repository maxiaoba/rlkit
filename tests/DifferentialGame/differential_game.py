"""
originally from: https://github.com/ml3705454/mapr2/blob/master/maci/environments/differential_game.py
"""

import numpy as np
from gym.spaces import Box
from rlkit.core.serializable import Serializable

class DifferentialGame(Serializable):
    def __init__(self, game_name, agent_num=2, action_low=-10, action_high=10):
        Serializable.quick_init(self, locals())
        self.game = game_name
        self.agent_num = agent_num
        # self.action_num = action_num
        self.action_range = [action_low, action_high]
        self.t = 0
        self.numplots = 0
        self.payoff = {}

        if self.game == 'zero_sum':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: a1 * a2
            self.payoff[1] = lambda a1, a2: -a1 * a2
        elif self.game == 'cooperative':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: a1 * a2
            self.payoff[1] = lambda a1, a2: a1 * a2
        elif self.game == 'trigonometric':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: np.cos(a2) * a1
            self.payoff[1] = lambda a1, a2: np.sin(a1) * a2
        elif self.game == 'mataching_pennies':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: (a1-0.5)*(a2-0.5)
            self.payoff[1] = lambda a1, a2: (a1-0.5)*(a2-0.5)
        elif self.game == 'rotational':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: 0.5 * a1 * a1 + 10 * a1 * a2
            self.payoff[1] = lambda a1, a2: 0.5 * a2 * a2 - 10 * a1 * a2
        elif self.game == 'wolf':
            assert self.agent_num == 2
            def V(alpha, beta, payoff):
                u = payoff[(0, 0)] - payoff[(0, 1)] - payoff[(1, 0)] + payoff[(1, 1)]
                return alpha * beta * u + alpha * (payoff[(0, 1)] - payoff[(1, 1)]) + beta * (
                            payoff[(1, 0)] - payoff[(1, 1)]) + payoff[(1, 1)]

            payoff_0 = np.array([[0, 3], [1, 2]])
            payoff_1 = np.array([[3, 2], [0, 1]])

            self.payoff[0] = lambda a1, a2: V(a1, a2, payoff_0)
            self.payoff[1] = lambda a1, a2: V(a1, a2, payoff_1)

        elif self.game == 'max2':
            assert self.agent_num == 2
            h1 = 0.8
            h2 = 1.
            s1 = 3.
            s2 = 1.
            x1 = -5.
            x2 = 5.
            y1 = -5.
            y2 = 5.
            c = 10.
            def max_f(a1, a2):
                f1 = h1 * (-(np.square(a1 - x1) / s1) - (np.square(a2 - y1) / s1))
                f2 = h2 * (-(np.square(a1 - x2) / s2) - (np.square(a2 - y2) / s2)) + c
                return max(f1, f2)
            self.payoff[0] = lambda a1, a2: max_f(a1, a2)
            self.payoff[1] = lambda a1, a2: max_f(a1, a2)
        else:
            raise NotImplementedError
        self.rewards = np.zeros((self.agent_num,))

    @property
    def observation_space(self):
        return Box(np.zeros(1), np.ones(1), dtype=np.float32)

    @property
    def action_space(self):
        return Box(-np.ones(1), np.ones(1), dtype=np.float32)

    def step(self, actions):
        assert len(actions) == self.agent_num
        # print('actions', actions)
        actions = np.array(actions).reshape((self.agent_num,)) * self.action_range[1]
        # print('scaled', actions)
        reward_n = np.zeros((self.agent_num,))
        for i in range(self.agent_num):
            # print('actions', actions)
            reward_n[i] = self.payoff[i](*tuple(actions))
        self.rewards = reward_n
        # print(reward_n)
        state_n = np.array([[1] for i in range(self.agent_num)])
        info = {}
        done_n = np.array([True] * self.agent_num)
        self.t += 1
        return state_n, reward_n, done_n, info

    def reset(self):
        return np.array([[1] for i in range(self.agent_num)])

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Action Range {}\n'.format(self.game, self.agent_num, self.action_range)
        return content
