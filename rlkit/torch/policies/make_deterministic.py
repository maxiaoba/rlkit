from torch import nn as nn
from rlkit.policies.base import Policy

class MakeDeterministic(nn.Module, Policy):
    def __init__(self, stochastic_policy):
        super().__init__()
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def reset(self):
    	self.stochastic_policy.reset()