"""
Torch argmax policy
"""
import numpy as np
from torch import nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy


class ArgmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf, use_preactivation=False):
        super().__init__()
        self.qf = qf
        self.use_preactivation = use_preactivation

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        if self.use_preactivation:
        	_, info = self.qf(obs,return_info=True)
        	q_values = info['preactivation']
        	q_values = q_values.squeeze(0)
        else:
	        q_values = self.qf(obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        return q_values_np.argmax(), {}
