import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np, np_ify
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
from rlkit.torch.policies.deterministic_policies import MlpPolicy

class SoftmaxPolicy(Policy, nn.Module):
    """
    A helper class using softmax output activation with a learned temperature
    """

    def __init__(
            self,
            module,
            learn_temperature=True,
    ):
        print("SoftmaxMlpPolicy: learn_temperature is {}".format(learn_temperature))
        super().__init__()
        self.module = module
        self.learn_temperature = learn_temperature

    def forward(self, obs, return_info=False):
        logits = self.module.forward(obs)
        if self.learn_temperature:
            logits = logits[:,:-1]/(torch.exp(logits[:,-1][:,None])+1e-3)
        else:
            logits = logits
        pis = torch.softmax(logits, dim=-1)
        info = dict(preactivation = logits)
        if return_info:
            return pis, info
        else:
            return pis

    def get_distribution(self, obs):
        _, info = self.forward(obs, return_info=True)
        logits = info['preactivation']
        return Categorical(logits=logits)

    def log_prob(self, obs, action):
        return self.get_distribution(obs).log_prob(action.squeeze(-1))

    def get_action(self, obs):
        pis = eval_np(self, obs[None])[0,:]
        action = np.random.choice(np.arange(pis.shape[0]),p=pis)
        return action, {}
