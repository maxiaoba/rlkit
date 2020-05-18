import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np, np_ify
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
from rlkit.torch.policies.deterministic_policies import MlpPolicy

class GumbelSoftmaxMlpPolicy(MlpPolicy):
    """
    A helper class using Gumbel Softmax
    """

    def __init__(
            self,
            hard,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        print('GumbelSoftmaxMlpPolicy: hard is {}'.format(hard))
        self.hard = hard

    def forward(self, obs, return_info=False, tau=1):
        action = super().forward(obs)
        preactivation = action
        action = F.gumbel_softmax(preactivation, tau=tau, hard=self.hard)
        info = dict(preactivation = preactivation)
        if return_info:
            return action, info
        else:
            return action

    