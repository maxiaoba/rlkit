import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np, np_ify
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
from rlkit.torch.networks import Mlp

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)

class SoftmaxMlpPolicy(MlpPolicy):
    """
    A helper class using softmax output activation with a learned temperature
    """

    def __init__(
            self,
            output_size,
            learn_temperature=True,
            *args,
            **kwargs
    ):
        print("SoftmaxMlpPolicy: learn_temperature is {}".format(learn_temperature))
        self.learn_temperature = learn_temperature
        if learn_temperature:
            output_size = output_size+1
        super().__init__(*args, output_size=output_size,**kwargs)

    def forward(self, obs, return_info=False):
        action = super().forward(obs)
        if self.learn_temperature:
            preactivation = action[:,:-1]/(torch.exp(action[:,-1][:,None])+1e-3)
        else:
            preactivation = action
        action = torch.softmax(preactivation, dim=-1)
        info = dict(preactivation = preactivation)
        if return_info:
            return action, info
        else:
            return action

    def one_hot(self, obs):
        probs = self.forward(obs)
        max_idx = torch.argmax(probs, -1, keepdim=True)
        one_hot = torch.FloatTensor(probs.shape).zero_().to(ptu.device)
        one_hot.scatter_(-1,max_idx,1)
        return one_hot