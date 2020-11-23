import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions.mix_tanh_normal import MixTanhNormal
from rlkit.torch.networks import Mlp

class MixTanhGaussianPolicy(Policy, nn.Module):
    def __init__(
            self,
            module,
            return_raw_action=False,
            log_std_max = 2,
            log_std_min = -20,
    ):
        super().__init__()
        self.module = module
        self.return_raw_action = return_raw_action
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def get_action(self, obs_np, deterministic=False):
        if self.return_raw_action:
            actions, raw_actions = self.get_actions(obs_np[None], deterministic=deterministic)
            return actions[0, :], {'raw_action':raw_actions[0,:]}
        else:
            actions = self.get_actions(obs_np[None], deterministic=deterministic)
            return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        if self.return_raw_action:
            actions, info = eval_np(self, obs_np, deterministic=deterministic,return_info=True)
            raw_actions = info['preactivation']
            return actions, raw_actions
        else:
            return eval_np(self, obs_np, deterministic=deterministic)

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_info=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_info: If True, return info
        """
        weight, mean, log_std = self.module(obs)
        # batch x num_d, batch x num_d x a_dim, batch x num_d x a_dim
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        log_prob = None
        entropy = None
        pre_tanh_value = None

        mix_tanh_normal = MixTanhNormal(weight, mean, std)
        if deterministic:
            indx = torch.argmax(weight,dim=1,keepdim=True).unsqueeze(-1)
            pre_tanh_value = mean.gather(1,indx)
            action = torch.tanh(pre_tanh_value)
        else:
            # if return_log_prob:
            if reparameterize is True:
                action, pre_tanh_value = mix_tanh_normal.rsample(
                    return_pretanh_value=True
                )
            else:
                action, pre_tanh_value = mix_tanh_normal.sample(
                    return_pretanh_value=True
                )
        if return_info:
            log_prob = mix_tanh_normal.log_prob(
                action,
                pre_tanh_value=pre_tanh_value
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)

        info = dict(
            weight=weight,mean=mean,log_std=log_std,log_prob=log_prob,entropy=entropy,
            preactivation=pre_tanh_value
            )
        if return_info:
            return action, info
        else:
            return action

    def log_prob(
            self,
            obs,
            action,
            raw_action=None
    ):
        """
        :param obs: Observation
        :param action: Action
        """
        weight, mean, log_std = self.module(obs)
        # batch x num_d, batch x num_d x a_dim, batch x num_d x a_dim
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        pre_tanh_value = raw_action

        mix_tanh_normal = MixTanhNormal(weight, mean, std)
        log_prob = mix_tanh_normal.log_prob(
            action,
            pre_tanh_value=pre_tanh_value
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return log_prob

    def get_distribution(self, obs):
        weight, mean, log_std = self.module(obs)
        # batch x num_d, batch x num_d x a_dim, batch x num_d x a_dim
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return MixTanhNormal(weight, mean, std)
