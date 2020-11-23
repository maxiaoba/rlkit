import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions.tanh_normal import TanhNormal
from rlkit.torch.networks import Mlp

class TanhGaussianPolicy(Policy, nn.Module):
    """
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    """
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
        mean, log_std = self.module(obs)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        log_prob = None
        entropy = None
        pre_tanh_value = None

        tanh_normal = TanhNormal(mean, std)
        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            # if return_log_prob:
            if reparameterize is True:
                action, pre_tanh_value = tanh_normal.rsample(
                    return_pretanh_value=True
                )
            else:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
        if return_info:
            log_prob = tanh_normal.log_prob(
                action,
                pre_tanh_value=pre_tanh_value
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)

        info = dict(
            mean=mean,log_std=log_std,log_prob=log_prob,entropy=entropy,
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
        mean, log_std = self.module(obs)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        pre_tanh_value = raw_action

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
            pre_tanh_value=pre_tanh_value
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return log_prob

    def get_distribution(self, obs):
        mean, log_std = self.module(obs)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return TanhNormal(mean, std)
