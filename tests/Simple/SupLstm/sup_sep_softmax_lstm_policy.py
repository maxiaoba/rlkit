import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np, np_ify, torch_ify
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
from rlkit.torch.policies.deterministic_policies import MlpPolicy

class SupSepSoftmaxLSTMPolicy(Policy, nn.Module):
    """
    LSTM policy with Categorical distributon using softmax
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            policy,
            sup_learner,
            label_num,
            label_dim,
    ):
        super().__init__()
        self.policy = policy
        self.sup_learner = sup_learner
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.label_num = label_num
        self.label_dim = label_dim

    def to_onehot_labels(self, labels):
        if labels.shape[-1] != self.label_dim:
            labels_onehot = torch.zeros(*labels.shape,self.label_dim)
            labels_onehot.scatter_(-1,labels.unsqueeze(-1).long(),1.)
        else:
            labels_onehot = labels
        return labels_onehot

    def _to_policy_inputs(self, obs_action, labels=None, return_info=False):
        obs, prev_actions = obs_action
        with torch.no_grad():
            dist, info = self.get_sup_distribution(obs_action,return_info=True)
        if labels is None:
            labels = torch.argmax(dist.probs, -1)
        else:
            labels = labels.clone()

        onehot_labels = self.to_onehot_labels(labels)
        onehot_labels = onehot_labels.reshape(*onehot_labels.shape[:-2],-1)
        obs = torch.cat((obs,onehot_labels),dim=-1)
        policy_inputs = (obs, prev_actions)
        if return_info:
            return policy_inputs, info
        else:
            return policy_inputs

    def forward(self, obs_action, labels, return_info=False):
        policy_inputs, sup_info = self._to_policy_inputs(obs_action, labels=labels, return_info=True)
        pis, policy_info = self.policy(policy_inputs, return_info=True)
        info = policy_info
        info['sup_preactivation'] = sup_info['preactivation']
        info['sup_h'] = sup_info['h']
        info['sup_c'] = sup_info['c']
        if return_info:
            return pis, info
        else:
            return pis

    def step(self, obs_action, labels=None, return_info=False):
        policy_inputs, sup_info = self._to_policy_inputs(obs_action, labels=labels, return_info=True)
        pis, policy_info = self.policy.step(policy_inputs, return_info=True)
        info = policy_info
        info['sup_preactivation'] = sup_info['preactivation']
        info['sup_h'] = sup_info['h']
        info['sup_c'] = sup_info['c']
        if return_info:
            return pis, info
        else:
            return pis

    def get_distribution(self, obs, labels=None):
        _, info = self.forward(obs, labels=labels, return_info=True)
        logits = info['preactivation']
        return Categorical(logits=logits)

    def log_prob(self, obs, action, labels=None):
        return self.get_distribution(obs, labels=labels).log_prob(action.squeeze(-1))

    def get_action(self, obs, labels=None, deterministic=False):
        assert len(obs.shape) == 1
        with torch.no_grad():
            obs_action = (torch_ify(obs)[None], self.policy.a_p[None])
            if labels is not None:
                labels = torch_ify(labels)[None]
            pis, info = self.step(obs_action, labels=labels, return_info=True)
        pis = np_ify(pis[0])
        if deterministic:
            action = np.argmax(pis)
        else:
            action = np.random.choice(np.arange(pis.shape[0]),p=pis)
        self.policy.a_p = torch_ify(np.array([action]))
        self.policy.h_p = info['h'].clone().reshape(-1)
        self.policy.c_p = info['c'].clone().reshape(-1)
        self.sup_learner.a_p = torch_ify(np.array([action]))
        self.sup_learner.h_p = info['sup_h'].clone().reshape(-1)
        self.sup_learner.c_p = info['sup_c'].clone().reshape(-1)
        return action, {}

    # def get_attention_weight(self, obs):
    #     if hasattr(self.policy[0], 'attentioner'):
    #         with torch.no_grad():
    #             policy_inputs = eval_np(self.obs_to_policy_inputs, obs[None])
    #             x, attention_weight = eval_np(self.policy[0], policy_inputs, return_attention_weights=True)
    #         return attention_weight
    #     else:
    #         return None

    def get_labels(self, obs):
        sup_probs = self.sup_prob(obs)
        return torch.argmax(sup_probs,-1)

    def get_sup_distribution(self, obs, return_info=False):
        if isinstance(obs, tuple):
            obs_action = obs
        else:
            assert len(obs.shape) == 2
            assert (self.sup_learner.a_p==self.policy.a_p).all()
            obs_action = (torch_ify(obs), self.sup_learner.a_p[None].repeat(obs.shape[0],1))
        obs = obs_action[0]
        if len(obs.shape) == 3:
            _, info = self.sup_learner(obs_action, return_info=True)
        elif len(obs.shape) == 2:
            _, info = self.sup_learner.step(obs_action, return_info=True)
        logits = info['preactivation']
        if return_info:
            return Categorical(logits=logits), info
        else:
            return Categorical(logits=logits)

    def sup_log_prob(self, obs, label):
        return self.get_sup_distribution(obs).log_prob(label)

    def sup_prob(self, obs):
        return self.get_sup_distribution(obs).probs

    def reset(self):
        self.policy.reset()
        self.sup_learner.reset()
