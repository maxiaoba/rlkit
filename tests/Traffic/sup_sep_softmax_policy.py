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

class SupSepSoftmaxPolicy(Policy, nn.Module):
    """
    A helper class using softmax output activation with a learned temperature
    """

    def __init__(
            self,
            policy,
            sup_learner,
            label_num,
            label_dim,
    ):
        super().__init__()
        self.policy = policy
        self.sup_learner = sup_learner
        self.label_num = label_num
        self.label_dim = label_dim

    def forward(self, obs, labels=None, return_info=False):
        obs_flat = obs
        batch_size, obs_dim = obs.shape
        obs = torch.reshape(obs,(batch_size, self.label_num+1, -1))
        valid_musk = (torch.sum(torch.abs(obs),dim=-1) != 0)[:,1:]
        if labels is None:
            with torch.no_grad():
                labels = self.get_labels(obs_flat)
        else:
            labels = labels.clone()
            valid_musk2 = ~torch.isnan(labels)
            assert torch.all(torch.eq(valid_musk, valid_musk2)), "obs label mismacth"

        labels[~valid_musk] = 0
        onehot_labels = torch.zeros(batch_size, self.label_num+1, self.label_dim)
        onehot_labels[:,1:,:].scatter_(-1,labels[:,:,None].long(),1.)
        onehot_labels[:,1:,:][~valid_musk] = 0.
        inputs = torch.cat((obs,onehot_labels),dim=-1).reshape(batch_size,-1)
        logits = self.policy(inputs)
        pis = torch.softmax(logits, dim=-1)
        info = dict(preactivation=logits, labels=labels)
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

    def get_action(self, obs, labels=None):
        if labels is None:
            pis = eval_np(self, obs[None])[0,:]
        else:
            pis = eval_np(self, obs[None], labels=labels[None])[0,:]
        action = np.random.choice(np.arange(pis.shape[0]),p=pis)
        return action, {}

    def get_attention_weight(self, obs):
        return None

    def get_labels(self, obs):
        sup_probs = self.sup_prob(obs)
        return torch.argmax(sup_probs,-1)

    def get_sup_distribution(self, obs):
        logits = self.sup_learner(obs)
        return Categorical(logits=logits)

    def sup_log_prob(self, obs, label):
        return self.get_sup_distribution(obs).log_prob(label)

    def sup_prob(self, obs):
        return self.get_sup_distribution(obs).probs
