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

    def to_policy_inputs(self, obs_action, labels, sup_latent, return_info=False):
        obs_flat, prev_actions = obs_action
        print(obs_flat, prev_actions)
        obs = torch.reshape(obs_flat,(*obs_flat.shape[:-1], self.label_num+1, -1))
        valid_musk = (torch.sum(torch.abs(obs),dim=-1) != 0)
        valid_musk = torch.index_select(valid_musk,-1,torch.arange(1,self.label_num+1))

        with torch.no_grad():
            dist, sup_info = self.get_sup_distribution(obs_action, sup_latent=sup_latent, return_info=True)
        if labels is None:
            labels = torch.argmax(dist.probs, -1)
        else:
            labels = labels.clone()
            # valid_musk2 = ~torch.isnan(labels)
            # assert torch.all(torch.eq(valid_musk, valid_musk2)), "obs label mismacth"
            # can't check this since out-of-actual-length labels are 0

        labels[~valid_musk] = 0
        onehot_labels = self.to_onehot_labels(labels)
        onehot_labels[~valid_musk] = 0.
        ego_labels = torch.zeros(*onehot_labels.shape[:-2],1,self.label_dim)
        onehot_labels = torch.cat((ego_labels,onehot_labels),-2)

        obs = torch.cat((obs,onehot_labels),dim=-1).reshape(*obs.shape[:-2],-1)

        policy_inputs = (obs, prev_actions)
        if return_info:
            return policy_inputs, sup_info
        else:
            return policy_inputs

    def forward(self, obs_action, labels, latent=None, sup_latent=None, return_info=False):
        if latent is None:
            latent = self.policy.latent_0
        if sup_latent is None:
            sup_latent = self.sup_learner.latent_0
        policy_inputs, sup_info = self.to_policy_inputs(obs_action, labels=labels, sup_latent=sup_latent, return_info=True)
        pis, policy_info = self.policy(policy_inputs, latent=latent, return_info=True)
        info = policy_info
        info['sup_preactivation'] = sup_info['preactivation']
        info['sup_latent'] = sup_info['latent']

        if return_info:
            return pis, info
        else:
            return pis

    def get_distribution(self, obs_action, latent=None, sup_latent=None, labels=None):
        _, info = self.forward(obs_action, latent=latent, sup_latent=sup_latent, labels=labels, return_info=True)
        logits = info['preactivation']
        return Categorical(logits=logits)

    def log_prob(self, obs_action, action, latent=None, sup_latent=None, labels=None):
        return self.get_distribution(obs_action, latent=latent, sup_latent=sup_latent, labels=labels).log_prob(action.squeeze(-1))

    def get_action(self, obs, labels=None, deterministic=False):
        assert len(obs.shape) == 1
        assert (self.policy.a_p == self.sup_learner.a_p).all()
        with torch.no_grad():
            obs_action = (torch_ify(obs)[None,None,:], self.policy.a_p[None,None,:])
            if labels is not None:
                labels = torch_ify(labels)[None,None,:]
            pis, info = self.forward(obs_action, labels=labels,
                                     latent=self.policy.latent_p,
                                     sup_latent=self.sup_learner.latent_p,
                                     return_info=True)
            sup_probs = Categorical(logits=info['sup_preactivation']).probs
        pis = np_ify(pis[0,0,:])
        sup_probs = np_ify(sup_probs[0,0,:,:])
        if deterministic:
            action = np.argmax(pis)
        else:
            action = np.random.choice(np.arange(pis.shape[0]),p=pis)
        self.policy.a_p = torch_ify(np.array([action]))
        self.policy.latent_p = info['latent']
        self.sup_learner.a_p = torch_ify(np.array([action]))
        self.sup_learner.latent_p = info['sup_latent']
        
        return action, {'intentions': sup_probs}

    # def get_attention_weight(self, obs):
    #     if hasattr(self.policy[0], 'attentioner'):
    #         with torch.no_grad():
    #             policy_inputs = eval_np(self.obs_to_policy_inputs, obs[None])
    #             x, attention_weight = eval_np(self.policy[0], policy_inputs, return_attention_weights=True)
    #         return attention_weight
    #     else:
    #         return None

    def get_sup_distribution(self, obs_action, sup_latent=None, return_info=False):
        _, info = self.sup_learner(obs_action, latent=sup_latent, return_info=True)
        logits = info['preactivation']
        if return_info:
            return Categorical(logits=logits), info
        else:
            return Categorical(logits=logits)

    def get_sup_labels(self, obs_action, sup_latent=None):
        sup_probs = self.sup_prob(obs_action, sup_latent=sup_latent)
        return torch.argmax(sup_probs,-1)

    def sup_log_prob(self, obs_action, label, sup_latent=None):
        return self.get_sup_distribution(obs_action, sup_latent=sup_latent).log_prob(label)

    def sup_prob(self, obs_action, sup_latent=None):
        return self.get_sup_distribution(obs_action, sup_latent=sup_latent).probs

    def reset(self):
        self.policy.reset()
        self.sup_learner.reset()
