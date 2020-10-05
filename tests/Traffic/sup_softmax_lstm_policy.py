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

class SupSoftmaxLSTMPolicy(Policy, nn.Module):
    """
    LSTM policy with Categorical distributon using softmax
    """

    def __init__(
            self,
            a_0,
            latent_0,
            obs_dim,
            action_dim,
            lstm_net,
            decoder,
            sup_learner,
    ):
        super().__init__()
        self.a_0 = torch_ify(a_0).clone().detach()
        self.latent_0 = tuple([torch_ify(h).clone().detach() for h in latent_0])
        self.a_p = self.a_0.clone().detach()
        self.latent_p = tuple([h.clone().detach() for h in self.latent_0])
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_net = lstm_net
        self.decoder = decoder
        self.sup_learner = sup_learner
        
    def to_onehot_actions(self, prev_actions):
        assert len(prev_actions.shape) == 3
        if prev_actions.shape[-1] != self.action_dim:
            prev_actions_onehot = torch.zeros(prev_actions.shape[0],prev_actions.shape[1],self.action_dim)
            prev_actions_onehot.scatter_(-1,prev_actions.long(),1.)
        else:
            prev_actions_onehot = prev_actions
        return prev_actions_onehot

    def forward(self, obs_action, post_net=None, latent=None, return_info=False):
        if post_net is None:
            post_net = self.decoder
        obs, prev_actions = obs_action
        # batch x T x |O| 
        # batch x (T-1) x |1|
        assert (len(obs.shape) == 3) and ((len(prev_actions.shape) == 3))
        
        prev_actions = self.to_onehot_actions(prev_actions) # batch x (T-1) x |A|
        if prev_actions.shape[1] == (obs.shape[1]-1):
            a_0_batch = self.a_0.reshape(1,1,self.action_dim).repeat(obs.shape[0],1,1)
            prev_actions = torch.cat((a_0_batch,prev_actions),dim=1)
        else:
            assert prev_actions.shape[1] == obs.shape[1]

        if latent is None:
            latent = self.latent_0
        output, latent_n = self.lstm_net.forward(obs, prev_actions, latent)
        # batch x T x dim, (batch x dim, batch x dim, ...)

        logits = post_net(output)
        pis = torch.softmax(logits, dim=-1)
        info = dict(preactivation = logits, latent = latent_n)
        if return_info:
            return pis, info
        else:
            return pis

    def get_distribution(self, obs_action, latent=None):
        _, info = self.forward(obs_action, latent=latent, return_info=True)
        logits = info['preactivation']
        return Categorical(logits=logits)

    def log_prob(self, obs_action, action, latent=None):
        return self.get_distribution(obs_action, latent=latent).log_prob(action.squeeze(-1))

    def get_action(self, obs, deterministic=False):
        assert len(obs.shape) == 1
        with torch.no_grad():
            obs_action = (torch_ify(obs)[None,None,:], self.a_p[None,None,:])
            pis, info = self.forward(obs_action, latent=self.latent_p, return_info=True)
        pis = np_ify(pis[0,0,:])
        if deterministic:
            action = np.argmax(pis)
        else:
            action = np.random.choice(np.arange(pis.shape[0]),p=pis)
        self.a_p = torch_ify(np.array([action]))
        self.latent_p = info['latent']
        return action, {}

    def get_sup_distribution(self, obs_action, latent=None):
        _, info = self.forward(obs_action, latent=latent, post_net=self.sup_learner, return_info=True)
        logits = info['preactivation']
        return Categorical(logits=logits)

    def sup_log_prob(self, obs_action, label, latent=None):
        return self.get_sup_distribution(obs_action, latent=latent).log_prob(label)

    def sup_prob(self, obs_action, latent=None):
        return self.get_sup_distribution(obs_action, latent=latent).probs

    def reset(self):
        self.a_p = self.a_0.clone().detach()
        self.latent_p = tuple([h.clone().detach() for h in self.latent_0])
