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

class SoftmaxLSTMPolicy(Policy, nn.Module):
    """
    LSTM policy with Categorical distributon using softmax
    """

    def __init__(
            self,
            a_0,
            h_0,
            c_0,
            obs_dim,
            action_dim,
            hidden_dim,
            post_net,
            num_layers=1,
    ):
        super().__init__()
        self.a_0 = torch_ify(a_0).clone().detach()
        self.h_0 = torch_ify(h_0).clone().detach()
        self.c_0 = torch_ify(c_0).clone().detach()
        self.a_p = self.a_0.clone().detach()
        self.h_p = self.h_0.clone().detach()
        self.c_p = self.c_0.clone().detach()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=obs_dim+action_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True)
        self.post_net = post_net
        
    def to_onehot(self, prev_actions):
        assert len(prev_actions.shape) == 3
        if prev_actions.shape[-1] != self.action_dim:
            prev_actions_onehot = torch.zeros(prev_actions.shape[0],prev_actions.shape[1],self.action_dim)
            prev_actions_onehot.scatter_(-1,prev_actions.long(),1.)
        else:
            prev_actions_onehot = prev_actions
        return prev_actions_onehot

    def forward(self, obs, return_info=False):
        obs, prev_actions = obs
        assert (len(obs.shape) == 3) and ((len(prev_actions.shape) == 3))
        assert obs.shape[1] == prev_actions.shape[1] + 1
        # trajectory batch x T x |O| and batch x (T-1) x |1|
        prev_actions = self.to_onehot(prev_actions) # batch x (T-1) x |A|
        a_0_batch = self.a_0.reshape(1,1,self.action_dim).repeat(obs.shape[0],1,1)
        prev_actions = torch.cat((a_0_batch,prev_actions),dim=1)
        obs = torch.cat((obs,prev_actions),dim=-1)
        h_0_batch = self.h_0.reshape(self.num_layers,self.hidden_dim)[:,None,:].repeat(1,obs.shape[0],1)
        # num_layers x batch x hidden_dim
        c_0_batch = self.c_0.reshape(self.num_layers,self.hidden_dim)[:,None,:].repeat(1,obs.shape[0],1)
        output, (h_n, c_n) = self.lstm.forward(obs, (h_0_batch, c_0_batch))

        logits = self.post_net(output)
        pis = torch.softmax(logits, dim=-1)
        info = dict(preactivation = logits, h=h_n, c=c_n)
        if return_info:
            return pis, info
        else:
            return pis

    def step(self, obs, return_info=False):
        obs, prev_actions = obs
        assert (len(obs.shape) == 2) and ((len(prev_actions.shape) == 2))
        # trajectory batch x |O|
        obs = obs[:,None,:] # batch x 1 x |O+A|
        prev_actions = prev_actions[:,None,:]
        obs = torch.cat((obs,prev_actions),dim=-1)
        h_p_batch = self.h_p.reshape(self.num_layers,self.hidden_dim)[:,None,:].repeat(1,obs.shape[0],1)
        # num_layers x batch x hidden_dim
        c_p_batch = self.c_p.reshape(self.num_layers,self.hidden_dim)[:,None,:].repeat(1,obs.shape[0],1)
        output, (h_n, c_n) = self.lstm.forward(obs, (h_p_batch, c_p_batch))
        output = output.reshape(obs.shape[0],-1)

        logits = self.post_net(output)
        pis = torch.softmax(logits, dim=-1)
        info = dict(preactivation = logits, h=h_n, c=c_n)
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

    def get_action(self, obs, deterministic=False):
        assert len(obs.shape) == 1
        with torch.no_grad():
            obs = (torch_ify(obs)[None], self.a_p[None])
            pis, info = self.step(obs, return_info=True)
        pis = np_ify(pis[0])
        if deterministic:
            action = np.random.choice(np.arange(pis.shape[0]),p=pis)
        else:
            action = np.argmax(pis)
        a_p = np.zeros(pis.shape[0])
        a_p[action] = 1
        self.a_p = torch_ify(a_p)
        self.h_p = info['h'].clone().reshape(-1)
        self.c_p = info['c'].clone().reshape(-1)
        return action, {}

    def reset(self):
        self.a_p = self.a_0.clone().detach()
        self.h_p = self.h_0.clone().detach()
        self.c_p = self.c_0.clone().detach()

if __name__ == '__main__':
    obs_dim = 5
    action_dim = 2
    hidden_dim = 3
    num_layers = 1
    post_net = torch.nn.Linear(hidden_dim, action_dim)
    a_0 = np.zeros(action_dim)
    h_0 = np.zeros(hidden_dim)
    c_0 = np.zeros(hidden_dim)
    policy = SoftmaxLSTMPolicy(
                a_0=a_0,
                h_0=h_0,
                c_0=c_0,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                post_net=post_net,
                num_layers = num_layers,
                )

    action = policy.get_action(np.ones(obs_dim))
    print(action)
    print(policy.a_p)
    print(policy.h_p)
    print(policy.c_p)
    action = policy.get_action(np.ones(obs_dim))
    print(action)
    print(policy.a_p)
    print(policy.h_p)
    print(policy.c_p)
    obs_batch = torch.zeros((8,6,obs_dim+action_dim))
    pis = policy(obs_batch)
    print(pis.shape)
