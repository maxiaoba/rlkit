import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from rlkit.torch.networks import Mlp

from network_utils import get_activation, build_conv_model

class LSTMNet(torch.nn.Module):
    def __init__(self, 
            obs_dim,
            action_dim,
            hidden_dim,
            num_layers,
                ):
        super(LSTMNet, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=obs_dim+action_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True)

    def forward(self, obs, prev_action, latent):
        # obs: batch x T x obs_dim
        # prev_action: batch x T x action_dim
        # latent: (batch x (num_layers*hidden_dim), batch x (num_layers*hidden_dim))
        batch_size, T, _ = obs.shape
        if prev_action is None:
            x = obs
        else:
            x = torch.cat((obs,prev_action),-1)
            
        h, c = latent
        if len(h.shape) == 1:
            h = h[None,:].repeat(batch_size,1)
        if len(c.shape) == 1:
            c = c[None,:].repeat(batch_size,1)
        h = h.reshape(batch_size, self.num_layers, self.hidden_dim).transpose(0,1).contiguous()
        c = c.reshape(batch_size, self.num_layers, self.hidden_dim).transpose(0,1).contiguous()

        self.lstm.flatten_parameters()
        o_n, (h_n, c_n) = self.lstm.forward(x, (h, c))
        # x_n: batch x T x dim
        # h_n, c_n: num_layers x batch x dim

        # flatten num_layers
        h_n = h_n.transpose(0,1).reshape(batch_size,int(self.num_layers*self.hidden_dim)).contiguous()
        c_n = c_n.transpose(0,1).reshape(batch_size,int(self.num_layers*self.hidden_dim)).contiguous()
        # bacth x (num_layers*dim)
        return o_n, (h_n, c_n)

