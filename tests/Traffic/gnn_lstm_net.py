import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from rlkit.torch.networks import Mlp

from network_utils import get_activation, build_conv_model

class GNNLSTMNet(torch.nn.Module):
    def __init__(self, 
            node_num,
            gnn,
            lstm_ego,
            lstm_other,
                ):
        super(GNNLSTMNet, self).__init__()
        self.node_num = node_num
        self.gnn = gnn
        self.lstm_ego = lstm_ego
        self.lstm_other = lstm_other

    def identity_match(self, obs, o ,h, c):
        # obs: batch x n_num x dim
        # o: batch x n_num x dim
        # h: batch x n_num x (num_layer*dim)
        # c: batch x n_num x (num_layer*dim)
        valid_mask = (torch.sum(torch.abs(obs),dim=-1) != 0)
        # o[~valid_mask] = 0. # do not modify o
        h[~valid_mask] = 0.
        c[~valid_mask] = 0.

        return obs, o, h, c, valid_mask

    def step(self, obs, action, o, h, c, t, verbose=False):
        # obs: batch x n_num x (dim+1)
        # action: batch x actio_dim
        # o: batch x n_num x (dim+1)
        # h: batch x n_num x (num_layer*dim+1)
        # c: batch x n_num x (num_layer*dim+1)
        batch_size, n_num, _ = obs.shape

        obs, o, h, c, valid_mask = self.identity_match(obs, o, h, c)
        # batch x n_num x dim
        gnn_input = torch.cat((obs,h),-1) # batch x n_num x dim

        if verbose:
            print('gnn_input: ',gnn_input[0])
        e = self.gnn(gnn_input) # batch x n_num x dim
        e[~valid_mask] = 0.
        if verbose:
            print('h: ',h[0])
            print('c: ',c[0])
            print('e: ',e[0])
        # ego
        e_ego = e[:,0,:].unsqueeze(1) 
        h_ego = h[:,0,:]
        c_ego = c[:,0,:]
        action_ego = action[:,None,:]
        # batch x 1 x dim
        o_ego_n, (h_ego_n, c_ego_n) = self.lstm_ego.forward(e_ego, action_ego, 
                                                            (h_ego, c_ego))
        h_ego_n = h_ego_n.unsqueeze(1)
        c_ego_n = c_ego_n.unsqueeze(1)
        # batch x 1 x dim, (batch x dim), (batch x dim)

        # other
        e_other = e[:,1:,:].reshape(int(batch_size*(n_num-1)),-1).unsqueeze(1)
        h_other = h[:,1:,:].reshape(int(batch_size*(n_num-1)),-1)
        c_other = c[:,1:,:].reshape(int(batch_size*(n_num-1)),-1)
        action_other = None
        # (batch*(n_num-1)) x 1 x dim

        o_other_n, (h_other_n, c_other_n) = self.lstm_other.forward(e_other, action_other,
                                                                    (h_other, c_other))
        # (batch*(n_num-1)) x 1 x dim, (batch*(n_num-1)) x dim, (batch*(n_num-1)) x dim

        o_other_n = o_other_n.reshape(batch_size,n_num-1,-1)
        h_other_n = h_other_n.reshape(batch_size,n_num-1,-1)
        c_other_n = c_other_n.reshape(batch_size,n_num-1,-1)

        o_n = torch.cat((o_ego_n,o_other_n),1)
        h_n = torch.cat((h_ego_n,h_other_n),1)
        c_n = torch.cat((c_ego_n,c_other_n),1)
        # batch x n_num x dim

        o_n[~valid_mask] = 0.
        h_n[~valid_mask] = 0.
        c_n[~valid_mask] = 0.
        if verbose:
            print('h_n: ',h_n[0])
            print('c_n: ',c_n[0])
            print('o_n: ',o_n[0])

        return o_n, h_n, c_n

    def forward(self, obs_sequence, action_sequence, latent): 
        # obs_sequence: batch x T x (n_num*(dim+1))
        # action_sequence: batch x T x action_dim
        # latent: (batch x n_num x dim, batch x n_num x dim)
        batch_size, T, _ = obs_sequence.shape
        obs_sequence = obs_sequence.reshape(batch_size, T, self.node_num, -1)

        o, h, c = latent
        if len(o.shape) == 2:
            o = o[None,:,:].repeat(batch_size,1,1)
        if len(h.shape) == 2:
            h = h[None,:,:].repeat(batch_size,1,1)
        if len(c.shape) == 2:
            c = c[None,:,:].repeat(batch_size,1,1)

        o_sequence = []
        for t in range(T):
            obs = obs_sequence[:,t,:,:]
            action = action_sequence[:,t,:]
            o, h, c = self.step(obs, action, o, h, c, t)
            o_sequence.append(o.unsqueeze(1))

        o_sequence = torch.cat(o_sequence,1)
        return o_sequence, (o, h, c)

