import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from rlkit.torch.networks import Mlp

from network_utils import get_activation, build_conv_model

class GNNLSTM2Net(torch.nn.Module):
    # from https://github.com/huang-xx/STGAT
    def __init__(self, 
            node_num,
            gnn,
            lstm1_ego,
            lstm1_other,
            lstm2_ego,
            lstm2_other,
                ):
        super(GNNLSTM2Net, self).__init__()
        self.node_num = node_num
        self.gnn = gnn
        self.lstm1_ego = lstm1_ego
        self.lstm1_other = lstm1_other
        self.lstm2_ego = lstm2_ego
        self.lstm2_other = lstm2_other

    def forward(self, obs_sequence, action_sequence, latent): 
        # obs_sequence: batch x T x (n_num*(dim+1))
        # action_sequence: batch x T x action_dim
        # latent: (batch x n_num x dim,) * 4
        batch_size, T, _ = obs_sequence.shape
        obs_sequence = obs_sequence.reshape(batch_size, T, self.node_num, -1)

        h1, c1, h2, c2 = latent
        if len(h1.shape) == 2:
            h1 = h1[None,:,:].repeat(batch_size,1,1)
            c1 = c1[None,:,:].repeat(batch_size,1,1)
            h2 = h2[None,:,:].repeat(batch_size,1,1)
            c2 = c2[None,:,:].repeat(batch_size,1,1)
        # batch x n_num x (num_layer*dim)

        # lstm1
        obs_ego_sequence = obs_sequence[:,:,0,:]
        h1_ego = h1[:,0,:]
        c1_ego = c1[:,0,:]
        o1_ego, (h1_ego_n, c1_ego_n) = self.lstm1_ego.forward(obs_ego_sequence, action_sequence, 
                                                                (h1_ego, c1_ego))
        # batch x T x dim, (batch x dim, batch x dim)
        o1_ego = o1_ego.unsqueeze(2)
        h1_ego_n = h1_ego_n.unsqueeze(1)
        c1_ego_n = c1_ego_n.unsqueeze(1)
        # batch x T x 1 x dim, batch x 1 x dim, batch x 1 x dim

        obs_other_sequence = obs_sequence[:,:,1:,:].transpose(1,2) # batch x (n_num-1) x T x dim
        obs_other_sequence = obs_other_sequence.reshape(batch_size*(self.node_num-1), T, -1)
        h1_other = h1[:,1:,:].reshape(batch_size*(self.node_num-1), -1)
        c1_other = c1[:,1:,:].reshape(batch_size*(self.node_num-1), -1)
        action_other = None
        o1_other, (h1_other_n, c1_other_n) = self.lstm1_other.forward(obs_other_sequence, action_other,
                                                                        (h1_other, c1_other))
        # (batch*(n_num-1)) x T x dim, ((batch*(n_num-1))x dim, (batch*(n_num-1)) x dim)
        o1_other = o1_other.reshape(batch_size, self.node_num-1, T, -1).transpose(1,2)
        h1_other_n = h1_other_n.reshape(batch_size, self.node_num-1, -1)
        c1_other_n = c1_other_n.reshape(batch_size, self.node_num-1, -1)
        # batch x T x (n_num-1) x dim, batch x (n_num-1) x dim, batch x (n_num-1) x dim)

        o1 = torch.cat((o1_ego,o1_other),2)
        h1_n = torch.cat((h1_ego_n,h1_other_n),1)
        c1_n = torch.cat((c1_ego_n,c1_other_n),1)
        # batch x T x n_num x dim, batch x n_num x dim, batch x n_num x dim

        # gnn
        valid_mask = (torch.sum(torch.abs(obs_sequence),dim=-1) != 0) # batch_size x T x n_num
        gnn_input = o1.reshape(batch_size*T, self.node_num, -1)
        e = self.gnn(gnn_input, valid_mask.reshape(batch_size*T, self.node_num)) # (batch*T) x n_num x dim
        e = e.reshape(batch_size, T, self.node_num, -1)
        e[~valid_mask] = 0.

        # lstm2
        e_ego = e[:,:,0,:]
        h2_ego = h2[:,0,:]
        c2_ego = c2[:,0,:]
        o2_ego, (h2_ego_n, c2_ego_n) = self.lstm2_ego.forward(e_ego, None, 
                                                                (h2_ego, c2_ego))
        # batch x T x dim, (batch x dim, batch x dim)
        o2_ego = o2_ego.unsqueeze(2)
        h2_ego_n = h2_ego_n.unsqueeze(1)
        c2_ego_n = c2_ego_n.unsqueeze(1)
        # batch x T x 1 x dim, batch x 1 x dim, batch x 1 x dim

        e_other = e[:,:,1:,:].transpose(1,2) # batch x (n_num-1) x T x dim
        e_other = e_other.reshape(batch_size*(self.node_num-1), T, -1)
        h2_other = h2[:,1:,:].reshape(batch_size*(self.node_num-1), -1)
        c2_other = c2[:,1:,:].reshape(batch_size*(self.node_num-1), -1)
        o2_other, (h2_other_n, c2_other_n) = self.lstm2_other.forward(e_other, None,
                                                                        (h2_other, c2_other))
        # (batch*(n_num-1)) x T x dim, ((batch*(n_num-1))x dim, (batch*(n_num-1)) x dim)
        o2_other = o2_other.reshape(batch_size, self.node_num-1, T, -1).transpose(1,2)
        h2_other_n = h2_other_n.reshape(batch_size, self.node_num-1, -1)
        c2_other_n = c2_other_n.reshape(batch_size, self.node_num-1, -1)
        # batch x T x (n_num-1) x dim, batch x (n_num-1) x dim, batch x (n_num-1) x dim)

        o2 = torch.cat((o2_ego,o2_other),2)
        h2_n = torch.cat((h2_ego_n,h2_other_n),1)
        c2_n = torch.cat((c2_ego_n,c2_other_n),1)
        # batch x T x n_num x dim, batch x n_num x dim, batch x n_num x dim

        o2[~valid_mask] = 0
        return o2, (h1_n, c1_n, h2_n, c2_n)

