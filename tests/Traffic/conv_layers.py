import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch.nn.init import xavier_uniform_, zeros_

import torch.nn as nn
import torch.nn.functional as F
from network_utils import get_activation

class GraphSage2(MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels,
                 activation='relu',
                 normalize_emb=False,
                 aggr='mean'):
        super(GraphSage2, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.message_lin = nn.Linear(2*in_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels+out_channels, out_channels)
        self.message_activation = get_activation(activation)
        self.normalize_emb = normalize_emb

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index):
        # x_j has shape [E, in_channels]
        # edge_index has shape [2, E]
        m_j = torch.cat((x_i, x_j),dim=-1)
        m_j = self.message_activation(self.message_lin(m_j))
        return m_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        aggr_out = self.agg_lin(torch.cat((aggr_out, x),dim=-1))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out
