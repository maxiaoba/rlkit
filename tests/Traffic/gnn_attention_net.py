import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from rlkit.torch.networks import Mlp

def get_activation(activation):
    print(activation)
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError

class GNNAttentionNet(torch.nn.Module):
    def __init__(self, 
                pre_graph_builder, 
                node_dim,
                conv_type='GCN',
                num_conv_layers=3,
                hidden_activation=None,
                output_activation=None,
                ):
        super(GNNAttentionNet, self).__init__()

        # graph builder
        self.pre_graph_builder = pre_graph_builder

        # convs
        self.node_input_dim = pre_graph_builder.output_dim
        self.node_dim = node_dim
        self.conv_type = conv_type
        self.num_conv_layers = num_conv_layers
        self.attentioner = self.build_attentioner(self.node_input_dim, self.node_dim)
        self.convs = self.build_convs(self.node_dim, self.num_conv_layers)
        self.hidden_activations = nn.ModuleList([get_activation(hidden_activation) for l in range(num_conv_layers+1)])
        self.output_activation = get_activation(output_activation)
        
    def build_attentioner(self, node_input_dim, node_dim):
        attentioner = pyg_nn.GATConv(node_input_dim, node_dim)
        return attentioner

    def build_convs(self, node_dim, num_conv_layers):
        convs = nn.ModuleList()
        for l in range(num_conv_layers):
            conv = self.build_conv_model(node_dim, node_dim)
            convs.append(conv)
        return convs

    def build_conv_model(self, node_in_dim, node_out_dim):
        if self.conv_type == 'GCN':
            return pyg_nn.GCNConv(node_in_dim,node_out_dim)
        else:
            raise NotImplementedError

    def forward(self, obs, return_attention_weights=False, **kwargs):
        batch_size = obs.shape[0]
        x, edge_index = self.pre_graph_builder(obs)
        x, edge_attention = self.attentioner(x, edge_index, return_attention_weights=True)
        x = self.hidden_activations[0](x)
        edge_index, attention_weights = edge_attention
        attention_weights = attention_weights.squeeze(-1)
        for l, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight = attention_weights)
            x = self.hidden_activations[l+1](x)
        x = x.reshape(batch_size,self.pre_graph_builder.node_num,self.node_dim)
        x = self.output_activation(x)
        if return_attention_weights:
            return x, edge_attention
        else:
            return x

