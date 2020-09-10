import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

def build_conv_model(conv_type, node_in_dim, node_out_dim):
    if conv_type == 'GSage':
        return pyg_nn.SAGEConv(node_in_dim,node_out_dim)
    elif conv_type == 'GCN':
        return pyg_nn.GCNConv(node_in_dim,node_out_dim)
    elif conv_type == 'GAT':
        return pyg_nn.GATConv(node_in_dim, node_out_dim)
    elif conv_type == 'GSage2':
        from conv_layers import GraphSage2
        return GraphSage2(node_in_dim, node_out_dim)
    elif conv_type == 'GSageW':
        from conv_layers import GraphSageW
        return GraphSageW(node_in_dim, node_out_dim)
    else:
        raise NotImplementedError

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