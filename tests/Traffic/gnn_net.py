import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from rlkit.torch.networks import Mlp

from network_utils import get_activation, build_conv_model

class GNNNet(torch.nn.Module):
    def __init__(self, 
                pre_graph_builder, 
                node_dim,
                conv_type='GSage',
                num_conv_layers=3,
                hidden_activation=None,
                output_activation=None,
                ):
        super(GNNNet, self).__init__()
        print('gnn')
        # graph builder
        self.pre_graph_builder = pre_graph_builder

        # convs
        self.node_input_dim = pre_graph_builder.output_dim
        self.node_dim = node_dim
        self.conv_type = conv_type
        self.num_conv_layers = num_conv_layers
        self.convs = self.build_convs(self.node_input_dim, self.node_dim, self.num_conv_layers)
        self.hidden_activations = nn.ModuleList([get_activation(hidden_activation) for l in range(num_conv_layers)])
        self.output_activation = get_activation(output_activation)
        

    def build_convs(self, node_input_dim, node_dim, num_conv_layers):
        convs = nn.ModuleList()
        conv = build_conv_model(self.conv_type, node_input_dim, node_dim)
        convs.append(conv)
        for l in range(1,num_conv_layers):
            conv = build_conv_model(self.conv_type, node_dim, node_dim)
            convs.append(conv)
        return convs

    def forward(self, obs, valid_mask=None):
        batch_size = obs.shape[0]
        x, edge_index = self.pre_graph_builder(obs, valid_mask)
        for l, conv in enumerate(self.convs):
            # self.check_input(x, edge_index)
            x = conv(x, edge_index)
            x = self.hidden_activations[l](x)
        x = x.reshape(batch_size,-1,self.node_dim)
        x = self.output_activation(x)
        return x

    def check_input(self, xs, edge_index):
        Os = {}
        for indx in range(edge_index.shape[1]):
            i=edge_index[1,indx].detach().numpy()
            j=edge_index[0,indx].detach().numpy()
            xi=xs[i].detach().numpy()
            xj=list(xs[j].detach().numpy())
            if str(i) not in Os.keys():
                Os[str(i)] = {'x_j':[]}
            Os[str(i)]['x_i'] = xi
            Os[str(i)]['x_j'] += xj

        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1,2,1)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_i'],label=str(i))
            plt.title('x_i')
        plt.legend()
        plt.subplot(1,2,2)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_j'],label=str(i))
            plt.title('x_j')
        plt.legend()
        plt.show()


