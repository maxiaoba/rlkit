import torch
import numpy as np
from graph_builder import TrafficGraphBuilder
from gnn_net import GNNNet

gb = TrafficGraphBuilder(input_dim=4,
						ego_init=torch.tensor([0.,1.]),
						other_init=torch.tensor([1.,0.]),
						edge_index=torch.tensor([[0,0,1,2],
											[1,2,0,0]]))

obs_batch = torch.tensor([[0,0,0,0,1,1,1,1,2,2,2,2],
						  [3,3,3,3,4,4,4,4,5,5,5,5]])

x, edge_index = gb(obs_batch)
print(x)
print(edge_index)

gnn = GNNNet( 
     		pre_graph_builder = gb, 
            node_dim = 16,
            output_dim = 2,
            post_mlp_kwargs = {'hidden_sizes':[64]},
            num_conv_layers=0)

q = gnn(obs_batch)
print(q)

