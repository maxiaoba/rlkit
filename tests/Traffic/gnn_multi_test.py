import torch
import numpy as np
from graph_builder_multi import MultiTrafficGraphBuilder
from gnn_net import GNNNet

gb = MultiTrafficGraphBuilder(
								input_dim=4,
								node_num=13,
								ego_init=torch.tensor([0.,1.]),
								other_init=torch.tensor([1.,0.]),
							)
print('full_edges: ',gb.full_edges)

obs_batch = torch.zeros(2,13,4)
obs_batch[0,0,:] = 1.
obs_batch[0,1,:] = 1.
obs_batch[0,2,:] = 1.
obs_batch[0,7,:] = 1.
obs_batch[1,0,:] = 1.
obs_batch[1,1,:] = 1.
obs_batch[1,2,:] = 1.
obs_batch[1,3,:] = 1.
obs_batch = obs_batch.reshape(2,-1)

valid_mask = gb.get_valid_node_mask(obs_batch)
print('valid_mask: ',valid_mask)

x, edge_index = gb(obs_batch)
print('x: ',x)
print('edge_index: ',edge_index)
