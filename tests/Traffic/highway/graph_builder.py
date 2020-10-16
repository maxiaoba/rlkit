import torch
import numpy as np
from numba import jit
import torch_geometric.utils as pyg_utils
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np, np_ify, torch_ify

class TrafficGraphBuilder(torch.nn.Module):
    def __init__(self,
                input_dim,
                node_num,
                ego_init=np.array([0.,1.]),
                other_init=np.array([1.,0.]),
                ):
        super(TrafficGraphBuilder, self).__init__()

        self.input_dim = input_dim
        self.node_num = node_num
        self.ego_init = torch_ify(ego_init)
        self.other_init = torch_ify(other_init)

        self.output_dim = input_dim + self.ego_init.shape[0]

    def forward(self, obs, valid_musk=None):
        # x: (batch*num_node) x output_dim
        # edge_index: 2 x node_edge
        # messages from nodes in edge_index[0] are sent to nodes in edge_index[1]
        
        batch_size, node_num, obs_dim = obs.shape

        x = torch.zeros(batch_size,self.node_num, self.output_dim).to(ptu.device)
        x[:,:,:self.input_dim] = obs
        x[:,0,self.input_dim:] = self.ego_init[None,:]
        x[:,1:,self.input_dim:] = self.other_init[None,None,:]
        x = x.reshape(int(batch_size*self.node_num),self.output_dim)

        # xs = obs[:,:,0]
        # ys = obs[:,:,1]
        # upper_indices = torch.where(ys > 4.)
        # lower_indices = torch.where((ys > 0.) and (ys <= 4.))
        obs = np_ify(obs)
        edge_index = get_edge_index(obs) #batch x 2 x max_edge_num
        edge_index = np.swapaxes(edge_index,0,1).reshape(2,-1)
        edge_index = np.unique(edge_index, axis=1)
        edge_index = torch_ify(edge_index).long()
        edge_index = pyg_utils.remove_self_loops(edge_index)[0]

        return x, edge_index

    def get_valid_node_mask(self, obs):
        # return a mask of all valid nodes
        # shape: batch x node_num
        valid_musk = (obs[:,:,1] != 0) # y!= 0
        return valid_musk

@jit(nopython=True)
def get_edge_index(obs):
    batch_size, node_num, obs_dim = obs.shape
    Xs = obs[:,:,0]
    Ys = obs[:,:,1]
    Edges = np.zeros((batch_size,2,node_num*(3+4)))
    for i in range(batch_size):
        xs = Xs[i]
        ys = Ys[i]
        sort_index = np.argsort(xs)
        sort_y = ys[sort_index]
        lane0_sort_mask = ((sort_y<=1./3.) * (sort_y>0.))
        lane1_sort_mask = ((sort_y>1./3.) * (sort_y<=2./3.))
        lane2_sort_mask = ((sort_y>=2./3.) * (sort_y<1.))

        lane0_sort_index = sort_index[lane0_sort_mask]
        lane1_sort_index = sort_index[lane1_sort_mask]
        lane2_sort_index = sort_index[lane2_sort_mask]
        lane01_sort_index = sort_index[(lane0_sort_mask | lane1_sort_mask)]
        lane12_sort_index = sort_index[(lane1_sort_mask | lane2_sort_mask)]

        lane0_edges = np.concatenate((np.expand_dims(np.concatenate((lane0_sort_index[:-1],lane0_sort_index[1:])),axis=0),
                                np.expand_dims(np.concatenate((lane0_sort_index[1:],lane0_sort_index[:-1])),axis=0)),axis=0)
        lane1_edges = np.concatenate((np.expand_dims(np.concatenate((lane1_sort_index[:-1],lane1_sort_index[1:])),axis=0),
                                np.expand_dims(np.concatenate((lane1_sort_index[1:],lane1_sort_index[:-1])),axis=0)),axis=0)
        lane2_edges = np.concatenate((np.expand_dims(np.concatenate((lane2_sort_index[:-1],lane2_sort_index[1:])),axis=0),
                                np.expand_dims(np.concatenate((lane2_sort_index[1:],lane2_sort_index[:-1])),axis=0)),axis=0)
        lane01_edges = np.concatenate((np.expand_dims(np.concatenate((lane01_sort_index[:-1],lane01_sort_index[1:])),axis=0),
                                np.expand_dims(np.concatenate((lane01_sort_index[1:],lane01_sort_index[:-1])),axis=0)),axis=0)
        lane12_edges = np.concatenate((np.expand_dims(np.concatenate((lane12_sort_index[:-1],lane12_sort_index[1:])),axis=0),
                                np.expand_dims(np.concatenate((lane12_sort_index[1:],lane12_sort_index[:-1])),axis=0)),axis=0)

        edges = np.concatenate((lane0_edges, lane1_edges, lane2_edges, lane01_edges, lane12_edges),axis=-1)+i*node_num
        Edges[i,:,:edges.shape[1]] = edges
    return Edges


