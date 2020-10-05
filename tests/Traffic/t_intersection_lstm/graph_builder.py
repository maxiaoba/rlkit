import torch
import numpy as np
import torch_geometric.utils as pyg_utils

class TrafficGraphBuilder(torch.nn.Module):
    def __init__(self, 
                input_dim,
                node_num,
                ego_init=torch.tensor([0.,1.]),
                other_init=torch.tensor([1.,0.]),
                ):
        super(TrafficGraphBuilder, self).__init__()

        self.input_dim = input_dim
        self.node_num = node_num
        self.ego_init = ego_init
        self.other_init = other_init

        self.output_dim = input_dim + self.ego_init.shape[0]
        self.full_edges = self.get_full_edges()

    def forward(self, obs):
        # x: (batch*num_node) x output_dim
        # edge_index: 2 x node_edge
        # messages from nodes in edge_index[0] are sent to nodes in edge_index[1]
        
        batch_size, node_num, obs_dim = obs.shape

        x = torch.zeros(batch_size,self.node_num, self.output_dim)
        x[:,:,:self.input_dim] = obs
        x[:,0,self.input_dim:] = self.ego_init[None,:]
        x[:,1:,self.input_dim:] = self.other_init[None,None,:]
        x = x.reshape(int(batch_size*self.node_num),self.output_dim)

        edge_index = self.full_edges[None,:].repeat(batch_size, 1, 1, 1) # batch x (node_num-1, 2, edge_num)
        index_offsets = torch.arange(batch_size)
        index_offsets = index_offsets * self.node_num
        edge_index = edge_index + index_offsets[:,None,None,None]
        valid_musk = self.get_valid_node_mask(obs) # batch x node_num-1
        edge_index = edge_index[valid_musk] # valid_veh_num x 2 x per_edge_num
        edge_index = edge_index.transpose(0,1).reshape(2,-1)
        # edge_index = pyg_utils.remove_self_loops(edge_index)[0]

        return x, edge_index

    def get_valid_node_mask(self, obs):
        # return a mask of all valid nodes
        # shape: batch x node_num-1 (assume node 0 always valid)
        batch_size, node_num, obs_dim = obs.shape
        valid_musk = (torch.sum(torch.abs(obs),dim=-1) != 0)
        return valid_musk[:,1:]
        # valid_musk = valid_musk[:,1:]
        # lower_valid_nums = torch.sum(valid_musk[:,:int(node_num/2)],dim=-1)
        # upper_valid_nums = torch.sum(valid_musk[:,int(node_num/2):],dim=-1)
        # return lower_valid_nums, upper_valid_nums

    def get_full_edges(self):
        # return edges if all vehicles are valid
        # shape (node_num-1, 2, per_edge_num)
        # each first dimension contains the out and in edges with ego (node 0) and out edge to others
        # there are self-loops! use torch_geometric.remove_self_loops later
        edges = torch.zeros(self.node_num-1, 2, 4, dtype=int)
        for i in range(self.node_num-1):
            idx = i+1
            edges[i,:,0] = torch.tensor([idx,0])
            edges[i,:,1] = torch.tensor([0,idx])
            if (idx==1):
                # the first vehicle in lower lane
                edges[i,:,2] = torch.tensor([idx,idx+1])
                edges[i,:,3] = torch.tensor([idx,int((self.node_num-1)/2)])
            elif (idx==int((self.node_num-1)/2)+1):
                # the first vehicle in upper lane
                edges[i,:,2] = torch.tensor([idx,idx+1])
                edges[i,:,3] = torch.tensor([idx,self.node_num-1])
            elif (idx==int((self.node_num-1)/2)):
                # the last vehicle in lower lane
                edges[i,:,2] = torch.tensor([idx,1])
                edges[i,:,3] = torch.tensor([idx,idx-1])
            elif (idx==self.node_num-1):
                # the last vehicle in upper lane
                edges[i,:,2] = torch.tensor([idx,int((self.node_num-1)/2)+1])
                edges[i,:,3] = torch.tensor([idx,idx-1])
            else:
                edges[i,:,2] = torch.tensor([idx,idx+1])
                edges[i,:,3] = torch.tensor([idx,idx-1])
        return edges



