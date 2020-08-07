import torch
import numpy as np

class TrafficGraphBuilder(torch.nn.Module):
	def __init__(self, 
				input_dim=4,
				ego_init=torch.tensor([0.,1.]),
				other_init=torch.tensor([1.,0.]),
				edge_index=torch.tensor([[0,0,1,2],
									 [1,2,0,0]])
				):
		super(TrafficGraphBuilder, self).__init__()

		self.input_dim = input_dim
		self.ego_init = ego_init
		self.other_init = other_init
		self.edge_index = edge_index

		self.output_dim = input_dim + self.ego_init.shape[0]

	def forward(self, obs):
		# x: (batch*num_node) x output_dim
		# edge_index: 2 x node_edge

		batch_size, obs_dim = obs.shape
		node_num = int(obs_dim/self.input_dim)

		x = torch.zeros(batch_size,node_num, self.output_dim)
		obs = obs.reshape(batch_size, node_num, self.input_dim)
		x[:,:,:self.input_dim] = obs
		x[:,0,self.input_dim:] = self.ego_init[None,:]
		x[:,1:,self.input_dim:] = self.other_init[None,None,:]
		x = x.reshape(int(batch_size*node_num),self.output_dim)

		edge_index = torch.zeros(batch_size,2,self.edge_index.shape[1],dtype=int)
		edge_index[:,:,:] = self.edge_index[None,:,:]
		index_offsets = torch.arange(batch_size)
		index_offsets = index_offsets * node_num
		edge_index = edge_index + index_offsets[:,None,None]
		edge_index = edge_index.transpose(0,1).contiguous() # 2 x batch x num_edge
		edge_index = edge_index.reshape(2,-1) # 2 x (batch * num_edge)

		return x, edge_index
