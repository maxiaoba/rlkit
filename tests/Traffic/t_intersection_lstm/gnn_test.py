import torch
import numpy as np
import torch_geometric.utils as pyg_utils
import networkx
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
from torch_geometric.data import Data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='t_intersection_lstm')
parser.add_argument('--noise', type=float, default=0.05)
parser.add_argument('--yld', type=float, default=0.5)
parser.add_argument('--ds', type=float, default=0.1)
args = parser.parse_args()
env_kwargs=dict(
    num_updates=1,
    obs_noise=args.noise,
    yld=args.yld,
    driver_sigma=args.ds,
)
from traffic.make_env import make_env
env = make_env(args.exp_name,**env_kwargs)
obs_dim = env.observation_space.low.size
action_dim = env.action_space.n
label_num = env.label_num
label_dim = env.label_dim

def check_graph(obs):
	obs_batch = torch.tensor([obs]).reshape(1,node_num,4)
	valid_mask = gb.get_valid_node_mask(obs_batch)
	print('valid_mask: ',valid_mask)

	x, edge_index = gb(obs_batch)
	print('x: ',x)
	print('edge_index: ',edge_index)

	data = Data(x=x, edge_index=edge_index)
	ng = pyg_utils.to_networkx(data)
	pos = {}
	for node in ng.nodes:
		pos[node] = obs_batch[0,node,:2].numpy()

	plt.figure()
	networkx.draw(ng,pos)
	plt.show()

from graph_builder import TrafficGraphBuilder
node_num = env.max_veh_num+1
gb = TrafficGraphBuilder(input_dim=4, node_num=node_num,
                        ego_init=torch.tensor([0.,1.]),
                        other_init=torch.tensor([1.,0.]),
                        )

obs = env.reset()
env.render()
check_graph(obs)
while True:
	obs,r,done,info = env.step(1)
	env.render()
	check_graph(obs)
