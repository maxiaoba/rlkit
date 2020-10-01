import torch
import numpy as np
import torch_geometric.utils as pyg_utils
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
from torch_geometric.data import Data
import networkx

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='t_intersection_multi')
parser.add_argument('--nob', action='store_true', default=False)
parser.add_argument('--obs', type=str, default='full')
parser.add_argument('--label', type=str, default='full')
parser.add_argument('--yld', type=float, default=0.5)
parser.add_argument('--ds', type=float, default=0.1)
args = parser.parse_args()
env_kwargs=dict(
    num_updates=1,
    normalize_obs=args.nob,
    observe_mode=args.obs,
    label_mode=args.label,
    yld=args.yld,
    driver_sigma=args.ds,
)
from traffic.make_env import make_env
env = make_env(args.exp_name,**env_kwargs)
obs_dim = env.observation_space.low.size
action_dim = env.action_space.n
label_num = env.label_num
label_dim = env.label_dim

from graph_builder_multi import MultiTrafficGraphBuilder
gb = MultiTrafficGraphBuilder(input_dim=4, node_num=env.max_veh_num+1,
                        ego_init=torch.tensor([0.,1.]),
                        other_init=torch.tensor([1.,0.]),
                        )

obs = env.reset()
obs_batch = torch.tensor([obs])
valid_mask = gb.get_valid_node_mask(obs_batch)
print('valid_mask: ',valid_mask)

x, edge_index = gb(obs_batch)
print('x: ',x)
print('edge_index: ',edge_index)

data = Data(x=x, edge_index=edge_index)
ng = pyg_utils.to_networkx(data)
networkx.draw_planar(ng)
plt.show()
