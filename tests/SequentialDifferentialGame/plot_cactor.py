import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='max2')
parser.add_argument('--log_dir', type=str, default='PRGGaussiank1oace')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pre_path = './Data/'+args.exp_name+'/'+args.log_dir
plot_file = pre_path+'/'+'seed'+str(args.seed)+'/cactor.png'

from sequential_differential_game import SequentialDifferentialGame
env = SequentialDifferentialGame(game_name=args.exp_name)

a1s = np.linspace(-1,1,100)
a2s = np.linspace(-1,1,100)
c1s = []
c2s = []

d_path = pre_path+'/'+'seed'+str(args.seed)+'/params.pkl'
data = torch.load(d_path,map_location='cpu')

c1net = data['trainer/cactor_n'][0]
c2net = data['trainer/cactor_n'][1]
from rlkit.torch.policies.make_deterministic import MakeDeterministic
c1net = MakeDeterministic(c1net)
c2net = MakeDeterministic(c2net)

with torch.no_grad():
	for a2 in a2s:
	    o_n = env.reset()
	    c1_input = torch.tensor([*o_n[0],*o_n[1],a2]).float()
	    c1, _ = c1net.get_action(c1_input)
	    c1s.append(c1[0])
	    
	for a1 in a1s:
	    o_n = env.reset()
	    c2_input = torch.tensor([*o_n[0],*o_n[1],a1]).float()
	    c2, _ = c2net.get_action(c2_input)
	    c2s.append(c2[0])

plt.figure()
plt.subplot(1,2,1)
plt.plot(a2s,c1s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a2')
plt.ylabel('c1')
plt.subplot(1,2,2)
plt.plot(a1s,c2s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('c2')
plt.savefig(plot_file)
plt.close()