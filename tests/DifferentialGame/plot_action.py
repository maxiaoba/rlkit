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
parser.add_argument('--exp_name', type=str, default='zero_sum')
parser.add_argument('--log_dir', type=str, default='PRGGaussiank1online_action')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pre_path = './Data/'+args.exp_name+'/'+args.log_dir
plot_file = pre_path+'/'+'seed'+str(args.seed)+'/action.png'

from differential_game import DifferentialGame
env = DifferentialGame(game_name=args.exp_name)

d_path = pre_path+'/'+'seed'+str(args.seed)+'/params.pkl'
data = torch.load(d_path,map_location='cpu')

p1 = data['trainer/trained_policy_n'][0]
p2 = data['trainer/trained_policy_n'][1]

sample_num = 100
a1s = []
a2s = []

o_n = env.reset()
for _ in range(sample_num):
    a1 = p1(torch.tensor([float(o_n[0])])).detach().numpy()
    a2 = p2(torch.tensor([float(o_n[0])])).detach().numpy()
    a1s.append(a1[0])
    a2s.append(a2[0])
plt.figure()
plt.scatter(a1s,a2s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.savefig(plot_file)
plt.close()