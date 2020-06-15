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
plot_file = pre_path+'/'+'seed'+str(args.seed)+'/cactor.png'

from differential_game import DifferentialGame
env = DifferentialGame(game_name=args.exp_name)

xs = np.linspace(-1,1,100)
ys = np.linspace(-1,1,100)
z1s = np.zeros((100,100))
z2s = np.zeros((100,100))

d_path = pre_path+'/'+'seed'+str(args.seed)+'/params.pkl'
data = torch.load(d_path,map_location='cpu')

c1net = data['trainer/cactor_n'][0]
c2net = data['trainer/cactor_n'][1]

for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        o_n = env.reset()
        c1_input = torch.tensor([float(o_n[0][0]),float(o_n[1][0]),y])[None,:]
        c1 = c1net(c1_input)
        c2_input = torch.tensor([float(o_n[0][0]),float(o_n[1][0]),x])[None,:]
        c2 = c2net(c2_input)
        z1s[j,i] = c1[0]
        z2s[j,i] = c2[0]
plt.figure()
plt.subplot(1,2,1)
plt.contourf(xs,ys,z1s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.subplot(1,2,2)
plt.contourf(xs,ys,z2s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.savefig(plot_file)
plt.close()