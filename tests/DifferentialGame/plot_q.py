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
plot_file1 = pre_path+'/'+'seed'+str(args.seed)+'/q1.png'
plot_file2 = pre_path+'/'+'seed'+str(args.seed)+'/q2.png'

from differential_game import DifferentialGame
env = DifferentialGame(game_name=args.exp_name)

xs = np.linspace(-1,1,100)
ys = np.linspace(-1,1,100)
z11s = np.zeros((100,100))
z12s = np.zeros((100,100))
z21s = np.zeros((100,100))
z22s = np.zeros((100,100))

d_path = pre_path+'/'+'seed'+str(args.seed)+'/params.pkl'
data = torch.load(d_path,map_location='cpu')

q11net = data['trainer/qf1_n'][0]
q12net = data['trainer/qf1_n'][1]
q21net = data['trainer/qf2_n'][0]
q22net = data['trainer/qf2_n'][1]

for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        o_n = env.reset()
        q_input = torch.tensor([float(o_n[0][0]),float(o_n[1][0]),x,y])[None,:]
        q11 = q11net(q_input)
        q12 = q12net(q_input)
        z11s[j,i] = q11[0]
        z12s[j,i] = q12[0]
        q21 = q21net(q_input)
        q22 = q22net(q_input)
        z21s[j,i] = q21[0]
        z22s[j,i] = q22[0]

plt.figure()
plt.subplot(1,2,1)
plt.contourf(xs,ys,z11s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.subplot(1,2,2)
plt.contourf(xs,ys,z12s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.savefig(plot_file1)
plt.close()

plt.figure()
plt.subplot(1,2,1)
plt.contourf(xs,ys,z21s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.subplot(1,2,2)
plt.contourf(xs,ys,z22s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.savefig(plot_file2)
plt.close()