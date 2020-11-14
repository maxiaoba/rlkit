import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
from rlkit.torch import pytorch_util as ptu
# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='max2')
parser.add_argument('--log_dir', type=str, default='PRGMixGaussiank1m2hidden32oace')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pre_path = './Data/'+args.exp_name+'/'+args.log_dir
plot_file1 = pre_path+'/'+'seed'+str(args.seed)+'/q1.png'
plot_file2 = pre_path+'/'+'seed'+str(args.seed)+'/q2.png'

from sequential_differential_game import SequentialDifferentialGame
env = SequentialDifferentialGame(game_name=args.exp_name)

xs = np.linspace(-1,1,100)
ys = np.linspace(-1,1,100)
q11s = np.zeros((100,100))
q12s = np.zeros((100,100))
q1mins = np.zeros((100,100))
q21s = np.zeros((100,100))
q22s = np.zeros((100,100))
q2mins = np.zeros((100,100))

d_path = pre_path+'/'+'seed'+str(args.seed)+'/params.pkl'
data = torch.load(d_path,map_location='cpu')

q11net = data['trainer/qf1_n'][0]
q12net = data['trainer/qf1_n'][1]
q21net = data['trainer/qf2_n'][0]
q22net = data['trainer/qf2_n'][1]
with torch.no_grad():
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            o_n = env.reset()
            q_input = torch.tensor([*o_n[0],*o_n[1],x,y]).float()[None,:]
            q11 = q11net(q_input)
            q12 = q12net(q_input)
            q11s[j,i] = q11[0]
            q12s[j,i] = q12[0]
            q1mins[j,i] = np.minimum(q11[0],q12[0])
            q21 = q21net(q_input)
            q22 = q22net(q_input)
            q21s[j,i] = q21[0]
            q22s[j,i] = q22[0]
            q2mins[j,i] = np.minimum(q21[0],q22[0])

plt.figure()
plt.subplot(1,3,1)
plt.contourf(xs,ys,q11s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.subplot(1,3,2)
plt.contourf(xs,ys,q12s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.subplot(1,3,3)
plt.contourf(xs,ys,q1mins)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.savefig(plot_file1)
plt.close()

plt.figure()
plt.subplot(1,3,1)
plt.contourf(xs,ys,q21s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.subplot(1,3,2)
plt.contourf(xs,ys,q22s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.subplot(1,3,3)
plt.contourf(xs,ys,q2mins)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('a2')
plt.savefig(plot_file2)
plt.close()