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
parser.add_argument('--ar', type=float, default=10.) # action range
parser.add_argument('--log_dir', type=str, default='PRGGaussiank1hidden32oaonaceersdadcigpna')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pre_path = './Data/'+args.exp_name+'ar'+str(args.ar)+'/'+args.log_dir
plot_file = pre_path+'/'+'seed'+str(args.seed)+'/alpha.png'

from sequential_differential_game import SequentialDifferentialGame
env = SequentialDifferentialGame(game_name=args.exp_name)

xs = np.linspace(-1,1,100)
ys = np.linspace(-1,1,100)
a1s = np.zeros((100,100))
ca1s = np.zeros((100,100))
a2s = np.zeros((100,100))
ca2s = np.zeros((100,100))

d_path = pre_path+'/'+'seed'+str(args.seed)+'/params.pkl'
data = torch.load(d_path,map_location='cpu')

a1net = data['trainer/log_alpha_n'][0]
ca1net = data['trainer/log_calpha_n'][0]
a2net = data['trainer/log_alpha_n'][1]
ca2net = data['trainer/log_calpha_n'][1]
with torch.no_grad():
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            a_input = torch.tensor([x,y,x,y]).float()[None,:]
            a1 = a1net(a_input)
            ca1 = ca1net(a_input)
            a2 = a2net(a_input)
            ca2 = ca2net(a_input)
            a1s[j,i] = a1[0]
            ca1s[j,i] = ca1[0]
            a2s[j,i] = a2[0]
            ca2s[j,i] = ca2[0]

plt.figure()

plt.subplot(2,2,1)
plt.contourf(xs,ys,a1s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('a1')

plt.subplot(2,2,2)
plt.contourf(xs,ys,a2s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('a2')

plt.subplot(2,2,3)
plt.contourf(xs,ys,ca1s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('ca1')

plt.subplot(2,2,4)
plt.contourf(xs,ys,ca2s)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('ca2')

plt.savefig(plot_file)
plt.close()