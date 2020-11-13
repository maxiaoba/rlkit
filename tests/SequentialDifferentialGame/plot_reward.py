import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np
import argparse
# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='zero_sum')
args = parser.parse_args()

plot_file = './Data/'+args.exp_name+'_reward.png'

from differential_game import DifferentialGame
env = DifferentialGame(game_name=args.exp_name)

xs = np.linspace(-1,1,100)
ys = np.linspace(-1,1,100)
z1s = np.zeros((100,100))
z2s = np.zeros((100,100))

for i,x in enumerate(xs):
    for j,y in enumerate(ys):
        env.reset()
        o_n,r_n,d_n,info = env.step([x,y])
        z1s[j,i] = r_n[0]
        z2s[j,i] = r_n[1]
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