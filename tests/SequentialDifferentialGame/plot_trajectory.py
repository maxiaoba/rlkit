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
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--num', type=int, default=1)
parser.add_argument('--log_dir', type=str, default='PRGMixGaussiank1m2hidden32oace')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pre_path = './Data/'+args.exp_name+'/'+args.log_dir
plot_file = pre_path+'/'+'seed'+str(args.seed)+'/traj.png'

from sequential_differential_game import SequentialDifferentialGame
env = SequentialDifferentialGame(game_name=args.exp_name)

a1s = np.linspace(-1,1,100)
a2s = np.linspace(-1,1,100)
logprobs1 = []
logprobs2 = []

d_path = pre_path+'/'+'seed'+str(args.seed)+'/params.pkl'
data = torch.load(d_path,map_location='cpu')

p1 = data['trainer/trained_policy_n'][0]
p2 = data['trainer/trained_policy_n'][1]
if args.eval:
    if not args.log_dir.startswith('MADDPG'):
        from rlkit.torch.policies.make_deterministic import MakeDeterministic
        p1 = MakeDeterministic(data['trainer/trained_policy_n'][0])
        p2 = MakeDeterministic(data['trainer/trained_policy_n'][1])

plt.figure()
for _ in range(args.num):
    o_n = env.reset()
    xs = [o_n[0][0]]
    ys = [o_n[0][1]]
    obs1 = torch.tensor(o_n[0]).float()
    obs2 = torch.tensor(o_n[1]).float()
    for i in range(5):
        a1, _ = p1.get_action(obs1)
        a2, _ = p2.get_action(obs2)
        o_n, r_n, done, info = env.step([a1,a2])
        xs.append(o_n[0][0])
        ys.append(o_n[0][1])
        obs1 = torch.tensor(o_n[0]).float()
        obs2 = torch.tensor(o_n[1]).float()
    plt.plot(xs, ys,'o-')
    plt.plot(xs[0],ys[0],'o',color='green')
    plt.plot(xs[-1],ys[-1],'o',color='black')

plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(plot_file)
plt.close()