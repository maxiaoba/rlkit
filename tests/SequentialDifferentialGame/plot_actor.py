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
parser.add_argument('--log_dir', type=str, default='PRGMixGaussiank1m2hidden32oace')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pre_path = './Data/'+args.exp_name+'/'+args.log_dir
plot_file = pre_path+'/'+'seed'+str(args.seed)+'/actor.png'

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

with torch.no_grad():
    for a1 in a1s:
        o_n = env.reset()
        obs = torch.tensor([o_n[0]]).float()
        action = torch.tensor([[a1]])
        logprob1 = p1.log_prob(obs,action)[0].item()
        logprobs1.append(logprob1)
        
    for a2 in a2s:
        o_n = env.reset()
        obs = torch.tensor([o_n[1]]).float()
        action = torch.tensor([[a2]])
        logprob2 = p2.log_prob(obs,action)[0].item()
        logprobs2.append(logprob2)

plt.figure()
plt.subplot(1,2,1)
plt.plot(a1s,logprobs1)
# plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
# plt.ylim(-1,1)
plt.xlabel('a1')
plt.ylabel('logprob1')
plt.subplot(1,2,2)
plt.plot(a2s,logprobs2)
# plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
# plt.ylim(-1,1)
plt.xlabel('a2')
plt.ylabel('logprob2')
plt.savefig(plot_file)
plt.close()