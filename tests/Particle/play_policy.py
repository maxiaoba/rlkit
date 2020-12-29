import torch
import numpy as np
import time
import pdb
import matplotlib 
matplotlib.rcParams.update({'font.family': 'serif'})
import matplotlib.pyplot as plt

from rlkit.torch.policies.tanh_gaussian_policy import TanhGaussianPolicy
from rlkit.torch.policies.make_deterministic import MakeDeterministic
from rlkit.torch.policies.gumbel_softmax_policy import GumbelSoftmaxMlpPolicy
from rlkit.policies.argmax import ArgmaxDiscretePolicy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='simple_spread')
parser.add_argument('--mpl', type=int, default=25) # max path length
parser.add_argument('--sample_num', type=int, default=1000)
parser.add_argument('--extra_name', type=str, default=None)
args = parser.parse_args()

seeds = [0,1,2,3,4]
P_paths = [
            'MADDPGlayer2hidden64oa',
            'MASACGaussianlayer2hidden64oaer',
            'MASACGaussianlayer2hidden64oadna',
            'PRGGaussiank1hidden64oaonaceerdcigpna',
            'PRGGaussiank1hidden64oaonacedcigdnapna',
            'PRG3Gaussianhidden64k0m0ceerdcigpna',
            'PRG3Gaussianhidden64k0m1ceerdcigpna',
            'PRG3Gaussianhidden64k0m0cedcigdnapna',
            'PRG3Gaussianhidden64k0m1cedcigdnapna',
            ]
policy_names = [
                'MADDPG',
                'MASACer',
                'MASACdna',
                'PRGer',
                'PRGdna',
                'PRG3k0m0er',
                'PRG3k0m1er',
                'PRG3k0m0dna',
                'PRG3k0m1dna',
            ]

extra_name = (args.extra_name if args.extra_name else 'self'+'-'+'-'.join(policy_names))

pre_path = './Data/'+args.exp_name+'_mpl'+str(args.mpl)
log_dir = pre_path+'/tests/'+extra_name+'_ss'+str(args.sample_num)

sample_num = args.sample_num
max_path_length = args.mpl

import os
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

import csv
csv_name = 'payoff.csv'

import sys
sys.path.append("./multiagent-particle-envs")
from make_env import make_env
from particle_env_wrapper import ParticleEnv
env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False))
num_agent = env.num_agent

with open(log_dir+'/'+csv_name, mode='w') as csv_file:
    fieldnames = ['seed'] + [p_path for p_path in P_paths]

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for seed in seeds:
        print('seed: ',seed)
        row_content = dict()
        row_content['seed'] = seed

        with torch.no_grad():
            for pid in range(len(P_paths)):
                d_path = pre_path+'/'+P_paths[pid]+'/seed'+str(seed)\
                        +'/params.pkl'
                data = torch.load(d_path,map_location='cpu')
                policy_n = data['trainer/trained_policy_n']
                if isinstance(policy_n[0],TanhGaussianPolicy):
                    policy_n = [MakeDeterministic(policy) for policy in policy_n]
                elif  isinstance(policy_n[0],GumbelSoftmaxMlpPolicy):
                    policy_n = [ArgmaxDiscretePolicy(policy,use_preactivation=True) for policy in policy_n]

                Cr = []
                for i in range(sample_num):
                    o_n = env.reset()
                    cr = 0.
                    for step in range(max_path_length):
                        actions = []
                        for p in range(num_agent):
                            a,_ = policy_n[p].get_action(o_n[p])
                            actions.append(a)
                        o_n, r_n, done, _ = env.step(actions)
                        # env.render()
                        # time.sleep(0.1)
                        cr += np.mean(r_n)
                        if done.all():
                            break
                    Cr.append(cr)
                avg_reward = np.mean(Cr)
                print(P_paths[pid],': ',avg_reward)
                row_content[P_paths[pid]] = avg_reward

        writer.writerow(row_content)

