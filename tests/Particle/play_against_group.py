import torch
import numpy as np
import time
import pdb
import matplotlib 
matplotlib.rcParams.update({'font.family': 'serif'})
import matplotlib.pyplot as plt

from rlkit.torch.policies.tanh_gaussian_policy import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.policies.gumbel_softmax_policy import GumbelSoftmaxMlpPolicy
from rlkit.policies.argmax import ArgmaxDiscretePolicy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='simple_push')
parser.add_argument('--sample_num', type=int, default=1000)
parser.add_argument('--extra_name', type=str, default='')
args = parser.parse_args()

if args.exp_name == 'simple_adversary':
    groups = [[0],[1,2]]
elif args.exp_name == 'simple_tag':
    groups = [[0,1,2],[3]]
elif args.exp_name == 'simple_push':
    groups = [[0],[1]]
print('groups: ',groups)

seeds = [0,1,2]
P_paths = [
            'MADDPG',
            # 'MADDPGonline_action',
            'MASAC',
            # 'MASAConline_action',
            # 'PRGk1',
            # 'PRGk1online_action',
            # 'PRGGaussiank1',
            # 'PRGGaussiank1online_action',
            'PRGGaussiank1online_actioncentropy'
            ]
policy_names = [
                'MADDPG',
                # 'MADDPGonline',
                'MASAC',
                # 'MASAConline',
                # 'PRGk1',
                # 'PRGk1online',
                # 'PRGGaussiank1',
                # 'PRGGaussiank1online',
                'PRGGaussiank1onlinecentropy'
            ]

extra_name = args.extra_name

pre_path = './Data/'+args.exp_name
log_dir = pre_path+'/tests/'+extra_name+'_ss'+str(args.sample_num)

sample_num = args.sample_num
max_path_length = 100

import os
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

import csv
csv_name = 'payoff.csv'

import sys
sys.path.append("./multiagent-particle-envs")
from make_env import make_env
from particle_env_wrapper import ParticleEnv
# env = ParticleEnv(make_env(args.exp_name,discrete_action_space=True,discrete_action_input=True))
env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False))

with open(log_dir+'/'+csv_name, mode='w') as csv_file:
    fieldnames0 = ['seed'] + ['p'+str(pid) for pid in range(len(P_paths))]
    fieldnames1 = []
    fieldnames2 = []
    for p1id in range(len(P_paths)):
        for p2id in range(len(P_paths)):
            fieldnames1.append('p1'+str(p1id)+'p2'+str(p2id)+'_1')
            fieldnames1.append('p1'+str(p1id)+'p2'+str(p2id)+'_2')

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames0+fieldnames1+fieldnames2)
    writer.writeheader()
    Mat1 = []
    Mat2 = []
    for seed in seeds:
        mat1 = np.zeros((len(P_paths),len(P_paths)))
        mat2 = np.zeros((len(P_paths),len(P_paths)))
        print('seed: ',seed)
        row_content = dict()
        row_content['seed'] = seed
        p_paths = []
        for pid in range(len(P_paths)):
            row_content['p'+str(pid)] = P_paths[pid]
            p_paths.append(P_paths[pid]+'/'+'seed'+str(seed))

        with torch.no_grad():
            # p1s = []
            # p2s = []
            players = []
            for pid in range(len(P_paths)):
                d_path = pre_path+'/'+P_paths[pid]+'/seed'+str(seed)\
                        +'/params.pkl'
                data = torch.load(d_path,map_location='cpu')
                policy_n = data['trainer/trained_policy_n']
                if isinstance(policy_n[0],TanhGaussianPolicy):
                    policy_n = [MakeDeterministic(policy) for policy in policy_n]
                elif  isinstance(policy_n[0],GumbelSoftmaxMlpPolicy):
                    policy_n = [ArgmaxDiscretePolicy(policy,use_preactivation=True) for policy in policy_n]
                # p1 = [policy_n[i] for i in groups[0]]
                # p2 = [policy_n[i] for i in groups[1]]
                # p1s.append(p1)
                # p2s.append(p2)
                players.append(policy_n)

            for p1id in range(len(P_paths)):
                for p2id in range(len(P_paths)):
                    player1 = players[p1id]
                    player2 = players[p2id]

                    Cr1, Cr2 = [],[]
                    for i in range(sample_num):
                        o_n = env.reset()
                        cr1, cr2 = 0, 0
                        for step in range(max_path_length):
                            actions = []
                            for sub_pid in groups[0]:
                                a1,_ = player1[sub_pid].get_action(o_n[sub_pid])
                                actions.append(a1)
                            for sub_pid in groups[1]:
                                a2,_ = player2[sub_pid].get_action(o_n[sub_pid])
                                actions.append(a2)
                            o_n, r_n, done, _ = env.step(actions)
                            cr1 += r_n[groups[0][0]]
                            cr2 += r_n[groups[1][0]]
                        Cr1.append(cr1)
                        Cr2.append(cr2)
                    p1_avg_reward = np.mean(Cr1)
                    p2_avg_reward = np.mean(Cr2)
                    mat1[p1id,p2id] = p1_avg_reward
                    mat2[p1id,p2id] = p2_avg_reward
                    print('p1'+str(p1id)+'p2'+str(p2id)+'_1',': ',p1_avg_reward)
                    print('p1'+str(p1id)+'p2'+str(p2id)+'_2',': ',p2_avg_reward)
                    row_content['p1'+str(p1id)+'p2'+str(p2id)+'_1'] = p1_avg_reward
                    row_content['p1'+str(p1id)+'p2'+str(p2id)+'_2'] = p2_avg_reward

        Mat1.append(mat1)
        Mat2.append(mat2)
        writer.writerow(row_content)

