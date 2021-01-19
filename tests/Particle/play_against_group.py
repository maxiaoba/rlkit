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
parser.add_argument('--exp_name', type=str, default='simple_push')
parser.add_argument('--boundary', action='store_true', default=False)
parser.add_argument('--num_ag', type=int, default=None)
parser.add_argument('--num_adv', type=int, default=None)
parser.add_argument('--num_l', type=int, default=None)
parser.add_argument('--mpl', type=int, default=25) # max path length
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--new', action='store_true', default=False)
parser.add_argument('--sample_num', type=int, default=1000)
parser.add_argument('--extra_name', type=str, default='')
args = parser.parse_args()

if args.num_ag:
    groups = [[i for i in range(args.num_adv)],[args.num_adv+i for i in range(args.num_ag)]]
else:
    if args.exp_name == 'simple_adversary':
        groups = [[0],[1,2]]
    elif args.exp_name == 'simple_tag':
        groups = [[0,1,2],[3]]
    elif args.exp_name == 'simple_push':
        groups = [[0],[1]]
print('groups: ',groups)

seeds = [0,1,2]
P_paths = [
            'MADDPGlayer2hidden64',
            'MADDPGlayer2hidden64oa',
            'MASACGaussianlayer2hidden64er',
            'MASACGaussianlayer2hidden64oaer',
            'PRGGaussianhidden64k1oaceerdcigpna',
            'PRGGaussianhidden64k1oaceerdcigpnadca',
            'PRG3Gaussianhidden64ceerdcigpna',
            'PRG3Gaussianhidden64ceerdcigpnadca',
            ]

extra_name = args.extra_name

pre_dir = './Data/'+args.exp_name\
            +('bd' if args.boundary else '')\
            +(('ag'+str(args.num_ag)) if args.num_ag else '')\
            +(('adv'+str(args.num_adv)) if args.num_adv else '')\
            +(('l'+str(args.num_l)) if args.num_l else '')\
            +'_mpl'+str(args.mpl)
log_dir = pre_dir+'/tests/'+extra_name\
            +('_ep{}'.format(args.epoch) if args.epoch else '')\
            +'_ss'+str(args.sample_num)
log_file = log_dir+'/results.pkl'

sample_num = args.sample_num
max_path_length = args.mpl

import os
if (not os.path.isdir(log_dir)):
    os.makedirs(log_dir)
if (not os.path.isfile(log_file)):
    results = {}
else:
    import joblib
    results = joblib.load(log_file)

import sys
sys.path.append("./multiagent-particle-envs")
from make_env import make_env
from particle_env_wrapper import ParticleEnv
world_args=dict(
    num_agents=args.num_ag,
    num_adversaries=args.num_adv,
    num_landmarks=args.num_l,
    boundary=([[-1.,-1.],[1.,1.]] if args.boundary else None)
)
env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False,world_args=world_args))


for seed in seeds:
    print('seed: ',seed)
    if seed in results.keys():
        pass
    else:
        results[seed] = dict()

    p_paths = []
    for pid in range(len(P_paths)):
        p_paths.append(P_paths[pid]+'/'+'seed'+str(seed))

    with torch.no_grad():
        players = []
        for pid in range(len(P_paths)):
            d_path = pre_dir+'/'+P_paths[pid]+'/seed'+str(seed)
            if args.epoch:
                d_path += '/itr_{}.pkl'.format(args.epoch)
            else:
                d_path += '/params.pkl'
            data = torch.load(d_path,map_location='cpu')
            policy_n = data['trainer/policy_n']
            if isinstance(policy_n[0],TanhGaussianPolicy):
                policy_n = [MakeDeterministic(policy) for policy in policy_n]
            elif  isinstance(policy_n[0],GumbelSoftmaxMlpPolicy):
                policy_n = [ArgmaxDiscretePolicy(policy,use_preactivation=True) for policy in policy_n]
            players.append(policy_n)

        for p1id in range(len(P_paths)):
            for p2id in range(len(P_paths)):
                pair_name = '{}-{}'.format(P_paths[p1id],P_paths[p2id])
                # print(pair_name)
                if (pair_name in results[seed].keys()) and (not args.new):
                    print('pass')
                    Cr1 = results[seed][pair_name]['r1']
                    Cr2 = results[seed][pair_name]['r2']
                    print('{}: r1: {:.2f}; r2: {:.2f}'.format(pair_name,np.mean(Cr1),np.mean(Cr2)))
                else:
                    results[seed][pair_name] = {}
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
                            # env.render()
                            # time.sleep(0.1)
                            cr1 += r_n[groups[0][0]]
                            cr2 += r_n[groups[1][0]]
                            if done.all():
                                break
                        Cr1.append(cr1)
                        Cr2.append(cr2)
                    print('{}: r1: {:.2f}; r2: {:.2f}'.format(pair_name,np.mean(Cr1),np.mean(Cr2)))
                    results[seed][pair_name]['r1'] = Cr1
                    results[seed][pair_name]['r2'] = Cr2

import pickle
f = open(log_file,"wb")
pickle.dump(results,f)
f.close()

