import os
import csv
import torch
import numpy as np
import time
import pdb
from rlkit.torch.policies.make_deterministic import MakeDeterministic
from rlkit.torch.core import eval_np, np_ify

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='t_intersection_lstm4')
parser.add_argument('--noise', type=float, default=0.05)
parser.add_argument('--yld', type=float, default=0.5)
parser.add_argument('--ds', type=float, default=0.1)
parser.add_argument('--dfd', type=float, default=0.1)
parser.add_argument('--dfi', type=float, default=0.3)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--pre_log', type=str, default='noise0.05yld0.5ds0.1')
parser.add_argument('--file', type=str, default='params')
args = parser.parse_args()

policies = [
            'PPOlayer1hidden48ep5000',
            'PPOSupVanillalayer1hidden48ep5000',
            'PPOSuplayer1hidden48ep5000',
            'PPOSupSep2layer1hidden28ep5000',
            'PPOGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            'PPOSupVanillaGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            'PPOSupGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            'PPOSupSep2GNN2llayer1hidden18GSagenode18glayer3actreluep5000',
            'PPOSupSep2LSTMGNN2layer1hidden28GSagenode18glayer3suphidden18suplayer1actreluep5000',
        ]
seeds = [0,1,2]
extra_name = ''

pre_dir = './Data/'+args.exp_name+args.pre_log
log_dir = extra_name+'noise'+str(args.noise)+'yld'+str(args.yld)+'ds'+str(args.ds)+'dfd'+str(args.dfd)+'dfi'+str(args.dfi)+'epoch'+str(args.epoch)
log_dir = '{}/Eval/{}'.format(pre_dir,log_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

with open('{}/result.csv'.format(log_dir), mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Policy', 'Num Path', 'Return', 'Success Rate', 'Collision Rate', 'Inference Accuracy'])

    from traffic.make_env import make_env
    env_kwargs=dict(
        num_updates=1,
        obs_noise=args.noise,
        yld=args.yld,
        driver_sigma=args.ds,
        des_front_gap_difference=args.dfd,
        des_front_gap_interval=args.dfi,
    )
    env = make_env(args.exp_name,**env_kwargs)
    max_path_length = 200
    for policy_path in policies:
        for seed in seeds:
            data_path = '{}/{}/seed{}_load/{}.pkl'.format(pre_dir,policy_path,seed,args.file)
            if os.path.exists(data_path):
                print('_load')
            else:
                data_path = '{}/{}/seed{}/{}.pkl'.format(pre_dir,args.log_dir,args.seed,args.file)
            data = torch.load(data_path,map_location='cpu')

            policy = data['trainer/policy']
            eval_policy = MakeDeterministic(policy)

            returns = []
            success_num = 0
            collision_num = 0
            inference_correct = 0
            inference_total = 0
            for _ in range(args.epoch):
                o = env.reset()
                policy.reset()
                path_length = 0
                done = False
                c_r = 0.
                while True:
                    path_length += 1
                    a, agent_info = eval_policy.get_action(o)
                    o, r, done, env_info = env.step(a)

                    if 'intentions' in agent_info.keys():
                        intention_probs = agent_info['intentions']
                        inffered_intentions = np.argmax(intention_probs,axis=-1)
                        true_intentions = env.get_sup_labels()
                        valid_mask = ~np.isnan(true_intentions)
                        true_intentions = true_intentions[valid_mask]
                        inffered_intentions = inffered_intentions[valid_mask]
                        inference_correct += np.sum(inffered_intentions==true_intentions)
                        inference_total += np.sum(valid_mask)
                    else:
                        inference_total += 1

                    c_r += r
                    if path_length > max_path_length or done:
                        returns.append(c_r)
                        if env_info['event'] == 'goal':
                            success_num += 1
                        elif env_info['event'] == 'collision':
                            collision_num += 1
                        break

            policy_name = '{}_seed{}_{}'.format(policy_path,seed,args.file)
            writer.writerow([policy_name, args.epoch, np.mean(returns),success_num/args.epoch,collision_num/args.epoch,inference_correct/inference_total])
