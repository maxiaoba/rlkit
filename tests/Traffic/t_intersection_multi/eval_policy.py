from rlkit.samplers.rollout_functions import rollout
import torch
import numpy as np
import time
import pdb
from rlkit.torch.policies.softmax_policy import SoftmaxPolicy
from sup_softmax_policy import SupSoftmaxPolicy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.policies.tanh_gaussian_policy import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.core import eval_np, np_ify

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='t_intersection_multi')
parser.add_argument('--extra_name', type=str, default='nobyld0.5ds0.1fullimportant')
parser.add_argument('--log_dir', type=str, default='PPO')
parser.add_argument('--file', type=str, default='params')
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--neval', type=int, default=100)
args = parser.parse_args()

pre_dir = './Data/'+args.exp_name+args.extra_name
data_path = '{}/{}/seed{}/{}.pkl'.format(pre_dir,args.log_dir,args.seed,args.file)
data = torch.load(data_path,map_location='cpu')

policy = data['trainer/policy']
policy = ArgmaxDiscretePolicy(policy,use_preactivation=True)

import sys
from traffic.make_env import make_env
import json
with open('{}/{}/seed{}/variant.json'.format(pre_dir,args.log_dir,args.seed)) as f:
  variant = json.load(f)
env = make_env(args.exp_name,**variant['env_kwargs'])

returns = []
for i in range(args.neval):
	path = rollout(env,policy,max_path_length=200)
	ret = np.sum(path['rewards'])
	returns.append(ret)

print(np.mean(returns),np.std(returns))

