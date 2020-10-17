import torch
import numpy as np
import time
import pdb
from rlkit.torch.policies.make_deterministic import MakeDeterministic
from rlkit.torch.core import eval_np, np_ify

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='highway')
parser.add_argument('--extra_name', type=str, default='noise0.05ds0.1')
parser.add_argument('--log_dir', type=str, default='PPO')
parser.add_argument('--file', type=str, default='params')
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pre_dir = './Data/'+args.exp_name+args.extra_name
import os
data_path = '{}/{}/seed{}_load/{}.pkl'.format(pre_dir,args.log_dir,args.seed,args.file)
if os.path.exists(data_path):
	print('_load')
else:
	data_path = '{}/{}/seed{}/{}.pkl'.format(pre_dir,args.log_dir,args.seed,args.file)
data = torch.load(data_path,map_location='cpu')

policy = data['trainer/policy']
eval_policy = MakeDeterministic(policy)

if 'trainer/sup_learner' in data.keys():
	sup_learner = data['trainer/sup_learner']
else:
	sup_learner = None

import sys
from traffic.make_env import make_env
import json
with open('{}/{}/seed{}/variant.json'.format(pre_dir,args.log_dir,args.seed)) as f:
  variant = json.load(f)
env = make_env(args.exp_name,**variant['env_kwargs'])
o = env.reset()
policy.reset()

max_path_length = 200
path_length = 0
done = False
c_r = 0.
while True:
	path_length += 1
	a, agent_info = eval_policy.get_action(o)
	o, r, done, _ = env.step(a)

	if 'intentions' in agent_info.keys():
		intentions = agent_info['intentions']
	else:
		intentions = None

	c_r += r
	env.render(extra_input={'intentions':intentions})
	print("step: ",path_length)
	print("intentions: ",intentions)
	print("a: ",a)
	# print("o: ",o)
	# print('r: ',r)
	print(done)
	# pdb.set_trace()
	time.sleep(0.1)
	if path_length > max_path_length or done:
		print('c_r: ',c_r)
		path_length = 0
		done = False
		c_r = 0.
		pdb.set_trace()
		o = env.reset()
		policy.reset()
