import torch
import numpy as np
import time
import pdb
from rlkit.torch.policies.softmax_policy import SoftmaxPolicy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.policies.tanh_gaussian_policy import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.core import eval_np, np_ify

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='t_intersection')
parser.add_argument('--extra_name', type=str, default='')
parser.add_argument('--log_dir', type=str, default='PPO')
parser.add_argument('--file', type=str, default='params')
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pre_dir = './Data/'+args.exp_name+args.extra_name
data_path = '{}/{}/seed{}/{}.pkl'.format(pre_dir,args.log_dir,args.seed,args.file)
data = torch.load(data_path,map_location='cpu')
if 'trainer/qf' in data.keys():
	qf = data['trainer/qf']
	policy = ArgmaxDiscretePolicy(qf)
else:
	policy = data['trainer/policy']
	if isinstance(policy, SoftmaxPolicy):
		policy = ArgmaxDiscretePolicy(policy,use_preactivation=True)
	elif isinstance(policy, TanhGaussianPolicy):
		policy = MakeDeterministic(policy)
if 'trainer/sup_learners' in data.keys():
	sup_learners = data['trainer/sup_learners']
else:
	sup_learners = None

import sys
from traffic.make_env import make_env
import json
with open('{}/{}/seed{}/variant.json'.format(pre_dir,args.log_dir,args.seed)) as f:
  variant = json.load(f)
env = make_env(args.exp_name,**variant['env_kwargs'])
o = env.reset()

max_path_length = 200
path_length = 0
done = False
c_r = 0.
while True:
	path_length += 1
	a, _ = policy.get_action(o)
	o, r, done, _ = env.step(a)
	if sup_learners:
		intentions = [eval_np(sup_learner, o) for sup_learner in sup_learners]
	else:
		intentions = None
	c_r += r
	env.render()
	print("step: ",path_length)
	print("intentions: ",intentions)
	print("a: ",a)
	print("o: ",o)
	print('r: ',r)
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