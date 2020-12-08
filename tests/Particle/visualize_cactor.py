import torch
import numpy as np
import time
import pdb
from rlkit.torch.policies.tanh_gaussian_policy import TanhGaussianPolicy
from rlkit.torch.policies.make_deterministic import MakeDeterministic
from rlkit.torch.policies.gumbel_softmax_policy import GumbelSoftmaxMlpPolicy
from rlkit.policies.argmax import ArgmaxDiscretePolicy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='simple_spread2')
parser.add_argument('--mpl', type=int, default=25) # max path length
parser.add_argument('--p_path', type=str, default='')
parser.add_argument('--cp_path', type=str, default='')
parser.add_argument('--cp', type=str, default='')
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

data_path = './Data/{}_mpl{}/{}/seed{}/params.pkl'.format(args.exp_name,args.mpl,args.p_path,args.seed)
data = torch.load(data_path,map_location='cpu')
policy_n = data['trainer/trained_policy_n']
policy_n = [MakeDeterministic(policy) for policy in policy_n]

data_path = './Data/{}_mpl{}/{}/seed{}/params.pkl'.format(args.exp_name,args.mpl,args.cp_path,args.seed)
data = torch.load(data_path,map_location='cpu')
cactor_n = data['trainer/cactor_n']
cactor_n = [MakeDeterministic(cactor) for cactor in cactor_n]

cp = list(map(int,args.cp.split("_")))
print(cp)

import sys
sys.path.append("./multiagent-particle-envs")
from make_env import make_env
from particle_env_wrapper import ParticleEnv
# env = ParticleEnv(make_env(args.exp_name,discrete_action_space=True,discrete_action_input=True))
env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False))
o_n = env.reset()
num_agent = env.num_agent

max_path_length = args.mpl
path_length = 0
done = np.array([False]*num_agent)
c_r = np.zeros(num_agent)
with torch.no_grad():
	while True:
		path_length += 1
		a_n = []
		for (policy,o) in zip(policy_n,o_n):
			a, _ = policy.get_action(o)
			a_n.append(a)
		ca_n = []
		for p,cactor in enumerate(cactor_n):
			other_ps = np.array([i for i in range(num_agent) if i!=p])
			other_a = [a_n[other_p] for other_p in other_ps]
			ca, _ = cactor.get_action(np.concatenate([*o_n,*other_a]))
			ca_n.append(ca)
		action_n = [ca_n[p] if (p in cp) else a_n[p] for p in range(num_agent)]
		print('a_n: ',a_n)
		print('ca_n: ',ca_n)
		print('action_n: ',action_n)
		o_n, r_n, done, _ = env.step(action_n)
		c_r += r_n
		env.render()
		print("step: ",path_length)
		print("a: ",a_n)
		print("o: ",o_n)
		print('r: ',r_n)
		print(done)
		time.sleep(0.1)
		# pdb.set_trace()
		if path_length > max_path_length or done.all():
			print('c_r: ',c_r)
			path_length = 0
			done = np.array([False]*num_agent)
			c_r = np.zeros(num_agent)
			o_n = env.reset()
			pdb.set_trace()