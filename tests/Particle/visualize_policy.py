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
parser.add_argument('--exp_name', type=str, default='simple_spread')
parser.add_argument('--num_ag', type=int, default=None)
parser.add_argument('--num_adv', type=int, default=None)
parser.add_argument('--num_l', type=int, default=None)
parser.add_argument('--mpl', type=int, default=25) # max path length
parser.add_argument('--log_dir', type=str, default='MASAC')
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pre_dir = './Data/'+args.exp_name\
            +(('ag'+str(args.num_ag)) if args.num_ag else '')\
            +(('adv'+str(args.num_adv)) if args.num_adv else '')\
            +(('l'+str(args.num_l)) if args.num_l else '')\
            +'_mpl'+str(args.mpl)
data_path = '{}/{}/seed{}/params.pkl'.format(pre_dir,args.log_dir,args.seed)
data = torch.load(data_path,map_location='cpu')
policy_n = data['trainer/trained_policy_n']
if isinstance(policy_n[0],TanhGaussianPolicy):
	policy_n = [MakeDeterministic(policy) for policy in policy_n]
elif  isinstance(policy_n[0],GumbelSoftmaxMlpPolicy):
	policy_n = [ArgmaxDiscretePolicy(policy,use_preactivation=True) for policy in policy_n]

import sys
sys.path.append("./multiagent-particle-envs")
from make_env import make_env
from particle_env_wrapper import ParticleEnv
world_args=dict(
    num_agents=args.num_ag,
    num_adversaries=args.num_adv,
    num_landmarks=args.num_l,
)
env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False,world_args=world_args))
o_n = env.reset()
env.render()
num_agent = env.num_agent

max_path_length = args.mpl
path_length = 0
done = np.array([False]*num_agent)
c_r = np.zeros(num_agent)
while True:
	path_length += 1
	a_n = []
	for (policy,o) in zip(policy_n,o_n):
		a, _ = policy.get_action(o)
		a_n.append(a)
	o_n, r_n, done, _ = env.step(a_n)
	c_r += r_n
	env.render()
	print("step: ",path_length)
	print("a: ",a_n)
	print("o: ",o_n)
	print('r: ',r_n)
	print(done)
	time.sleep(0.1)
	if path_length > max_path_length or done.all():
		print('c_r: ',c_r)
		path_length = 0
		done = np.array([False]*num_agent)
		c_r = np.zeros(num_agent)
		o_n = env.reset()
		pdb.set_trace()