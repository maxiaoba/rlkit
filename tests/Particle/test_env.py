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
parser.add_argument('--boundary', action='store_true', default=False)
parser.add_argument('--num_ag', type=int, default=None)
parser.add_argument('--num_adv', type=int, default=None)
parser.add_argument('--num_l', type=int, default=None)
parser.add_argument('--mpl', type=int, default=25) # max path length
args = parser.parse_args()

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
o_n = env.reset()
env.render()
num_agent = env.num_agent

max_path_length = args.mpl
path_length = 0
done = np.array([False]*num_agent)
c_r = np.zeros(num_agent)
max_o = 0.
while True:
	path_length += 1
	a_n = []
	for i in range(num_agent):
		# a = input("Action for agent {}:\n".format(i))
		# a = np.array(list(map(float,a.split(' '))))
		# a_n.append(a)
		a = env.action_space.sample()
		a_n.append(a)
	o_n, r_n, done, _ = env.step(a_n)
	c_r += r_n
	env.render()
	print("step: ",path_length)
	print("a: ",a_n)
	print("o: ",np.max(np.abs(o_n)))
	if np.max(np.abs(o_n)) > max_o:
		max_o = np.max(np.abs(o_n))
	print('r: ',r_n)
	print(done)
	# pbd.set_trace()
	time.sleep(0.1)
	if path_length > max_path_length or done.all():
		print('c_r: ',c_r)
		print('max_o: ',max_o)
		path_length = 0
		done = np.array([False]*num_agent)
		c_r = np.zeros(num_agent)
		max_o = 0.
		o_n = env.reset()
		pdb.set_trace()