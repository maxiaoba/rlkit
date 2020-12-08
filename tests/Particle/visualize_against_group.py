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
parser.add_argument('--p1', type=str, default='MASAC')
parser.add_argument('--p2', type=str, default='PRG')
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

if args.exp_name == 'simple_adversary':
    groups = [[0],[1,2]]
elif args.exp_name == 'simple_tag':
    groups = [[0,1,2],[3]]
elif args.exp_name == 'simple_push':
    groups = [[0],[1]]
print('groups: ',groups)

policy_n = []

data_path1 = './Data/{}_mpl{}/{}/seed{}/params.pkl'.format(args.exp_name,args.mpl,args.p1,args.seed)
data1 = torch.load(data_path1,map_location='cpu')
policy_n1 = data1['trainer/trained_policy_n']
if isinstance(policy_n1[0],TanhGaussianPolicy):
    policy_n1 = [MakeDeterministic(policy) for policy in policy_n1]
elif  isinstance(policy_n1[0],GumbelSoftmaxMlpPolicy):
    policy_n1 = [ArgmaxDiscretePolicy(policy,use_preactivation=True) for policy in policy_n1]

data_path2 = './Data/{}_mpl{}/{}/seed{}/params.pkl'.format(args.exp_name,args.mpl,args.p2,args.seed)
data2 = torch.load(data_path2,map_location='cpu')
policy_n2 = data2['trainer/trained_policy_n']
if isinstance(policy_n2[0],TanhGaussianPolicy):
    policy_n2 = [MakeDeterministic(policy) for policy in policy_n2]
elif  isinstance(policy_n2[0],GumbelSoftmaxMlpPolicy):
    policy_n2 = [ArgmaxDiscretePolicy(policy,use_preactivation=True) for policy in policy_n2]

policy_n = []
for p1 in groups[0]:
    policy_n.append(policy_n1[p1])
for p2 in groups[1]:
    policy_n.append(policy_n2[p2])

import sys
sys.path.append("./multiagent-particle-envs")
from make_env import make_env
from particle_env_wrapper import ParticleEnv
# env = ParticleEnv(make_env(args.exp_name,discrete_action_space=True,discrete_action_input=True))
env = ParticleEnv(make_env(args.exp_name,discrete_action_space=False))
o_n = env.reset()
num_agent = env.num_agent

max_path_length = 100
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

