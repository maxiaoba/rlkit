import numpy as np
import time
import argparse
from rlkit.envs.wrappers import NormalizedBoxEnv

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='Ant')
parser.add_argument('--ml', type=int, default=1000)
args = parser.parse_args()

import gym
env = NormalizedBoxEnv(gym.make(args.exp_name+'-v2'))
o = env.reset()
# print(env.observation_space.high)
max_path_length = args.ml
path_length = 0
done = False
c_r = 0.0

while (path_length < max_path_length) and (not done):
	path_length += 1
	a = env.action_space.sample()
	o, r, done, _ = env.step(a)
	c_r += r
	# env.render()
	print("step: ",path_length)
	print("o_max: ",np.max(o),np.argmax(o))
	print("o_mean: ",np.mean(o))
	print("a: ",a)
	print('r: ',r)
	print(done)
	time.sleep(0.1)
print('c_r: ',c_r)