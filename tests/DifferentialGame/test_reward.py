import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 20})
from matplotlib import pyplot as plt
import numpy as np
import argparse
# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='max2')
args = parser.parse_args()

from differential_game import DifferentialGame
env = DifferentialGame(game_name=args.exp_name)

a1 = 0.5
a2 = 0
env.reset()
o_n,r_n,d_n,info = env.step([a1,a2])
print('a1 {}, a2 {}, r1 {}, r2 {}'.format(a1,a2,r_n[0],r_n[1]))

a1 = -0.5
a2 = 0
env.reset()
o_n,r_n,d_n,info = env.step([a1,a2])
print('a1 {}, a2 {}, r1 {}, r2 {}'.format(a1,a2,r_n[0],r_n[1]))

a1 = 0.5
a2s = np.arange(-1,1,0.001)
r1s, r2s = [], []
for a2 in a2s:
	env.reset()
	o_n,r_n,d_n,info = env.step([a1,a2])
	r1s.append(r_n[0])
	r2s.append(r_n[1])
print('a1 {}, a2 {}, r1 {}, r2 {}'.format(a1,'unifrom',np.mean(r1s),np.mean(r2s)))

a1 = -0.5
a2s = np.arange(-1,1,0.001)
r1s, r2s = [], []
for a2 in a2s:
	env.reset()
	o_n,r_n,d_n,info = env.step([a1,a2])
	r1s.append(r_n[0])
	r2s.append(r_n[1])
print('a1 {}, a2 {}, r1 {}, r2 {}'.format(a1,'unifrom',np.mean(r1s),np.mean(r2s)))

