import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='max2')
parser.add_argument('--log_dir', type=str, default='PRGGaussiank1oace')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pre_path = './Data/'+args.exp_name+'/'+args.log_dir
plot_file = pre_path+'/'+'seed'+str(args.seed)+'/action.png'
a1_field = 'evaluation/Actions 0 Mean'
a2_field = 'evaluation/Actions 1 Mean'
a1s = []
a2s = []

file_path = pre_path+'/'+'seed'+str(args.seed)+'/progress.csv'
print(file_path)
if os.path.exists(file_path):
    print(file_path)
    itrs = []
    losses = []
    loss = []
    with open(file_path) as csv_file:
        if '\0' in open(file_path).read():
            print("you have null bytes in your input file")
            csv_reader = csv.reader(x.replace('\0', '') for x in csv_file)
        else:
            csv_reader = csv.reader(csv_file, delimiter=',')

        for (i,row) in enumerate(csv_reader):
            if i == 0:
                entry_dict = {}
                for index in range(len(row)):
                    entry_dict[row[index]] = index
                # print(entry_dict)
            else:
                a1s.append(float(row[entry_dict[a1_field]]))
                a2s.append(float(row[entry_dict[a2_field]]))

fig = plt.figure()
plot, = plt.plot(a1s,a2s)
plt.plot(a1s[0],a2s[0],'o',color='green')
plt.plot(a1s[-1],a2s[-1],'o',color='red')
plt.xlabel('a1')
plt.ylabel('a2') 
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1,1)
plt.ylim(-1,1)
fig.savefig(plot_file)
plt.close(fig)