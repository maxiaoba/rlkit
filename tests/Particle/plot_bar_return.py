import numpy as np
import matplotlib 
matplotlib.rcParams.update({'font.family': 'serif','font.size': 18})
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='simple_spread')
parser.add_argument('--mpl', type=int, default=25) # max path length
parser.add_argument('--subset', type=str, default=None) # subset of policies
parser.add_argument('--sample_num', type=int, default=1000)
parser.add_argument('--extra_name', type=str, default='')
args = parser.parse_args()

P_paths = [
            'MADDPGlayer2hidden64oa',
            'MASACGaussianlayer2hidden64oaer',
            'MASACGaussianlayer2hidden64oadna',
            'PRGGaussiank1hidden64oaonaceerdcigpna',
            'PRGGaussiank1hidden64oaonacedcigdnapna',
            'PRG3Gaussianhidden64k0m0ceerdcigpna',
            'PRG3Gaussianhidden64k0m1ceerdcigpna',
            'PRG3Gaussianhidden64k0m0cedcigdnapna',
            'PRG3Gaussianhidden64k0m1cedcigdnapna',
            ]
policy_names = [
                'MADDPG',
                'MASACer',
                'MASACdna',
                'PRGer',
                'PRGdna',
                'PRG3k0m0er',
                'PRG3k0m1er',
                'PRG3k0m0dna',
                'PRG3k0m1dna',
            ]

if args.subset:
    subset = np.array(list(map(int,args.subset.split('_'))))
    sub_policy_names = [policy_names[i] for i in subset]
    extra_plot_name = args.subset
else:
    subset = np.arange(len(policy_names))
    sub_policy_names = policy_names
    extra_plot_name = ''

extra_name = (args.extra_name if args.extra_name else 'self-'+'-'.join(policy_names))

pre_path = './Data/'+args.exp_name+'_mpl'+str(args.mpl)
log_dir = pre_path+'/tests/'+extra_name+'_ss'+str(args.sample_num)

import csv
csv_name = 'payoff.csv'

with open(log_dir+'/'+csv_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    Returns = []
    for (rowid,row) in enumerate(csv_reader):
        if rowid == 0:
            entry_dict = {}
            for index in range(len(row)):
                entry_dict[row[index]] = index
        else:
            seed = row[entry_dict["seed"]]
            returns = np.zeros(len(subset))
            for i in range(len(subset)):
                returns[i] = float(row[entry_dict[P_paths[subset[i]]]])
            Returns.append(returns)

Returns = np.array(Returns)
Returns = (Returns-np.min(Returns))/(np.max(Returns)-np.min(Returns))

plt.figure()
plt.bar(sub_policy_names,np.mean(Returns,0),
        yerr=np.std(Returns,0)/np.sqrt(Returns.shape[0]),capsize=10.0,)
plt.ylim(0,1)
plot_name = extra_plot_name+'return.pdf'
plt.savefig(log_dir+'/'+plot_name, bbox_inches='tight')
plt.close()
