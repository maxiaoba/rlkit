import numpy as np
import matplotlib.pyplot as plt
from rlkit.util import alpharank 
# this need open_spiel, open need to add python path
# export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>
# export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel>/build/python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='simple_push')
parser.add_argument('--mpl', type=int, default=25) # max path length
parser.add_argument('--subset', type=str, default=None) # subset of policies
parser.add_argument('--sample_num', type=int, default=1000)
parser.add_argument('--extra_name', type=str, default=None)
args = parser.parse_args()

P_paths = [
            'MADDPGlayer2hidden64oa',
            'MASACGaussianlayer2hidden64oaer',
            'MASACGaussianlayer2hidden64oadna',
            'PRGGaussiank1hidden64oaonaceerdcigpna',
            'PRGGaussiank1hidden64oaonacedcigdnapna',
            'PRG3Gaussianhidden64k0m0cedcigdnapna',
            'PRG3Gaussianhidden64k0m1cedcigdnapna',
            ]
policy_names = [
                'MADDPG',
                'MASACer',
                'MASACdna',
                'PRGer',
                'PRGdna',
                'PRG3k0m0dna',
                'PRG3k0m1dna',
            ]

if args.subset:
    subset = np.array(list(map(int,args.subset.split('_'))))
    sub_policy_names = [policy_names[i] for i in subset]
    extra_plot_name = args.subset+'_'
else:
    subset = np.arange(len(policy_names))
    sub_policy_names = policy_names
    extra_plot_name = ''

extra_name = (args.extra_name if args.extra_name else '-'.join(policy_names))

pre_path = './Data/'+args.exp_name+'_mpl'+str(args.mpl)
log_dir = pre_path+'/tests/'+extra_name+'_ss'+str(args.sample_num)

import csv
csv_name = 'payoff.csv'

with open(log_dir+'/'+csv_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    Mat1 = []
    Mat2 = []
    for (rowid,row) in enumerate(csv_reader):
        if rowid == 0:
            entry_dict = {}
            for index in range(len(row)):
                entry_dict[row[index]] = index
        else:
            seed = row[entry_dict["seed"]]
            mat1 = np.zeros((len(subset),len(subset)))
            mat2 = np.zeros((len(subset),len(subset)))
            for i in range(len(subset)):
                for j in range(len(subset)):
                    mat1[i,j] = float(row[entry_dict['p1'+str(subset[i])+'p2'+str(subset[j])+'_1']])
                    mat2[i,j] = float(row[entry_dict['p1'+str(subset[i])+'p2'+str(subset[j])+'_2']])
            Mat1.append(mat1)
            Mat2.append(mat2)

# for trial,(mat1, mat2) in enumerate(zip(Mat1,Mat2)):
#     pi, alpha, rank_fig = alpharank.sweep_pi_vs_alpha([mat1, mat2])
#     plt.savefig(log_dir+'/'+extra_plot_name+'rank'+str(trial)+'.pdf', bbox_inches='tight')
#     plt.close()
    # alpharank.compute_and_report_alpharank([mat1, mat2], alpha=alpha, verbose=True)
    # plt.savefig(log_dir+'/'+'net'+str(trial)+'.pdf', bbox_inches='tight')
    # plt.close()

Mat1 = np.array(Mat1)
Mat2 = np.array(Mat2)
print(Mat1.shape,Mat2.shape)
Mat1 = np.mean(Mat1,axis=0)
Mat2 = np.mean(Mat2,axis=0)

pi, alpha, rank_fig = alpharank.sweep_pi_vs_alpha([Mat1, Mat2])
plt.savefig(log_dir+'/'+extra_plot_name+'rank.pdf', bbox_inches='tight')
plt.close()
# net_fig = alpharank.compute_and_report_alpharank([Mat1, Mat2], alpha=alpha, verbose=True)
# plt.savefig(log_dir+'/'+'net.pdf', bbox_inches='tight')
# plt.close()

