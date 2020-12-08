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
parser.add_argument('--sample_num', type=int, default=1000)
parser.add_argument('--extra_name', type=str, default='')
args = parser.parse_args()

P_paths = [
            'MADDPGlayer2hidden64oa',
            'MASACGaussianlayer2hidden64oadna',
            'MASACGaussianlayer2hidden64oaerdna',
            'PRGGaussiank1hidden64oaonacedcigdnapna',
            'PRGGaussiank1hidden64oaonaceerdcigdnapna',
            ]
policy_names = [
                'MADDPG',
                'MASACdna',
                'MASACerdna'
                'PRGdna',
                'PRGerdna',
            ]

extra_name = args.extra_name

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
            mat1 = np.zeros((len(policy_names),len(policy_names)))
            mat2 = np.zeros((len(policy_names),len(policy_names)))
            for i in range(len(policy_names)):
                for j in range(len(policy_names)):
                    mat1[i,j] = float(row[entry_dict['p1'+str(i)+'p2'+str(j)+'_1']])
                    mat2[i,j] = float(row[entry_dict['p1'+str(i)+'p2'+str(j)+'_2']])
            Mat1.append(mat1)
            Mat2.append(mat2)

for trial,(mat1, mat2) in enumerate(zip(Mat1,Mat2)):
    pi, alpha, rank_fig = alpharank.sweep_pi_vs_alpha([mat1, mat2])
    plt.savefig(log_dir+'/'+'rank'+str(trial)+'.pdf', bbox_inches='tight')
    plt.close()
    # alpharank.compute_and_report_alpharank([mat1, mat2], alpha=alpha, verbose=True)
    # plt.savefig(log_dir+'/'+'net'+str(trial)+'.pdf', bbox_inches='tight')
    # plt.close()

Mat1 = np.array(Mat1)
Mat2 = np.array(Mat2)
print(Mat1.shape,Mat2.shape)
Mat1 = np.mean(Mat1,axis=0)
Mat2 = np.mean(Mat2,axis=0)

pi, alpha, rank_fig = alpharank.sweep_pi_vs_alpha([Mat1, Mat2])
plt.savefig(log_dir+'/'+'rank.pdf', bbox_inches='tight')
plt.close()
# net_fig = alpharank.compute_and_report_alpharank([Mat1, Mat2], alpha=alpha, verbose=True)
# plt.savefig(log_dir+'/'+'net.pdf', bbox_inches='tight')
# plt.close()

