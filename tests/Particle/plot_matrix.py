import numpy as np
import matplotlib 
matplotlib.rcParams.update({'font.family': 'serif'})
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='simple_push')
parser.add_argument('--sample_num', type=int, default=1000)
parser.add_argument('--extra_name', type=str, default='')
args = parser.parse_args()

P_paths = [
            'MADDPG',
            'MADDPGonline_action',
            'MASAC',
            'MASAConline_action',
            # 'PRGk1',
            'PRGk1online_action',
            # 'PRGGaussiank1',
            'PRGGaussiank1online_action',
            'PRGGaussiank1online_actioncentropy'
            ]
policy_names = [
                'MADDPG',
                'MADDPGonline',
                'MASAC',
                'MASAConline',
                # 'PRGk1',
                'PRGk1online',
                # 'PRGGaussiank1',
                'PRGGaussiank1online',
                'PRGGaussiank1onlinecentropy'
            ]

extra_name = args.extra_name

pre_path = './Data/'+args.exp_name
log_dir = pre_path+'/tests/'+extra_name+'_ss'+str(args.sample_num)

import csv
csv_name = 'payoff.csv'

def plot_matrix(mat,policy_names):
    # fig, ax = plt.subplots(figsize=(5, 5))
    fig, ax = plt.subplots()
    im = ax.imshow(mat, interpolation="None", cmap="viridis", alpha=0.7)
    for (j, i), label in np.ndenumerate(mat):
        ax.text(i, j, round(label, 3), ha="center", va="center", fontsize=10)

    ax.set_xlabel("Player #2", fontsize=20)
    ax.set_ylabel("Player #1", fontsize=20)

    # plt.xticks(ticks=np.arange(len(policy_names)), labels=policy_names, fontsize=15)
    # plt.yticks(ticks=np.arange(len(policy_names)), labels=policy_names, fontsize=15,
    #             rotation='vertical',verticalalignment='center')
    plt.xlim(-0.5,len(policy_names)-0.5)
    plt.ylim(-0.5,len(policy_names)-0.5)
    #fig.colorbar(im, ax=ax)
    return fig

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

            plot_name = 'payoff'+str(seed)+'_p1.pdf'
            fig = plot_matrix(mat1,policy_names)
            plt.savefig(log_dir+'/'+plot_name, bbox_inches='tight')
            plt.close()
            plot_name = 'payoff'+str(seed)+'_p2.pdf'
            fig = plot_matrix(mat2,policy_names)
            plt.savefig(log_dir+'/'+plot_name, bbox_inches='tight')
            plt.close()

Mat1 = np.array(Mat1)
Mat2 = np.array(Mat2)
print(Mat1.shape,Mat2.shape)
Mat1 = np.mean(Mat1,axis=0)
Mat2 = np.mean(Mat2,axis=0)


plot_name = 'payoff_p1.pdf'
fig = plot_matrix(Mat1,policy_names)
plt.savefig(log_dir+'/'+plot_name, bbox_inches='tight')
plt.close()

plot_name = 'payoff_p2.pdf'
fig = plot_matrix(Mat2,policy_names)
plt.savefig(log_dir+'/'+plot_name, bbox_inches='tight')
plt.close()