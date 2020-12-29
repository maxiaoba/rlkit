import numpy as np
import matplotlib 
matplotlib.rcParams.update({'font.family': 'serif','font.size': 9})
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='simple_push')
parser.add_argument('--mpl', type=int, default=25) # max path length
parser.add_argument('--sample_num', type=int, default=1000)
parser.add_argument('--extra_name', type=str, default='')
args = parser.parse_args()

P_paths = [
            'MADDPGlayer2hidden64',
            'MADDPGlayer2hidden64oa',
            'MASACGaussianlayer2hidden64er',
            'MASACGaussianlayer2hidden64oaer',
            # 'MASACGaussianlayer2hidden64oadna',
            'PRGGaussiank1hidden64oaonaceerdcigpna',
            # 'PRGGaussiank1hidden64oaonacedcigdnapna',
            # 'PRG3Gaussianhidden64k0m0cedcigdnapna',
            # 'PRG3Gaussianhidden64k0m1cedcigdnapna',
            'PRG3Gaussianhidden64k0m0ceerdcigpna',
            'PRG3Gaussianhidden64k0m1ceerdcigpna',
            ]
policy_names = [
                'MADDPG',
                'MADDPG-OA',
                'MASAC',
                'MASAC-OA',
                # 'MASACdna',
                'R2G',
                # 'PRGdna',
                # 'PRG3k0m0dna',
                # 'PRG3k0m1dna',
                'PRG3k0m0er',
                'PRG3k0m1er',
            ]
extra_plot_name = '-'.join(policy_names)+'_'

extra_name = args.extra_name

pre_path = './Data/'+args.exp_name+'_mpl'+str(args.mpl)
log_dir = pre_path+'/tests/'+extra_name+'_ss'+str(args.sample_num)
log_file = log_dir+'/results.pkl'

def plot_matrix(mat,policy_names):
    # fig, ax = plt.subplots(figsize=(5, 5))
    fig, ax = plt.subplots()
    # mat = np.array([[0,1,2],[3,4,5],[6,7,8]])
    mat = (mat-np.min(mat))/(np.max(mat)-np.min(mat))
    mat = np.mean(mat,0)
    im = ax.imshow(mat, interpolation="nearest", cmap="viridis", alpha=0.7)
    for (j, i), label in np.ndenumerate(mat):
        ax.text(i, j, round(label, 3), ha="center", va="center", fontsize=10)

    ax.set_xlabel("Player #2", fontsize=20)
    ax.set_ylabel("Player #1", fontsize=20)

    plt.xticks(ticks=np.arange(len(policy_names)), labels=policy_names, fontsize=10,
                rotation=45,)
    plt.yticks(ticks=np.arange(len(policy_names)), labels=policy_names, fontsize=10,
                rotation=45,verticalalignment='center')
    # plt.xlim(-0.5,len(policy_names)-0.5)
    # plt.ylim(-0.5,len(policy_names)-0.5)
    #fig.colorbar(im, ax=ax)
    return fig

import joblib
results = joblib.load(log_file)
Mat1, Mat2 = [], []
for seed in results.keys():
    mat1 = np.zeros((len(P_paths),len(P_paths),args.sample_num))
    mat2 = np.zeros((len(P_paths),len(P_paths),args.sample_num))
    for p1id in range(len(P_paths)):
        for p2id in range(len(P_paths)):
            pair_name = '{}-{}'.format(P_paths[p1id],P_paths[p2id])
            r1 = results[seed][pair_name]['r1']
            r2 = results[seed][pair_name]['r2']
            mat1[p1id,p2id,:] = r1[:]
            mat2[p1id,p2id,:] = r2[:]
    Mat1.append(mat1) # [n x n x ss] * num_seed
    Mat2.append(mat2) # [n x n x ss] * num_seed

# Mat1 = np.concatenate(Mat1,axis=2) # [n x n x ss*num_seed]
# Mat2 = np.concatenate(Mat2,axis=2) # [n x n x ss*num_seed]
# print(Mat1.shape,Mat2.shape)
Mat1 = np.mean(np.array(Mat1),-1) # [num_seed x n x n]
Mat2 = np.mean(np.array(Mat2),-1) # [num_seed x n x n]
print(Mat1.shape,Mat2.shape)

plot_name = extra_plot_name+'payoff_p1.pdf'
fig = plot_matrix(Mat1,policy_names)
plt.savefig(log_dir+'/'+plot_name, bbox_inches='tight')
plt.close()

plot_name = extra_plot_name+'payoff_p2.pdf'
fig = plot_matrix(Mat2,policy_names)
plt.savefig(log_dir+'/'+plot_name, bbox_inches='tight')
plt.close()