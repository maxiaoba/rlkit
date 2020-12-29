import numpy as np
import matplotlib 
matplotlib.rcParams.update({'font.family': 'serif','font.size': 9})
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='simple_push')
parser.add_argument('--mpl', type=int, default=25) # max path length
parser.add_argument('--ci', type=int, default=4) # center_index
parser.add_argument('--only', type=str, default=None) # 'agent' or 'adversary'
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
            # 'PRG3Gaussianhidden64k0m0ceerdcigpna',
            # 'PRG3Gaussianhidden64k0m1ceerdcigpna',
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
                # 'PRG3k0m0er',
                # 'PRG3k0m1er',
            ]
colors = ['C0', 'C1', 'C2', 'C4', 'C3']
center_index = args.ci

extra_plot_name = ''
for pid,policy_name in enumerate(policy_names):
    if pid != 0:
        extra_plot_name += '-'
    if pid == args.ci:
        extra_plot_name += '[{}]'.format(policy_name)
    else:
        extra_plot_name += policy_name
if args.only:
    extra_plot_name += '_'+args.only
extra_plot_name += '_'

extra_name = args.extra_name

pre_path = './Data/'+args.exp_name+'_mpl'+str(args.mpl)
log_dir = pre_path+'/tests/'+extra_name+'_ss'+str(args.sample_num)
log_file = log_dir+'/results.pkl'

def plot_pair_return(mat1,mat2,policy_names, center_index):
    if args.only:
        fig = plt.figure(figsize=(len(policy_names),6))
    else:
        fig = plt.figure(figsize=(2.*len(policy_names),6))
    # fig = plt.figure()
    fid = 0
    mat1 = (mat1-np.min(mat1))/(np.max(mat1)-np.min(mat1))
    mat2 = (mat2-np.min(mat2))/(np.max(mat2)-np.min(mat2))
    # mat1_mean = np.mean(mat1,2)
    # mat1_std = np.std(mat1,2)
    # mat2_mean = np.mean(mat2,2)
    # mat2_std = np.std(mat2,2)
    mat1_mean = np.mean(mat1,0)
    mat1_std = np.std(mat1,0)/np.sqrt(mat1.shape[0])
    mat2_mean = np.mean(mat2,0)
    mat2_std = np.std(mat2,0)/np.sqrt(mat2.shape[0])
    for i in range(len(policy_names)):
        if i == center_index:
            pass
        else:
            fid += 1
            if args.only == 'agent':
                pass
            else:
                if args.only == 'adversary':
                    plt.subplot(2,(len(policy_names)-1)/2,fid)
                else:
                    plt.subplot(2,len(policy_names)-1,fid)
                p1ip2i_1 = mat1_mean[i,i]
                p1ip2i_std_1 = mat1_std[i,i]
                p1cp2i_1 = mat1_mean[center_index,i]
                p1cp2i_std_1 = mat1_std[center_index,i]
                plt.bar([policy_names[i],
                         policy_names[center_index]
                        ],
                        [p1ip2i_1,p1cp2i_1],
                        yerr=[p1ip2i_std_1,p1cp2i_std_1],
                        capsize=10.0,
                        color=[colors[i],colors[center_index]],
                        )
                plt.title('Agent: {}'.format(policy_names[i]))
                plt.ylim(0,1)
                if fid%2 == 1:
                    plt.ylabel('Normalized Adversary Return')
                else:
                    plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().set_aspect(2.0, 'box')

            if args.only == 'adversary':
                pass
            else:
                if args.only == 'agent':
                    plt.subplot(2,(len(policy_names)-1)/2,fid)
                else:
                    plt.subplot(2,len(policy_names)-1,fid+len(policy_names)-1)
                p1ip2i_2 = mat2_mean[i,i]
                p1ip2i_std_2 = mat2_std[i,i]
                p1ip2c_2 = mat2_mean[i,center_index]
                p1ip2c_std_2 = mat2_std[i,center_index]
                plt.bar([policy_names[i],
                         policy_names[center_index]
                        ],
                        [p1ip2i_2,p1ip2c_2],
                        yerr=[p1ip2i_std_2,p1ip2c_std_2],
                        capsize=10.0,
                        color=[colors[i],colors[center_index]],
                        )
                plt.title('Adversary: {}'.format(policy_names[i]))
                plt.ylim(0,1)
                if fid%2 == 1:
                    plt.ylabel('Normalized Agent Return')
                else:
                    plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().set_aspect(2.0, 'box')
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

plot_name = extra_plot_name+'pair_return.pdf'
fig = plot_pair_return(Mat1,Mat2,policy_names, center_index)
plt.savefig(log_dir+'/'+plot_name, bbox_inches='tight')
plt.close()
