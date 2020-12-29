import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.family': 'serif','font.size': 18})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 10
max_itr = 2e4

fields = [
            'evaluation/Average Returns 0',
            # 'evaluation/Average Returns 1',
            # 'evaluation/Average Returns 2',
            # 'evaluation/Average Returns 3',
            ]
field_names = [
            'Average Return 0',
            # 'Average Return 1',
            # 'Average Return 2',
            # 'Average Return 3',
            ]

itr_name = 'epoch'
min_loss = [-1000,-1000,-1000,-1000]
max_loss = [1000,1000,1000,1000]
exp_name = "simple_spread_mpl25"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

policies = [
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
# policy_names = policies
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

extra_name = ''
seeds = [0,1,2,3,4]

# colors = []
# for pid in range(len(policies)):
#     colors.append('C'+str(pid))
colors = ['C0', 'C1', 'C2', 'C4', 'C3']

pre_name = ''
post_name = ''

for fid,field in enumerate(fields):
    print(field)
    # fig = plt.figure(fid)
    fig = plt.figure(fid,figsize=(2.*len(policy_names),6))
    Losses_n = []
    for (policy_index,policy) in enumerate(policies):
        policy_path = pre_name+policy+post_name
        Itrs = []
        Losses = []
        min_itr = np.inf
        for trial in seeds:
            folder_path = prepath+'/'+policy_path+'/'+'seed'+str(trial)
            print(folder_path)
            if os.path.exists(folder_path):
                print(policy+'_'+str(trial))
                itrs = []
                losses = []
                loss = []
                last_itr = 0
                while folder_path is not None:
                    print(folder_path)
                    file_path = folder_path+'/progress.csv'
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
                                itr = last_itr+i-1#int(float(row[entry_dict[itr_name]]))
                                if itr > max_itr:
                                    break
                                loss.append(np.clip(float(row[entry_dict[field]]),
                                                    min_loss[fid],max_loss[fid]))
                                if itr % itr_interval == 0:
                                    itrs.append(itr)
                                    loss = np.mean(loss)
                                    losses.append(loss)
                                    loss = []
                        last_itr = itr
                    folder_path = folder_path+'_load'
                    if not os.path.exists(folder_path):
                        folder_path = None
                if len(losses) < min_itr:
                    min_itr = len(losses)
                Losses.append(losses)
        Losses = [losses[min_itr-1] for losses in Losses]
        Losses_n.append(Losses)

    Losses_n = np.array(Losses_n) # num_policy x num_seed
    Losses_n = (Losses_n-np.min(Losses_n))/(np.max(Losses_n)-np.min(Losses_n))
    plt.bar(policy_names,np.mean(Losses_n,1),
            yerr=np.std(Losses_n,1)/np.sqrt(Losses_n.shape[1]),
            capsize=10.0,
            color=colors,
            )
    plt.ylabel('Normalized Evaluation Return')
    plt.ylim(0,1) 
    fig.savefig(plot_path+'/'+extra_name+field_names[fid]+'_last.pdf', bbox_inches='tight')
    plt.close(fig)