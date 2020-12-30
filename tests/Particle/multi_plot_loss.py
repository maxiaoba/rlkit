import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 10
max_itr = 2e4

fields = [
            # 'trainer/K0 Loss Weight 0',
            # 'trainer/Alpha 0 Mean',
            # 'trainer/CAlpha 0 Mean',
            # 'exploration/Returns 0 Max',
            # 'exploration/Rewards 0 Max',
            # 'evaluation/Returns 0 Max',
            'evaluation/Average Returns 0',
            'evaluation/Average Returns 1',
            'evaluation/Average Returns 2',
            'evaluation/Average Returns 3',
            'evaluation/Average Returns 4',
            ]
field_names = [
            # 'K0 0',
            # 'Alpha 0',
            # 'CAlpha 0',
            # 'Expl Max Return 0',
            # 'Expl Max Reward 0',
            # 'Eval Max Reward 0',
            'Average Return 0',
            'Average Return 1',
            'Average Return 2',
            'Average Return 3',
            'Average Return 4',
            ]

itr_name = 'epoch'
min_loss = [-1000,-1000,-1000,-1000,-1000]
max_loss = [1000,1000,1000,1000,1000]
exp_name = "simple_spread_mpl25"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

policies = [
            # 'MADDPGlayer2hidden64',
            'MADDPGlayer2hidden64oa',
            # 'MASACGaussianlayer2hidden64er',
            'MASACGaussianlayer2hidden64oaer',
            # 'MASACGaussianlayer2hidden64oadna',
            'PRGGaussiank1hidden64oaonaceerdcigpna',
            # 'PRGGaussiank1hidden64oaonacedcigdnapna',
            # 'PRG3Gaussianhidden64ceerdcigpna',
            # 'PRG3Gaussianhidden64k0m0cedcigdnapna',
            # 'PRG3Gaussianhidden64k0m1cedcigdnapna',
            'PRG3Gaussianhidden64k0m0ceerdcigpna',
            'PRG3Gaussianhidden64k0m1ceerdcigpna',
        ]
# policy_names = policies
policy_names = [
                # 'MADDPG',
                'MADDPG-OA',
                # 'MASAC',
                'MASAC-OA',
                # 'MASACdna',
                'R2G',
                # 'PRGdna',
                # 'R2G3',
                # 'PRG3k0m0dna',
                # 'PRG3k0m1dna',
                'PRG3k0m0er',
                'PRG3k0m1er',
            ]
extra_name = ''
seeds = [0,1,2,3,4]

colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

pre_name = ''
post_name = ''

for fid,field in enumerate(fields):
    print(field)
    fig = plt.figure(fid)
    legends = []
    plts = []
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
        Losses = [losses[:min_itr] for losses in Losses]
        itrs = itrs[:min_itr]
        Losses = np.array(Losses)
        print(Losses.shape)
        y = np.mean(Losses,0)
        yerr = np.std(Losses,0)
        plot, = plt.plot(itrs,y,colors[policy_index])
        plt.fill_between(itrs,y+yerr,y-yerr,linewidth=0,
                            facecolor=colors[policy_index],alpha=0.3)
        plts.append(plot)
        legends.append(policy_names[policy_index])

    plt.legend(plts,legends,loc='best')
    plt.xlabel('Itr')
    plt.ylabel(field_names[fid]) 
    fig.savefig(plot_path+'/'+extra_name+field_names[fid]+'.pdf')
    plt.close(fig)