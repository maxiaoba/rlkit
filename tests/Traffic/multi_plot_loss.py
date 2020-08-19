import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 100
max_itr = 2e4

fields = [
            'evaluation/Average Returns',
            # 'evaluation/Actions Max',
            # 'evaluation/Actions Min',
            # 'evaluation/Num Fail',
            # 'exploration/Average Returns',
            # 'exploration/Returns Max',
            # 'exploration/Returns Min',
            # 'exploration/Num Fail',
            # 'trainer/SUP LossAfter',
            # 'trainer/LossBefore',
            # 'trainer/LossAfter',
            # 'trainer/KLBefore',
            # 'trainer/KL'
            ]
field_names = [
            'Eval Average Return',
            # 'Eval Action Max',
            # 'Eval Action Min',
            # 'Eval Fail',
            # 'Expl Average Return',
            # 'Expl Max Return',
            # 'Expl Min Return',
            # 'Expl Fail',
            # 'Sup LossAfter',
            # 'LossBefore',
            # 'LossAfter',
            # 'KLBefore',
            # 'KL',
            ]
itr_name = 'epoch'
min_loss = [-1000]*100
max_loss = [1000]*100
exp_name = "t_intersection_multiyld1.0full"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

policies = [
            # 'PPO',
            # 'PPOSup',
            # 'PPOSupeb0.1',
            # 'PPOGNN',
            # 'PPOSupGNN',
            # 'PPOSupGNNeb0.1',
            'PPOSupOnline',
            'PPOSupOnlineeb0.1',
            'PPOSupOnlineGNN',
            'PPOSupOnlineGNNeb0.1',
            'TRPO',
            'TRPOSup',
            'TRPOSupeb0.1',
            'TRPOGNN',
            'TRPOSupGNN',
            'TRPOSupGNNeb0.1',
        ]
policy_names = policies

seeds = [2]
colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

extra_name = '2'

pre_name = ''
post_name = ''

plot_names = []

for fid,field in enumerate(fields):
    print(field)
    fig = plt.figure(fid)
    legends = []
    plts = []
    plot_names.append(extra_name+field_names[fid])
    for (policy_index,policy) in enumerate(policies):
        policy_path = pre_name+policy+post_name
        Itrs = []
        Losses = []
        min_itr = np.inf
        for trial in seeds:
            file_path = prepath+'/'+policy_path+'/'+'seed'+str(trial)+'/progress.csv'
            print(file_path)
            if os.path.exists(file_path):
                print(policy+'_'+str(trial))
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
                            itr = i-1#int(float(row[entry_dict[itr_name]]))
                            if itr > max_itr:
                                break
                            loss.append(np.clip(float(row[entry_dict[field]]),
                                                min_loss[fid],max_loss[fid]))
                            if itr % itr_interval == 0:
                                itrs.append(itr)
                                loss = np.mean(loss)
                                losses.append(loss)
                                loss = []
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
    fig.savefig(plot_path+'/'+plot_names[fid]+'.pdf')
    plt.close(fig)