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
            'evaluation/Num Success',
            # 'evaluation/Num Timeout',
            # 'evaluation/Num Fail',
            'exploration/Average Returns',
            # 'exploration/Returns Max',
            # 'exploration/Returns Min',
            # 'exploration/Num Fail',
            'exploration/Num Success',
            'trainer/SUP LossAfter',
            'trainer/LossBefore',
            'trainer/LossAfter',
            'trainer/KLBefore',
            'trainer/KL'
            ]
field_names = [
            'Eval Average Return',
            # 'Eval Action Max',
            # 'Eval Action Min',
            'Eval Success',
            # 'Eval Timeout',
            # 'Eval Fail',
            'Expl Average Return',
            # 'Expl Max Return',
            # 'Expl Min Return',
            # 'Expl Fail',
            'Expl Success',
            'Sup LossAfter',
            'LossBefore',
            'LossAfter',
            'KLBefore',
            'KL',
            ]
itr_name = 'epoch'
min_loss = [-1000]*100
max_loss = [1000]*100
exp_name = "t_intersection_lstmnoise0.05yld0.5ds0.1"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

policies = [
            # 'PPOlayer1hidden48ep5000',
            'PPOGNNllayer1hidden32GSagenode24glayer3actreluep5000',
            # 'PPOGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            # 'PPOSuplayer1hidden48ep5000',
            'PPOSupGNNllayer1hidden24GSagenode32glayer3actreluep5000',
            'PPOSupGNNllayer1hidden32GSagenode24glayer3actreluep5000',
            # 'PPOSupGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            # 'PPOSupSep2layer1hidden28ep5000',
            'PPOSupSep2GNNllayer1hidden16GSagenode24glayer3actreluep5000',
            'PPOSupSep2GNNllayer1hidden24GSagenode16glayer3actreluep5000',
            # 'PPOSupSep2GNN2llayer1hidden16GSagenode20glayer3actreluep5000'
        ]
policy_names = policies

seeds = [0,1,2]
colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

extra_name = 'gnnlstm'

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
                            try:
                                if field == 'evaluation/Num Success':
                                    num_path = float(row[entry_dict['evaluation/Num Paths']])
                                    loss.append(np.clip(float(row[entry_dict[field]])/num_path,
                                                    min_loss[fid],max_loss[fid]))
                                elif field == 'exploration/Num Success':
                                    num_path = float(row[entry_dict['exploration/Num Paths']])
                                    loss.append(np.clip(float(row[entry_dict[field]])/num_path,
                                                    min_loss[fid],max_loss[fid]))
                                else:
                                    loss.append(np.clip(float(row[entry_dict[field]]),
                                                    min_loss[fid],max_loss[fid]))
                            except:
                                pass
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
        # plt.fill_between(itrs,y+yerr,y-yerr,linewidth=0,
                            # facecolor=colors[policy_index],alpha=0.3)
        plts.append(plot)
        legends.append(policy_names[policy_index])

    plt.legend(plts,legends,loc='best')
    plt.xlabel('Itr')
    plt.ylabel(field_names[fid]) 
    fig.savefig(plot_path+'/'+plot_names[fid]+'.pdf')
    plt.close(fig)