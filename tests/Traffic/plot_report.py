import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 13})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 100
max_itr = 1e4

fields = [
            'evaluation/Average Returns',
            'evaluation/Num Success',
            'trainer/SUP LossAfter',
            ]
field_names = [
            'Eval Average Return',
            'Eval Success Rate',
            'Supervised Learning Loss',
            ]
legend_fields = [0,1]
itr_name = 'epoch'
min_loss = [0]*100
max_loss = [1000]*100
exp_name = "t_intersection_multinobyld0.5ds0.1fullfull"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

# policies = [
#             'PPOhidden72ep5000',
#             # 'PPOSupVanillahidden64ep5000',
#             'PPOSuphidden64ep5000',
#             # 'PPOSupOnlinehidden64ep5000',
#             'PPOSupSep2hidden40ep5000',
#             ]
# policy_names = [
#             'PPO',
#             # 'MLP + Sup Vanilla',
#             'PPO + Shared Supervised Learning',
#             # 'MLP + Sup Online',
#             'PPO + Separated Supervised Learning'
#             ]
# extra_name = 't_intersection_mlp'
# colors = ['C0','C1','C2']

# policies = [
#             'PPOGSagenode48layer3actreluep5000',
#             # 'PPOSupVanillaGSagenode48layer3actreluep5000',
#             'PPOSupGSagenode48layer3actreluep5000',
#             # 'PPOSupOnlineGSagenode48layer3actreluep5000',
#             'PPOSupSep2GSagenode32layer3actreluep10000',
#             ]
# policy_names = [
#             'PPO',
#             # 'GNN + Sup Vanilla',
#             'PPO + Shared Supervised Learning',
#             # 'GNN + Sup Online',
#             'PPO + Separated Supervised Learning'
#             ]
# extra_name = 't_intersection_gsage'
# colors = ['C3','C4','C5']

# policies = [
#             'PPOGSageWnode40layer3attentionactreluep5000',
#             # 'PPOSupVanillaGSageWnode40layer3attentionactreluep5000',
#             'PPOSupGSageWnode40layer3attentionactreluep5000',
#             # 'PPOSupOnlineGSageWnode40layer3attentionactreluep5000',
#             'PPOSupSep2GSageWGSagenode32layer3attentionactreluep5000',
#             ]
# policy_names = [
#             'PPO',
#             # 'GNN + Sup Vanilla',
#             'PPO + Shared Supervised Learning',
#             # 'GNN + Sup Online',
#             'PPO + Separated Supervised Learning'
#             ]
# extra_name = 't_intersection_gsagew'
# colors = ['C6','C7','C8']

policies = [
            # 'PPOhidden72ep5000',
            'PPOSuphidden64ep5000',
            'PPOSupSep2hidden40ep5000',
            # 'PPOGSagenode48layer3actreluep5000',
            # 'PPOSupGSagenode48layer3actreluep5000',
            # 'PPOSupSep2GSagenode32layer3actreluep10000',
            # 'PPOGSageWnode40layer3attentionactreluep5000',
            'PPOSupGSageWnode40layer3attentionactreluep5000',
            'PPOSupSep2GSageWGSagenode32layer3attentionactreluep5000',
            # 'PPOSupSep2MLPGSagehidden40node32layer3actreluep5000',
            ]
policy_names = [
            # 'MLP',
            'MLP + Shared Supervised Learning',
            'MLP + Separated Supervised Learning',
            # 'GNN',
            'GNN + Shared Supervised Learning',
            'GNN + Separated Supervised Learning',
            # 'MLPGNN',
            ]
colors = ['C1','C2','C7','C8']
extra_name = 't_intersection_sup'

# policies = [
#             # 'PPOhidden72ep5000',
#             'PPOSuphidden64ep5000',
#             # 'PPOSupSep2hidden40ep5000',
#             # 'PPOGSagenode48layer3actreluep5000',
#             'PPOSupGSagenode48layer3actreluep5000',
#             # 'PPOSupSep2GSagenode32layer3actreluep10000',
#             # 'PPOGSageWnode40layer3attentionactreluep5000',
#             'PPOSupGSageWnode40layer3attentionactreluep5000',
#             # 'PPOSupSep2GSageWGSagenode32layer3attentionactreluep5000',
#             # 'PPOSupSep2MLPGSagehidden40node32layer3actreluep5000',
#             ]
# policy_names = [
#             'MLP',
#             'GNN',
#             'GNN + attention',
#             ]
# colors = ['C1','C4','C7']
# extra_name = 't_intersection_share'

seeds = [0,1,2]

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
                ### load ###
                load_file_path = prepath+'/'+policy_path+'/'+'seed'+str(trial)+'_load/progress.csv'
                if os.path.exists(load_file_path):
                    last_itr = itr
                    print(policy+'_'+str(trial)+'_load')
                    with open(load_file_path) as csv_file:
                        if '\0' in open(load_file_path).read():
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
                                itr = last_itr+i #int(float(row[entry_dict[itr_name]]))
                                if itr > max_itr:
                                    break
                                try:
                                    if field == 'evaluation/Num Success':
                                        num_path = float(row[entry_dict['evaluation/Num Paths']])
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
                ### load ###
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

    if fid in legend_fields:
        plt.legend(plts,legends,loc='best')
    plt.xlabel('Itr')
    plt.ylabel(field_names[fid]) 
    fig.savefig(plot_path+'/'+plot_names[fid]+'.pdf')
    plt.close(fig)