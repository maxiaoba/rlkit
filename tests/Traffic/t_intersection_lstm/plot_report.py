import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 15})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 200
max_itr = 1e4

fields = [
            # 'evaluation/Average Returns',
            'evaluation/Num Success',
            # 'exploration/Average Returns',
            # 'exploration/Num Success',
            'trainer/SUP AccuracyAfter',
            ]
field_names = [
            # 'Evaluation Average Return',
            'Evaluation Success Rate',
            # 'Exploration Average Return',
            # 'Exploration Success Rate',
            'Intention Inference Accuracy',
            ]
itr_name = 'epoch'
min_loss = [-1000]*100
max_loss = [1000]*100
exp_name = "t_intersection_lstm4noise0.05yld0.5ds0.1"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

# policies = [
#             # 'PPOlayer1hidden48ep5000',
#             # 'PPOSupVanillalayer1hidden48ep5000',
#             # 'PPOSuplayer1hidden48ep5000',
#             'PPOSupSep2layer1hidden28ep5000',
#             # 'PPOGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
#             # 'PPOSupVanillaGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
#             # 'PPOSupGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
#             'PPOSupSep2GNN2llayer1hidden18GSagenode18glayer3actreluep5000',
#             'PPOSupSep2LSTMGNN2layer1hidden28GSagenode18glayer3suphidden18suplayer1actreluep5000',
#         ]
# policy_names = [
#             # 'PPO + LSTM',
#             # 'PPO + LSTM \nShared Inference + LSTM',
#             # 'PPO + LSTM \nCoupled Inference + LSTM',
#             'PPO + LSTM \nSeparated Inference + LSTM',
#             # 'PPO + STGSage',
#             # 'PPO + STGSage \nShared Inference + STGSage',
#             # 'PPO + STGSage \nCoupled Inference + STGSage',
#             'PPO + STGSage \nSeparated Inference + STGSage',
#             'PPO + LSTM \nSeparated Inference + STGSage',
#         ]
# colors = [
#         # 'C0',
#         # 'C1',
#         # 'C2',
#         'C3',
#         # 'C4',
#         # 'C5',
#         # 'C6',
#         'C7',
#         'C8',
#         ]
# extra_name = 'Separated Inference'

policies = [
            'PPOGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            'PPOGNN2llayer1hidden26GCNnode26glayer3actreluep10000',
            'PPOGNN2llayer1hidden26GATnode26glayer3actreluep10000',
            # 'PPOSupVanillaGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            # 'PPOSupVanillaGNN2llayer1hidden26GCNnode26glayer3actreluep10000',
            # 'PPOSupVanillaGNN2llayer1hidden26GATnode26glayer3actreluep10000'
            # 'PPOSupGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            # 'PPOSupGNN2llayer1hidden26GCNnode26glayer3actreluep10000',
            # 'PPOSupGNN2llayer1hidden26GATnode26glayer3actreluep10000'
            # 'PPOSupSep2GNN2llayer1hidden18GSagenode18glayer3actreluep5000',
            # 'PPOSupSep2GNN2llayer1hidden18GCNnode18glayer3actreluep10000',
            # 'PPOSupSep2GNN2llayer1hidden18GATnode18glayer3actreluep10000',
            # 'PPOSupSep2LSTMGNN2layer1hidden28GSagenode18glayer3suphidden18suplayer1actreluep5000',
            # 'PPOSupSep2LSTMGNN2layer1hidden28GCNnode18glayer3suphidden18suplayer1actreluep10000',
            # 'PPOSupSep2LSTMGNN2layer1hidden28GATnode18glayer3suphidden18suplayer1actreluep10000',
        ]
policy_names = [
            'PPO + STGSage',
            'PPO + STGCN',
            'PPO + STGAT',
            # 'PPO + STGSage \nShared Inference + STGSage',
            # 'PPO + STGCN \nShared Inference + STGCN',
            # 'PPO + STGAT \nShared Inference + STGAT',
            # 'PPO + STGSage \nCoupled Inference + STGSage',
            # 'PPO + STGCN \nCoupled Inference + STGCN',
            # 'PPO + STGAT \nCoupled Inference + STGAT',
            # 'PPO + STGSage \nSeparated Inference + STGSage',
            # 'PPO + STGCN \nSeparated Inference + STGCN',
            # 'PPO + STGAT \nSeparated Inference + STGAT',
            # 'PPO + LSTM \nSeparated Inference + STGSage',
            # 'PPO + LSTM \nSeparated Inference + STGSCN',
            # 'PPO + LSTM \nSeparated Inference + STGSAT',
        ]
colors = [
        'C4',
        # 'C5',
        # 'C6',
        # 'C7',
        # 'C8',
        'C9',
        'C10',
        ]
extra_name = 'PPOGNN'

seeds = [0,1,2]
pre_name = ''
post_name = ''

plot_names = []

fig = plt.figure(figsize=(20,5))
for fid,field in enumerate(fields):
    print(field)
    plt.subplot(1,len(fields),fid+1)
    # fig = plt.figure(fid)
    legends = []
    plts = []
    plot_names.append(extra_name+field_names[fid])
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
        # plot, = plt.plot(itrs,y,colors[policy_index])
        plot, = plt.plot(itrs,y,colors[policy_index],label=policy_names[policy_index])
        plt.fill_between(itrs,y+yerr,y-yerr,linewidth=0,
                            facecolor=colors[policy_index],alpha=0.3)
        legends.append(policy_names[policy_index])

    # if fid == len(fields)-1:
    #     plt.legend(plts,legends, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xlabel('Itr')
    plt.ylabel(field_names[fid]) 
    # fig.savefig(plot_path+'/'+plot_names[fid]+'.pdf',bbox_inches='tight')
    # plt.close(fig)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=len(labels))
# plt.legend(plts,legends, bbox_to_anchor=(1.01, 1), loc='upper left')
fig.savefig(plot_path+'/'+extra_name+'.pdf',bbox_inches='tight')
plt.close(fig)