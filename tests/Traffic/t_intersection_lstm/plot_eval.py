import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 20})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 200
max_itr = 1e4

fields = [
            # 'Return',
            'Success Rate',
            # 'Collision Rate',
            'Inference Accuracy'
            ]
field_names = [
            # 'Evaluation Average Return',
            'Evaluation Success Rate',
            # 'Evaluation Collision Rate',
            'Latent Inference Accuracy',
            ]
itr_name = 'epoch'
min_loss = [-1000]*100
max_loss = [1000]*100
exp_name = "t_intersection_lstm4noise0.05yld0.5ds0.1"

prepath = "./Data/"+exp_name+"/Eval"
plot_path = "./Data/"+exp_name+"/Eval"

# result_paths = [
#                 'noise0.05yld0.2ds0.1dfd0.1dfi0.3epoch500',
#                 'noise0.05yld0.4ds0.1dfd0.1dfi0.3epoch500',
#                 'noise0.05yld0.6ds0.1dfd0.1dfi0.3epoch500',
#                 'noise0.05yld0.8ds0.1dfd0.1dfi0.3epoch500',]
# ylabels = ['Reletive Success Rate','Reletive Inference Accuracy']
# xlabel = 'P(CONSERVATIVE)'
# xs = [0.2,0.4,0.6,0.8]
# extra_name = 'yld_drift'

result_paths = [
                'noise0.05yld0.5ds0.1dfd0.1dfi0.1epoch500',
                'noise0.05yld0.5ds0.1dfd0.1dfi0.3epoch500',
                'noise0.05yld0.5ds0.1dfd0.1dfi0.5epoch500',
                ]
ylabels = ['Reletive Success Rate','Reletive Inference Accuracy']
xlabel = 'Front Gap Sample Interval'
xs = [0.1,0.3,0.5]
extra_name = 'dfi_drift'

# result_paths = [
#                 'noise0.05yld0.5ds0.1dfd0.05dfi0.3epoch500',
#                 'noise0.05yld0.5ds0.1dfd0.1dfi0.3epoch500',
#                 'noise0.05yld0.5ds0.1dfd0.3dfi0.3epoch500',
#                 'noise0.05yld0.5ds0.1dfd0.5dfi0.3epoch500',
#                 ]
# ylabels = ['Reletive Success Rate','Reletive Inference Accuracy']
# xlabel = 'Front Gap Mean Difference'
# xs = [0.05,0.1,0.3,0.5]
# extra_name = 'dfd_drift'

policies = [
            'PPOlayer1hidden48ep5000',
            'PPOSupVanillalayer1hidden48ep5000',
            'PPOSuplayer1hidden48ep5000',
            'PPOSupSep2layer1hidden28ep5000',
            'PPOGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            'PPOSupVanillaGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            'PPOSupGNN2llayer1hidden24GSagenode24glayer3actreluep5000',
            'PPOSupSep2GNN2llayer1hidden18GSagenode18glayer3actreluep5000',
            'PPOSupSep2LSTMGNN2layer1hidden28GSagenode18glayer3suphidden18suplayer1actreluep5000',
        ]
policy_names = [
            'PPO + LSTM',
            'PPO + LSTM \nShared Inference + LSTM',
            'PPO + LSTM \nCoupled Inference + LSTM',
            'PPO + LSTM \nSeparated Inference + LSTM',
            'PPO + STGSage',
            'PPO + STGSage \nShared Inference + STGSage',
            'PPO + STGSage \nCoupled Inference + STGSage',
            'PPO + STGSage \nSeparated Inference + STGSage',
            'PPO + LSTM \nSeparated Inference + STGSage',
        ]
colors = [
        'C0',
        'C1',
        'C2',
        'C3',
        'C4',
        'C5',
        'C6',
        'C7',
        'C8',
        ]


seeds = [0,1,2]
pre_name = ''
post_name = ''

results = {}
for policy in policies:
    results[policy] = {}
    for field in fields:
        results[policy][field]=[]

for result_path in result_paths:
    file_path = prepath+'/'+result_path+'/result.csv'
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
                policy, seed, _ = row[entry_dict['Policy']].split('_')
                if policy in policies:
                    for field in fields:
                        results[policy][field].append(float(row[entry_dict[field]]))

fig = plt.figure(figsize=(10*len(fields),5))
plot_names = []
avg_losses = {}
for field in fields:
    Base_Losses = []
    for p in policies:
        if np.sum(results[p][field]) == 0:
            continue
        Base_Losses.append(np.reshape(results[p][field],(len(xs),len(seeds))).transpose())
    avg_losses[field] = np.mean(np.mean(Base_Losses, 0), 0)

for fid,field in enumerate(fields):
    print(field)
    plt.subplot(1,len(fields),fid+1)
    # fig = plt.figure(fid)
    legends = []
    plts = []
    plot_names.append(extra_name+field_names[fid])
    for (policy_index,policy) in enumerate(policies):
        Losses = np.reshape(results[policy][field],(len(xs),len(seeds))).transpose()
        Losses = Losses/avg_losses[field]
        if np.sum(Losses) == 0:
            continue

        # print(policy,field,Losses)
        y = np.mean(Losses,0)
        yerr = np.std(Losses,0)
        # plot, = plt.plot(itrs,y,colors[policy_index])
        plt.errorbar(xs,y,yerr,color=colors[policy_index], fmt='o-', markersize=3, capsize=10,label=policy_names[policy_index])
        # plot, = plt.plot(xs,y,colors[policy_index],label=policy_names[policy_index],marker='o')
        # plt.fill_between(xs,y+yerr,y-yerr,linewidth=0,
        #                     facecolor=colors[policy_index],alpha=0.3)
        legends.append(policy_names[policy_index])

    # if fid == len(fields)-1:
    #     plt.legend(plts,legends, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabels[fid]) 
    # fig.savefig(plot_path+'/'+plot_names[fid]+'.pdf',bbox_inches='tight')
    # plt.close(fig)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
# plt.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc='upper left')
# fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=len(labels))
fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.3), loc='upper center', ncol=4)
fig.savefig(plot_path+'/'+extra_name+'.pdf',bbox_inches='tight')
plt.close(fig)