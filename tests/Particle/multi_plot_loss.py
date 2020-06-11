import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 10
max_itr = 2e4

fields = [
            'evaluation/Average Returns 0',
            ]
itr_name = 'epoch'
min_loss = [-1000]
max_loss = [0]
exp_name = "simple_spread"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

policies = [
            'MADDPG',
            'MASAC',
            'PRGk1',
            'PRGk1online_action',
            # 'PRGk1target_action',
            'PRGGaussiank1',
            'PRGGaussiank1online_action',
            # 'PRGGaussiank1target_action'
        ]
seeds = [0,1,2,3,4]
policy_names = policies
colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

extra_name = ''

pre_name = ''
post_name = ''

plot_name = extra_name

for fid,field in enumerate(fields):
    print(field)
    fig = plt.figure(fid,figsize=(5,5))
    legends = []
    plts = []
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
        # plt.fill_between(itrs,y+yerr,y-yerr,linewidth=0,
        #                     facecolor=colors[policy_index],alpha=0.3)
        plts.append(plot)
        legends.append(policy_names[policy_index])

    plt.legend(plts,legends,loc='best')
    plt.xlabel('Itr')
    plt.ylabel(field) 
    fig.savefig(plot_path+'/'+plot_name+'_'+"_".join(field.split('/'))+'.pdf')
    plt.close(fig)