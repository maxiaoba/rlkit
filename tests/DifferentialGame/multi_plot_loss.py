import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 1
max_itr = 100

fields = [
            # 'evaluation/Actions 0 Mean',
            'evaluation/Actions 1 Mean',
            # 'evaluation/Average Returns 0',
            ]
field_names = [
            'Average absolute action value'
            ]
use_abs = True
plot_err = False
itr_name = 'epoch'
min_loss = [-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]
max_loss = [np.inf,np.inf,np.inf,np.inf,np.inf]
exp_name = "zero_sum"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

policies = [
            # 'MADDPG',
            'MADDPGonline_action',
            # 'MASAC',
            'MASAConline_action',
            # 'PRGk1online_action',
            # 'PRGGaussiank1online_action',
            'PRGGaussiank1online_actioncentropy',
        ]
# policy_names = policies
policy_names = [
                'MADDPG',
                # 'MADDPGonline',
                'MASAC',
                # 'MASAConline',
                # 'PRGk1online',
                # 'PRGGaussiank1',
                # 'PRGGaussiank1online',
                'PRGGaussiank1onlinecentropy',
            ]
seeds = [0,1,2,3,4]

colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

extra_name = 'absa1'

pre_name = ''
post_name = ''

plot_name = extra_name

fig = plt.figure()
for fid,(field,field_name) in enumerate(zip(fields,field_names)):
    print(field)
    plt.subplot(len(fields),1,fid+1)
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
        if use_abs:
            Losses = np.abs(Losses)
        print(Losses.shape)
        y = np.mean(Losses,0)
        yerr = np.std(Losses,0)
        plot, = plt.plot(itrs,y,colors[policy_index])
        if plot_err:
            plt.fill_between(itrs,y+yerr,y-yerr,linewidth=0,
                                facecolor=colors[policy_index],alpha=0.3)
        plts.append(plot)
        legends.append(policy_names[policy_index])
            # y = np.array(losses)
            # if trial == 0:
            #     plot, = plt.plot(itrs,y,colors[policy_index],label=policy_names[policy_index])
            # else:
            #     plot, = plt.plot(itrs,y,colors[policy_index])

    plt.legend(plts,legends,loc='best')
    # plt.legend()
    plt.xlabel('Itr')
    plt.ylabel(field_name) 
fig.savefig(plot_path+'/'+plot_name+'.pdf')
plt.close(fig)