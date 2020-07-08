import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 10
max_itr = 300

field = 'evaluation/Success Rate'
external_field = "eval_successes"
field_name = 'Success Rate'
itr_name = ''
min_loss = 0
max_loss = 1
exp_name = "2pHard"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

externals = [
            'vdnq_unsym1_r2',
            'mixq_unsym1_r2',
            'pr2q_unsym1_r2',
        ]
external_names = [
            'VDN',
            'QMIX',
            'PR2',
        ]
external_path = '/Users/xiaobaima/.julia/dev/BlockedRoad/scripts/log/2pHard'
policies = [
            'MASACDiscreteUnsym1rs100.0',
            'MASACDiscreteUnsym1online_actionrs100.0',
            'PRGDiscreteUnsym1k1online_actionhardrs100.0',
            'PRGDiscreteUnsym1k1online_actionsoftrs100.0',
        ]
policy_names = [
            'MASAC',
            'MASAConline',
            'PRGhard',
            'PRGsoft',
        ]
extra_name = 'usym1'

seeds = [0,1,2,3,4]
colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

plot_name = 'joint'+extra_name

fig = plt.figure(figsize=(6,5))
legends = []
plts = []

from load_tensorboard import load_tensorboard
for (external_index,external) in enumerate(externals):
    file_path = external_path+'/'+external
    print(file_path)
    Losses = load_tensorboard(file_path,external_field,itr_interval,seeds)
    Losses = Losses[:,:max_itr]/100.
    itrs = np.arange(0,Losses.shape[1])
    y = np.mean(Losses,0)
    yerr = np.std(Losses,0)
    # plot, = plt.plot(itrs,y,colors[policy_index])
    # plt.fill_between(itrs,y+yerr,y-yerr,linewidth=0,
    #                     facecolor=colors[policy_index],alpha=0.3)
    plot, = plt.plot(itrs,y)
    plts.append(plot)
    legends.append(external_names[external_index])

for (policy_index,policy) in enumerate(policies):
    Itrs = []
    Losses = []
    min_itr = np.inf
    for trial in seeds:
        file_path = prepath+'/'+policy+'/'+'seed'+str(trial)+'/progress.csv'
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
                                            min_loss,max_loss))
                        if itr % itr_interval == 0:
                            itrs.append(itr)
                            loss = np.mean(loss)
                            losses.append(loss)
                            loss = []
                print(len(losses))
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
    # plt.fill_between(itrs,y+yerr,y-yerr,linewidth=0,
    #                     facecolor=colors[policy_index],alpha=0.3)
    plot, = plt.plot(itrs,y)
    plts.append(plot)
    legends.append(policy_names[policy_index])

plt.legend(plts,legends,loc='best')
plt.xlabel('Itr')
plt.ylabel(field_name) 
fig.savefig(plot_path+'/'+plot_name+'_'+"_".join(field.split('/'))+'.pdf')
plt.close(fig)