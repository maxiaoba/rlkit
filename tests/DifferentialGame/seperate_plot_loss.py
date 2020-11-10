import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 1
max_itr = 100

fields = [
            'evaluation/Actions 0 Mean',
            'evaluation/Actions 1 Mean',
            # 'trainer/Raw Cactor Loss 0',
            # 'trainer/Raw Cactor Loss 1',
            'evaluation/Average Returns 0',
            'evaluation/Average Returns 1',
            ]
field_names = [
            'a0',
            'a1',
            # 'cactor0_loss',
            # 'cactor1_loss',
            'r0',
            'r1',
            ]
use_abs = True
plot_err = False
itr_name = 'epoch'
min_loss = [-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]
max_loss = [np.inf,np.inf,np.inf,np.inf,np.inf]
exp_name = "max2"

prepath = "./Data/"+exp_name

policies = [
            'MADDPG',
            'MADDPGoa',
            'MASAC',
            'MASACoa',
            'PRGGaussiank1ce',
            'PRGGaussiank1oace',
            'PRGGaussiank1tace',
        ]
policy_names = policies
seeds = [0,1,2,3,4]

colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

pre_name = ''
post_name = ''

for fid,(field,field_name) in enumerate(zip(fields,field_names)):
    for (policy_index,policy) in enumerate(policies):
        policy_path = pre_name+policy+post_name
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
                y = np.array(losses)
                fig = plt.figure()
                plot, = plt.plot(itrs,y,colors[policy_index])
                plt.xlabel('Itr')
                plt.ylabel(field_name) 
                plot_path = prepath+'/'+policy_path+'/'+'seed'+str(trial)
                fig.savefig(plot_path+'/'+field_name+'.png')
                plt.close(fig)