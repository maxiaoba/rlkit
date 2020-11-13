import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 1
max_itr = 1000

fields = [
            'exploration/Actions 0 Max',
            'exploration/Actions 0 Min',
            'evaluation/Actions 0 Max',
            'evaluation/Actions 0 Min',
            # 'exploration/Returns 0 Max',
            # 'exploration/Returns 0 Min',
            # 'evaluation/Average Returns 0',
            # 'trainer/Q1 Predictions 0 Max',
            # 'trainer/Q1 Predictions 0 Min',
            # 'trainer/Q2 Predictions 0 Max',
            # 'trainer/Q2 Predictions 0 Min',
            ]
field_names = [
            'Expl a0 max',
            'Expl a0 min',
            'Eval a0 max',
            'Eval a0 min',
            # 'Expl Return Max',
            # 'Expl Return Min',
            # 'Eval Average Return',
            # 'Q1 max',
            # 'Q1 min',
            # 'Q2 max',
            # 'Q2 min'
            ]
use_abs = False
plot_err = True
itr_name = 'epoch'
min_loss = [-np.inf]*100
max_loss = [np.inf]*100
exp_name = "max2"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

policies = [
            # 'MADDPGhidden32oa',
            'MASACGaussianhidden32oa',
            'MASACMixGaussianm2hidden32oa',
            'PRGGaussiank1hidden32oace',
            'PRGGaussiank1hidden32oaonace',
            'PRGMixGaussiank1m2hidden32oace',
            'PRGMixGaussiank1m2hidden32oaonace'
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