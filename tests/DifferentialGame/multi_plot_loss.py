import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 1
max_itr = 1000

fields = [
            # 'evaluation/Actions 0 Mean',
            # 'evaluation/Actions 1 Mean',
            'exploration/Returns 0 Max',
            'evaluation/Average Returns 0',
            # 'trainer/Q1 Predictions 0 Max',
            # 'trainer/Q1 Predictions 0 Min',
            # 'trainer/Q2 Predictions 0 Max',
            # 'trainer/Q2 Predictions 0 Min',
            # 'trainer/Alpha 0',
            # 'trainer/Alpha 1',
            ]
field_names = [
            # 'a0',
            # 'a1',
            'Expl Return Max',
            'Eval Average Return',
            # 'Q1 max',
            # 'Q1 min',
            # 'Q2 max',
            # 'Q2 min'
            # 'alpha0',
            # 'alpha1',
            ]
use_abs = False
plot_err = True
itr_name = 'epoch'
min_loss = [-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]
max_loss = [np.inf,np.inf,np.inf,np.inf,np.inf]
exp_name = "max2"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

policies = [
            # 'MADDPGhidden32oa',
            # 'MASACGaussianhidden32oa',
            # 'MASACMixGaussianm2hidden32oa',
            # 'PRGGaussiank1hidden32oace',
            # 'PRGMixGaussiank1m2hidden32oace'
            # 'MASACGaussianhidden32oaalpha3.0',
            # 'MASACGaussianhidden32oaalpha3.0fa',
            # 'MASACGaussianhidden32oaalpha5.0',
            # 'MASACGaussianhidden32oaalpha5.0fa',
            # 'MASACMixGaussianm2hidden32oaalpha3.0',
            # 'MASACMixGaussianm2hidden32oaalpha3.0fa',
            # 'MASACMixGaussianm2hidden32oaalpha5.0',
            # 'MASACMixGaussianm2hidden32oaalpha5.0fa',
            # 'PRGGaussiank1hidden32oacealpha3.0',
            # 'PRGGaussiank1hidden32oacealpha3.0dcig',
            'PRGGaussiank1hidden32oacealpha3.0fa',
            # 'PRGGaussiank1hidden32oacealpha5.0',
            # 'PRGGaussiank1hidden32oacealpha5.0dcig',
            'PRGGaussiank1hidden32oacealpha5.0fa',
            # 'PRGMixGaussiank1m2hidden32oacealpha3.0',
            # 'PRGMixGaussiank1m2hidden32oacealpha3.0fa',
            # 'PRGMixGaussiank1m2hidden32oacealpha5.0',
            # 'PRGMixGaussiank1m2hidden32oacealpha5.0fa'
            # 'PRG2Gaussiank1hidden32ce',
            # 'PRG3Gaussiank1hidden32ce',
            'PRG3Gaussiank1hidden32oacealpha3.0fa',
        ]
policy_names = policies
seeds = [0,1,2,3,4]

colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

extra_name = 'prg3'

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
    plt.xlabel('Itr')
    plt.ylabel(field_name)
    if field == fields[-1]:
        plt.legend(plts,legends,loc='best')
        # plt.legend() 
fig.savefig(plot_path+'/'+plot_name+'.pdf')
plt.close(fig)