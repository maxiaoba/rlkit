import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 10
max_itr = 2e4

fields = [
            'evaluation/Success Rate',
            'exploration/Success Rate',
            'evaluation/Average Returns 0',
            'exploration/Average Returns 0',
            # "trainer/QF1 Gradient 0",
            # "trainer/Policy Gradient 0",
            # "trainer/QF1 Loss 0",
            # "trainer/QF2 Loss 0",
            # "trainer/Raw Policy Loss 0",
            # "trainer/Entropy Loss 0",
            # "trainer/Alpha 0",
            ]
itr_name = 'epoch'
min_loss = [0,0,-100,-100,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]
max_loss = [1,1,100,100,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]
exp_name = "2pHard"

prepath = "./Data/"+exp_name

policies = [
            'MASACDiscreteUnsym1online_actionrs100.0',
            'PRGDiscreteUnsym1k1online_actionhardrs100.0',
            'PRGDiscreteUnsym1k1online_actionsoftrs100.0',
            'MASACDiscreteUnsym2online_actionrs100.0',
            'PRGDiscreteUnsym2k1online_actionhardrs100.0',
            'PRGDiscreteUnsym2k1online_actionsoftrs100.0',
        ]
seeds = [0,1,2,3,4]
policy_names = policies

extra_name = ''

pre_name = ''
post_name = ''

plot_name = extra_name

for fid,field in enumerate(fields):
    print(field)
    for (policy_index,policy) in enumerate(policies):
        fig = plt.figure(fid,figsize=(5,5))
        legends = []
        plts = []
        policy_path = pre_name+policy+post_name
        plot_path = prepath+'/'+policy_path
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
                    print(len(losses))
                    y = np.array(losses)
                    plot, = plt.plot(itrs,y)
                    plts.append(plot)
                    legends.append(policy_names[policy_index]+'_'+str(trial))

        plt.legend(plts,legends,loc='best')
        plt.xlabel('Itr')
        plt.ylabel(field) 
        fig.savefig(plot_path+'/'+plot_name+'_'+"_".join(field.split('/'))+'.pdf')
        plt.close(fig)