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
min_loss = [-100,0,-100,0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]
max_loss = [1,1,1,1,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]
exp_name = "2pHard"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

policies = [
            # 'MASACDiscreters10.0',
            'MASACDiscreters100.0',
            'MASACDiscreteonline_actionrs100.0',
            # 'MADDPGGumbelharddouble_q',
            # 'MADDPGGumbelharddouble_qonline_action',
            # 'PRGGumbelk1harddouble_q',
            # 'PRGGumbelk1harddouble_qonline_action',
            # 'PRGGumbelk1harddouble_qtarget_action',
            # 'PRGDiscretek0rs100.0',
            # 'PRGDiscretek1rs100.0',
            'PRGDiscretek1online_actionrs100.0',
            # 'PRGDiscretek1target_actionrs100.0',
            # 'PRGDiscretek1target_qrs100.0',
            # 'PRGDiscretek1online_actiontarget_qrs100.0',
            # 'PRGDiscretek1target_actiontarget_qrs100.0',
            # 'MASACDiscreteSimplifiedActionrs100.0',
            'PRGDiscretek1online_actionhardrs100.0',
            'PRGDiscretek1online_actionsoftrs100.0',
        ]
seeds = [0,1,2,3,4]
policy_names = policies
colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

extra_name = 'Discrete'

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