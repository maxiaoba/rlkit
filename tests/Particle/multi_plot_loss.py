import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 10
max_itr = 2e4

fields = [
            'trainer/Policy Loss 0',
            'trainer/Policy Loss 1',
            'trainer/Policy Loss 2',
            'trainer/Policy Loss 3',
            'trainer/Policy Loss 4',
            "trainer/Q1 Predictions 0 Mean",
            "trainer/Q1 Predictions 1 Mean",
            "trainer/Q1 Predictions 2 Mean",
            "trainer/Q1 Predictions 3 Mean",
            "trainer/Q1 Predictions 4 Mean",
            # "trainer/Q1 Predictions 0 Mean",
            # "trainer/Q1 Predictions 1 Mean",
            # "trainer/Q1 Predictions 2 Mean",
            # "trainer/Q1 Predictions 3 Mean",
            # "trainer/Q1 Predictions 4 Mean",
            # "trainer/Alpha 0 Mean","trainer/Alpha 1 Mean","trainer/Alpha 2 Mean","trainer/Alpha 3 Mean","trainer/Alpha 4 Mean",
            # "trainer/Entropy Loss 0","trainer/Entropy Loss 1","trainer/Entropy Loss 2","trainer/Entropy Loss 3","trainer/Entropy Loss 4",
            # 'evaluation/Average Returns 0',
            # 'evaluation/Average Returns 1',
            # 'evaluation/Average Returns 2',
            # 'evaluation/Average Returns 3',
            # 'evaluation/Average Returns 4',
            # 'time/epoch (s)',
            ]
field_names = [
            'Policy Loss 0',
            'Policy Loss 1',
            'Policy Loss 2',
            'Policy Loss 3',
            'Policy Loss 4',
            "Q1 0","Q1 1","Q1 2","Q1 3","Q1 4",
            # 'Alpha 0','Alpha 1','Alpha 2','Alpha 3','Alpha 5', 
            # 'Entropy Loss 0','Entropy Loss 1','Entropy Loss 2','Entropy Loss 3','Entropy Loss 5', 
            # 'Average Return 0',
            # 'Average Return 1',
            # 'Average Return 2',
            # 'Average Return 3',
            # 'Average Return 4',
            # 'time epoch (s)',
            ]

itr_name = 'epoch'
min_loss = [-np.inf]*20
max_loss = [np.inf]*20
exp_name = "simple_spreadag5l5_mpl25"

prepath = "./Data/"+exp_name
plot_path = "./Data/"+exp_name

policies = [
            'MADDPGlayer2hidden64',
            'MADDPGlayer2hidden64oa',
            'MASACGaussianlayer2hidden64er',
            'MASACGaussianlayer2hidden64oaer',
            'PRGGaussianhidden64k1oaceerdcigpna',
            'PRG3Gaussianhidden64ceerdcigpna',
            # 'testhidden64ceerdcigpna',
        ]
# policy_names = policies
policy_names = [
                'MADDPG',
                'MADDPG-OA',
                'MASAC',
                'MASAC-OA',
                'R2G',
                'R2G3',
                # 'test',
            ]
extra_name = ''
seeds = [0,1,2]

colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

pre_name = ''
post_name = ''

for fid,field in enumerate(fields):
    print(field)
    fig = plt.figure(fid)
    legends = []
    plts = []
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
                                if field in entry_dict.keys():
                                    loss.append(np.clip(float(row[entry_dict[field]]),
                                                        min_loss[fid],max_loss[fid]))
                                else:
                                    loss.append(0.)
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
        plot, = plt.plot(itrs,y,colors[policy_index])
        plt.fill_between(itrs,y+yerr,y-yerr,linewidth=0,
                            facecolor=colors[policy_index],alpha=0.3)
        plts.append(plot)
        legends.append(policy_names[policy_index])

    plt.legend(plts,legends,loc='best')
    plt.xlabel('Itr')
    plt.ylabel(field_names[fid]) 
    fig.savefig(plot_path+'/'+extra_name+field_names[fid]+'.pdf')
    plt.close(fig)