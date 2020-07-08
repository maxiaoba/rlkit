import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 20})
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os.path as osp

def load_tensorboard(file_path,field,average,seeds):
	Vals = []
	for trial in seeds:
		print(trial)
		if trial > 0:
			event_acc = EventAccumulator(file_path+"_"+str(trial))
		else:	
			event_acc = EventAccumulator(file_path)
		event_acc.Reload()
		if field in event_acc.Tags()['scalars']:
			w_times, step_nums, vals = zip(*event_acc.Scalars(field))
			vals = np.convolve(vals, np.ones((average,))/average, mode='same')
			Vals.append(vals)
			print(vals.shape)
	return np.array(Vals)
