import torch
import numpy as np
from torch.distributions import Categorical

logits = torch.zeros(2,3,4)
logits[0,0,:] = torch.tensor([1,-1,-1,-1])
logits[0,1,:] = torch.tensor([-1,1,-1,-1])
logits[0,2,:] = torch.tensor([-1,-1,1,-1])
logits[1,0,:] = torch.tensor([1,-1,-1,-1])*2
logits[1,1,:] = torch.tensor([-1,1,-1,-1])*2
logits[1,2,:] = torch.tensor([-1,-1,1,-1])*2

d = Categorical(logits=logits)
labels = torch.zeros(2,3)
labels[0,0] = np.nan
valid_mask = ~torch.isnan(labels)
labels[~valid_mask] = 0
log_prob = d.log_prob(labels)
print(log_prob)
print(log_prob[valid_mask])
entropy = d.entropy()
print(entropy)

