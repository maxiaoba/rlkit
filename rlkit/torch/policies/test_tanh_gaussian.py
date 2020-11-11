import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.layers import *

network = torch.nn.Sequential(
			SplitLayer(layers=[nn.Linear(1,2),nn.Linear(1,2),nn.Linear(1,2)]),
			ConcatLayer(dim=-1)
			)

input_tensor = torch.tensor([[0.],[0.]])
print(network(input_tensor))
