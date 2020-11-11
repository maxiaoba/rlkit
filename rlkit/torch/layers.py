import torch
from torch import nn as nn
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np, np_ify, torch_ify

class ReshapeLayer(torch.nn.Module):
    def __init__(self, shape, keep_dim=-1):
        super(ReshapeLayer, self).__init__()
        self.shape = shape
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.reshape(*x.shape[:self.keep_dim],*self.shape)

class FlattenLayer(torch.nn.Module):
    def __init__(self, keep_dim=1):
        super(FlattenLayer, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.reshape(*x.shape[:self.keep_dim],-1)

class SelectLayer(torch.nn.Module):
    def __init__(self, dim, index):
        super(SelectLayer, self).__init__()
        self.dim = dim
        self.index = torch.tensor(index).to(ptu.device)

    def forward(self, x):
        return torch.index_select(x, self.dim, self.index)

class SplitLayer(torch.nn.Module):
    def __init__(self, layers, need_gradients=True):
        super(SplitLayer, self).__init__()
        if not isinstance(need_gradients, list):
            need_gradients = [need_gradients] * len(layers)
        self.need_gradients = need_gradients
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        output = []
        for layer, need_gradient in zip(self.layers, self.need_gradients):
            if need_gradient:
                output.append(layer(x))
            else:
                with torch.no_grad():
                    output.append(layer(x))
        return output

class ConcatLayer(torch.nn.Module):
    def __init__(self, dim=-1):
        super(ConcatLayer, self).__init__()
        self.dim = dim

    def forward(self, xs):
        return torch.cat(xs, dim=self.dim)

