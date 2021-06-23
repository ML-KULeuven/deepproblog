import torch
import torch.nn as nn

from deepproblog.utils.standard_networks import MLP


class EncodeModule(nn.Module):
    def __init__(self, in_size, mid_size, out_size, activation=None):
        super().__init__()
        if activation == "tanh":
            self.mlp = MLP(in_size, mid_size, out_size, activation=nn.Tanh)
        else:
            self.mlp = MLP(in_size, mid_size, out_size)
        self.in_size = in_size

    def __call__(self, *x):
        input = torch.zeros(self.in_size)
        for j, i in enumerate(x):
            i = int(i)
            input[j * 10 + i] = 1
        return self.mlp(input)
