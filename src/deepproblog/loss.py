import math

import torch
from torch import nn as nn


class JSD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-8):
        logN = math.log(float(x.shape[0]))

        y = torch.mean(x, 0)
        y = y * (y + eps).log() / logN
        y = y.sum()

        x = x * (x + eps).log() / logN
        x = x.sum(1).mean()
        return 1.0 - x + y


class InfoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-8):
        x = torch.mean(x, 0)
        logN = math.log(float(x.shape[0]))
        x = x * (x + eps).log() / logN
        neg_entropy = x.sum()
        return 1.0 + neg_entropy


class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-8):
        logN = math.log(float(x.shape[0]))
        x = x * (x + eps).log() / logN
        neg_entropy = x.sum(1)
        return -neg_entropy.mean()


infoLoss = InfoLoss()
entropyLoss = EntropyLoss()
jsd = JSD()


def MSE(p, target):
    return (p - target) ** 2, 2 * (p - target)
