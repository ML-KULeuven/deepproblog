import random

import torch.nn as nn

from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term, Constant


class Successor(Dataset):
    def __getitem__(self, item):
        d1, d2 = self.parent[item * 2], self.parent[item * 2 + 1]
        return d1[0], d2[0], int(d2[1] - d1[1])

    def to_query(self, i, *args):
        neg = False
        if self.negatives:
            neg = bool(i % 3)
        i1, i2 = 2 * i, 2 * i + 1
        label = self[i][2]
        if neg:
            label = random.randint(-9, 9)
        return Query(
            Term("successor", Term("a"), Term("b"), Constant(label)),
            {
                Term("a"): Term(self.name, Constant(i1)),
                Term("b"): Term(self.name, Constant(i2)),
            },
            p=0.0 if neg else 1.0,
        )

    def __len__(self):
        return len(self.parent) // 2

    def __init__(self, parent, name, negatives=False):
        self.parent = parent
        self.name = name
        self.negatives = negatives


class SuccessorNet(nn.Module):
    def __init__(self, N, store):
        super(SuccessorNet, self).__init__()
        self.store = store
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # s6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        self.final = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, N),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.final(x)
        return x
