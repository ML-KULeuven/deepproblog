import torch
from torch import nn as nn


class Separate_Baseline(nn.Module):
    def __init__(self, batched=False, probabilities=True):
        super(Separate_Baseline, self).__init__()
        self.batched = batched
        self.probabilities = probabilities
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 2, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19),
        )
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x, y):
        if not self.batched:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        x = self.encoder(x)
        y = self.encoder(y)
        x = x + y
        x = x.view(-1, 16 * 8 * 2)
        x = self.classifier(x)
        if self.probabilities:
            x = self.activation(x)
        if not self.batched:
            x = x.squeeze(0)

        return x


class Separate_Baseline_Multi(nn.Module):
    def __init__(self, n=4):
        super(Separate_Baseline_Multi, self).__init__()
        self.n = n
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16 * 4 * 4 * self.n // 2, 100),
        )
        self.classifier2 = nn.Sequential(
            nn.ReLU(), nn.Linear(100 * 2, 128), nn.ReLU(), nn.Linear(128, 199)
        )

    def forward(self, imgs1, imgs2):
        imgs1 = [self.encoder(x) for x in imgs1]
        imgs2 = [self.encoder(x) for x in imgs2]
        x1, x2 = torch.cat(imgs1, 2), torch.cat(imgs2, 2)
        x1, x2 = (
            x1.view(-1, 16 * 4 * 4 * self.n // 2),
            x2.view(-1, 16 * 4 * 4 * self.n // 2),
        )
        x1, x2 = self.classifier(x1), self.classifier(x2)
        x = torch.cat([x1, x2], 1)
        x = self.classifier2(x)
        return x
