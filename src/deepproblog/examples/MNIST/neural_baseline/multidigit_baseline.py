from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from deepproblog.examples.MNIST.data import addition
from deepproblog.examples.MNIST.neural_baseline.baseline_models import (
    Separate_Baseline_Multi,
)
from deepproblog.utils.logger import Logger
from deepproblog.utils.stop_condition import StopOnPlateau


def test_addition(dset):
    confusion = np.zeros(
        (199, 199), dtype=np.uint32
    )  # First index actual, second index predicted
    correct = 0
    n = 0
    for d1, d2, l in dset:
        d1 = [x.unsqueeze(0) for x in d1]
        d2 = [x.unsqueeze(0) for x in d2]
        outputs = net.forward(d1, d2)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    print("Accuracy: ", acc)
    return acc


test_dataset = addition(2, "test")

Train = namedtuple("Train", ["logger"])
modelname = "Separate_multi"

for N in [14900]:
    # for N in [150, 1500, 14000]:
    batch_size = 2
    test_period = 5000
    log_period = N // (batch_size * 10)
    train_dataset = addition(2, "train").subset(N)
    val_dataset = addition(2, "train").subset(N, N + 100)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    running_loss = 0.0
    log = Logger()
    i = 1
    net = Separate_Baseline_Multi()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    stop_condition = StopOnPlateau("Accuracy", patience=5)
    train_obj = Train(log)
    j = 1
    while not stop_condition.is_stop(train_obj):
        print("Epoch {}".format(j))
        for d1, d2, label in trainloader:
            optimizer.zero_grad()

            outputs = net(d1, d2)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += float(loss)
            if i % log_period == 0:
                print("Iteration: ", i, "\tAverage Loss: ", running_loss / log_period)
                log.log("loss", i, running_loss / log_period)
                running_loss = 0
            if i % test_period == 0:
                log.log("Accuracy", i, test_addition(val_dataset))
            i += 1
        j += 1
    log.comment("Accuracy\t{}".format(test_addition(test_dataset)))
    log.write_to_file("log/{}_{}".format(modelname, N))
