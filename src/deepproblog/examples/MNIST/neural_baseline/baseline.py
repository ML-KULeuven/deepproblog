from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

from deepproblog.examples.MNIST.data import addition
from deepproblog.examples.MNIST.neural_baseline.baseline_models import Separate_Baseline
from deepproblog.utils.logger import Logger
from deepproblog.utils.stop_condition import StopOnPlateau


def test_addition(dset):
    confusion = np.zeros(
        (19, 19), dtype=np.uint32
    )  # First index actual, second index predicted
    correct = 0
    n = 0
    for i1, i2, l in dset:
        i1 = i1[0]
        i2 = i2[0]
        i1 = Variable(i1.unsqueeze(0))
        i2 = Variable(i2.unsqueeze(0))
        outputs = net.forward(i1, i2)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    print("Accuracy: ", acc)
    return acc


test_dataset = addition(1, "test")

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    Train = namedtuple("Train", ["logger"])
    model, modelname = Separate_Baseline, "Separate"

    # for N in [50, 100, 200, 500, 1000]:
    for N in [500]:
        train_dataset = addition(1, "train").subset(N)
        val_dataset = addition(1, "train").subset(N, N + 100)
        for batch_size in [4]:
            test_period = N // batch_size
            log_period = N // (batch_size * 10)
            trainloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            running_loss = 0.0
            log = Logger()
            i = 1
            net = model(batched=True, probabilities=False)
            optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-2)
            criterion = nn.CrossEntropyLoss()
            stop_condition = StopOnPlateau("Accuracy", patience=5)
            train_obj = Train(log)
            j = 1
            while not stop_condition.is_stop(train_obj):
                print("Epoch {}".format(j))
                for i1, i2, l in trainloader:
                    i1 = i1[0]
                    i2 = i2[0]
                    i1, i2, l = Variable(i1), Variable(i2), Variable(l)
                    optimizer.zero_grad()

                    outputs = net(i1, i2)
                    loss = criterion(outputs, l)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss)
                    if i % log_period == 0:
                        print(
                            "Iteration: ",
                            i,
                            "\tAverage Loss: ",
                            running_loss / log_period,
                        )
                        log.log("loss", i, running_loss / log_period)
                        running_loss = 0
                    if i % test_period == 0:
                        log.log("Accuracy", i, test_addition(val_dataset))
                    i += 1
                j += 1
            torch.save(
                net.state_dict(), "../models/pretrained/addition_{}.pth".format(N)
            )
            log.comment("Accuracy\t{}".format(test_addition(test_dataset)))
