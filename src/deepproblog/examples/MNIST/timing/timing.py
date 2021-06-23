import time
from statistics import mean, stdev

import torch

from deepproblog.engines import ExactEngine
from deepproblog.examples.MNIST.data import addition, MNIST_train, MNIST_test
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network

for N in [1, 2]:

    train_set = addition(N, "train").subset(100).to_queries()

    network = MNIST_Net()

    net = Network(network, "mnist_net", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    model = Model("models/addition.pl", [net])
    model.set_engine(ExactEngine(model), cache=True)

    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)

    timing = []
    for q in train_set:
        start = time.time()
        r = model.solve([q])
        timing.append(time.time() - start)
    mean, stdev = mean(timing), stdev(timing)
    with open("timing_{}.txt".format(N), "w") as f:
        f.write("{}\t{}\n".format(mean, stdev))
        f.write("\n".join(str(x) for x in timing))
