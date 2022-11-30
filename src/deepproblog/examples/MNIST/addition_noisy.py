from random import randint

import torch
from problog.logic import Constant

from deepproblog.dataset import DataLoader
from deepproblog.dataset import NoiseMutatorDecorator, MutatingDataset
from deepproblog.engines import ExactEngine
from deepproblog.examples.MNIST.data import MNISTOperator, MNIST_train, MNIST_test
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.optimizer import SGD
from deepproblog.query import Query
from deepproblog.train import train_model


def noise(_, query: Query):
    new_query = query.replace_output([Constant(randint(0, 18))])
    return new_query


dataset = MNISTOperator(
    dataset_name="train",
    function_name="addition_noisy",
    operator=sum,
    size=1,
)
noisy_dataset = MutatingDataset(dataset, NoiseMutatorDecorator(0.2, noise))
queries = DataLoader(noisy_dataset, 2)

network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
model = Model("models/noisy_addition.pl", [net])

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

model.set_engine(ExactEngine(model))
model.optimizer = SGD(model, 1e-3)

train = train_model(model, queries, 1, log_iter=100)
