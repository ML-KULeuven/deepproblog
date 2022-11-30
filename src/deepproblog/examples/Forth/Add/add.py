import torch

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Forth import EncodeModule
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

train = 2
test = 8

train_queries = QueryDataset("data/train{}_test{}_train.txt".format(train, test))
test_queries = QueryDataset("data/train{}_test{}_test.txt".format(train, test))
val = QueryDataset("data/train{}_test{}_dev.txt".format(train, test))


net1 = EncodeModule(30, 50, 10, "tanh")
network1 = Network(net1, "neural1")
network1.optimizer = torch.optim.Adam(net1.parameters(), lr=0.02)

net2 = EncodeModule(22, 10, 2, "tanh")
network2 = Network(net2, "neural2")
network2.optimizer = torch.optim.Adam(net2.parameters(), lr=0.02)

model = Model("choose.pl", [network1, network2])
test_model = Model(
    "choose.pl", [Network(net1, "neural1", k=1), Network(net2, "neural2", k=1)]
)
model.set_engine(ExactEngine(model), cache=True)
test_model.set_engine(ExactEngine(test_model), cache=False)
train_obj = train_model(
    model,
    DataLoader(train_queries, 50),
    40,
    log_iter=20,
    test=lambda x: [("Accuracy", get_confusion_matrix(test_model, val).accuracy())],
    test_iter=100,
)
