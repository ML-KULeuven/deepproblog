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
dev_queries = QueryDataset("data/train{}_test{}_dev.txt".format(train, test))
test_queries = QueryDataset("data/train{}_test{}_test.txt".format(train, test))

fc1 = EncodeModule(20, 20, 2)

model = Model(
    "compare.pl",
    [Network(fc1, "swap_net", optimizer=torch.optim.Adam(fc1.parameters(), 1.0))],
)
model.set_engine(ExactEngine(model), cache=True)

test_model = Model("compare.pl", [Network(fc1, "swap_net", k=1)])
test_model.set_engine(ExactEngine(test_model), cache=False)

train_obj = train_model(
    model,
    DataLoader(train_queries, 16),
    40,
    log_iter=50,
    test_iter=len(train_queries),
    test=lambda x: [
        ("Accuracy", get_confusion_matrix(test_model, dev_queries).accuracy())
    ],
)
