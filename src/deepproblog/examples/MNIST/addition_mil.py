import sys
from json import dumps
from random import choice

import torch
from deepproblog.dataset import Dataset, DataLoader
from deepproblog.engines import ApproximateEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import (
    addition,
    datasets,
    MNIST_train,
    MNIST_test,
    MNIST,
)
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.examples.MNIST.neural_baseline.baseline_models import Separate_Baseline
from deepproblog.heuristics import NeuralHeuristic
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.query import Query
from deepproblog.utils import get_configuration, format_time_precise, config_to_string
from problog.logic import list2term, Term


class AnySum(Dataset):
    def __init__(self, parent_dset, bag_size):
        self.parent_dset = parent_dset
        self.bag_size = bag_size

    def __len__(self):
        return len(self.parent_dset) // self.bag_size

    def to_query(self, i):
        i *= self.bag_size
        all_images = []
        results = []
        for j in range(self.bag_size):
            query = self.parent_dset.to_query(i + j).substitute()
            images = [query.query.args[0], query.query.args[1]]
            all_images.append(list2term(images))
            results.append(query.query.args[2])
        result = choice(results)
        correct_bags = [
            all_images[i] for i in range(len(all_images)) if results[i] == result
        ]
        query = Query(Term("anysum", list2term(all_images), result))
        query.correct_bags = correct_bags
        return query


def heuristic_function(model, image1, image2, result):
    i1, l1 = datasets[image1.args[0].functor][int(image1.args[0].args[0])]
    i2, l2 = datasets[image2.args[0].functor][int(image2.args[0].args[0])]

    label = int(result)
    p = model(i1, i2)[label]
    return p


mnist_addition_test = addition(1, "test")


def test_heuristic(net):
    import numpy as np

    dset = mnist_addition_test
    confusion = np.zeros(
        (19, 19), dtype=np.uint32
    )  # First index actual, second index predicted
    correct = 0
    n = 0
    N = 100
    for i in range(N):
        i1, i2, l = dset[i]
        i1 = i1[0]
        i2 = i2[0]
        outputs = net.forward(i1, i2)
        _, out = torch.max(outputs.data, 0)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    return acc


predicate_val = MNIST("train").subset(100)
network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
heuristic_net = Separate_Baseline()
heuristic_net = Network(
    heuristic_net,
    "heuristic",
    optimizer=torch.optim.Adam(heuristic_net.parameters(), lr=1e-3),
)
model = Model("models/any_sum.pl", [net, heuristic_net])
neural = NeuralHeuristic(
    {"addition": lambda *x: heuristic_function(heuristic_net.network_module, *x)}, model
)

i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
configurations = {
    "bag_size": [16, 32],
    "heuristic": [neural],
    "pretrain": [500],
    "exploration": [True, False],
    "run": range(5),
}
configuration = get_configuration(configurations, i)
name = "anysum_" + config_to_string(configuration) + "_" + format_time_precise()
print(name)

if configuration["pretrain"] > 0:
    heuristic_net.network_module.load_state_dict(
        torch.load(
            "models/pretrained/addition_{}.pth".format(configuration["pretrain"])
        )
    )

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)
engine = ApproximateEngine(
    model, 1, configuration["heuristic"], exploration=configuration["exploration"]
)
model.set_engine(engine)

dataset = AnySum(addition(1, "train"), configuration["bag_size"])

loader = DataLoader(dataset, 2, True)

train = train_model(
    model,
    loader,
    2,
    log_iter=100,
    test_iter=500,
    test=lambda x: [
        ("Predicate", get_confusion_matrix(x, predicate_val, verbose=0).accuracy()),
        ("Heuristic", test_heuristic(heuristic_net.network_module)),
    ],
    profile=0,
)

model.save_state("snapshot/" + name + ".pth")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.write_to_file("log/" + name)
