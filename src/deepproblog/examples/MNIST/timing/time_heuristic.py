from time import time

import torch

from deepproblog.arithmetic_circuit import ArithmeticCircuit
from deepproblog.engines.approximate_engine import ApproximateEngine
from deepproblog.engines.exact_engine import ExactEngine
from deepproblog.examples.MNIST.data import (
    addition,
    MNIST_train,
    MNIST_test,
)
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.heuristics import GeometricMean
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.utils import Table


def analyse(model: Model, query):
    start = time()
    ground = model.solver.engine.ground(query)
    ground_time = time() - start

    start = time()
    ac = ArithmeticCircuit(ground, None)
    compile_time = time() - start
    start = time()
    eval_time = time() - start
    return ground_time, compile_time, eval_time


table = Table("length", "method", "ground time", "compile time", "eval time")
methods = [
    (ExactEngine, dict(), [1, 2, 3, 4]),
    (ApproximateEngine, {"k": 1, "heuristic": GeometricMean()}, [1, 2, 3, 4, 5, 6]),
    (ApproximateEngine, {"k": 2, "heuristic": GeometricMean()}, [1, 2, 3, 4, 5, 6]),
    (ApproximateEngine, {"k": 3, "heuristic": GeometricMean()}, [1, 2, 3, 4, 5, 6]),
]
for engine, kwargs, lengths in methods:
    for length in lengths:
        queries = addition(length, "train").to_queries()
        test = addition(1, "test").subset(10)

        network = MNIST_Net()
        network.load_state_dict(torch.load("models/pretrained/all_8.pth"))
        net = Network(network, "mnist_net", batching=True)
        net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        model = Model("models/addition.pl", [net])
        model.set_engine(engine(model, **kwargs))
        model.add_tensor_source("train", MNIST_train)
        model.add_tensor_source("test", MNIST_test)

        if "k" in kwargs:
            name = "{}, k: {}".format(engine.__name__, kwargs["k"])
        else:
            name = engine.__name__
        for q in queries[:10]:
            table.store(length, name, *analyse(model, q))

print(table.format("length", "method", "ground time"))
print(table.format("length", "method", "compile time"))
print(table.format("length", "method", "eval time"))
