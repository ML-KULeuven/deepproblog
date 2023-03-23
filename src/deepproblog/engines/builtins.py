from typing import Callable, TYPE_CHECKING

import torch
from problog.logic import Term, Constant, is_list, term2list

from deepproblog.engines import Engine

if TYPE_CHECKING:
    from deepproblog.model import Model


def embed(engine: Engine, term: Term):
    embedding = engine.model.get_embedding(term)[0, :]
    return Term("tensor", Constant(engine.tensor_store.store(embedding)))


def to_tensor(model: "Model", a):
    if type(a) is Term:
        if is_list(a):
            a = term2list(a)
        else:
            return model.get_tensor(a)
    # elif type(a) is Functor:
    #     return engine.tensor_store[int(a.args[0])]
    if type(a) is list:
        out = [to_tensor(model, x) for x in a]
        return [x for x in out if x is not None]
    else:
        return float(a)


def tensor_wrapper(engine: Engine, func: Callable, *args):
    model = engine.model
    inputs = [to_tensor(model, a) for a in args]
    out = func(*inputs)
    return model.store_tensor(out)


def tensor_wrapper_nondet(engine: Engine, func: Callable, *args):
    model = engine.model
    inputs = [to_tensor(model, a) for a in args]
    out = func(*inputs)
    return out


def rbf(x, y):
    return torch.exp(-torch.norm(x - y, 2))


def add(x, y):
    return x + y


def mul(x, y):
    return x * y


def dot(x, y):
    return torch.dot(x, y)


def sigmoid(x):
    return torch.sigmoid(x)


def max(x):
    x = torch.stack(x, 0)
    x, _ = torch.max(x, 0)
    return x


def mean(x):
    x = torch.stack(x, 0)
    x = torch.mean(x, 0)
    return x


def one_hot(i, n):
    x = torch.zeros(int(n))
    x[int(i)] = 1.0
    return x


def cat(tensors):
    return torch.cat(tensors)


def stack(tensors):
    return torch.stack(tensors)


def list_to_tensor(list_term):
    return torch.tensor(list_term)


def tensor_index(tensor, index):
    index = [int(x) for x in index]
    return tensor[index]


def less_than(tensor1, tensor2):
    if tensor1 < tensor2:
        return [()]
    return []

def register_tensor_predicates(engine: Engine):
    engine.register_foreign(lambda *x: embed(engine, *x), "embed", 1, 1)
    engine.register_foreign(lambda *x: tensor_wrapper(engine, rbf, *x), "rbf", 2, 1)
    engine.register_foreign(lambda *x: tensor_wrapper(engine, add, *x), "add", 2, 1)
    engine.register_foreign(lambda *x: tensor_wrapper(engine, mul, *x), "mul", 2, 1)
    engine.register_foreign(lambda *x: tensor_wrapper(engine, dot, *x), "dot", 2, 1)
    engine.register_foreign(lambda *x: tensor_wrapper(engine, max, *x), "max", 1, 1)
    engine.register_foreign(
        lambda *x: tensor_wrapper(engine, sigmoid, *x), "sigmoid", 1, 1
    )
    engine.register_foreign(lambda *x: tensor_wrapper(engine, mean, *x), "mean", 1, 1)
    engine.register_foreign(lambda *x: tensor_wrapper(engine, stack, *x), "stack", 1, 1)
    engine.register_foreign(lambda *x: tensor_wrapper(engine, cat, *x), "cat", 1, 1)
    engine.register_foreign(
        lambda *x: tensor_wrapper(engine, one_hot, *x), "one_hot", 2, 1
    )
    engine.register_foreign(lambda *x: tensor_wrapper(engine, list_to_tensor, *x), "list_to_tensor", 1, 1)
    engine.register_foreign(lambda *x: tensor_wrapper(engine, tensor_index, *x), "tensor_index", 2, 1)
    engine.register_foreign_nondet(lambda *x: tensor_wrapper_nondet(engine, less_than, *x), "less_than", 2, 0)