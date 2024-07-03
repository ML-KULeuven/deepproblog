from typing import TYPE_CHECKING

import torch
from problog.logic import Term, term2list

from deepproblog.engines import Engine

if TYPE_CHECKING:
    pass


# def embed(engine: Engine, term: Term):
#     embedding = engine.model.get_embedding(term)[0, :]
#     return Term("tensor", Constant(engine.tensor_store.store(embedding)))
#
#
# def to_tensor(model: "Model", a):
#     if type(a) is Term:
#         if is_list(a):
#             a = term2list(a)
#         else:
#             return model.get_tensor(a)
#     # elif type(a) is Functor:
#     #     return engine.tensor_store[int(a.args[0])]
#     if type(a) is list:
#         out = [to_tensor(model, x) for x in a]
#         return [x for x in out if x is not None]
#     else:
#         return float(a)
#
#
# def tensor_wrapper(engine: Engine, func: Callable, *args):
#     model = engine.model
#     inputs = [to_tensor(model, a) for a in args]
#     out = func(*inputs)
#     return model.store_tensor(out)


def rbf(x, y):
    if isinstance(x, Term):
        x = torch.tensor(float(x))
    if isinstance(y, Term):
        y = torch.tensor(float(y))
    return torch.exp(-torch.norm(x.flatten() - y.flatten(), 2))


def sample_normal(size, _):
    size = int(size)
    mean = torch.zeros(size)
    std = torch.ones(size)
    return torch.normal(mean, std)


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


#
# def register_tensor_predicates(engine: Engine):
#     engine.register_foreign(lambda *x: embed(engine, *x), "embed", 1, 1)
#     engine.register_foreign(lambda *x: tensor_wrapper(engine, rbf, *x), "rbf", 2, 1)
#     engine.register_foreign(lambda *x: tensor_wrapper(engine, add, *x), "add", 2, 1)
#     engine.register_foreign(lambda *x: tensor_wrapper(engine, mul, *x), "mul", 2, 1)
#     engine.register_foreign(lambda *x: tensor_wrapper(engine, dot, *x), "dot", 2, 1)
#     engine.register_foreign(lambda *x: tensor_wrapper(engine, max, *x), "max", 1, 1)
#     engine.register_foreign(
#         lambda *x: tensor_wrapper(engine, sigmoid, *x), "sigmoid", 1, 1
#     )
#     engine.register_foreign(lambda *x: tensor_wrapper(engine, mean, *x), "mean", 1, 1)
#     engine.register_foreign(lambda *x: tensor_wrapper(engine, stack, *x), "stack", 1, 1)
#     engine.register_foreign(lambda *x: tensor_wrapper(engine, cat, *x), "cat", 1, 1)
#     engine.register_foreign(
#         lambda *x: tensor_wrapper(engine, one_hot, *x), "one_hot", 2, 1
#     )


def get_tensor_wrapper(name):
    def tensor_wrapper(*args):
        return Term('tensor', Term(name, *args))

    return tensor_wrapper


def list_to_tensor(list_term):
    return torch.tensor(term2list(list_term))


def tensor_index(tensor, index):
    index = [int(x) for x in term2list(index)]
    return tensor[index]

# TODO: Added non_det support for register_foreign
# def less_than(tensor1, tensor2):
#     print(tensor1, tensor2)
#     if tensor1 < tensor2:
#         return [()]
#     return []
#

all_builtins = {"rbf": (rbf, 2, 1),
                "sample_normal": (sample_normal, 2, 1),
                "list_to_tensor": (list_to_tensor, 1, 1),
                # "less_than": (less_than, 2, 0),
                "tensor_index": (tensor_index, 2, 1), }


def register_tensor_predicates(engine: Engine):
    for builtin in all_builtins:
        engine.register_foreign(get_tensor_wrapper(builtin), builtin, all_builtins[builtin][1],
                                all_builtins[builtin][2])
