import math
from typing import Optional

import torch

from deepproblog.optimizer import Optimizer
from deepproblog.semiring import Semiring, Result
from problog.logic import Constant, Term


def get_hook(optimizer: Optimizer, i):
    def hook(grad):
        optimizer.add_parameter_gradient(i, grad)

    return hook


class GraphSemiring(Semiring):
    def __init__(self, model, substitution, values):
        Semiring.__init__(self, model, substitution, values)

    def negate(self, a):
        return 1.0 - a

    def one(self):
        return 1.0

    def zero(self):
        return 0.0

    def plus(self, a, b):
        if self.is_zero(b):
            return a
        if self.is_zero(a):
            return b
        return a + b

    def times(self, a, b):
        if self.is_one(b):
            return a
        if self.is_one(a):
            return b
        return a * b

    def value(self, a, key=None):
        if type(a) is Constant:
            return float(a)
        elif type(a) is float:
            return a
        elif type(a) is Term:
            if a.functor == "nn":
                net, inputs = a.args[0], a.args[1]
                inputs = inputs.apply_term(self.substitution)
                val = self.values[(net, inputs)]
                i = 0
                if len(a.args) == 3:
                    i = int(a.args[2])
                return val[i]
            elif a.functor == "t":
                i = int(a.args[0])
                p = torch.tensor(self.model.parameters[i], requires_grad=True)
                p.register_hook(get_hook(self.model.optimizer, i))
                return p
            elif a.functor == "tensor":
                return self.model.get_tensor(a)
            elif a.functor == "'/'":  # Deals with probabilities formatted as franctions
                return float(a)
            else:
                raise Exception("unhandled term {}".format(a.functor))
        else:
            return float(a.compute_value())

    def is_one(self, a):
        return 1.0 - self.eps <= float(a) <= 1.0 + self.eps

    def is_zero(self, a):
        return -self.eps <= float(a) <= self.eps

    def is_dsp(self):
        return True

    def normalize(self, a, z):
        if self.is_one(z):
            return a
        return a / z

    @staticmethod
    def cross_entropy(
        result: Result,
        target: float,
        weight: float,
        q: Optional[Term] = None,
        eps: float = 1e-12,
    ) -> float:

        result = result.result
        if len(result) == 0:
            print("No results found for {}".format(q))
            return 0
        if q is None:
            if len(result) == 1:
                q, p = next(iter(result.items()))
            else:
                raise ValueError(
                    "q is None and number of results is {}".format(len(result))
                )
        else:
            p = result[q]
        if type(p) is float:
            loss = (
                -(target * math.log(p + eps) + (1.0 - target) * math.log(1.0 - p + eps))
                * weight
            )
        else:
            if target == 1.0:
                loss = -torch.log(p + eps) * weight
            elif target == 0.0:
                loss = -torch.log(1.0 - p + eps) * weight
            else:
                loss = (
                    -(
                        target * torch.log(p + eps)
                        + (1.0 - target) * torch.log(1.0 - p + eps)
                    )
                    * weight
                )
            loss.backward(retain_graph=True)
        return float(loss)

    @staticmethod
    def mse(
        result: Result, target: float, weight: float, q: Optional[Term] = None
    ) -> float:

        result = result.result
        if len(result) == 0:
            print("No results found for {}".format(q))
            return 0
        if q is None:
            if len(result) == 1:
                q, p = next(iter(result.items()))
            else:
                raise ValueError(
                    "q is None and number of results is {}".format(len(result))
                )
        else:
            p = result[q]
        loss = (p - target) ** 2 * weight
        if type(p) is not float:
            loss.backward(retain_graph=True)
        return float(loss)
