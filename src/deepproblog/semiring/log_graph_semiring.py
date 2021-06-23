import math
from math import log, exp

import torch

from deepproblog.arithmetic_circuit import ArithmeticCircuit
from problog.logic import Constant, Term
from . import Semiring


class LogGraphSemiring(Semiring):
    eps = 1e-8
    eps2 = -1e8

    class Result(object):
        def __len__(self):
            return len(self.result)

        def __init__(self, result):
            self.log_result = result
            self.result = dict()
            for k in result:
                if type(result[k]) is float:
                    self.result[k] = exp(result[k])
                else:
                    self.result[k] = torch.exp(result[k])

        def cross_entropy(self, target, weight, q=None, retain_graph=False):
            if len(self.result) == 0:
                print("No results found for {}".format(q))
                return 0
            if q is None:
                if len(self.result) == 1:
                    q, p = next(iter(self.result.items()))
                    logp = self.log_result[q]
                else:
                    raise ValueError(
                        "q is None and numer of results is {}".format(len(self.result))
                    )
            else:
                p = self.result[q.substitute().query]
                logp = self.log_result[q.substitute().query]
            if type(logp) is float:
                loss = (
                    -(target * logp + (1.0 - target) * math.log1p(-p - 1e-12)) * weight
                )
            else:
                if target == 1.0:
                    loss = -logp * weight
                else:
                    loss = (
                        -(target * logp + (1.0 - target) * torch.log1p(-p - 1e-12))
                        * weight
                    )
                # bce_loss = torch.nn.BCELoss()
                # loss = bce_loss(p, torch.FloatTensor([target]))
                loss.backward(retain_graph=retain_graph)

            return float(loss)

        def __iter__(self):
            return iter(self.result.keys())

    @classmethod
    def get_ac(cls, ground):
        return ArithmeticCircuit(ground, cls)

    def __init__(self, model, substitution, values, tensor_store, optimizer=None):
        Semiring.__init__(self, model, substitution, values)
        self.tensor_store = tensor_store
        self.optimizer = optimizer

    def negate(self, a):
        if self.is_zero(a):
            return self.one()
        if self.is_one(a):
            return self.zero()
        res = torch.log1p(-torch.exp(a))
        if torch.isnan(res):
            raise Exception("nan")
        return res
        # return a + torch.log(torch.exp(-a)-1.0)

    def one(self):
        return 0.0

    def zero(self):
        return -1e10

    def plus(self, a, b):
        if self.is_zero(b):
            return a
        if self.is_zero(a):
            return b
        if a > b:
            return a + torch.log1p(torch.exp(b - a))
        else:
            return b + torch.log1p(torch.exp(a - b))

    def times(self, a, b):
        if self.is_one(b):
            return a
        if self.is_one(a):
            return b
        return a + b

    def value(self, a, key=None):
        if type(a) is Constant:
            return log(float(a))
        elif type(a) is float:
            return log(a)
        elif type(a) is Term:
            if a.functor == "nn":
                net, inputs, output = a.args
                inputs = inputs.apply_term(self.substitution)
                val = self.values[(net, inputs)]
                i = int(output)
                return torch.log(val[0]["p"][i])
            elif a.functor == "t":
                i = int(a.args[0])

                def hook(grad):
                    self.optimizer.add_param_grad(i, grad)

                p = torch.tensor(self.model.parameters[i], requires_grad=True)
                p.register_hook(hook)
                return torch.log(p)
            else:
                raise Exception("unhandled term {}".format(a.functor))
        else:
            return float(a.compute_value())

    def is_one(self, a):
        return float(a) > -LogGraphSemiring.eps

    def is_zero(self, a):
        return float(a) <= LogGraphSemiring.eps2

    def is_dsp(self):
        return True

    def normalize(self, a, z):
        if self.is_one(z):
            return a
        # print('normalizing with ', self.one()-float(z))
        return a - z
