import math

import numpy as np

from problog.evaluator import Semiring as ProbLogSemiring
from problog.logic import Constant, Term
from . import Semiring


def ce(p, target):
    return -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p)), (
        -target / p + (1.0 - target) / (1.0 - p)
    )


class GradientSemiring(Semiring, ProbLogSemiring):
    class Result(object):
        def __init__(
            self,
            result,
            ground_time=None,
            compile_time=None,
            eval_time=None,
            proof=None,
        ):
            self.result = result
            self.ground_time = ground_time
            self.compile_time = compile_time
            self.eval_time = eval_time
            self.proof = proof

        def __iter__(self):
            return iter(self.result.keys())

    def __init__(self, model, substitution, values, tensor_store):
        Semiring.__init__(self, model, substitution, values)
        self.eps = 1e-8
        self.tensor_store = tensor_store
        self.dtype = np.float32

    def zero(self):
        # return 0.0, np.zeros(self.length, dtype=self.dtype)
        return 0.0, IndexedVector()

    def one(self):
        return 1.0, IndexedVector()

    def plus(self, a, b):
        return a[0] + b[0], a[1] + b[1]

    def times(self, a, b):
        return a[0] * b[0], a[1] * b[0] + b[1] * a[0]

    def value(self, a, key=None):
        if type(a) is Constant:
            return float(a), IndexedVector()
        elif type(a) is float:
            return a, IndexedVector()
        elif type(a) is Term:
            if a.functor == "nn":
                net, inputs, output = a.args
                inputs = inputs.apply_term(self.substitution)
                val = self.values[(net, inputs)]
                domain = self.model.networks[str(net)].domain
                # i = domain.index(output)
                i = int(output)
                diff = np.zeros(len(domain), dtype=self.dtype)
                diff[i] = 1.0
                gradient = IndexedVector()
                gradient[(net, inputs)] = diff
                if len(val) > 1:
                    print(val)
                return float(val[0]["p"][i]), gradient
            elif a.functor == "tensor":
                i = int(a.args[0])
                tensor = self.tensor_store[i]
                if tensor.numel() > 1:
                    raise ValueError(
                        "A tensor used as probability should only have one element"
                    )
                diff = np.ones(1, dtype=self.dtype)
                gradient = IndexedVector()
                gradient[("tensor", i)] = diff
                return float(tensor), gradient
            elif a.functor == "t":
                i = int(a.args[0])
                p = self.model.parameters[i]
                diff = np.ones(1, dtype=self.dtype)
                gradient = IndexedVector()
                gradient[i] = diff
                return p, gradient
            else:
                raise Exception("unhandled term {}".format(a.functor))
        else:
            return float(a.compute_value()), IndexedVector()

    def negate(self, a):
        return 1.0 - a[0], a[1] * -1.0

    def is_dsp(self):
        return True

    def is_one(self, a):
        return (1.0 - self.eps < float(a[0]) < 1.0 + self.eps) and a[1].is_zero()

    def is_zero(self, a):
        return (-self.eps < a[0] < self.eps) and a[1].is_zero()

    # def normalize(self, a, z):
    #     print(a,z)

    # def normalize(self, a, z):
    #     if self.is_one(z):
    #         return a
    #     diff = np.zeros(self.shape.length)
    #     for i in range(self.shape.length):
    #         diff[i] = (a[1][i]*z[0]-z[1][i]*a[0])/(z[0]**2)
    #     return a[0]/z[0], diff


class IndexedVector(object):
    def __init__(self):
        self.d = dict()

    def __getitem__(self, item):
        return self.d[item]

    def __setitem__(self, key, value):
        self.d[key] = value

    def __add__(self, other):
        result = IndexedVector()
        for k in self:
            try:
                result[k] = self[k] + other[k]
            except KeyError:
                result[k] = self[k]
        for k in other:
            if k not in self:
                result[k] = other[k]
        return result

    def __mul__(self, other):
        result = IndexedVector()
        for k in self:
            result[k] = self[k] * float(other)
        return result

    def __iter__(self):
        return iter(self.d)

    def is_zero(self, eps=1e-6):
        for k in self:
            for v in self[k]:
                if v > eps or v < -eps:
                    return False
        return True

    def __repr__(self):
        return "[" + ", ".join("{}: {}".format(k, self[k]) for k in self) + "]"
