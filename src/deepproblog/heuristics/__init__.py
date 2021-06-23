from math import isnan

import torch


class Heuristic(object):
    def __init__(self, name):
        self.name = name

    def get_hyperparameters(self):
        return {"type": type(self).__name__}

    def __repr__(self):
        return self.name


class PartialProbability(Heuristic):
    def __init__(self):
        super().__init__("partial_probability")


ucs = PartialProbability()


class GeometricMean(Heuristic):
    def __init__(self):
        super().__init__("geometric_mean")


geometric_mean = GeometricMean()


class LearnedHeuristic(Heuristic):
    def __init__(self, name, predicates):
        super().__init__(name)
        self.predicates = predicates

    def count(self, batch, acs):
        raise NotImplementedError("Count method is not implemented.")

    def get_value(self, node):
        raise NotImplementedError

    def get_hyperparameters(self):
        return {"type": type(self).__name__, "predicates": self.predicates}


class NeuralHeuristic(LearnedHeuristic):
    def __init__(self, heuristic_functions, model):
        predicates = [x for x in heuristic_functions]
        name = "extern([{}])".format(",".join(p for p in predicates))
        super().__init__(name, predicates)
        self.heuristic_functions = heuristic_functions
        self.heuristic_optimizers = dict()
        self.cache = dict()
        self.model = model
        self.seen = set()
        self.t = 0
        self.freeze = False

    def count(self, batch, acs):
        if self.freeze:
            return None
        acs = list(zip(*acs))
        loss = 0
        N = 0
        for i, (ac, semiring) in enumerate(acs):
            names = ac.get_named()
            for name in names:
                if name in self.cache:
                    target = 1.0
                    value = self.cache[name]
                    if target == 1.0:
                        l = -torch.log(value)
                    else:
                        l = -(
                            target * torch.log(value)
                            + (1.0 - target) * torch.log(1.0 - value)
                        )
                    if isnan(l):
                        raise ValueError("nan loss")
                    loss += l
                    N += 1
        if N > 0:
            loss /= N
            loss.backward(retain_graph=True)
        self.cache.clear()
        self.t += 1

    def get_value(self, node):

        value = self.heuristic_functions[node.functor](*node.args)
        self.cache[node] = value
        return float(value)

    def arg_to_tensor(self, arg):
        if arg.functor == "tensor":
            tensor = self.model.get_tensor(arg).detach()
            return tensor
        return self.model.get_embedding(arg).detach()
