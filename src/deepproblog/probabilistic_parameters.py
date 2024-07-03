import random
from typing import Iterable

import torch
import torch.nn.utils.parametrize as parametrize
from problog.logic import Constant, Clause, Term, Or, InstantiationError
from problog.program import LogicProgram, SimpleProgram


class NormalizeADs(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, parameters):
        new_parameters = parameters.clone()
        for i, j in self.groups:
            new_parameters[i:j] /= new_parameters[i:j].sum()
        return new_parameters


class ProbabilisticParameters(torch.nn.Module):
    def __init__(
        self, start_values: Iterable[float], groups: Iterable[tuple[int, int]]
    ):
        super().__init__()
        self.probabilistic_parameters = torch.nn.Parameter(torch.tensor(start_values))
        parametrize.register_parametrization(self, "probabilistic_parameters", NormalizeADs(groups))

    def forward(self, indices):
        return self.probabilistic_parameters[indices]

    def __len__(self):
        return self.probabilistic_parameters.numel()


def extract_probabilistic_parameters(
    program: LogicProgram,
) -> tuple[LogicProgram, list[float], list[tuple[int, int]]]:
    translated = SimpleProgram()
    parameters = []
    groups = []
    for n in program:
        if type(n) is Term:
            if (
                n.probability is not None
                and type(n.probability) is Term
                and n.probability.functor == "t"
            ):
                p = n.probability.with_args(Constant(len(parameters)))
                parameters.append(get_parameter_value(n.probability.args[0]))
                n = n.with_probability(p)
            translated.add_statement(n)
        elif type(n) is Clause:
            if (
                n.head.probability is not None
                and type(n.head.probability) is Term
                and n.head.probability.functor == "t"
            ):
                p = n.head.probability.with_args(Constant(len(parameters)))
                parameters.append(get_parameter_value(n.head.probability.args[0]))
                head = n.head.with_probability(p)
                n = Clause(head, n.body)
            translated.add_statement(n)
        elif type(n) is Or:
            new_list = []
            new_group = []
            n = n.to_list()
            for x in n:
                if (
                    x.probability is not None
                    and type(x.probability) is Term
                    and x.probability.functor == "t"
                ):
                    new_group.append(len(parameters))
                    p = x.probability.with_args(Constant(len(parameters)))
                    parameters.append(get_parameter_value(x.probability.args[0]))
                    new_list.append(x.with_probability(p))
                else:
                    new_list.append(x)
            if len(new_group) > 0:
                groups.append((new_group[0], new_group[-1]))
            n = Or.from_list(new_list)
            translated.add_statement(n)
        else:
            translated.add_statement(n)

    return translated, parameters, groups


def get_parameter_value(val: Constant):
    try:
        return float(val)
    except InstantiationError:
        return random.random()
