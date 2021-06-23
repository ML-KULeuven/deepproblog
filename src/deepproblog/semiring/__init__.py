from abc import ABC, abstractmethod
from typing import Optional, Dict, Union

import torch

from deepproblog.query import Query
from problog.evaluator import Semiring as ProbLogSemiring
from problog.formula import LogicFormula
from problog.logic import Term


class Semiring(ProbLogSemiring, ABC):
    """
    The semiring object defines the operations for the evaluation of arithmetic circuits.
    """

    def __init__(self, model, substitution, values):
        """
        :param model: The model in which the evaluation happens.
        :param substitution: The substitution to apply to the arithmetic circuit before evaluation.
        :param values: The output values of the neural network to use in the evaluation.
        """
        self.model = model
        self.eps = 1e-5
        self.values = values
        self.substitution = substitution

    @staticmethod
    @abstractmethod
    def cross_entropy(
        result: "Result",
        target: float,
        weight: float,
        q: Optional[Query] = None,
        eps: float = 1e-6,
    ):
        """
        Calculates the cross_entropy between the predicted and target probabilities.
        Also performs the backwards pass for the given result.
        :param result: The result to calculate loss on.
        :param target: The target probability.
        :param weight: The weight of this examplE. A float that is multiplied with the loss before backpropagation.
        :param q: If there's more than one query in result, calculate the loss for this query.
        :param eps: The epsilon used in the cross-entropy loss calculation.
        :return:
        """
        pass


class Result(object):
    """
    A class that contains the result and timing info for evaluating a query.
    """

    def __init__(
        self,
        result: Dict[Term, Union[float, torch.Tensor]],
        semiring: Semiring,
        ground_time: Optional[float] = None,
        compile_time: Optional[float] = None,
        eval_time: Optional[float] = None,
        proof: Optional[LogicFormula] = None,
    ):
        """Construct object

        :param result: Dictionary of results, the key is the term and the value is the probability.
        :param semiring: Semiring object in use
        :param ground_time:
        :param compile_time:
        :param eval_time:
        :param proof:

        Note! The term indexing the result object may not be the same as in your query. There are
        a few reasons for this:
        * Your query had substitutions, the term is going to be the substituted variant.
        * You have a non-ground query, your query could be partially ground (giving you multiple answers).
        """
        self.result = result
        self.semiring = semiring
        self.ground_time = ground_time
        self.compile_time = compile_time
        self.eval_time = eval_time
        self.proof = proof

    def __iter__(self):
        return iter(self.result.keys())

    def __len__(self):
        return len(self.result)

    def __repr__(self):
        return repr(self.result)
