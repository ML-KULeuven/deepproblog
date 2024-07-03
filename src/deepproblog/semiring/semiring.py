from abc import ABC, abstractmethod
from typing import Optional

from problog.evaluator import Semiring as ProbLogSemiring

from deepproblog.query import Query
from deepproblog.semiring.result import Result


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
        self.eps = 1e-15
        self.values = values
        self.substitution = substitution

    # @staticmethod
    # @abstractmethod
    # def cross_entropy(
    #     result: "Result",
    #     target: float,
    #     weight: float,
    #     q: Optional[Query] = None,
    #     eps: float = 1e-6,
    # ):
    #     """
    #     Calculates the cross_entropy between the predicted and target probabilities.
    #     Also performs the backwards pass for the given result.
    #     :param result: The result to calculate loss on.
    #     :param target: The target probability.
    #     :param weight: The weight of this examplE. A float that is multiplied with the loss before backpropagation.
    #     :param q: If there's more than one query in result, calculate the loss for this query.
    #     :param eps: The epsilon used in the cross-entropy loss calculation.
    #     :return:
    #     """
    #     pass
