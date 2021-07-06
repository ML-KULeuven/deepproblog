from typing import TYPE_CHECKING, Sequence

from deepproblog.query import Query
from deepproblog.tensor import TensorStore
from problog.formula import LogicFormula
from problog.logic import Term
from problog.program import LogicProgram

if TYPE_CHECKING:
    from deepproblog.model import Model


class Engine(object):
    """
    An asbtract engine base class.
    """

    def __init__(self, model: "Model"):
        """

        :param model: The model that this engine will solve queries for.
        """
        self.model = model
        self.tensor_store = TensorStore()

    def perform_count(self, queries: Sequence[Query], acs):
        pass

    def prepare(self, program: LogicProgram) -> LogicProgram:
        """
        Modifies the given program to a format suited for querying in this engine.
        :param program: The program to be modified
        :return: The modified program
        """
        raise NotImplementedError("prepare is an abstract method")

    def ground(self, query: Query, **kwargs) -> LogicFormula:
        """

        :param query: The query to ground.
        :param kwargs:
        :return: A logic formula representing the ground program.
        """
        raise NotImplementedError("ground is an abstract method")

    def register_foreign(
        self, func: callable, function_name: str, arity_in: int, arity_out: int
    ):
        """
        Makes a Python function available to the grounding engine.
        :param func: The Python function to be made available.
        :param function_name: The name of the predicate that will be used to address this function in logic.
        :param arity_in: The number of input arguments to func
        :param arity_out The number of return values of func
        :return:
        """
        raise NotImplementedError("register_foreign is an abstract method")

    def get_tensor(self, tensor_term: Term):
        if tensor_term.functor == "tensor":
            return self.tensor_store[tensor_term.args[0]]
        return tensor_term

    def get_hyperparameters(self) -> dict:
        raise NotImplementedError("get_hyperparameters is an abstract method")

    def eval(self):
        """
        Set the engine to eval mode.
        :return:
        """
        pass

    def train(self):
        """
        Set th engine to train mode.
        :return:
        """
        pass

    @staticmethod
    def can_cache() -> bool:
        return False
