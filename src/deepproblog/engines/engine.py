from time import time
from typing import TYPE_CHECKING, Iterable

from problog.formula import LogicFormula

from ..arithmetic_circuit import ArithmeticCircuit
from ..query import Query
from ..utils.cache import DictCache, FileCache

if TYPE_CHECKING:
    from deepproblog.model import Model


class Engine(object):
    """
    An abstract engine base class.
    """

    def __init__(
        self, model: "Model", cache_memory: bool = False, cache_root: str = None
    ):
        """

        :param model: The model that this engine will solve queries for.
        """
        self.model = model

        def key_func(x):
            return str(x.query)

        self.ac_builder = self.build_ac
        if cache_root is not None:
            self.ac_builder = FileCache(self.ac_builder, cache_root, key_func)
        if cache_memory:
            self.ac_builder = DictCache(self.ac_builder, key_func)

    def ground(self, query: Query) -> LogicFormula:
        """

        :param query: The query to ground.
        :return: A logic formula representing the ground program.
        """
        raise NotImplementedError("ground is an abstract method")

    def query(self, query: Query) -> ArithmeticCircuit:
        return self.ac_builder(query)

    def query_batch(self, batch: Iterable[Query]) -> Iterable[ArithmeticCircuit]:
        for query in batch:
            yield self.query(query)

    def build_ac(self, query: Query) -> ArithmeticCircuit:
        """
        Builds the arithmetic circuit.
        :param query: The query for which to build the arithmetic circuit.
        :return: The arithmetic circuit for the given query.
        """
        ground = self.ground(query)
        ac = ArithmeticCircuit(
            ground,
            collect=query.collect,
            name=query.query,
        )
        return ac

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

    def get_hyperparameters(self) -> dict:
        return {"type": self.__class__.__name__}
