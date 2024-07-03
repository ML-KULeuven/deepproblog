from collections import defaultdict
from typing import Optional, List, Tuple, Dict, TYPE_CHECKING

from problog.formula import LogicDAG, LogicFormula
from problog.logic import Term, term2list
from problog.sdd_formula import SDD

from .semiring import GraphSemiring
from .semiring.result import Result

if TYPE_CHECKING:
    from .model import Model


class ArithmeticCircuit(object):
    def __init__(
        self,
        formula: LogicFormula,
        collect=False,
        name=None,
    ):
        """
        :param formula: The ground logic formula that will be compiled.
        :param ground_time: Optional. Keep track of time it took to ground out formula. Used for timing statistics.
        :param sdd_auto_gc: Controls if the PySDD auto-GC feature should be turned on (may be needed for large problems)
        """
        self.proof = LogicDAG.create_from(formula, keep_named=True)
        if collect:
            query_nodes = list(q[1] for q in self.proof.queries())
            key = self.proof.add_or(query_nodes)
            self.proof.clear_queries()
            self.proof.add_query(name, key)
        self.sdd = SDD.create_from(self.proof, sdd_auto_gc=False)

    def __setstate__(self, state):
        self.__dict__ = state

    def evaluate(
        self,
        model: "Model",
        substitution: Optional[dict] = None,
    ) -> Result:
        """
        Evaluates the arithmetic circuit.
        :param substitution: Optional dict. The substitution applied to the parameterized AC. See apply_term.
        :return:
        """
        substitution = dict() if substitution is None else substitution
        neural_probabilities = self.extract_neural_probabilities(substitution)
        values = model.evaluate_nn(neural_probabilities)
        semiring = GraphSemiring(model, substitution, values)
        evaluation = self.sdd.evaluate(semiring=semiring)
        evaluation = {k.apply_term(substitution): evaluation[k] for k in evaluation}
        return Result(
            evaluation,
            semiring,
            self.proof,
        )

    def extract_neural_probabilities(self, substitution: dict) -> dict[str, set[tuple[Term, ...]]]:
        """
        :return: Returns a set of all ground neural predicates that need to be evaluated.
        """
        neural_probabilities = defaultdict(set)
        weights = self.sdd.get_weights()
        for w in weights:
            w = weights[w]
            if isinstance(w, Term):
                if w.functor == "nn":
                    net_name = w.args[0].functor
                    arguments = term2list(w.args[1].apply_term(substitution))
                    neural_probabilities[net_name].add(tuple(arguments))
        return neural_probabilities

    # def extract_neural(self) -> List[Tuple[Term, Term]]:
    #     """
    #     :return: Returns a set of all ground neural predicates that need to be evaluated.
    #     """
    #     neural_eval = []
    #     weights = self.sdd.get_weights()
    #     for w in weights:
    #         w = weights[w]
    #         if type(w) is Term:
    #             if w.functor == "nn":
    #                 self._add_ordered_evaluation(w.args[0], w.args[1], neural_eval)
    #
    #     return neural_eval

    # def _add_ordered_evaluation(self, name, arguments, evals):
    #     # Check arguments for tensors that need to be evaluated
    #     for argument in term2list(arguments, deep=False):
    #         if argument.functor == "tensor":
    #             argument = argument.args[0]
    #             if argument.functor == "nn":
    #                 self._add_ordered_evaluation(*argument.args, evals)
    #     k = (name, arguments)
    #     if k not in evals:
    #         evals.append(k)

    # def get_named(self) -> Dict[Term, int]:
    #     """
    #     :return: A dictionary mapping all named nodes in the SDD to their node id.
    #     """
    #     named = dict()
    #     for n in self.sdd.get_names():
    #         named[n[0]] = n[1]
    #     return named

    def save(self, filename):
        manager = self.sdd.get_manager().get_manager()
        i = list(self.sdd.queries())[0][1] - 1

        inode = self.sdd.get_manager().nodes[i]
        constraint_inode = self.sdd.get_constraint_inode()
        node = self.sdd.get_manager().conjoin(inode, constraint_inode)
        manager.minimize()
        var_names = [
            (var, self.sdd.get_node(atom).probability)
            for var, atom in self.sdd.var2atom.items()
        ]

        manager.vtree().save((filename + ".vtree").encode())
        manager.save((filename + ".sdd").encode(), node)
        manager.save((filename + ".constraint").encode(), constraint_inode)
        with open(filename + ".tsv", "w") as f:
            f.write("\n".join(str(v) + "\t" + str(p) for v, p in var_names))
