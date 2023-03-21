from __future__ import annotations

import re
from collections import defaultdict
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union, Generator, Iterable

from problog.core import ProbLogObject
from problog.logic import And
from problog.logic import Term, Clause, Or, AnnotatedDisjunction, Var
from problog.logic import term2list, ArithmeticError
from problog.program import LogicProgram
from pyswip import Prolog, registerForeign, Variable

from .heuristics import Heuristic
from .swip import parse, pyswip_to_term

ALL_FACTS = "fa(_,_,_,_)"

ALL_CLAUSES = "cl(_,_,_)"

ids = 0
current_program = None


def get_heuristic_func(heuristic_obj):
    def get_heuristic(node, heuristic: Variable):
        node = pyswip_to_term(node)
        value = heuristic_obj.get_value(node)
        heuristic.unify(value)

    get_heuristic.arity = 2
    return get_heuristic


class SWIProgramException(Exception):
    """Exception for prolog issues from SWIProgram"""


# Regex for query variable replacement
_RE_QUERY = re.compile(r"\$VAR\((.*?)\)")


class SWIProgram(ProbLogObject):
    FactOrClause = Union[Tuple[str, int, str, str], Tuple[str, int, float, str, str]]

    def __init__(
        self,
        db: Optional[LogicProgram] = None,
        heuristic: Optional[Heuristic] = None,
        parent: Optional[SWIProgram] = None,
    ):
        global ids
        self.id = "p{}".format(ids)
        self.groups = 0
        ids += 1
        self.prolog = Prolog()
        if parent is not None:
            self.facts_and_clauses = parent.facts_and_clauses
            self.db = parent.db
            self.ad_heads = parent.ad_heads
            self.i = parent.i
            self.d = parent.d
        else:

            self.facts_and_clauses: List[Optional[SWIProgram.FactOrClause]] = []
            self.db = db
            self.ad_heads = defaultdict(list)
            self.i = 0
            self.d = dict()
            self.parse_db()
        registerForeign(get_heuristic_func(heuristic), name="get_heuristic_extern")
        self.heuristic = heuristic

    # ASSERTING

    def assert_clause(self, clause):
        self.prolog.assertz(self.to_prolog(clause))

    def assert_fact(self, fact):
        self.prolog.assertz(self.to_prolog(fact))

    def add_program(
        self, db: Iterable[Union[Term, Or, Clause]]
    ) -> Generator[Tuple[str, int], None, None]:
        """
        Add all clauses and facts from a LogicProgram
        :return: The indices of the added facts and clauses.
        """
        for n in db:
            if type(n) is Clause:
                yield self.add_clause(n)
            elif type(n) is Or:
                yield from self.add_or(n)
            elif type(n) is Term:
                yield self.add_fact(n)
            elif type(n) is AnnotatedDisjunction:
                yield from self.add_or(n)
            else:
                raise SWIProgramException("Unhandled node type ", type(n))

    def add_fact(
        self,
        node: Term,
        group_id: Optional[int] = None,
        ad_i: Optional[int] = None,
        ad_vars: Optional[List[Var]] = None,
    ):
        probability = node.probability
        if probability is None:
            probability = 1.0
        ad_identifier = "none"
        if group_id is not None:
            ad_vars = [str(x) for x in ad_vars]
            ad_identifier = "ad({},{},[{}])".format(group_id, ad_i, ",".join(ad_vars))

        return self._add_fact(
            probability, str(node.with_probability(None)), ad_identifier
        )

    def _add_fact(
        self, probability: float, fact, ad_identifier, i: Optional[int] = None
    ) -> FactOrClause:
        if i is None:
            i = self.new_entry()
            self.facts_and_clauses.append(("fa", i, probability, fact, ad_identifier))
        else:
            assert self.facts_and_clauses[i - 1] is None
            self.facts_and_clauses[i - 1] = ("fa", i, probability, fact, ad_identifier)
        global current_program
        new_fact = ("fa", i, probability, fact, ad_identifier)
        if current_program == self:
            self.assert_fact(new_fact)
        return new_fact

    def add_or(self, node: Union[AnnotatedDisjunction, Or]):
        if type(node) is AnnotatedDisjunction:
            if node.body.functor == "true":
                disjuncts = node.heads
            else:
                raise SWIProgramException("ADs with bodies not yet supported")
        else:
            disjuncts = node.to_list()
        group_id = self.groups
        self.groups += 1
        variables = set()
        for disjunct in disjuncts:
            variables |= disjunct.variables()
        variables = list(variables)
        for i, disjunct in enumerate(disjuncts):
            yield self.add_fact(disjunct, group_id, i, variables)

    def add_clause(self, node: Clause, i=None) -> Optional[FactOrClause]:
        if node.head.functor == "_directive":
            return None
        if type(node.body) is Term:
            body = [node.body]
        elif type(node.body) is And:
            body = node.body.to_list()
        else:
            raise SWIProgramException("Unhandled body type")
        return self._add_clause(
            str(node.head), "[{}]".format(",".join(str(x) for x in body)), i=i
        )

    def _add_clause(
        self, head: str, body: str, i: Optional[int] = None
    ) -> FactOrClause:
        global current_program
        if i is None:
            i = self.new_entry()
            self.facts_and_clauses.append(("cl", i, head, body))
        else:
            assert self.facts_and_clauses[i - 1] is None
            self.facts_and_clauses[i - 1] = ("cl", i, head, body)
        new_clause = ("cl", i, head, body)
        if current_program == self:
            self.assert_clause(new_clause)
        return new_clause

    def assert_all(self):
        global current_program

        if current_program != self:
            current_program = self
            self.retract_all()
            for line in self.get_lines():
                self.prolog.assertz(line)

    # RETRACTING

    def retract_fact(self, i: int):
        self.prolog.retractall("fa({},_,_,_)".format(i))

    def remove_fact(self, i: int):
        global current_program
        if current_program == self:
            self.retract_fact(i)
        return self._remove_clause_or_fact(i)

    def _remove_clause_or_fact(self, i: int):
        removed = self.facts_and_clauses[i - 1]
        self.facts_and_clauses[i - 1] = None
        return removed

    def remove_clause(self, i: int):
        global current_program
        if current_program == self:
            self.retract_clause(i)
        return self._remove_clause_or_fact(i)

    def retract_clause(self, i: int):
        self.prolog.retractall("cl({},_,_)".format(i))

    def retract_all(self):
        self.prolog.retractall(ALL_CLAUSES)
        self.prolog.retractall(ALL_FACTS)

    # Proof building

    def build_tree(self, proof, target):
        name, proof = proof.args
        try:
            key = target.names[name]
        except KeyError:
            key = target.add_or([], placeholder=True, readonly=False)
            target.names[name] = key
            target.add_name(name, key)
        t = proof.functor
        if t == "::":
            i = (proof.args[0], proof.args[2])
            try:
                p = float(proof.args[1])
            except ArithmeticError:
                p = proof.args[1]
            group = proof.args[3]
            if group.functor == "ad":
                group = group.args[0], tuple(term2list(group.args[2], deep=False))
            else:
                group = None
            new_name = Term("fact", *i)
            try:
                new = target.names[new_name]
            except KeyError:
                i = target.get_next_atom_identifier()
                new = target.add_atom(i, p, group=group)
                target.names[new_name] = new
                target.add_name(new_name, new)
            target.add_disjunct(key, new)
        elif t == "and":
            body = proof.args[0]
            body = term2list(body, False)
            new_name = Term("and", *(p.args[0] for p in body))
            try:
                new = target.names[new_name]
                for b in body:
                    self.build_tree(b, target)
            except KeyError:
                new = target.add_and([self.build_tree(b, target) for b in body])
                target.names[new_name] = new
                target.add_name(new_name, new)
            target.add_disjunct(key, new)
        elif t == "foreign":
            new = target.TRUE
            target.add_disjunct(key, new)
        elif t == "neg":
            new = -self.build_tree(proof.args[0], target)
            target.add_disjunct(key, new)
        elif t == "cycle":
            pass
        elif t == "builtin":
            new = target.TRUE
            target.add_disjunct(key, new)
        elif t == "extern":
            new = target.TRUE
            target.add_disjunct(key, new)
        elif t == "nn":
            # net, inp, i = proof.args
            target.add_atom()
        else:
            raise SWIProgramException("Unhandled node type " + str(proof))
        return key

    def add_proof_trees(self, trees, target, label=None):
        target.names = dict()
        for tree in trees:
            depth, tree = tree.args
            query, proof = tree.args
            key = self.build_tree(tree, target)
            target.add_name(query, key, label=label)
        return target

    def query(self, query: str, profile=0) -> List[Dict[str, Any]]:
        """Perform a Prolog query and translate result back to Problog objects

        :param query: Query to perform
        :param profile: Allows enabling profiling.

        :return: A list of results to the query.
        """
        self.assert_all()

        # Replaces $VAR(X) with actual variables
        # Needed when specified queries are non ground
        query = _RE_QUERY.sub(r"X\1", query)
        #
        start = 0
        if profile > 0:
            start = time()
            if profile > 1:
                # query = 'profile((between(1,100,_),{},fail);true)'.format(query)
                query = f"profile({query})"
        result = list(self.prolog.query(query))
        if profile > 0:
            print(f"Query: {query} answered in {time() - start} seconds")
            if profile > 1:
                input()
        if result:
            out = []
            for r in result:
                out_partial = {}
                for k in r:
                    v = result[0][k]
                    if type(v) is list:
                        if len(v) > 1 and isinstance(v[0], str):
                            raise TypeError("Oops, it appears you are using the wrong version of PySwip.\n"
                                            "Please make sure you are using PySwip from https://github.com/ML-KULeuven/pyswip\n"
                                            "To install, first remove your current PySwip version, then install the correct version.\n"
                                            "You can try doing\n'pip install git+https://github.com/ML-KULeuven/pyswip'.\n"
                                            "For some reason, this does not always resolve the issue. If you still get this error,\n"
                                            "clone the repo locally, then compile from your local codebase doing\n"
                                            "'pip install [path to local clone]'.")
                        out_partial[k] = [p for p in term2list(parse(result[0][k]))]
                    else:
                        out_partial[k] = parse(v)
                out.append(out_partial)
            return out
        else:
            return []

    # Other

    @staticmethod
    def to_prolog(f: FactOrClause) -> str:
        if f[0] == "fa":
            return "fa({},{},{},{})".format(*f[1:])
        elif f[0] == "cl":
            return "cl({},{},{})".format(*f[1:])

    def get_lines(self) -> List[str]:
        lines = [self.to_prolog(c) for c in self.facts_and_clauses if c is not None]
        return lines

    def __str__(self):
        return "\n".join(line + "." for line in self.get_lines())

    def parse_db(self):
        """
        Parse the database (LogicProgram) into a valid SWI-Prolog
        :return: List of added clauses (Also updates the current object)
        """
        if self.db is not None:
            return list(self.add_program(self.db))

    @staticmethod
    def registerForeign(func, name, arity=None, **kwargs):
        registerForeign(func, name, arity, **kwargs)

    def extend(self):
        print("Extending")
        return SWIProgram(parent=self)

    def new_entry(self) -> int:
        self.i += 1
        return self.i
