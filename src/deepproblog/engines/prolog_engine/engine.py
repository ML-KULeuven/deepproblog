from pathlib import Path
from typing import Optional

from problog.engine import GenericEngine
from problog.formula import LogicFormula
from problog.logic import Term, Var, Constant
from problog.program import LogicProgram
from pyswip import Prolog

from .heuristics import Heuristic
from .swi_program import SWIProgram

root = Path(__file__).parent


def escape_strings_in_term(term: Term):
    if type(term) is Term:
        return Term(term.functor, *(escape_strings_in_term(x) for x in term.args))
    elif type(term) is Var:
        return term
    elif type(term) is Constant:
        if term.is_string():
            value = term.value
            if value[0] != '"' and value[-1] != '"':
                return Constant('"' + value + '"')
        return term
    raise ValueError("{} not handled in escape_strings_in_term".format(type(term)))


class PrologEvaluationException(Exception):
    """Exception from PrologEngine for unexpected result when evaluating a query."""


class PrologEngine(GenericEngine):
    def __init__(
        self,
        k,
        heuristic: Optional[Heuristic],
        exploration: bool,
        timeout=None,
        ignore_timeout=False,
    ):
        super().__init__()
        self.k = k
        self.heuristic = heuristic
        self.prolog = Prolog()
        self.timeout = timeout
        self.ignore_timeout = ignore_timeout
        self.exploration = exploration
        path = root / "prolog_files" / "engine_heap.pl"
        self.prolog.consult(path.as_posix())

    def prepare(self, db):
        program = SWIProgram(db, heuristic=self.heuristic)
        return program

    def ground(self, sp: LogicProgram, term, target=None, label=None, *args, **kwargs):
        if type(sp) != SWIProgram:
            sp = self.prepare(sp)
        if target is None:
            target = LogicFormula(keep_all=True)
        proofs = self.get_proofs(term, sp)
        result = sp.add_proof_trees(proofs, target=target, label=label)
        return result

    def ground_all(
        self,
        sp: LogicProgram,
        target=None,
        queries=None,
        evidence=None,
        *args,
        **kwargs,
    ):
        if type(sp) != SWIProgram:
            sp = self.prepare(sp)
        if target is None:
            target = LogicFormula()
        if queries is None:
            queries = [
                q[0].args[0]
                for q in self.ground(
                    sp, Term("query", Var("X")), *args, **kwargs
                ).queries()
            ]
        for q in queries:
            self.ground(sp, q, target, *args, **kwargs)
        return target

    def get_proofs(self, q: Term, program: SWIProgram, profile=0):
        exploration = "true" if self.exploration else "false"
        q_term = escape_strings_in_term(q)
        query_str = "prove({},{},Proofs,{},{})".format(
            q_term, self.k, self.heuristic.name, exploration
        )
        if self.timeout is not None:
            query_str = "call_with_time_limit({},{})".format(self.timeout, query_str)
        try:
            res = program.query(query_str, profile=profile)
        except TimeoutError:
            if self.ignore_timeout:
                return []
            else:
                raise TimeoutError()
        except OverflowError:
            return []
        if len(res) != 1:
            raise PrologEvaluationException(
                f"Expected exactly 1 result, got {len(res)}"
            )
        return res[0]["Proofs"]
