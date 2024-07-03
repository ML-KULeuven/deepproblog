import contextlib
from typing import Iterable, Union

from deepproblog.engines.prolog_engine.swi_program import SWIProgram
from deepproblog.model import Model
from problog.clausedb import ClauseDB
from problog.extern import problog_export
from problog.logic import AnnotatedDisjunction, Clause, Term


@contextlib.contextmanager
def with_terms(model: Model, terms: Iterable[Term]):
    """
    Execute code block with terms registered into the Problog program.

    :param model: Model to modify
    :param terms: Iterable of terms to register
    """
    program: Union[ClauseDB, SWIProgram] = model.solver.program
    if isinstance(program, SWIProgram):
        # Approximate engine case
        identifiers = list(x[0:2] for x in program.add_program(terms))
        model.solver.cache.invalidate()
        try:
            yield
        finally:
            for type_, idx in identifiers:
                if type_ == "cl":
                    program.remove_clause(idx)
                elif type_ == "fa":
                    program.remove_fact(idx)
            model.solver.cache.invalidate()
    elif isinstance(program, ClauseDB):
        # Exact engine case
        assert program == problog_export.database
        childdb = program.extend()
        for term in terms:
            if isinstance(term, (Clause, AnnotatedDisjunction)):
                childdb.add_clause(term)
            elif isinstance(term, Term):
                childdb.add_fact(term)
            else:
                raise NotImplementedError(f"Unsupported case: {type(term)}")
        model.solver.cache.invalidate()
        problog_export.database = childdb
        model.solver.program = childdb
        try:
            yield
        finally:
            model.solver.program = program
            problog_export.database = program
            model.solver.cache.invalidate()
    else:
        raise NotImplementedError(
            "with_terms is currently only implemented for ApproximateEngine & ExactEngine"
        )
