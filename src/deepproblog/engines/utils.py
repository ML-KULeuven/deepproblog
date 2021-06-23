import contextlib
from typing import Iterable

from deepproblog.engines.prolog_engine.swi_program import SWIProgram
from deepproblog.model import Model
from problog.logic import Term


@contextlib.contextmanager
def with_terms(model: Model, terms: Iterable[Term]):
    """
    Execute code block with terms registered into the Problog program.

    :param model: Model to modify
    :param terms: Iterable of terms to register

    TODO: Only works with ApproximateEngine currently.
    """
    program: SWIProgram = model.solver.program
    if isinstance(program, SWIProgram):
        # cdb = ClauseDB(builtins={})
        # for c in terms:
        #     cdb.add_statement(c)
        identifiers = list(x[0:2] for x in program.add_program(terms))
        model.solver.cache.invalidate()
        try:
            yield
        finally:
            for type_, idx in identifiers:
                if type_ == "cl":
                    program.retract_clause(idx)
                elif type_ == "fa":
                    program.retract_fact(idx)
    else:
        raise NotImplementedError(
            "with_terms is currently only implemented for ApproximateEngine"
        )
