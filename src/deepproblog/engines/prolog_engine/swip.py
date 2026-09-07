from typing import Dict, Tuple, Union

from problog.logic import (
    Term,
    Constant,
    list2term,
    term2list,
    Var,
    is_list,
    Clause,
    And,
    Or,
)
from problog.parser import PrologParser
from problog.program import ExtendedPrologFactory
from pyswip import Functor, Atom, Variable
from pyswip.easy import Term as PySwipTerm, putTerm
from pyswip.core import PL_copy_term_ref, PL_new_term_ref, PL_unify

PySwipObject = Union[Functor, Atom, Variable, int, float, list, bytes]
ProblogObject = Union[Term, Constant, Var, Clause, And, Or]


def pyswip_to_term(
    pyswip_obj: PySwipObject, with_variables=False
) -> Union[ProblogObject, Tuple[ProblogObject, Dict[Var, Variable]]]:
    variables = dict()
    if type(pyswip_obj) is Functor:
        args = []
        for a in pyswip_obj.args:
            args2, variables2 = pyswip_to_term(a, True)
            args.append(args2)
            variables.update(variables2)
        operator = pyswip_obj.name.get_value()
        if operator == ":-":
            new_term = Clause(*args)
        elif operator == ",":
            new_term = And.from_list(args)
        elif operator == ";":
            new_term = Or.from_list(args)
        else:
            new_term = Term(operator, *args)
    elif type(pyswip_obj) is Atom:
        new_term = Term(pyswip_obj.get_value())
    elif type(pyswip_obj) is int or type(pyswip_obj) is float:
        new_term = Constant(pyswip_obj)
    elif type(pyswip_obj) is list:
        lst = []
        for o in pyswip_obj:
            e, vars2 = pyswip_to_term(o, True)
            lst.append(e)
            variables.update(vars2)
        new_term = list2term(lst)
    elif type(pyswip_obj) is Variable:
        new_term = Var(
            pyswip_obj.chars if pyswip_obj.chars else f"Var{pyswip_obj.handle}"
        )
        variables = {new_term: pyswip_obj}
    elif type(pyswip_obj) is bytes:
        new_term = Constant(pyswip_obj.decode("utf-8"))
    else:
        raise Exception(f"Unhandled type {type(pyswip_obj)} from object {pyswip_obj}")
    if with_variables:
        return new_term, variables
    else:
        return new_term


def term_to_pyswip(term: ProblogObject) -> PySwipObject:
    if type(term) is Term:
        if is_list(term):
            return [term_to_pyswip(x) for x in term2list(term, False)]
        args = [term_to_pyswip(arg) for arg in term.args]
        if not args:
            return Atom(term.functor)

        functor = Functor(term.functor, arity=term.arity)
        return functor(*args)
    elif type(term) is Constant:
        return term.functor
    elif type(term) is Var:
        return Variable(name=term.name)
    else:
        raise Exception(
            f"Unhandled type {type(term)} from object {term} -> Robin has to fix it"
        )


_parser = PrologParser(ExtendedPrologFactory())


def parse(to_parse: Union[str, PySwipObject]) -> ProblogObject:
    if type(to_parse) is str:
        return _parser.parseString(str(to_parse) + ".")[0]
    return pyswip_to_term(to_parse)


def bindings_to_dict(bindings: list) -> Dict[str, PySwipObject]:
    """Turn the raw binding list of a query into a dictionary.

    Queries are run with normalize=False, because normalization renders
    compound terms as strings, which cannot be turned back into ProbLog terms.
    The raw result is a list of Name=Value terms.
    """
    result = dict()
    for binding in bindings:
        name, value = binding.args
        result[name.value if isinstance(name, Atom) else str(name)] = value
    return result


def unify(variable: Variable, value: PySwipObject) -> None:
    """Unify a variable with a value, including compound terms.

    Variable.unify does not handle compound terms, so those are put into a
    term reference and unified directly.
    """
    if not isinstance(value, PySwipTerm):
        variable.unify(value)
        return
    if variable.handle is None:
        handle = PL_new_term_ref()
    else:
        handle = PL_copy_term_ref(variable.handle)
    reference = PL_new_term_ref()
    putTerm(reference, value)
    PL_unify(handle, reference)
    variable.handle = handle


def _value(pyswip_obj: PySwipObject):
    try:
        return pyswip_obj.value
    except AttributeError:
        return pyswip_obj


def _unifier(arity, *args):
    """Handler for =/2 terms.

    PySwip installs one itself, but it raises an AttributeError when the first
    argument has no value, such as the integer in a(X) :- X = 3.
    """
    assert arity == 2
    return {_value(args[0]): _value(args[1])}


Functor.func[Functor("=", 2).handle] = _unifier
