import pytest
from deepproblog.engines import ApproximateEngine
from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.query import Query
from problog.logic import Term
from deepproblog.solver import SolverException


def _create_model(program, cache=False) -> Model:
    """Setup code: Load a program minimally"""
    model = Model(program, [], load=False)
    engine = ApproximateEngine(model, 10, geometric_mean)
    model.set_engine(engine, cache=cache)
    return model


def test_cache_error():
    with pytest.raises(SolverException):
        _create_model("", cache=True)


def test_ad():
    program = """
0.5 :: a; 0.5 :: b.
c :- a,b.
d :- a,a.
    """
    model = _create_model(program)
    q = Query(Term("c"))
    r = model.solve([q])[0].result[Term("c")]
    assert pytest.approx(0.0) == r
    q = Query(Term("d"))
    r = model.solve([q])[0].result[Term("d")]
    assert pytest.approx(0.5) == r


def test_ad2():
    program = """
0.5 :: a; 0.5 :: b.
0.5 :: a; 0.5 :: b.
c :- a,b.
d :- a,a.
    """
    model = _create_model(program)
    q = Query(Term("c"))
    r = model.solve([q])[0].result[Term("c")]
    assert pytest.approx(0.5) == r
    q = Query(Term("d"))
    r = model.solve([q])[0].result[Term("d")]
    assert pytest.approx(0.75) == r


def test_ad3():
    program = """
0.5 :: a(X); 0.5 :: b(X).
c :- a(1),b(1).
d :- a(1),b(2).
    """
    model = _create_model(program)
    q = Query(Term("c"))
    r = model.solve([q])[0].result[Term("c")]
    assert r == pytest.approx(0.0)
    q = Query(Term("d"))
    r = model.solve([q])[0].result[Term("d")]
    assert r == pytest.approx(0.25)


def test_fact():
    program = """
0.8::a(_).
c :- a(1),a(2).
d :- a(1),a(1).
    """
    model = _create_model(program)
    q = Query(Term("c"))
    r = model.solve([q])[0].result[Term("c")]
    assert pytest.approx(0.64) == r
    q = Query(Term("d"))
    r = model.solve([q])[0].result[Term("d")]
    assert pytest.approx(0.8) == r


def test_fact2():
    program = """
0.6::a(_).
0.5::a(_).
c :- a(1),a(2).
d :- a(1),a(1).
    """
    model = _create_model(program)
    q = Query(Term("c"))
    r = model.solve([q])[0].result[Term("c")]
    assert pytest.approx(0.64) == r
    q = Query(Term("d"))
    r = model.solve([q])[0].result[Term("d")]
    assert pytest.approx(0.8) == r
