import pytest
from deepproblog.engines import ApproximateEngine

import pytest
if ApproximateEngine is None:
    pytest.skip('Approximate Inference is not available as PySDD is not installed', allow_module_level=True)

from deepproblog.model import Model
from deepproblog.query import Query
from problog.logic import Term, Constant, Var
from deepproblog.solver import SolverException


def _create_model(program, cache=False) -> Model:
    """Setup code: Load a program minimally"""
    model = Model(program, [], load=False)
    engine = ApproximateEngine(model, 10, ApproximateEngine.geometric_mean)
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
0.8::d(_).
e :- d(1),d(2).
f :- d(1),e.
    """
    model = _create_model(program)
    q = Query(Term("e"))
    r = model.solve([q])[0].result
    assert pytest.approx(0.64) == r[Term("e")]
    q = Query(Term("f"))
    r = model.solve([q])[0].result
    assert pytest.approx(0.64) == r[Term("f")]


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


def test_foreign():
    program = """
a(T,Out) :- evaluate(T, Out).
    """

    def evaluate(x):
        print("Inside evaluate", x)
        return Constant(int(x) ** 2)

    model = _create_model(program)
    model.register_foreign(evaluate, "evaluate", 1, 1)
    q = Query(Term("a", Constant(2), Var("X")))
    r = model.solve([q])[0].result[Term("a", Constant(2), Constant(4))]
    assert pytest.approx(1.0) == r
    q = Query(Term("a", Constant(2), Constant(4)))
    r = model.solve([q])[0].result[Term("a", Constant(2), Constant(4))]
    assert pytest.approx(1.0) == r
    q = Query(Term("a", Constant(2), Constant(5)))
    r = model.solve([q])[0]
    assert len(r.result) == 0


def test_foreign_text():
    program = """
a(Text,I,Word) :- ith_word(Text,I,Word).
    """

    def ith_word(text, i):
        text = text.value
        word = text.split(" ")[int(i)]
        return Constant(word)

    model = _create_model(program)
    model.register_foreign(ith_word, "ith_word", 2, 1)
    queries = [
        Query(Term("a", Constant("one two three"), Constant(x), Var("X")))
        for x in range(3)
    ]
    r = model.solve(queries)
    assert (
        pytest.approx(1.0)
        == r[0].result[
            Term("a", Constant("one two three"), Constant(0), Constant("one"))
        ]
    )
    assert (
        pytest.approx(1.0)
        == r[1].result[
            Term("a", Constant("one two three"), Constant(1), Constant("two"))
        ]
    )
    assert (
        pytest.approx(1.0)
        == r[2].result[
            Term("a", Constant("one two three"), Constant(2), Constant("three"))
        ]
    )


def test_assignment():
    program = """
a(X) :- X=3.
    """

    def evaluate(x):
        print("Inside evaluate", x)
        return Constant(x ** 2)

    model = _create_model(program)
    model.register_foreign(evaluate, "evaluate", 1, 1)
    q = Query(Term("a", Var("X")))
    r = model.solve([q])[0].result[Term("a", Constant(3))]
    assert pytest.approx(1.0) == r
