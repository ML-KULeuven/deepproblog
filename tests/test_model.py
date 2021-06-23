import pytest

from deepproblog.engines import ExactEngine, ApproximateEngine
from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.query import Query
from deepproblog.utils import parse

_simple_program = """
0.5 :: a.

equal(X,X).

0.1 :: multiple_answers(dummy1).
0.2 :: multiple_answers(dummy2). 
"""


@pytest.fixture(
    params=[
        {
            "engine_factory": lambda model: ApproximateEngine(model, 2, geometric_mean),
            "cache": False,
        },
        {"engine_factory": lambda model: ExactEngine(model), "cache": False},
        {"engine_factory": lambda model: ExactEngine(model), "cache": True},
    ]
)
def model(request) -> Model:
    """Simple fixture creating both the approximate and the exact engine"""
    model = Model(_simple_program, [], load=False)
    engine = request.param["engine_factory"](model)
    model.set_engine(engine, cache=request.param["cache"])
    return model


def test_model_basics(model):
    # These should be set up after running the fixture.
    assert model.solver is not None
    assert model.program is not None


def test_solve(model):
    q1 = Query(parse("equal(dummy,dummy)."))
    q2 = Query(parse("equal(dummy,notequal)."))
    q3 = Query(parse("multiple_answers(X)."))
    q4 = Query(parse("a."))
    results = model.solve([q1, q2, q3, q4])
    assert len(results) == 4
    # Q1:
    assert len(results[0].result) == 1  # Provable
    assert q1.query in results[0].result
    assert pytest.approx(1.0) == results[0].result[q1.query]
    # Q2:
    r2 = results[1].result
    assert len(r2) == 0 or (
        len(r2) == 1 and r2[q2.query] == pytest.approx(0.0)
    )  # Unprovable
    # Q3:
    r3 = results[2].result
    assert len(r3) == 2
    assert r3[parse("multiple_answers(dummy1).")] == pytest.approx(0.1)
    assert r3[parse("multiple_answers(dummy2).")] == pytest.approx(0.2)
    # Q4:
    r4 = results[3].result
    assert len(r4) == 1
    assert r4[q4.query] == pytest.approx(0.5)
