import pytest

from deepproblog.engines import Engine, ExactEngine, ApproximateEngine
from deepproblog.model import Model
from deepproblog.query import Query
from deepproblog.utils import parse

_simple_program = """
0.5 :: a.

equal(X,X).

0.1 :: multiple_answers(dummy1).
0.2 :: multiple_answers(dummy2). 
"""


@pytest.fixture()
def model() -> Model:
    """Simple fixture creating both the approximate and the engine"""
    model = Model(_simple_program, [])
    return model


@pytest.fixture(
    params=[
        #{"engine_factory": lambda model: ApproximateEngine(model, 2, ApproximateEngine.geometric_mean)},
        {"engine_factory": lambda model: ExactEngine(model, cache_memory=True)},
        {"engine_factory": lambda model: ExactEngine(model, cache_memory=False)},
    ]
)
def engine(model, request) -> Engine:
    return request.param["engine_factory"](model)


def test_model_basics(model: Model, engine: Engine):
    # These should be set up after running the fixture.
    assert model.program is not None
    assert engine.model is not None


def test_solve(model: Model, engine: Engine):
    q1 = Query(parse("equal(dummy,dummy)."))
    q2 = Query(parse("equal(dummy,notequal)."))
    q3 = Query(parse("multiple_answers(X)."))
    q4 = Query(parse("a."))
    acs = engine.query_batch([q1, q2, q3, q4])
    results = [ac.evaluate(model) for ac in acs]
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
