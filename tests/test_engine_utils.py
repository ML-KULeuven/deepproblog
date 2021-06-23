import pytest

import deepproblog.engines.engine
import deepproblog.engines.utils
from deepproblog.engines import ApproximateEngine
from deepproblog.engines.prolog_engine.swi_program import SWIProgram
from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.query import Query
from deepproblog.utils import parse

initial_program = """
parent(ann,steve).
parent(ann,amy).
parent(amy,amelia).
"""


def _create_model() -> Model:
    """Setup code: Load a program minimally"""
    model = Model(initial_program, [], load=False)
    engine = ApproximateEngine(model, 1, geometric_mean)
    model.set_engine(engine)
    return model


def test_with_terms():
    """Test to ensure the context manager correctly registers and un-registers clauses"""
    model = _create_model()
    program: SWIProgram = model.solver.program

    pl_query_term = parse("a(2,2).")
    pl_query_term2 = parse("b(2,3).")

    def _verify_not_registered():
        """Test to verify that the atoms were not registered"""
        assert len(program.query("fa(_,_,a(2,2),none).")) == 0
        assert len(program.query("cl(_,b(_,_),_).")) == 0
        r = model.solve([Query(pl_query_term), Query(pl_query_term2)])
        assert len(r) == 2
        assert len(r[0].result) == 0
        assert len(r[1].result) == 0

    _verify_not_registered()

    with deepproblog.engines.utils.with_terms(
        model, [parse("a(2, 2)."), parse("b(X, Y) :- Y is X + 1.")]
    ):
        assert len(program.query("fa(_,_,a(2,2),none).")) == 1
        assert len(program.query("cl(_,b(_,_),_).")) == 1
        r = model.solve([Query(pl_query_term), Query(pl_query_term2)])
        assert len(r) == 2
        assert len(r[0].result) == 1
        assert pytest.approx(1.0) == r[0].result[pl_query_term]
        assert len(r[1].result) == 1
        assert pytest.approx(1.0) == r[0].result[pl_query_term]

    _verify_not_registered()


def test_with_terms_grandparent():
    model = _create_model()
    program: SWIProgram = model.solver.program

    # The first statement is provable, the second is not.
    pl_query_term = parse("grandparent(ann,amelia).")
    pl_query_term2 = parse("grandparent(ann,steve).")

    def _verify_not_registered():
        """Test to verify that the atoms were not registered"""
        assert len(program.query("cl(_,grandparent(_,_),_).")) == 0
        r = model.solve([Query(pl_query_term), Query(pl_query_term2)])
        assert len(r) == 2
        assert len(r[0].result) == 0
        assert len(r[1].result) == 0

    _verify_not_registered()

    with deepproblog.engines.utils.with_terms(
        model, [parse("grandparent(X, Y) :- parent(X,Z), parent(Z,Y).")]
    ):
        assert len(program.query("cl(_,grandparent(_,_),_).")) == 1
        r = model.solve([Query(pl_query_term), Query(pl_query_term2)])
        assert len(r) == 2
        assert len(r[0].result) == 1
        assert pytest.approx(1.0) == r[0].result[pl_query_term]
        assert len(r[1].result) == 0

    _verify_not_registered()
