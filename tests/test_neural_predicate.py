import pytest
from deepproblog.utils.standard_networks import DummyNet
from problog.logic import Term, Var
from deepproblog.engines import ExactEngine, ApproximateEngine
from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.query import Query
from deepproblog.network import Network
import numpy as np

program = """
nn(dummy1,[X],Y,[a,b,c]) :: net1(X,Y).
nn(dummy2,[X]) :: net2(X).
nn(dummy3,[X],Y) :: net3(X,Y).

test1(X1,Y1,X2,Y2) :- net1(X1,Y1), net1(X2,Y2).
test2(X1,X2) :- net2(X1), net2(X2).
test3(X,Y) :- net3(X,Y).
"""

dummy_values1 = {Term("i1"): [0.8, 0.15, 0.05], Term("i2"): [0.2, 0.3, 0.5]}
dummy_net1 = Network(DummyNet(dummy_values1), "dummy1")

dummy_values2 = {Term("i1"): [0.8], Term("i2"): [0.4]}
dummy_net2 = Network(DummyNet(dummy_values2), "dummy2")

dummy_values3 = {Term("i1"): [1.0, 2.0, 3.0, 4.0], Term("i2"): [-1.0, 0.0, 1.0]}
dummy_net3 = Network(DummyNet(dummy_values3), "dummy3")


@pytest.fixture(
    params=[
        {
            "engine_factory": lambda model: ApproximateEngine(
                model, 10, geometric_mean
            ),
            "cache": False,
        },
        {"engine_factory": lambda model: ExactEngine(model), "cache": False},
        {"engine_factory": lambda model: ExactEngine(model), "cache": True},
    ]
)
def model(request) -> Model:
    """Simple fixture creating both the approximate and the exact engine"""
    model = Model(program, [dummy_net1, dummy_net2, dummy_net3], load=False)
    engine = request.param["engine_factory"](model)
    model.set_engine(engine, cache=request.param["cache"])
    return model


def test_model_basics(model):
    # These should be set up after running the fixture.
    assert model.solver is not None
    assert model.program is not None


def test_ad_network(model: Model):
    q1 = Query(Term("test1", Term("i1"), Term("a"), Term("i2"), Term("b")))  # p = 0.24
    q2 = Query(Term("test1", Term("i1"), Term("a"), Term("i2"), Term("a")))  # p = 0.16
    q3 = Query(Term("test1", Term("i1"), Term("a"), Term("i1"), Term("b")))  # p = 0
    results = model.solve([q1, q2, q3])
    r1 = float(results[0].result[q1.query])
    r2 = float(results[1].result[q2.query])
    r3 = float(results[2].result[q3.query])
    assert pytest.approx(0.24) == r1
    assert pytest.approx(0.16) == r2
    assert pytest.approx(0) == r3


def test_fact_network(model: Model):
    q1 = Query(Term("test2", Term("i1"), Term("i2")))  # p = 0.32
    q2 = Query(Term("test2", Term("i1"), Term("i1")))  # p = 0.8
    results = model.solve([q1, q2])
    r1 = float(results[0].result[q1.query])
    r2 = float(results[1].result[q2.query])
    assert pytest.approx(0.32) == r1
    assert pytest.approx(0.8) == r2


def test_det_network(model: Model):
    q1 = Query(Term("test3", Term("i1"), Var("X")))
    q2 = Query(Term("test3", Term("i2"), Var("X")))
    results = model.solve([q1, q2])
    r1 = list(results[0].result)[0].args[1]
    r2 = list(results[1].result)[0].args[1]
    r1 = model.get_tensor(r1)
    r2 = model.get_tensor(r2)
    assert all(r1.detach().numpy() == [1.0, 2.0, 3.0, 4.0])
    assert all(r2.detach().numpy() == [-1.0, 0.0, 1.0])


def test_det_network_substitution(model: Model):
    if not model.solver.cache.cache:
        q1 = Query(Term("test3", Term("a"), Var("X")), {Term("a"): Term("i1")})
        q2 = Query(Term("test3", Term("a"), Var("X")), {Term("a"): Term("i2")})
        results = model.solve([q1, q2])
        r1 = list(results[0].result)[0].args[1]
        r2 = list(results[1].result)[0].args[1]
        r1 = model.get_tensor(r1)
        r2 = model.get_tensor(r2)
        assert all(r1.detach().numpy() == [1.0, 2.0, 3.0, 4.0])
        assert all(r2.detach().numpy() == [-1.0, 0.0, 1.0])
