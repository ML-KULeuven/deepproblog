import pytest
from deepproblog.default_networks import DummyNet
from problog.logic import Term, Var
from deepproblog.engines import Engine, ExactEngine
from deepproblog.model import Model
from deepproblog.query import Query
from deepproblog.network import Network

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


@pytest.fixture()
def model() -> Model:
    """Simple fixture creating both the approximate and the engine"""
    model = Model(program, [dummy_net1, dummy_net2, dummy_net3])
    return model


@pytest.fixture(
    params=[
       # {"engine_factory": lambda model: ApproximateEngine(model, 2, ApproximateEngine.geometric_mean)},
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


def test_ad_network(model: Model, engine: Engine):
    q1 = Query(Term("test1", Term("i1"), Term("a"), Term("i2"), Term("b")))  # p = 0.24
    q2 = Query(Term("test1", Term("i1"), Term("a"), Term("i2"), Term("a")))  # p = 0.16
    q3 = Query(Term("test1", Term("i1"), Term("a"), Term("i1"), Term("b")))  # p = 0
    acs = engine.query_batch([q1, q2, q3])
    results = [ac.evaluate(model) for ac in acs]
    r1 = float(results[0].result[q1.query])
    r2 = float(results[1].result[q2.query])
    r3 = float(results[2].result[q3.query])
    assert pytest.approx(0.24) == r1
    assert pytest.approx(0.16) == r2
    assert pytest.approx(0) == r3


def test_fact_network(model: Model, engine: Engine):
    q1 = Query(Term("test2", Term("i1"), Term("i2")))  # p = 0.32
    q2 = Query(Term("test2", Term("i1"), Term("i1")))  # p = 0.8
    acs = engine.query_batch([q1, q2])
    results = [ac.evaluate(model) for ac in acs]
    r1 = float(results[0].result[q1.query])
    r2 = float(results[1].result[q2.query])
    assert pytest.approx(0.32) == r1
    assert pytest.approx(0.8) == r2


def test_det_network(model: Model, engine: Engine):
    batch = [Query(Term("test3", Term("i1"), Var("X"))), Query(Term("test3", Term("i2"), Var("X")))]
    acs = engine.query_batch(batch)
    results = [ac.evaluate(model) for ac in acs]
    r1 = list(results[0].result)[0].args[1]
    r2 = list(results[1].result)[0].args[1]
    r1 = model.get_tensor(r1)
    r2 = model.get_tensor(r2)
    assert all(r1.detach().numpy() == [1.0, 2.0, 3.0, 4.0])
    assert all(r2.detach().numpy() == [-1.0, 0.0, 1.0])


def test_det_network_substitution(model: Model, engine: Engine):
    q1 = Query(Term("test3", Term("a"), Var("X")), {Term("a"): Term("i1")})
    q2 = Query(Term("test3", Term("a"), Var("X")), {Term("a"): Term("i2")})
    acs = engine.query_batch([q1, q2])
    results = [ac.evaluate(model) for ac in acs]
    r1 = list(results[0].result)[0].args[1]
    r2 = list(results[1].result)[0].args[1]
    r1 = model.get_tensor(r1.apply_term(q1.substitution))
    r2 = model.get_tensor(r2.apply_term(q2.substitution))
    assert all(r1.detach().numpy() == [1.0, 2.0, 3.0, 4.0])
    assert all(r2.detach().numpy() == [-1.0, 0.0, 1.0])
