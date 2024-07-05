import pytest
from deepproblog.utils.standard_networks import DummyNet, DummyTensorNet
from problog.logic import Term, Var, Constant
from deepproblog.engines import ExactEngine, ApproximateEngine
from deepproblog.model import Model
from deepproblog.query import Query
from deepproblog.network import Network

import torch

program = """
nn(dummy1,[X],Y,[a,b,c]) :: net1(X,Y).
nn(dummy2,[X]) :: net2(X).
nn(dummy3,[X],Y) :: net3(X,Y).
nn(dummy4,[X,Y]) :: net4(X,Y).

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


dummy_tensors = {(Term("a"),): torch.Tensor([0.1, 0.2, 0.3, 0.4]), (Term("b"),): torch.Tensor([0.25, 0.25, 0.25, 0.25])}


class IndexNet(torch.nn.Module):

    def forward(self, t, index):
        # index = int(index)
        index = torch.LongTensor([int(i) for i in index])
        return t.index_select(dim=1, index=index)


dummy_net4 = Network(IndexNet(), "dummy4", batching=True)


@pytest.fixture(
    params=[
        {
            "name": "approximate",
            "engine_factory": lambda model: ApproximateEngine(
                model, 10, ApproximateEngine.geometric_mean
            ),
            "cache": False,
        },
        {"name": "no_cache", "engine_factory": lambda model: ExactEngine(model), "cache": False},
        {"name": "cache", "engine_factory": lambda model: ExactEngine(model), "cache": True},
    ]
)
def model(request) -> Model:
    """Simple fixture creating both the approximate and the exact engine"""
    if ApproximateEngine is None and request.param["name"] == "approximate":
        pytest.skip("ApproximateEngine is not available as PySWIP is not installed")
    model = Model(program, [dummy_net1, dummy_net2, dummy_net3, dummy_net4], load=False)
    engine = request.param["engine_factory"](model)
    model.set_engine(engine, cache=request.param["cache"])
    model.add_tensor_source('dummy', dummy_tensors)
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
    results = model.solve([Query(Term("test3", Term("i1"), Var("X"))), Query(Term("test3", Term("i2"), Var("X")))])
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

def test_multi_input_network(model: Model):
    dummy_tensor = lambda x: Term("tensor", Term("dummy", x))
    q1 = Query(Term("net4", dummy_tensor(Term("a")), Constant(1)))
    q2 = Query(Term("net4", dummy_tensor(Term("b")), Constant(2)))
    results = model.solve([q1, q2])
    r1 = float(results[0].result[q1.query])
    r2 = float(results[1].result[q2.query])
    assert pytest.approx(0.2) == r1
    assert pytest.approx(0.25) == r2