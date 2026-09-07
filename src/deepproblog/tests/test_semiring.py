import math

import pytest
import torch

from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.optimizer import SGD
from deepproblog.query import Query
from deepproblog.semiring.graph_semiring import (
    GraphSemiring,
    is_exact_one,
    is_exact_zero,
)
from problog.logic import Constant, Term


class Images(object):
    """Minimal tensor source."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, item):
        return self.tensors[int(item[0])]


def _create_model(program) -> Model:
    """Setup code: Load a program minimally"""
    model = Model(program, [], load=False)
    model.set_engine(ExactEngine(model))
    return model


def _solve(program, term=Term("q")) -> float:
    model = _create_model(program)
    return model.solve([Query(term)])[0].result[term]


@pytest.mark.parametrize("p", [1e-8, 1e-6, 5e-6, 1e-5, 1.0001e-5, 0.1, 0.3])
def test_small_probabilities_are_summed(p):
    """Regression test for #17: plus should not drop terms smaller than eps."""
    program = """
{p} :: a; {p} :: b; {rest} :: c.
q :- a.
q :- b.
    """.format(p=p, rest=1.0 - 2.0 * p)
    assert pytest.approx(2.0 * p, rel=1e-6) == _solve(program)


@pytest.mark.parametrize("p", [1.0 - 1e-8, 1.0 - 1e-6, 0.99999, 0.9])
def test_probabilities_near_one_are_multiplied(p):
    """Regression test for #17: times should not drop factors within eps of 1."""
    program = """
{p} :: a.
0.5 :: b.
q :- a, b.
    """.format(p=p)
    assert pytest.approx(0.5 * p, rel=1e-6) == _solve(program)


def test_small_probabilities_keep_their_gradient():
    """Regression test for #17: every branch of the AD contributes a gradient."""
    program = """
t(0.00000001) :: a; t(0.00000001) :: b; t(0.99999998) :: c.
q :- a.
q :- b.
    """
    model = _create_model(program)
    model.optimizer = SGD(model, 1.0)
    p = model.solve([Query(Term("q"))])[0].result[Term("q")]
    assert pytest.approx(2e-8, rel=1e-6) == float(p)
    p.backward()
    # Both a and b are part of the proof, so both should receive a gradient.
    assert pytest.approx(1.0) == float(model.optimizer._params_grad[0])
    assert pytest.approx(1.0) == float(model.optimizer._params_grad[1])


def test_is_exact_zero_and_one():
    assert is_exact_zero(0.0)
    assert not is_exact_zero(1e-8)
    assert is_exact_one(1.0)
    assert not is_exact_one(1.0 - 1e-8)
    # Tensors are never shortcut: their gradient is not zero, even at 0.0 or 1.0.
    assert not is_exact_zero(torch.tensor(0.0))
    assert not is_exact_one(torch.tensor(1.0))


def test_neutral_elements_keep_tensors_in_the_graph():
    semiring = GraphSemiring(None, None, None)
    zero = torch.tensor(0.0, requires_grad=True)
    one = torch.tensor(1.0, requires_grad=True)
    assert semiring.plus(zero, 0.5).grad_fn is not None
    assert semiring.times(one, 0.5).grad_fn is not None
    # Neutral floats are still shortcut.
    assert semiring.plus(0.5, semiring.zero()) == 0.5
    assert semiring.times(0.5, semiring.one()) == 0.5


class SaturatedNet(torch.nn.Module):
    """A network whose softmax output is within eps of 0 and of 1."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits))

    def forward(self, *_):
        return torch.softmax(self.logits, -1)


def _saturated_model(logits, program):
    net = SaturatedNet(logits)
    model = Model(program, [Network(net, "sat")], load=False)
    model.set_engine(ExactEngine(model))
    return model, net


def test_saturated_network_keeps_gradient_and_finite_loss():
    """A saturated network should still be trainable, not silently truncated."""
    program = """
nn(sat,[X],Y,[a,b]) :: c(X,Y).
q :- c(i,b).
    """
    model, net = _saturated_model([0.0, -20.0], program)
    result = model.solve([Query(Term("q"))])[0]
    p = result.result[Term("q")]
    assert pytest.approx(float(torch.softmax(net.logits, -1)[1]), rel=1e-5) == float(p)
    # The loss stays finite and the gradient reaches the (saturated) network.
    loss = GraphSemiring.cross_entropy(result, 1.0, 1.0, q=Term("q"))
    assert not math.isnan(loss) and not math.isinf(loss)
    assert net.logits.grad is not None
    assert float(net.logits.grad.abs().sum()) > 0.0


def test_exhaustive_ad_stays_a_probability():
    """Round-off must not push probabilities outside [0, 1]."""
    program = """
nn(sat,[X],Y,[c0,c1,c2,c3,c4,c5,c6,c7,c8,c9]) :: c(X,Y).
q :- c(i,_).
r :- \\+ q.
    """
    logits = [-2.0 * i for i in range(10)]  # spans 1.0 down to ~1e-8
    model, _ = _saturated_model(logits, program)
    results = model.solve([Query(Term("q")), Query(Term("r"))])
    p_q = float(results[0].result[Term("q")])
    assert abs(p_q - 1.0) < 1e-5
    r = results[1].result
    p_r = float(r[Term("r")]) if r else 0.0
    assert -1e-5 < p_r < 1e-5


def test_training_through_the_circuit_is_stable():
    """Smoke test: sum-only supervision should train without NaNs diverging."""
    program = """
nn(net,[X],Y,[0,1,2]) :: digit(X,Y).
addition(X1,X2,S) :- digit(X1,Y1), digit(X2,Y2), S is Y1+Y2.
    """
    torch.manual_seed(0)
    centers = torch.eye(3) * 3.0
    labels = [i % 3 for i in range(12)]
    images = [centers[l] + torch.randn(3) * 0.1 for l in labels]

    net = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Softmax(-1))
    network = Network(net, "net", batching=True)
    network.optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)
    model = Model(program, [network], load=False)
    model.set_engine(ExactEngine(model))
    model.add_tensor_source("d", Images(images))

    queries = [
        Query(
            Term(
                "addition",
                Term("tensor", Term("d", Constant(2 * i))),
                Term("tensor", Term("d", Constant(2 * i + 1))),
                Constant(labels[2 * i] + labels[2 * i + 1]),
            )
        )
        for i in range(len(images) // 2)
    ]

    losses = []
    for _ in range(5):
        epoch_loss = 0.0
        for q in queries:
            model.optimizer.zero_grad()
            result = model.solve([q])[0]
            p = float(result.result[q.query])
            assert 0.0 <= p <= 1.0, "probability outside [0, 1]: {}".format(p)
            loss = GraphSemiring.cross_entropy(result, 1.0, 1.0, q=q.query)
            assert not math.isnan(loss) and not math.isinf(loss)
            epoch_loss += loss
            model.optimizer.step()
        losses.append(epoch_loss / len(queries))
    assert losses[-1] < losses[0]
