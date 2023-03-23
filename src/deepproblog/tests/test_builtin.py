import pytest
from deepproblog.engines import ExactEngine

from deepproblog.model import Model
from deepproblog.query import Query
from problog.logic import Term, Var

def _create_model(program) -> Model:
    """Setup code: Load a program minimally"""
    model = Model(program, [], load=False)
    engine = ExactEngine(model)
    model.set_engine(engine)
    return model



def test_tensor_index():
    program = """
P :: a(I) :-  list_to_tensor([0.25,0.1,0.4,0.15], Tensor), between(0,3,I), tensor_index(Tensor,[I],P).
    """
    model = _create_model(program)
    q = Query(Term("a", Var("X")))
    r = model.solve([q])[0].result
    expected_result = [0.25, 0.1, 0.4, 0.15]
    for k in r:
        i = int(k.args[0])
        assert pytest.approx(expected_result[i]) == float(r[k])


def test_less_than():
    program = """
    a :- list_to_tensor([0.8],T1), list_to_tensor([0.2],T2), less_than(T1,T2).
    b :- list_to_tensor([0.2],T1), list_to_tensor([0.8],T2), less_than(T1,T2).
    """
    model = _create_model(program)
    q1 = Query(Term("a"))
    q2 = Query(Term("b"))
    assert pytest.approx(0.0) == model.solve([q1])[0].result[Term("a")]
    assert pytest.approx(1.0) == model.solve([q2])[0].result[Term("b")]