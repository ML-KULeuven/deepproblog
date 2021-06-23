from unittest import TestCase

from deepproblog.engines.prolog_engine import PrologEngine
from deepproblog.heuristics import ucs
from problog import get_evaluatable
from problog.logic import Term, Constant, Var, Clause
from problog.program import PrologString


class TestSWIProgram(TestCase):
    def compare_query_probability(self, engine, program, query, target):
        ground = engine.ground(program, query, label="query")
        sdd = get_evaluatable().create_from(ground)
        p = sdd.evaluate()[query]
        self.assertAlmostEqual(target, p)

    def has_solution(self, engine, program, query, answer):
        ground = engine.ground(program, query, label="query")
        sdd = get_evaluatable().create_from(ground)
        p = sdd.evaluate()
        print(p)
        self.assertAlmostEqual(p[answer], 1.0)

    def test_add_clause(self):
        program = """
a :- b, c.
0.5::a.
0.5::b.
0.5::c.
0.8::d.
query(a).
"""
        program = PrologString(program)
        engine = PrologEngine(k=10, heuristic=ucs, exploration=False)
        program = engine.prepare(program)

        self.compare_query_probability(engine, program, Term("a"), 0.625)
        clause_i = program.add_clause(Clause(Term("a"), Term("d")))[1]
        self.compare_query_probability(engine, program, Term("a"), 0.925)
        program.remove_clause(clause_i)
        self.compare_query_probability(engine, program, Term("a"), 0.625)

    def test_non_ground_is(self):
        program = "b(X,Y) :- Y is X + 1."
        program = PrologString(program)
        engine = PrologEngine(k=5, heuristic=ucs, exploration=False)
        program = engine.prepare(program)

        self.has_solution(
            engine,
            program,
            Term("b", Constant(2), Var("Y")),
            Term("b", Constant(2), Constant(3)),
        )
