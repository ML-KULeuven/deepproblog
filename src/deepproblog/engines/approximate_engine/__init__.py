from problog.logic import term2list
from pyswip import Variable, registerForeign

from deepproblog.engines.approximate_engine.prolog_engine import (
    PrologEngine,
    pyswip_to_term,
    term_to_pyswip,
)
from deepproblog.engines.approximate_engine.prolog_engine.heuristics import (
    GeometricMean,
    PartialProbability,
    Heuristic,
)
from ..engine import Engine


class ApproximateEngine(Engine):
    geometric_mean = GeometricMean()
    ucs = PartialProbability()

    def __init__(
        self,
        model,
        k,
        heuristic: Heuristic,
        exploration=False,
        timeout=None,
        ignore_timeout=False,
    ):
        Engine.__init__(self, model)
        self.heuristic = heuristic
        self.k = k
        self.engine = PrologEngine(k, heuristic, exploration, timeout, ignore_timeout)
        heuristic.engine = self.engine
        self.register_neural_predicates()

    def ground(self, query):
        return self.engine.ground(
            self.model.program, query.substitute().query, label="query"
        )

    @staticmethod
    def get_wrapped_func(func, arity_in, arity_out):
        def wrapped_func(*arguments):
            input_args, output_args = arguments[:arity_in], arguments[arity_in:]
            input_args = [pyswip_to_term(x) for x in input_args]
            result = func(*input_args)
            if type(result) is not func:
                result = (result,)
            result = [term_to_pyswip(r) for r in result]
            for o, r in zip(output_args, result):
                if type(o) is Variable:
                    o.unify(r)
                else:
                    if o != r:
                        return False

        return wrapped_func

    def register_foreign(self, func, function_name, arity_in, arity_out):
        wrapped_func = self.get_wrapped_func(func, arity_in, arity_out)
        wrapped_func.arity = arity_in + arity_out
        registerForeign(wrapped_func, function_name)
        builtin_name = "{}({})".format(
            function_name, ",".join(["_"] * (arity_in + arity_out))
        )
        list(
            self.engine.prolog.query(
                "assertz(allowed_builtin({}))".format(builtin_name)
            )
        )

    def get_hyperparameters(self) -> dict:
        parameters = {
            "type": "ApproximateEngine",
            "heuristic": self.heuristic.get_hyperparameters(),
            "k": self.k,
        }
        return parameters

    def eval(self):
        list(self.engine.prolog.query("set_flag(mode,eval)"))

    def train(self):
        list(self.engine.prolog.query("set_flag(mode,train)"))

    def register_neural_predicates(self):
        def func(network, inputs, output_var):
            network = network.value
            inputs, variables = pyswip_to_term(inputs, with_variables=True)
            assert len(variables) == 0
            net = self.model.networks[network]
            inputs = ([arg] for arg in term2list(inputs, False))
            probabilities = net(*inputs)[0]
            if probabilities.shape[0] == 1:
                output_var.unify(float(probabilities))
            else:
                output_var.unify([float(x) for x in probabilities])

        func.arity = 3
        registerForeign(func, "evaluate_network")
