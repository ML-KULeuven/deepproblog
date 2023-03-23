from problog.logic import (
    Term,
    AnnotatedDisjunction,
    term2list,
    Clause,
    Or,
    Constant,
    And,
    list2term,
)
from problog.program import SimpleProgram
from pyswip import Variable, registerForeign

from deepproblog.engines.engine import Engine
from deepproblog.engines.prolog_engine import (
    PrologEngine,
    pyswip_to_term,
    term_to_pyswip,
)
from deepproblog.engines.prolog_engine.heuristics import (
    GeometricMean,
    PartialProbability,
    Heuristic,
    LearnedHeuristic,
)
from deepproblog.engines.prolog_engine.swi_program import SWIProgram
from deepproblog.tensor import TensorStore


def wrap_tensor(x, store: TensorStore):
    if type(x) is list:
        return list2term([wrap_tensor(e, store) for e in x])
    else:
        return Term("tensor", Constant(store.store(x)))


def unwrap_tensor(x, model):
    if x.functor == "tensor":
        return model.get_tensor(x)
    else:
        return x


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

    def perform_count(self, batch, acs):
        if isinstance(self.heuristic, LearnedHeuristic):
            self.heuristic.count(batch, acs)

    def prepare(self, db):
        translated = SimpleProgram()

        for e in db:
            new_e = e
            if type(e) is Term or type(e) is Clause:
                body = None
                if type(e) is Clause:
                    body = e.body
                    e = e.head
                p = e.probability
                if p is not None and p.functor == "nn":
                    if len(p.args) == 4:
                        net, inputs, output, domain = p.args
                        heads = []
                        self.model.networks[str(net)].domain = term2list(domain, False)
                        for domain_n, domain_element in enumerate(
                                term2list(domain, False)
                        ):
                            head = e.with_probability(
                                p.with_args(net, inputs, Constant(domain_n))
                            )
                            head = head.apply_term({output: domain_element})
                            heads.append(head)
                        if type(e) is Clause:
                            new_e = AnnotatedDisjunction(heads, e.body)
                        else:
                            new_e = Or.from_list(heads)
                    elif len(p.args) == 3:
                        net, inputs, output = p.args
                        self.model.networks[str(net)].det = True
                        head = e.with_probability(None)
                        body2 = Term(
                            "extern", Term("{}_extern".format(net), inputs, output)
                        )
                        if body is not None:
                            body = And(body, body2)
                        else:
                            body = body2
                        new_e = Clause(head, body)
                    elif len(p.args) == 2:
                        # net = p.args[0]
                        new_e = e
                    else:
                        raise ValueError(
                            "A neural predicate with {} arguments is not supported.".format(
                                len(p.args)
                            )
                        )
            translated.add_clause(new_e)
        db = self.engine.prepare(translated)
        self.register_networks(db)
        self.register_parameter(db)

        def evaluate(network, inputs2):
            inputs3 = tuple(term2list(inputs2, False))
            probabilities = self.model.networks[network]([inputs3])
            self.model.networks[network].cache[inputs2] = probabilities[0]
            probabilities = probabilities.tolist()[0]
            return probabilities

        db.eval_mode = evaluate
        self.register_tensor(db)
        return db

    def ground(self, query, label=None, **kwargs):
        return self.engine.ground(
            self.model.solver.program, query.substitute().query, label=label, **kwargs
        )

    def get_network_func_3(self, network):
        def func(arguments, output_var):
            arguments, variables = pyswip_to_term(arguments, with_variables=True)
            net = self.model.networks[network]
            out = net([term2list(arguments, False)])[0]
            out = wrap_tensor(out, self.tensor_store)
            out = term_to_pyswip(out)
            output_var.unify(out)

        func.arity = 2
        return func

    def get_network_func_4(self, network):
        def func(inputs, output_var):
            inputs, variables = pyswip_to_term(inputs, with_variables=True)
            assert len(variables) == 0
            net = self.model.networks[network]
            probabilities = net([term2list(inputs, False)])
            probabilities = probabilities[0]
            output_var.unify([float(x) for x in probabilities])

        func.arity = 2
        return func

    def get_network_func_2(self, network):
        def func(inputs, output_var):
            inputs, variables = pyswip_to_term(inputs, with_variables=True)
            assert len(variables) == 0
            net = self.model.networks[network]
            probabilities = net([term2list(inputs, False)])
            probabilities = probabilities[0]
            output_var.unify(float(probabilities[0]))

        func.arity = 2
        return func

    def register_networks(self, db: SWIProgram):
        for network in self.model.networks:
            name = network + "_extern"
            if self.model.networks[network].domain is not None:
                func = self.get_network_func_4(network)
                db.registerForeign(func, name)
            elif self.model.networks[network].det:
                func = self.get_network_func_3(network)
                db.registerForeign(func, name)
            else:
                func = self.get_network_func_2(network)
                db.registerForeign(func, name)

    def register_parameter(self, db: SWIProgram):
        def get_parameter(parameter_id, p):
            probability = self.model.parameters[parameter_id]
            p.unify(probability)

        get_parameter.arity = 2
        db.registerForeign(get_parameter, "get_parameter")

    def register_tensor(self, db):
        def get_tensor_probability(parameter_id, p):
            probability = float(self.tensor_store[parameter_id])
            p.unify(probability)

        get_tensor_probability.arity = 2
        db.registerForeign(get_tensor_probability, "get_tensor_probability")

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

    def register_foreign_nondet(self, func, function_name, arity_in, arity_out):
        #TODO Implement
        pass

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
