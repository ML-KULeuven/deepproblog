import importlib
import tempfile
from itertools import chain
from pathlib import Path
from typing import (
    Iterable,
    Optional,
    Mapping,
    Union,
)

import numpy as np
import torch
from problog.logic import Term, Clause, Constant, Or, term2list
from problog.program import PrologString, LogicProgram, SimpleProgram
from tqdm import tqdm

from .dataset import Dataset, QueryDataset, DataLoader
from .default_datasets import default_datasets
from .engines import Engine, ExactEngine
from .engines.builtins import all_builtins
from .logger import Logger
from .network import Network
from .probabilistic_parameters import (
    extract_probabilistic_parameters,
    ProbabilisticParameters,
)
from .query import Query
from .utils.stop_condition import StopOnPlateau, EpochStop


class Model(object):
    def __init__(
        self,
        program_string: Optional[str],
        networks: Optional[Iterable[Network]] = None,
        logger: Optional[Logger] = None,
        path: str = None,
    ):
        """
        :param program_string: A string representing a DeepProbLog program or the path to a file containing a program.
        :param networks: A collection of networks that will be used to evaluate the neural predicates.
        """
        # State
        self.program_string = program_string
        self.logger = logger if logger is not None else Logger()
        self.included_files: dict[str, str] = dict()

        # Generated
        self.path = path
        self.networks: dict[str, Network] = dict()
        if networks is not None:
            self.networks = {network.name: network for network in networks}
        self.tensor_sources = dict()
        self._evidence: list[Iterable[Query]] = list()

        self.program = None
        self.parameters = None
        self.parse_program(program_string)
        self.builtins = all_builtins

    def parse_program(self, program_string: Optional[str]):
        if program_string is not None:
            program: LogicProgram = PrologString(program_string)
            program, parameters, groups = extract_probabilistic_parameters(program)
            self.program = self.transform_program(program)
            self.parameters = ProbabilisticParameters(parameters, groups)

    def __getstate__(self):
        state = {
            "program_string": self.program_string,
            "included_files": self.included_files,
            "network_states": {
                network: self.networks[network].__getstate__()
                for network in self.networks
            },
            "logger_type": type(self.logger),
            "logger_state": self.logger.__getstate__(),
        }
        return state

    def __setstate__(self, state):
        self.__init__(None)
        self.included_files = state["included_files"]
        self.parse_program(state["program_string"])
        network_states = state["network_states"]
        for network in network_states:
            self.networks[network].__setstate__(network_states[network])
        self.logger = state["logger_type"]()
        self.logger.__setstate__(state["logger_state"])

    @classmethod
    def from_file(
        cls,
        path,
        networks: Optional[Iterable[Network]] = None,
        logger: Optional[Logger] = None,
    ):
        with open(path) as f:
            program_string = f.read()
        return cls(program_string, networks, logger, path)

    def get_evidence(self) -> Iterable[Query]:
        yield from chain(*self._evidence)

    def evaluate_nn(
        self, neural_predicates: dict[str, set[tuple[Term, ...]]]
    ) -> dict[str, dict[tuple[Term, ...], torch.Tensor]]:
        """
        :param neural_predicates: Dictionary of neural probabilities to evaluate
        :return: A dictionary with the evaluated neural probabilities
        """
        evaluated = dict()
        intermediate_cache = dict()
        for net_name in neural_predicates:
            input_terms = list(neural_predicates[net_name])
            zipped_terms = zip(*input_terms)
            input_tensors = (
                [self.get_tensor(arg, intermediate_cache) for arg in args]
                for args in zipped_terms
            )
            output_tensors = self.networks[net_name](*input_tensors)
            evaluated[net_name] = dict(zip(input_terms, output_tensors))
        return evaluated

    def eval(self):
        """
        Set the mode of all networks in the model to eval.
        """
        for n in self.networks:
            self.networks[n].eval()

    def train(self):
        """
        Set the mode of all networks in the model to train.
        :return:
        """
        for n in self.networks:
            self.networks[n].train()

    def freeze(self):
        for n in self.networks:
            self.networks[n].freeze()

    def step(self):
        for n in self.networks:
            self.networks[n].step()

    def zero_grad(self):
        for n in self.networks:
            self.networks[n].zero_grad()

    def _get_tensor(
        self, term: Term, cache: Optional[dict[Term, torch.tensor]]
    ) -> Union[torch.Tensor, Term]:
        """

        :param term: A term of the form tensor(_).
        If the tensor is of the form tensor(a(*args)), then it well look into tensor source 'a'.
        :return:  Returns the stored tensor identifier by the term.
        """
        if term.functor == "tensor":
            tensor_id = term.args[0]
            if tensor_id.functor == "nn":
                net_name = tensor_id.args[0].functor
                args = term2list(tensor_id.args[1], False)
                args = ([self.get_tensor(arg, cache)] for arg in args)
                return self.networks[net_name](*args)[0]
            elif tensor_id.functor in self.tensor_sources:
                return self.tensor_sources[tensor_id.functor][tensor_id.args]
            elif tensor_id.functor in self.builtins:
                args = tuple(self.get_tensor(arg, cache) for arg in tensor_id.args)
                return self.builtins[tensor_id.functor][0](*args)
            else:
                raise ValueError("Unknown tensor: {}".format(tensor_id))
        else:
            return term

    def get_tensor(
        self, term: Term, intermediate_cache: Optional[dict[Term, torch.tensor]] = None
    ) -> torch.Tensor:
        if intermediate_cache is None:
            return self._get_tensor(term, intermediate_cache)
        else:
            tensor = intermediate_cache.get(term)
            if tensor is None:
                tensor = self._get_tensor(term, intermediate_cache)
                intermediate_cache[term] = tensor
            return tensor

    def add_tensor_source(self, name: str, source: Mapping[Term, torch.Tensor]):
        """
        Adds a named tensor source to the model.
        :param name: The name of the added tensor source.
        :param source: The tensor source to add
        :return:
        """
        self.tensor_sources[name] = source

    def get_hyperparameters(self) -> dict:
        """
        Recursively build a dictionary containing the most important hyperparameters in the model.
        :return: A dictionary that contains the values of the most important hyperparameters of the model.
        """
        parameters = dict()
        parameters["networks"] = [
            self.networks[network].get_hyperparameters() for network in self.networks
        ]
        parameters["program"] = self.program.to_prolog()
        return parameters

    def predict(self, dataset: Dataset, engine: Optional[Engine] = None, bar=True):
        if engine is None:
            engine = ExactEngine(self, cache_memory=True)

        iterator = dataset.to_queries()
        result = list()
        if bar:
            iterator = tqdm(iterator)
        for i, gt_query in enumerate(iterator):
            test_query = gt_query.variable_output()
            answer = engine.query(test_query).evaluate(self, gt_query.substitution)
            answer = max(answer.result, key=lambda x: answer.result[x])
            answer = dataset.label_to_indicator(gt_query.output_value(answer))
            result.append(answer)

        return np.array(result, dtype=int)

    def query(self, query: Query, engine: Optional[Engine] = None):
        if engine is None:
            engine = ExactEngine(self, cache_memory=True)
        return engine.query(query).evaluate(self, query.substitution)

    def fit(
        self,
        dataset: Optional[Dataset] = None,
        engine: Optional[Engine] = None,
        batch_size: int = 16,
        shuffle=True,
        stop_condition=None,
        loss_function=None,
    ):
        if engine is None:
            engine = ExactEngine(self, cache_memory=True)
        if dataset is None:
            dataset = QueryDataset(self.get_evidence())
        if stop_condition is None:
            stop_condition = StopOnPlateau("loss", 1e-4, 4)
        elif type(stop_condition) is int:
            stop_condition = EpochStop(stop_condition)

        if loss_function is None:
            loss_function = torch.nn.BCELoss()

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        while not stop_condition.is_stop(self.logger):
            self.logger.log_epoch()
            for batch in dataloader:
                self.train()
                self.zero_grad()

                arithmetic_circuits = zip(batch, engine.query_batch(batch))
                evaluated = [
                    ac.evaluate(self, query.substitution)
                    for query, ac in arithmetic_circuits
                ]
                predictions = torch.stack([result.prediction() for result in evaluated])
                target = torch.tensor([query.p for query in batch])

                loss = loss_function(predictions, target)
                loss.backward()
                self.step()

                self.logger.log_iteration(float(loss), evaluated)

        return self

    def transform_program(self, program):
        translated = SimpleProgram()
        for clause in program:
            for translated_clause in self.transform_clause(clause):
                translated.add_clause(translated_clause)
                print(translated_clause)
        return translated

    def transform_clause(self, clause):
        if is_directive(clause):
            yield from self.parse_directive(clause)
        else:
            if isinstance(clause, Clause):
                if is_neural_predicate(clause.head):
                    yield from translate_neural_predicate(clause.head, prefix="aux_")
                    new_body = clause.body & Term(
                        "aux_" + clause.head.functor, *clause.head.args
                    )
                    yield Clause(clause.head.with_probability(None), new_body)
                else:
                    yield clause
            else:
                if is_neural_predicate(clause):
                    yield from translate_neural_predicate(clause)
                else:
                    yield clause

    def parse_directive(self, clause: Clause):
        directive = clause.body
        if directive.functor == "include_dataset":
            self.add_dataset(directive.args[0].functor)
        elif directive.functor == "include_network":
            self.add_network(directive.args[0], *directive.args[1:])
        elif directive.functor == "include_evidence":
            self.add_evidence(directive.args[0])
        else:
            yield clause

    def add_dataset(self, dataset_name: str):
        self.add_tensor_source(dataset_name, default_datasets[dataset_name])

    def add_network(
        self, network_name: Term, path: Term, function_name: Optional[Term] = None
    ):
        network_name = network_name.functor
        if path.functor == "default":
            default_directory = Path(__file__).parent / "default_networks"
            path = default_directory / "{}.py".format(path.args[0])
        else:
            path = self.get_path(path)
        module_loader = importlib.machinery.SourceFileLoader(network_name, str(path))
        module = module_loader.load_module()
        if function_name is None:
            network, optimizer = module.get_network()
        else:
            network, optimizer = getattr(module, function_name.functor.strip("'"))()
        network = Network(network, network_name, optimizer)
        self.networks[network_name] = network

    def add_evidence(self, path: Term):
        path = self.get_path(path)
        with open(path) as f:
            file_content = f.read()

        self._evidence.append(self.load_queries(file_content))

    def get_path(self, path_term: Term):
        path_term = path_term.functor.strip("'")
        if path_term in self.included_files:
            temp_path = Path(tempfile.gettempdir()) / path_term
            with open(temp_path, "w") as f:
                f.write(self.included_files[path_term])
            return temp_path
        else:
            path = Path(self.path).parent / path_term
            with open(path) as f:
                file_content = f.read()
            self.included_files[path_term] = file_content
            return path

    def load_queries(self, prolog: str) -> Iterable[Query]:
        clauses = PrologString(prolog)
        for clause in clauses:
            if is_directive(clause):
                yield from self.parse_directive(clause)
            else:
                yield Query(clause).substitute_tensors()


def is_directive(clause: Term):
    return isinstance(clause, Clause) and clause.head.functor == "_directive"


def is_neural_predicate(term: Term):
    return term.probability is not None and term.probability.functor == "nn"


def translate_neural_predicate(term: Term, prefix=""):
    p = term.probability
    if len(p.args) == 4:
        yield from translate_neural_ad(term, prefix)
    elif len(p.args) == 3:
        yield from translate_neural_function(term, prefix)
    elif len(p.args) == 2:
        yield from translate_neural_fact(term, prefix)
    else:
        raise ValueError(
            "A neural predicate with {} arguments is not supported.".format(len(p.args))
        )


def translate_neural_ad(term: Term, prefix=""):
    p = term.probability
    net, inputs, output, domain = p.args

    heads = []
    for i, domain_element in enumerate(term2list(domain, False)):
        head = Term(
            prefix + term.functor, *term.args, p=p.with_args(net, inputs, Constant(i))
        )
        head = head.apply_term({output: domain_element})
        heads.append(head)

    yield Or.from_list(heads)


def translate_neural_function(term: Term, prefix=""):
    p = term.probability
    net, inputs, output = p.args
    term = Term(prefix + term.functor, *term.args)
    yield term.apply_term({output: Term("tensor", p.with_args(net, inputs))})


def translate_neural_fact(term: Term, prefix=""):
    yield Term(prefix + term.functor, *term.args, p=term.probability)
