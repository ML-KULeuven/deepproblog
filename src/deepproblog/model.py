import json
import os
import pickle
import time
from collections import defaultdict
from io import BytesIO
from os import PathLike
from random import random
from typing import Collection, Optional, Tuple, List, Mapping, Any, Sequence, Union
from zipfile import ZipFile

import torch

from deepproblog.engines.builtins import register_tensor_predicates
from problog.logic import term2list, Term, Clause, Constant, InstantiationError, Or
from problog.program import PrologString, PrologFile, SimpleProgram, LogicProgram
from .dataset import Dataset, ImageDataset
from .embeddings import TermEmbedder
from .engines import Engine
from .network import Network
from .optimizer import Optimizer
from .query import Query
from .semiring import Result
from .solver import Solver
from .utils import check_path


class Model(object):
    def __init__(
        self,
        program_string: Union[str, os.PathLike],
        networks: Collection[Network],
        embeddings: Optional[TermEmbedder] = None,
        load: bool = True,
    ):
        """

        :param program_string: A string representing a DeepProbLog program or the path to a file containing a program.
        :param networks: A collection of networks that will be used to evaluate the neural predicates.
        :param embeddings: A TermEmbedder used to embed Terms in the program.
        :param load: If true, then it will attempt to load the program from 'program_string',
         else, it will consider program_string to be the program itself.
        """
        self.networks = dict()
        if load:
            self.program: LogicProgram = PrologFile(str(program_string))
        else:
            self.program: LogicProgram = PrologString(program_string)
        self.parameters = []
        self.parameter_groups = []
        self._extract_parameters()
        for network in networks:
            self.networks[network.name] = network
            network.model = self
        self.solver: Optional[Solver] = None
        self.eval_mode = False
        self.embeddings = embeddings
        self.tensor_sources = dict()
        self.optimizer = Optimizer(self)

    def get_embedding(self, term: Term):
        return self.embeddings.get_embedding(term)

    def evaluate_nn(self, to_evaluate: List[Tuple[Term, Term]]):
        """
        :param to_evaluate: List of neural predicates to evaluate
        :return: A dictionary with the elements of to_evaluate as keys, and the output of the NN as values.
        """
        result = dict()
        evaluations = defaultdict(list)
        # Group inputs per net to send in batch
        for net_name, inputs in to_evaluate:
            net = self.networks[str(net_name)]
            if net.det:
                tensor_name = Term("nn", net_name, inputs)
                if tensor_name not in self.solver.engine.tensor_store:
                    evaluations[net_name].append(inputs)
            else:
                if inputs in net.cache:
                    result[(net_name, inputs)] = net.cache[inputs]
                    del net.cache[inputs]
                else:
                    evaluations[net_name].append(inputs)
        for net in evaluations:
            network = self.networks[str(net)]
            out = network([term2list(x, False) for x in evaluations[net]])
            for i, k in enumerate(evaluations[net]):
                if network.det:
                    tensor_name = Term("nn", net, k)
                    self.solver.engine.tensor_store.store(out[i], tensor_name)
                else:
                    result[(net, k)] = out[i]
        return result

    def set_engine(self, engine: Engine, **kwargs):
        """
        Initializes the solver of this model with the given engine and additional arguments.
        :param engine: The engine that will be used to ground queries in this model.
        :param kwargs: Additional arguments passed to the solver.
        :return:
        """
        self.solver = Solver(self, engine, **kwargs)
        register_tensor_predicates(engine)

    def solve(self, batch: Sequence[Query]) -> List[Result]:
        return self.solver.solve(batch)

    def ground_dataset(self, dataset: Dataset):
        total_time = 0
        compile_times = []
        ground_times = []
        for q in dataset.to_queries():
            start = time.time()
            result = self.solver.cache.get(q)
            total_time += time.time() - start
            if not result.from_cache:
                compile_times.append(result.compile_time)
                ground_times.append(result.ground_time)
        return {
            "total_time": total_time,
            "ground_times": ground_times,
            "compile_times": compile_times,
        }

    def save_state(self, filename: Union[str, PathLike], complete=False):
        """
        Saves the state of this model to a zip file with the given filename. This only includes the probabilistic
            parameters and all parameters of the neural networks, but not the model architecture or neural architectures
        :param filename: The filename to save the model to.
        :param complete: If true, save neural networks with information needed to resume training.
        :return:
        """
        check_path(filename)
        with ZipFile(filename, "w") as zipf:
            with zipf.open("parameters", "w") as f:
                pickle.dump(self.parameters, f)
            for n in self.networks:
                with zipf.open(n, "w") as f:
                    self.networks[n].save(f, complete=complete)

    def load_state(self, filename: Union[str, PathLike]):
        """
        Restore the state of this model from the given filename. This only includes the probabilistic parameters
            and all parameters of the neural networks, but not the model architecture or neural architectures.
        :param filename: The filename to restore the model from.
        :return:
        """
        with ZipFile(filename) as zipf:
            with zipf.open("parameters") as f:
                self.parameters = pickle.load(f)
            for n in self.networks:
                with zipf.open(n) as f:
                    self.networks[n].load(BytesIO(f.read()))

    def eval(self):
        """
        Set the mode of all networks in the model to eval.
        """
        self.eval_mode = True
        for n in self.networks:
            self.networks[n].eval()
        self.solver.engine.eval()

    def train(self):
        """
        Set the mode of all networks in the model to train.
        :return:
        """
        self.eval_mode = False
        for n in self.networks:
            self.networks[n].train()
        self.solver.engine.train()

    def register_foreign(
        self, func: callable, function_name: str, arity_in: int, arity_out: int
    ):
        self.solver.engine.register_foreign(func, function_name, arity_in, arity_out)

    def __str__(self):
        return "\n".join(str(line) for line in self.program)

    def get_tensor(self, term: Term) -> torch.Tensor:
        """

        :param term: A term of the form tensor(_).
        If the tensor is of the form tensor(a(*args)), then it well look into tensor source a.
        :return:  Returns the stored tensor identifier by the term.
        """
        if len(term.args) > 0 and term.args[0].functor in self.tensor_sources:
            return self.tensor_sources[term.args[0].functor][term.args[0].args]
        return self.solver.get_tensor(term)

    def store_tensor(self, tensor: torch.Tensor) -> Term:
        """
        Stores a tensor in the tensor store and returns and identifier.
        :param tensor: The tensor to store.
        :return: The Term that is the identifier by which this tensor can be uniquely identified in the logic.
        """
        return Term("tensor", Constant(self.solver.engine.tensor_store.store(tensor)))

    def add_tensor_source(
        self, name: str, source: Union[ImageDataset, Mapping[Any, torch.Tensor]]
    ):
        """
        Adds a named tensor source to the model.
        :param name: The name of the added tensor source.
        :param source: The tensor source to add
        :return:
        """
        self.tensor_sources[name] = source

    def get_hyperparameters(self) -> dict:
        """
        Recursively build a dictionary containing the most important hyperparameters in the mode.
        :return: A dictionary that contains the values of the most important hyperparameters of the model.
        """
        parameters = dict()
        parameters["solver"] = (
            None if self.solver is None else self.solver.get_hyperparameters()
        )
        parameters["networks"] = [
            self.networks[network].get_hyperparameters() for network in self.networks
        ]
        parameters["program"] = self.program.to_prolog()
        return parameters

    def hyperparameters_to_file(self, filename):
        """
        Write the output of the get_hyperparameter() method in JSON format to a file.
        :param filename: The path to write the hyperparameters to.
        :return:
        """
        with open(filename, "w") as f:
            f.write(json.dumps(self.get_hyperparameters()))

    def _extract_parameters(self):
        translated = SimpleProgram()
        for n in self.program:
            if type(n) is Term:
                if (
                    n.probability is not None
                    and type(n.probability) is Term
                    and n.probability.functor == "t"
                ):
                    i = self._add_parameter(n.probability.args[0])
                    p = n.probability.with_args(Constant(i))
                    n = n.with_probability(p)
                translated.add_statement(n)
            elif type(n) is Clause:
                if (
                    n.head.probability is not None
                    and type(n.head.probability) is Term
                    and n.head.probability.functor == "t"
                ):
                    i = self._add_parameter(n.head.probability.args[0])
                    p = n.head.probability.with_args(Constant(i))
                    head = n.head.with_probability(p)
                    n = Clause(head, n.body)
                translated.add_statement(n)
            elif type(n) is Or:
                new_list = []
                new_group = []
                for x in n.to_list():
                    if (
                        x.probability is not None
                        and type(x.probability) is Term
                        and x.probability.functor == "t"
                    ):
                        i = self._add_parameter(x.probability.args[0])
                        new_group.append(i)
                        p = x.probability.with_args(Constant(i))
                        new_list.append(x.with_probability(p))
                    else:
                        new_list.append(x)
                if len(new_group) > 0:
                    self.parameter_groups.append(new_group)
                n = Or.from_list(new_list)
                translated.add_statement(n)
            else:
                translated.add_statement(n)
        self.program = translated

    def _add_parameter(self, val):
        i = len(self.parameters)
        try:
            val = float(val)
        except InstantiationError:
            val = random()
        self.parameters.append(val)
        return i
