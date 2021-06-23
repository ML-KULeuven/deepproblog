from __future__ import annotations

from os import PathLike
from typing import Any, Dict, IO, Iterator, Optional, Union, List

import torch


def get_tensor_function(network: Network):
    def tensor_function(*args):
        return tuple(network.model.get_tensor(arg) for arg in args)

    return tensor_function


class Network(object):
    def __init__(
        self,
        network_module: torch.nn.Module,
        name: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        k: Optional[int] = None,
        batching: bool = False,
    ):
        """

        :param network_module: The neural network module.
        :param name: The name of the network as used in the neural predicate nn(name, ...)
        :param optimizer: The optimizer that updates the neural network parameters.
        :param scheduler: An optional learn rate scheduler for the optimizer.
        :param k: If k is set, only the top k results of the neural network will be used.
        :param batching: If batching is true, the inputs will be stacked and evaluated in the network in a batch.
        Otherwise, they are evaluated one by one.
        """
        self.network_module = network_module
        self.name = name
        # self.function = function
        # if function is None:
        self.function = get_tensor_function(self)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = None
        self.is_cuda = False
        self.device = None
        self.n = 0
        self.domain = None
        self.cache = dict()
        self.k = k
        self.eval_mode = False
        self.batching = batching
        self.det = False

    def zero_grad(self):
        """
        Call zero grad on the optimizer
        :return:
        """
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Return the parameters of the network module
        :return:
        """
        return self.network_module.parameters()

    def step(self):
        """
        Call the step function of the optimizer
        :return:
        """
        if self.optimizer is not None:
            self.optimizer.step()

    def step_epoch(self):
        """
        Call the step function of the scheduler
        :return:
        """
        if self.scheduler is not None:
            self.scheduler.step()

    def load(self, location: Union[PathLike, IO[bytes]]):
        """
        Restore the parameters of this network from a file.
        :param location: The location of the parameter file.
        :return:
        """
        loaded: Dict[str, Any] = torch.load(location)
        self.network_module.load_state_dict(loaded["model_state_dict"])
        if "optimizer_state_dict" in loaded:
            assert self.optimizer is not None
            self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        if "scheduler_state_dict" in loaded:
            assert self.scheduler is not None
            self.scheduler.load_state_dict(loaded["scheduler_state_dict"])

    def save(self, location: Union[PathLike, IO[bytes]], complete=False):
        """
        Save the parameters of this network to the given location.
        :param location: The location to save the parameter file to.
        :param complete: If true, also save information needed to resume training.
        :return:
        """
        to_save = {"model_state_dict": self.network_module.state_dict()}
        if complete:
            if self.optimizer is not None:
                to_save["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                to_save["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(to_save, location)

    def __call__(self, to_evaluate: list) -> list:
        """
        Evaluate the network on the given inputs.
        :param to_evaluate: A list that contains the inputs that the neural network should be evaluated on.
        :return:
        """
        if self.batching:
            batched_inputs: List[torch.Tensor] = [
                self.function(*e)[0] for e in to_evaluate
            ]
            stacked_inputs = torch.stack(batched_inputs)
            if self.is_cuda:
                stacked_inputs = stacked_inputs.cuda(device=self.device)
            evaluated = self.network_module(stacked_inputs)
        else:
            evaluated = [self.network_module(*self.function(*e)) for e in to_evaluate]
        return evaluated

    def cuda(self, device=None):
        """
        Mode the network to a cuda device if CUDA is available.
        :param device:  The cuda device to move the network to.
        """
        if torch.cuda.is_available():
            self.is_cuda = True
            self.device = device
            self.network_module.cuda(device)
            print("Moving ", self.name, " to GPU")
        else:
            print("CUDA is unavailable")
        return self

    def eval(self):
        """
        Set the network to eval mode.
        """
        self.network_module.eval()
        self.eval_mode = True

    def train(self):
        """
        Set the network to train mode.
        """
        self.network_module.train()
        self.eval_mode = False

    def get_hyperparameters(self):
        parameters = {
            "name": self.name,
            "module": str(self.network_module),
            "optimizer": str(self.optimizer),
            "k": self.k,
        }
        return parameters


# class NetworkEvaluation(object):
#     """
#     An object that keeps track of which inputs the neural networks need to be evaluated on.
#     """
#
#     def __init__(self, batching: bool = False):
#         self.evaluated = None
#         self.added = []
#         self.batching = batching
#
#     def add(self, e):
#         self.added.append(e)
#
#     def evaluate(self, function: Callable, network: Network):
#         net = network.network_module
#         if self.batching:
#             batched_inputs = [function(*e) for e in self.added]
#             self.evaluated = net(torch.stack(batched_inputs))
#         else:
#             self.evaluated = torch.stack([net(*function(*e)) for e in self.added], 0)
#
#     def __getitem__(self, item):
#         return self.evaluated[item]
#
#     def item(self):
#         return self[0]
