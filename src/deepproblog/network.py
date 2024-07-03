from __future__ import annotations

from typing import Iterator, Optional

import torch


class Network(object):
    """Wraps a PyTorch neural network for use with DeepProblog"""

    def __init__(
        self,
        network_module: torch.nn.Module,
        name: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """Create a Network object

        :param network_module: The neural network module.
        :param name: The name of the network as used in the neural predicate nn(name, ...)
        :param optimizer: The optimizer that updates the neural network parameters.
        :param scheduler: An optional learn rate scheduler for the optimizer.
        """
        # State
        self.name = name
        self.network_module = network_module
        self.optimizer = optimizer

        self.is_cuda = False
        self.device = None
        self.eval_mode = False

    def __getstate__(self):
        state = {
            "name": self.name,
            "network_module": self.network_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return state

    def __setstate__(self, state):
        self.name = state["name"]
        self.network_module.load_state_dict(state["network_module"])
        self.optimizer.load_state_dict(state["optimizer"])

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

    # def step_epoch(self):
    #     """
    #     Call the step function of the scheduler
    #     :return:
    #     """
    #     if self.scheduler is not None:
    #         self.scheduler.step()

    # def load(self, location: Union[PathLike, IO[bytes]]):
    #     """
    #     Restore the parameters of this network from a file.
    #     :param location: The location of the parameter file.
    #     :return:
    #     """
    #     loaded: Dict[str, Any] = torch.load(location)
    #     self.network_module.load_state_dict(loaded["model_state_dict"])
    #     if "optimizer_state_dict" in loaded:
    #         assert self.optimizer is not None
    #         self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
    #     if "scheduler_state_dict" in loaded:
    #         assert self.scheduler is not None
    #         self.scheduler.load_state_dict(loaded["scheduler_state_dict"])
    #
    # def save(self, location: Union[PathLike, IO[bytes]], complete=False):
    #     """
    #     Save the parameters of this network to the given location.
    #     :param location: The location to save the parameter file to.
    #     :param complete: If true, also save information needed to resume training.
    #     :return:
    #     """
    #     to_save = {"model_state_dict": self.network_module.state_dict()}
    #     if complete:
    #         if self.optimizer is not None:
    #             to_save["optimizer_state_dict"] = self.optimizer.state_dict()
    #         if self.scheduler is not None:
    #             to_save["scheduler_state_dict"] = self.scheduler.state_dict()
    #     torch.save(to_save, location)

    def __call__(self, *arguments) -> tuple[torch.Tensor, ...]:
        batched_output: torch.Tensor = self.network_module(*arguments)
        return batched_output.unbind()

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

    def freeze(self):
        self.network_module.requires_grad = False

    def get_hyperparameters(self):
        parameters = {
            "name": self.name,
            "module": str(self.network_module),
            "optimizer": str(self.optimizer),
        }
        return parameters
