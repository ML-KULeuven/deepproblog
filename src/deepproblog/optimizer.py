from collections import defaultdict
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from .model import Model


class Optimizer(object):
    """
    The optimizer responsible for optimizing the neural parameters in the model.
    """

    def __init__(self, model: "Model"):
        """
        :param model: The model whose parameters to optimize.
        """
        self.model = model
        self._params_grad = defaultdict(float)
        self.epoch = 0

    def step_epoch(self):
        """
        Perform an epoch step. This calls the step_epoch method of all the networks in the model.
        :return:
        """
        self.epoch += 1
        for _, network in self.model.networks.items():
            network.step_epoch()

    def add_parameter_gradient(self, k, grad: Tensor):
        """
        Accumulate the gradient for a probabilistic parameter.
        :param k: The key of the parameter.
        :param grad: The gradient to accumulate.
        :return:
        """
        pass

    def zero_grad(self):
        """
        Zeroes all gradients.
        :return:
        """
        self._params_grad.clear()
        for _, network in self.model.networks.items():
            network.zero_grad()
        if self.model.embeddings is not None:
            self.model.embeddings.zero_grad()

    def step(self):
        """
        Perform an optimization step for all parameters.
        :return:
        """
        for _, network in self.model.networks.items():
            network.step()
        if self.model.embeddings is not None:
            self.model.embeddings.step()


class SGD(Optimizer):
    """
    An optimizer that also optimizes the probabilistic parameters in the model using stochastic gradient descent.
    """

    def __init__(self, model: "Model", param_lr: float):
        """

        :param model: The model whose parameters will be optimized.
        :param param_lr: The learning rate for the parameters.
        """
        Optimizer.__init__(self, model)
        self.param_lr = param_lr

    def get_lr(self) -> float:
        """

        :return: The learning rate for the probabilistic parameters.
        """
        return self.param_lr

    def add_parameter_gradient(self, k, grad: Tensor):
        self._params_grad[k] += grad

    def step(self):
        Optimizer.step(self)
        for k in self._params_grad:
            self.model.parameters[k] -= float(self.get_lr() * self._params_grad[k])
            self.model.parameters[k] = max(min(self.model.parameters[k], 1.0), 0.0)
        for group in self.model.parameter_groups:
            p_sum = sum(self.model.parameters[x] for x in group)
            for param in group:
                self.model.parameters[param] /= p_sum
