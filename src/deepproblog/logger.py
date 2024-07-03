import time
from collections import defaultdict
from statistics import mean
from typing import Iterable

from .semiring import Result


class Logger(object):
    def __init__(self):
        self.epoch_data = []
        self.iteration = 0
        self.epoch = 0

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def log_iteration(self, loss: float, results: Iterable[Result]):
        self.epoch_data[-1]["loss"].append(loss)
        self.epoch_data[-1]["time"].append(time.time())

        self.iteration += 1

    def log_epoch(self):
        self.epoch_data.append(defaultdict(list))
        self.epoch += 1

    def get_attribute_per_epoch(self, attribute):
        return [mean(epoch[attribute]) for epoch in self.epoch_data]

    def get_attribute(self, attribute):
        return sum((epoch[attribute] for epoch in self.epoch_data), [])


class VerboseLogger(Logger):
    def __init__(self, log_every: int = 0):
        super().__init__()
        self.log_every = log_every

    def log_iteration(self, loss: float, results: Iterable[Result]):
        super().log_iteration(loss, results)
        if self.log_every > 0 and self.iteration % self.log_every == 0:
            mean_loss = mean(self.get_attribute("loss")[-self.log_every :])
            total_time = (
                self.get_attribute("time")[-1]
                - self.get_attribute("time")[-self.log_every]
            )
            print(
                "\tIteration: {}\t\tTime: {:.2f}\t\tLoss: {:.4f}".format(
                    str(self.iteration).rjust(6), total_time, mean_loss
                )
            )


    def log_epoch(self):
        super().log_epoch()
        if self.epoch > 1:
            mean_loss = mean(self.epoch_data[-2]["loss"])
            total_time = (
                self.epoch_data[-2]["time"][-1] - self.epoch_data[-2]["time"][0]
            )
            print(
                "Total epoch time: {:.2f}\t\tEpoch mean loss: {:.4f}".format(
                    total_time, mean_loss
                )
            )
        print("Epoch: {}".format(self.epoch))
