import signal
import time
from typing import List, Callable, Union

from deepproblog.dataset import DataLoader
from deepproblog.model import Model
from deepproblog.query import Query
from deepproblog.utils.logger import Logger
from deepproblog.utils.stop_condition import EpochStop
from deepproblog.utils.stop_condition import StopCondition


class TrainObject(object):
    """
    An object that performs the training of the model and keeps track of the state of the training.
    """

    def __init__(self, model: Model):
        self.model = model
        self.logger = Logger()
        self.accumulated_loss = 0
        self.i = 1
        self.start = 0
        self.prev_iter_time = 0
        self.epoch = 0
        self.previous_handler = None
        self.interrupt = False
        self.hooks = []
        self.timing = [0, 0, 0]

    def get_loss(self, batch: List[Query], backpropagate_loss: Callable) -> float:
        """
        Calculates and propagates the loss for a given batch of queries and loss function.
        :param batch: The batch of queries.
        :param backpropagate_loss:  The loss function. It should also perform the backpropagation.
        :return: The average loss over the batch
        """
        total_loss = 0
        result = self.model.solve(batch)
        for r in result:
            self.timing[0] += r.ground_time / len(batch)
            self.timing[1] += r.compile_time / len(batch)
            self.timing[2] += r.eval_time / len(batch)
        result = [
            (result[i], batch[i]) for i in range(len(batch)) if len(result[i]) > 0
        ]
        for r, q in result:
            total_loss += backpropagate_loss(
                r, q.p, weight=1 / len(result), q=q.substitute().query
            )
        return total_loss

    def get_loss_with_negatives(
        self, batch: List[Query], backpropagate_loss: Callable
    ) -> float:
        """
        Calculates and propagates the loss for a given batch of queries and loss function.
        This includes negative examples. Negative examples are found by using the query.replace_var method.
        :param batch: The batch of queries.
        :param backpropagate_loss:  The loss function. It should also perform the backpropagation.
        :return: The average loss over the batch
        """
        total_loss = 0

        result = self.model.solve([q.variable_output() for q in batch])
        result = [(result[i], batch[i]) for i in range(len(batch))]

        for r, q in result:
            expected = q.substitute().query
            try:
                total_loss += backpropagate_loss(
                    r, q.p, weight=1 / len(result), q=expected
                )
            except KeyError:
                self.get_loss([q], backpropagate_loss)
            neg_proofs = [x for x in r if x != expected]
            for neg in neg_proofs:
                # print('penalizing wrong answer {} vs {}'.format(q.substitute().query, k))
                total_loss += backpropagate_loss(
                    r, 0, weight=1 / (len(result) * len(neg_proofs)), q=neg
                )
        return total_loss

    def train(
        self,
        loader: DataLoader,
        stop_criterion: Union[int, StopCondition],
        verbose: int = 1,
        loss_function_name: str = "cross_entropy",
        with_negatives: bool = False,
        log_iter: int = 100,
        initial_test: bool = True,
        **kwargs
    ) -> Logger:

        self.previous_handler = signal.getsignal(signal.SIGINT)
        loss_function = getattr(self.model.solver.semiring, loss_function_name)

        self.accumulated_loss = 0
        self.timing = [0, 0, 0]
        self.epoch = 0
        self.start = time.time()
        self.prev_iter_time = time.time()
        epoch_size = len(loader)
        if "test" in kwargs and initial_test:
            value = kwargs["test"](self.model)
            self.logger.log_list(self.i, value)
            print("Test: ", value)

        if type(stop_criterion) is int:
            stop_criterion = EpochStop(stop_criterion)
        print("Training ", stop_criterion)

        while not (stop_criterion.is_stop(self) or self.interrupt):
            epoch_start = time.time()
            self.model.optimizer.step_epoch()
            if verbose and epoch_size > log_iter:
                print("Epoch", self.epoch + 1)
            for batch in loader:
                if self.interrupt:
                    break
                self.i += 1
                self.model.train()
                self.model.optimizer.zero_grad()
                if with_negatives:
                    loss = self.get_loss_with_negatives(batch, loss_function)
                else:
                    loss = self.get_loss(batch, loss_function)
                self.accumulated_loss += loss

                self.model.optimizer.step()
                self.log(verbose=verbose, log_iter=log_iter, **kwargs)
                for j, hook in self.hooks:
                    if self.i % j == 0:
                        hook(self)

                if stop_criterion.is_stop(self):
                    break
            if verbose and epoch_size > log_iter:
                print("Epoch time: ", time.time() - epoch_start)
            self.epoch += 1
        if "snapshot_name" in kwargs:
            filename = "{}_final.mdl".format(kwargs["snapshot_name"])
            print("Writing snapshot to " + filename)
            self.model.save_state(filename)

        signal.signal(signal.SIGINT, self.previous_handler)
        return self.logger

    def log(
        self, snapshot_iter=None, log_iter=100, test_iter=1000, verbose=1, **kwargs
    ):
        iter_time = time.time()

        if (
            "snapshot_name" in kwargs
            and snapshot_iter is not None
            and self.i % snapshot_iter == 0
        ):
            filename = "{}_iter_{}.mdl".format(kwargs["snapshot_name"], self.i)
            print("Writing snapshot to " + filename)
            self.model.save_state(filename)
        if verbose and self.i % log_iter == 0:
            print(
                "Iteration: ",
                self.i,
                "\ts:%.4f" % (iter_time - self.prev_iter_time),
                "\tAverage Loss: ",
                self.accumulated_loss / log_iter,
            )
            if len(self.model.parameters):
                print("\t".join(str(parameter) for parameter in self.model.parameters))
            self.logger.log("time", self.i, iter_time - self.start)
            self.logger.log("loss", self.i, self.accumulated_loss / log_iter)
            self.logger.log("ground_time", self.i, self.timing[0] / log_iter)
            self.logger.log("compile_time", self.i, self.timing[1] / log_iter)
            self.logger.log("eval_time", self.i, self.timing[2] / log_iter)
            # for k in self.model.parameters:
            #     self.logger.log(str(k), self.i, self.model.parameters[k])
            #     print(str(k), self.model.parameters[k])
            self.accumulated_loss = 0
            self.timing = [0, 0, 0]
            self.prev_iter_time = iter_time
        if "test" in kwargs and self.i % test_iter == 0:
            value = kwargs["test"](self.model)
            self.logger.log_list(self.i, value)
            print("Test: ", value)

    def write_to_file(self, *args, **kwargs):
        self.logger.write_to_file(*args, **kwargs)


def train_model(
    model: Model,
    loader: DataLoader,
    stop_condition: Union[int, StopCondition],
    **kwargs
) -> TrainObject:
    train_object = TrainObject(model)
    train_object.train(loader, stop_condition, **kwargs)
    return train_object
