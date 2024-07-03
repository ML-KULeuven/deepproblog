from ..logger import Logger


class StopCondition(object):
    def is_stop(self, train_object: "TrainObject"):
        raise NotImplementedError()

    def __add__(self, other: "StopCondition"):
        return Or(self, other)

    def __or__(self, other: "StopCondition"):
        return Or(self, other)

    def __mul__(self, other: "StopCondition"):
        return And(self, other)

    def __and__(self, other: "StopCondition"):
        return And(self, other)


class EpochStop(StopCondition):
    def __init__(self, max_epoch: int):
        self.max_epoch = max_epoch

    def __str__(self):
        return "for {} epoch(s)".format(self.max_epoch)

    def is_stop(self, logger: Logger):
        return logger.epoch >= self.max_epoch


class StopOnPlateau(StopCondition):
    def __init__(self, attribute: str, delta=0.01, patience=5, aggregate=min):
        self.attribute = attribute
        self.delta = delta
        self.patience = patience
        self.aggregate = aggregate

    def __str__(self):
        return "until {} plateaus for {} epochs".format(self.attribute, self.patience)

    def is_stop(self, logger: Logger):
        if logger.epoch <= self.patience:
            return False
        old_best = self.aggregate(logger.get_attribute_per_epoch(self.attribute)[:-self.patience])
        all_best = self.aggregate(logger.get_attribute_per_epoch(self.attribute))

        if abs(old_best - all_best) < self.delta:
            print("No improvement for {} steps. Stopping.".format(self.patience))
            return True
        return False


class StopOnNoChange(StopCondition):
    def __init__(self, attribute: str, delta=0.01, patience=5):
        self.attribute = attribute
        self.delta = delta
        self.patience = patience

    def __str__(self):
        return "until no change in {} for {} epochs".format(self.attribute, self.patience)

    def is_stop(self, logger: Logger):
        if logger.epoch < self.patience:
            return False
        data = logger.get_attribute_per_epoch(self.attribute)[-self.patience:]

        if (max(data) - min(data)) < self.delta:
            print("No change for {} steps. Stopping.".format(self.patience))
            return True

        return False


class Or(StopCondition):
    def __init__(self, *criteria: "StopCondition"):
        self.criteria = criteria

    def __str__(self):
        return " or ".join([str(c) for c in self.criteria])

    def is_stop(self, train_object: "TrainObject"):
        for c in self.criteria:
            if c.is_stop(train_object):
                return True
        return False


class And(StopCondition):
    def __init__(self, *criteria: "StopCondition"):
        self.criteria = criteria

    def __str__(self):
        return " and ".join([str(c) for c in self.criteria])

    def is_stop(self, train_object: "TrainObject"):
        for c in self.criteria:
            if not c.is_stop(train_object):
                return False
        return True


class Threshold(StopCondition):
    def __init__(self, attribute: str, threshold, lower_bound=False, duration=1):
        self.attribute = attribute
        self.threshold = threshold
        self.lower_bound = lower_bound
        self.no_data = 0
        self.duration = duration

    def __str__(self):
        if self.lower_bound:
            return "until {} <= {} for {} steps".format(self.attribute, self.threshold, self.duration)
        else:
            return "until {} >= {} for {} steps".format(self.attribute, self.threshold, self.duration)

    def is_stop(self, train_object: "TrainObject"):
        data = train_object.logger.get_attribute(self.attribute)
        if len(data) == 0:
            self.no_data += 1
            if self.no_data > 2:
                pass
                # print('Received no data about {} for {} steps'.format(self.attribute, self.no_data))
            return False
        for i in range(self.duration):
            if self.lower_bound:
                if data[-i - 1] > self.threshold:
                    return False
            else:
                if data[-i - 1] < self.threshold:
                    return False
        return True
