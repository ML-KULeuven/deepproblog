class StopCondition(object):
    def is_stop(self, train_object):
        raise NotImplementedError()

    def __add__(self, other):
        return Or(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __mul__(self, other):
        return And(self, other)

    def __and__(self, other):
        return And(self, other)


class EpochStop(StopCondition):
    def __init__(self, max_epoch):
        self.max_epoch = max_epoch

    def __str__(self):
        return "for {} epoch(s)".format(self.max_epoch)

    def is_stop(self, train_object):
        return train_object.epoch >= self.max_epoch


class StopOnPlateau(StopCondition):
    def __init__(self, attribute, delta=0.01, patience=5, warm_up=5):
        self.attribute = attribute
        self.delta = delta
        self.patience = patience
        self.best = warm_up
        self.warm_up = warm_up

    def __str__(self):
        return "until plateau in {}".format(self.attribute)

    def is_stop(self, train_object):
        data = train_object.logger.get_attribute(self.attribute)
        if len(data) <= self.warm_up:
            return False
        if self.best == -1 or (
            data[-1] > data[self.best] + data[self.best] * self.delta
        ):
            self.best = len(data) - 1

        if len(data) - self.best >= self.patience:
            print("No improvement for {} steps. Stopping.".format(self.patience))
            return True
        return False


class StopOnNoChange(StopCondition):
    def __init__(self, attribute, delta=0.01, patience=5, warm_up=5):
        self.attribute = attribute
        self.delta = delta
        self.patience = patience
        self.last_change = warm_up
        self.warm_up = warm_up

    def __str__(self):
        return "until no change in {}".format(self.attribute)

    def is_stop(self, train_object):
        data = train_object.logger.get_attribute(self.attribute)
        if len(data) <= self.warm_up:
            return False
        if abs(data[-1] - data[self.last_change]) > data[self.last_change] * self.delta:
            self.last_change = len(data) - 1

        if len(data) - self.last_change > self.patience:
            print("No change for {} steps. Stopping.".format(self.patience))
            return True
        return False


class Or(StopCondition):
    def __init__(self, *criteria):
        self.criteria = criteria

    def __str__(self):
        return " or ".join([str(c) for c in self.criteria])

    def is_stop(self, train_object):
        for c in self.criteria:
            if c.is_stop(train_object):
                return True
        return False


class And(StopCondition):
    def __init__(self, *criteria):
        self.criteria = criteria

    def __str__(self):
        return " and ".join([str(c) for c in self.criteria])

    def is_stop(self, train_object):
        for c in self.criteria:
            if not c.is_stop(train_object):
                return False
        return True


class Threshold(StopCondition):
    def __init__(self, attribute, max, duration=1):
        self.attribute = attribute
        self.max = max
        self.no_data = 0
        self.duration = duration

    def __str__(self):
        return "until {} reaches {}".format(self.attribute, self.max)

    def is_stop(self, train_object):
        data = train_object.logger.get_attribute(self.attribute)
        if len(data) == 0:
            self.no_data += 1
            if self.no_data > 2:
                pass
                # print('Received no data about {} for {} steps'.format(self.attribute, self.no_data))
            return False
        for i in range(self.duration):
            if data[-i - 1] < self.max:
                return False
        return True
