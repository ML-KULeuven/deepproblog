import bisect

import numpy as np

from deepproblog.utils import check_path, split


class Logger(object):
    def __init__(self):
        self.log_data = dict()
        self.indices = list()
        self.comments = list()

    def log(self, name, index, value):
        if name not in self.log_data:
            self.log_data[name] = dict()
        i = bisect.bisect_left(self.indices, index)
        if i >= len(self.indices) or self.indices[i] != index:
            self.indices.insert(i, index)
        self.log_data[name][index] = value

    def log_list(self, i, l):
        if l is not None:
            for e in l:
                self.log(e[0], i, e[1])

    def comment(self, comment):
        self.comments += comment.split("\n")

    def get_attribute(self, attribute, include_indices=False):
        sorted_attribute = []
        if attribute not in self.log_data:
            return []
        indices = []
        for i in self.indices:
            if i in self.log_data[attribute]:
                indices.append(i)
                sorted_attribute.append(self.log_data[attribute][i])
        if include_indices:
            return indices, sorted_attribute
        return sorted_attribute

    def get_union(self, att1, att2):
        indices = []
        sorted_attribute = []
        for i in self.indices:
            if i in self.log_data[att1] and i in self.log_data[att2]:
                indices.append(i)
                sorted_attribute.append(
                    (self.log_data[att1][i], self.log_data[att2][i])
                )

        return indices, sorted_attribute

    def __getitem__(self, item):
        return self.get_attribute(item, True)

    def __str__(self):
        lines = ["#{}".format(comment) for comment in self.comments]
        columns = list(self.log_data.keys())
        lines += ["i," + ",".join(columns)]
        for i in self.indices:
            row = [str(i)]
            for c in columns:
                row.append(str(self.log_data[c].get(i, "")))
            lines.append(",".join(row))
        return "\n".join(lines)

    def has_attribute(self, attribute):
        return attribute in self.log_data

    def write_to_file(self, name):
        # datetime = strftime('_%y%m%d_%H%M%S')
        filename = name + ".log"
        check_path(filename)
        with open(filename, "w") as f:
            f.write(str(self))

    def read_from_file(self, fname):
        headers = None
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if line[0] == "#":
                    self.comments.append(line[1:])
                    continue
                if headers is None:
                    headers = split(line, ",")
                    continue
                data = line.split(",")
                index = 0
                for i, h in enumerate(headers):
                    if h == "i":
                        index = int(data[i])
                        continue
                    if len(data[i]) > 0:
                        self.log(h, index, float(data[i]))


def equalize(data):  # Makes all sets of data contain same indices
    indices = set()
    for d in data:
        for i in d[0]:
            indices.add(i)
    indices = sorted(indices)
    new_data = []
    for d_ind, d_data in data:
        new_d = []
        prev_value = 0
        for i in indices:
            try:
                prev_value = d_data[d_ind.index(i)]
            except ValueError:
                pass
            new_d.append(prev_value)
        new_data.append(new_d)
    return indices, new_data


def aggregate(data, percentiles):
    matrix = np.zeros((len(data[0]), len(data)))
    for i, d in enumerate(data):
        matrix[:, i] = d
    stats = np.percentile(matrix, percentiles, 1)
    return stats


def aggregate_loggers(loggers, x, y, percentiles=(0.5,)):
    x_data = [logger[x] for logger in loggers]
    y_data = [logger[y] for logger in loggers]
    data = [tuple(equalize([x_data[i], y_data[i]])[1]) for i in range(len(x_data))]
    t, data = equalize(data)
    return t, aggregate(data, percentiles)
