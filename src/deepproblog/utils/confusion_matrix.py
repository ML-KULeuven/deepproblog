from typing import List, Union

import numpy as np

from deepproblog.utils import TabularFormatter


class ConfusionMatrix(object):
    def __init__(self, classes: Union[int, List[str]] = 0):
        if isinstance(classes, int):
            self.n = classes
            self.classes = list(range(self.n))
        else:
            self.classes = classes
            self.n = len(classes)
        self.matrix = np.zeros((self.n, self.n), dtype=np.uint)

    def get_index(self, c):
        if c not in self.classes:
            self.grow(c)
        return self.classes.index(c)

    def grow(self, c):
        self.classes.append(c)
        self.n = len(self.classes)
        new_matrix = np.zeros((self.n, self.n), dtype=np.uint)
        new_matrix[0 : self.n - 1, 0 : self.n - 1] = self.matrix
        self.matrix = new_matrix

    def add_item(self, predicted, actual):
        actual_i = self.get_index(actual)
        predicted_i = self.get_index(predicted)

        self.matrix[predicted_i, actual_i] += 1

    def __str__(self):
        formatter = TabularFormatter()
        data = [[""] * (self.n + 2), ["", ""] + self.classes]
        data[0][(self.n + 1) // 2 + 1] = "Actual"
        for row in range(self.n):
            data.append(
                [" ", self.classes[row]]
                + [str(self.matrix[row, col]) for col in range(self.n)]
            )
        data[len(data) // 2][0] = "Predicted"
        return formatter.format(data)

    def accuracy(self):
        correct = 0
        for i in range(self.n):
            correct += self.matrix[i, i]
        total = self.matrix.sum()
        acc = correct / total
        print("Accuracy: ", acc)
        return acc
