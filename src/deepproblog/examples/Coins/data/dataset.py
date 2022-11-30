import os

import torchvision.transforms as transforms

from deepproblog.dataset import ImageDataset
from deepproblog.query import Query
from problog.logic import Term, Constant

path = os.path.dirname(os.path.abspath(__file__))

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


class Coins(ImageDataset):
    def __init__(
        self, subset,
    ):
        super().__init__("{}/image_data/{}/".format(path, subset), transform=transform)
        self.data = []
        self.subset = subset
        with open("{}/label_data/{}.csv".format(path, subset)) as f:
            for line in f:
                c1, c2 = [l.strip() for l in line.split(",")]
                outcome = "loss"
                if c1 == c2:
                    outcome = "win"
                self.data.append((c1, c2, outcome))

    def to_query(self, i):
        c1, c2, outcome = self.data[i]
        sub = {Term("a"): Term("tensor", Term(self.subset, Constant(i)))}
        return Query(Term("game", Term("a"), Term(outcome)), sub)

    def __len__(self):
        return len(self.data)


train_dataset = Coins("train")
test_dataset = Coins("test")
