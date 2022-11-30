import json
from collections import defaultdict
from pathlib import Path

import torchvision.transforms as transforms
from PIL import Image
from problog.logic import Term, Constant, list2term

from deepproblog.dataset import Dataset
from deepproblog.query import Query

root = Path(__file__).parent


def create_split_train_val():
    if not ((root / "expr_train2.json").exists() and (root / "expr_val.json").exists()):
        train = list()
        val = list()

        with open(root / "expr_train.json") as f:
            data = json.load(f)

        for i, d in enumerate(data):
            if i % 10 == 0:
                val.append(d)
            else:
                train.append(d)

        print("Train size", len(train))
        print("Val size", len(val))

        with open(root / "expr_train2.json", "w") as f:
            json.dump(train, f)
        with open(root / "expr_val.json", "w") as f:
            json.dump(val, f)


class HWFImages(object):
    def __init__(self, in_memory=True, transform=None):
        self.data = dict()
        self.image_root = root / "Handwritten_Math_Symbols"
        self.in_memory = in_memory
        self.transform = transform
        if in_memory:
            for subdir in self.image_root.iterdir():
                name = subdir.name
                for image in subdir.iterdir():
                    name += "/" + image.name
                    image = Image.open(image)
                    if transform is not None:
                        image = transform(image)
                    self.data[name] = image

    def get_image(self, path):

        if self.in_memory:
            return self.data[path]
        else:
            image = Image.open(self.image_root / path)
            if self.transform is not None:
                image = transform(image)
            return image

    def __getitem__(self, item):
        path = str(item[0].functor).strip('"')
        return self.get_image(path)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
hwf_images = HWFImages(False, transform)


class Expression(object):
    def __init__(self, data):
        self.id = data["id"]
        self.images = data["img_paths"]
        self.expr = data["expr"]
        self.res = float(data["res"])
        self.length = len(self.images)

    def to_query(self):
        images = [Term("tensor", Term("hwf", Constant(path))) for path in self.images]
        term = Term("expression", list2term(images), Constant(self.res))
        return Query(term)

    def __repr__(self):
        return " ".join(self.expr)

    def labeled_images(self):
        return zip(self.images, self.expr)


class HWFDataset(Dataset):
    def __len__(self):
        return len(self.expressions)

    def to_query(self, i):
        return self.expressions[i].to_query()

    def __init__(self, name, filter):
        create_split_train_val()
        self.name = name
        self.expressions = []
        self.lengths = defaultdict(list)
        self.images = set()
        with open(root / "expr_{}.json".format(name)) as f:
            data = json.load(f)
            for d in data:
                expression = Expression(d)
                if filter(expression.length):
                    self.lengths[expression.length].append(expression)
                    # self.lengths[expression.length].append(len(self.expressions))
                    self.expressions.append(expression)
                    self.images.update(expression.labeled_images())
