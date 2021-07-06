import random
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple, TextIO, Union, Callable, Iterable

from PIL import Image

import problog
from deepproblog.query import Query
from problog.logic import Term, Constant


class Dataset(ABC):
    __slots__ = ()

    def __str__(self):
        return "\n".join(str(self.to_query(i)) for i in range(min(len(self), 5)))

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def to_query(self, i: int) -> Query:
        """

        :param i: Index i
        :return: The i-th example as a Query object
        """
        pass

    def to_queries(self) -> List[Query]:
        """

        :return: A list of all queries in the dataset.
        """
        return [self.to_query(i) for i in range(len(self))]

    def subset(self, i: int, j: Optional[int] = None) -> "Dataset":
        """

        :param i: index i
        :param j: index j
        :return: If j is None, returns a subset with the indices [0,i], else returns a subset with the indices [i, j]
        """
        if j is None:
            j = i
            i = 0
        return Subset(self, i, j)

    def __add__(self, other: "Dataset") -> "Dataset":
        """

        :param other: The other dataset.
        :return: Returns a dataset that is the combination of self and other
        """
        return Extension(self, other)

    def fold(self, n: int, i: int) -> Tuple["Dataset", "Dataset"]:
        """

        :param n: The number of folds to make.
        :param i: Which of the folds is the held-out set.
        :return: A tuple of the training fold and test fold datasets.
        """
        return Fold(self, n, i, False), Fold(self, n, i, True)

    def to_file_repr(self, i: int) -> str:
        """

        :param i: index
        :return: Returns a string representation of file i suitable for writing to a file. Defaults to to_query.
        """
        return str(self.to_query(i).to_text_query())

    def write_to_file(self, f: TextIO):
        """

        :param f: File to write the dataset. to
        """
        for i in range(len(self)):
            f.write(self.to_file_repr(i) + "\n")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        raise NotImplementedError()


class DataLoader(object):

    __slots__ = ("dataset", "batch_size", "length", "shuffle", "epoch", "rng", "i")

    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool = True, seed=None
    ):
        """

        :param dataset: The dataset that this loader will iterate over.
        :param batch_size: The batch size.
        :param shuffle: If true, the queries are shuffled, otherwise, they returned in order.
        :param seed: Seed for random shuffle.
        """
        self.dataset = dataset.to_queries()
        self.batch_size = batch_size
        self.length = len(self.dataset)
        self.shuffle = shuffle
        self.dataset = self.dataset[: self.length]
        self.epoch = 0
        self.rng = random.Random(seed)
        self._set_iter()

    def _shuffling(self):
        if callable(self.shuffle):
            return self.shuffle(self.epoch)
        return self.shuffle

    def _set_iter(self):
        if self._shuffling():
            indices = list(range(self.length))
            self.rng.shuffle(indices)
            self.i = iter(indices)
        else:
            self.i = iter(range(self.length))

    def __next__(self):
        if self.i is None:
            self.epoch += 1
            self._set_iter()
            raise StopIteration
        batch = list()
        try:
            for i in range(self.batch_size):
                batch.append(self.dataset[next(self.i)])
            return batch
        except StopIteration:
            if len(batch) == 0:
                self.epoch += 1
                self._set_iter()
                raise StopIteration
            else:
                self.i = None
            return batch

    def __iter__(self):
        return self

    def __len__(self):
        return int(ceil(self.length / self.batch_size))

    def __repr__(self):
        return "DataLoader: " + str(self.dataset[0])


class Subset(Dataset):
    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def to_query(self, i, *args):
        return self.dataset.to_query(i + self.i, *args)

    def to_file_repr(self, i):
        return self.dataset.to_file_repr(i + self.i)

    def __getitem__(self, item):
        # if item >= len(self):
        #     raise IndexError()
        return self.dataset[item + self.i]

    def __len__(self):
        return self.j - self.i

    def __init__(self, dataset, i, j):
        self.i = i
        self.j = min(j, len(dataset))
        self.dataset = dataset


class Extension(Dataset):
    def __len__(self):
        return len(self.a) + len(self.b)

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __getitem__(self, item):
        if item < len(self.a):
            return self.a[item]
        else:
            return self.b[item - len(self.a)]

    def to_query(self, i, *args):
        if i < len(self.a):
            return self.a.to_query(i, *args)
        else:
            return self.b.to_query(i - len(self.a), *args)

    def to_file_repr(self, i):
        if i < len(self.a):
            return self.a.to_file_repr(i)
        else:
            return self.b.to_file_repr(i - len(self.a))


class Fold(Dataset):
    def __len__(self):
        return self._len

    def to_file_repr(self, i):
        return self.parent.to_file_repr(self._get_index(i))

    def _get_index(self, i):
        i, j = i // (len(self.indices)), i % (len(self.indices))
        return i * self.n + self.indices[j]

    def to_query(self, i):
        return self.parent.to_query(self._get_index(i))

    def __init__(self, parent, n, i, split):
        self.parent = parent
        self.n = n
        self.i = i
        if split:
            self.indices = [i]
        else:
            self.indices = [x for x in range(n) if x != i]

        self._parent_len = len(parent)
        self._len = len(parent) // n * len(self.indices)
        extra = len(parent) % n
        if i < extra:
            extra = 1 if split else extra - 1
        else:
            extra = 0 if split else extra

        self._len += extra
        self.split = split


class MutatingDataset(Dataset):
    """
    Generic dataset adapter that mutates an underlying dataset.

    Intended use cases involve generating noisy datasets as well as negative examples.
    """

    __slots__ = ("inner_dataset", "mutator")

    def __init__(self, inner_dataset: Dataset, mutator: Callable[[int, Query], Query]):
        super().__init__()
        self.inner_dataset = inner_dataset
        self.mutator = mutator

    def __len__(self):
        return len(self.inner_dataset)

    def to_query(self, i: int) -> Query:
        return self.mutator(i, self.inner_dataset.to_query(i))

    def __getitem__(self, item):
        raise NotImplementedError("__getitem__ is not implemented for MutatingDataset")


class NoiseMutatorDecorator:
    """Dataset mutator that will mutate with a certain probability"""

    __slots__ = ("p", "seed", "inner_mutator")

    def __init__(
        self,
        p: float,
        inner_mutator: Callable[[int, Query], Query],
        seed: Optional[int] = None,
    ):
        """Constructor

        :param p: Probability with which to mutate the sample
        :param inner_mutator: Function that does actual mutation.
        :param seed: Seed for RNG
        """
        self.p = p
        if seed is None:
            seed = random.SystemRandom().randint(0, 2 ** 64)
        self.seed = seed
        self.inner_mutator = inner_mutator

    def __call__(self, index: int, query: Query) -> Query:
        rng = random.Random(self.seed ^ index)
        if rng.random() < self.p:
            return self.inner_mutator(index, query)
        else:
            return query


class NumericIncorrectOutputMutator:
    """Dataset mutator that replaces numeric output with an incorrect value. Only constants are supported."""

    __slots__ = ("domain", "change_p", "seed")

    def __init__(self, domain: List[int], change_p=False, seed: Optional[int] = None):
        """Constructor

        :param domain: Domain of outputs to choose from
        :param change_p: If true, set the new query to have p=1-original.p
        :param seed: Random seed
        """
        self.domain = domain
        self.change_p = change_p
        if seed is None:
            seed = random.SystemRandom().randint(0, 2 ** 64)
        self.seed = seed

    def __call__(self, index: int, query: Query) -> Query:
        # We want a stable behaviour given a specific index
        rng = random.Random(self.seed ^ index)
        new_args = list(query.query.args)
        for i in query.output_ind:
            arg = new_args[i]
            assert isinstance(arg, Constant)
            v = arg.value
            new_v = v
            while new_v == v:
                new_v = rng.choice(self.domain)
            new_args[i] = Constant(new_v)

        new_term = Term(query.query.functor, *new_args)
        return Query(
            query=new_term,
            substitution=query.substitution,
            p=(1.0 - query.p) if self.change_p else query.p,
            output_ind=query.output_ind,
        )


class ImageDataset(Dataset, ABC):
    def __init__(self, root, extension="png", transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.extension = extension

    def __getitem__(self, index):
        if type(index) is tuple:
            index = index[0]
        p = self.root / "{}.{}".format(index, self.extension)
        with open(p, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        return img


def load(filename: Union[Path, str], max_size=None):
    parser = problog.parser.PrologParser(problog.program.ExtendedPrologFactory())
    all_queries = list()
    with open(filename) as f:
        for i, line in enumerate(f):
            if max_size is not None and i >= max_size:
                break
            parsed: Iterable[Term] = parser.parseString(line)
            for term in parsed:
                query_prob = 1.0
                if term.probability:
                    query_prob = term.probability.value
                    term = term.with_probability()
                if term.functor == "subs":
                    t = term.args[0]
                    subs = problog.logic.term2list(term.args[1], False)
                    sub_dict = dict()
                    for sub in subs:
                        a, b = problog.logic.term2list(sub, False)
                        sub_dict[a] = b
                    all_queries.append(Query(t, substitution=sub_dict, p=query_prob))
                else:
                    all_queries.append(Query(term, p=query_prob))
    return all_queries


class QueryDataset(Dataset):
    """Dataset that loads queries from a file or a list"""

    def __len__(self):
        return len(self.queries)

    def to_query(self, i):
        return self.queries[i]

    def __init__(self, queries: Union[List[Query], Path, str]):
        super().__init__()
        if isinstance(queries, list):
            self.queries = queries
        else:
            self.queries = load(queries)
