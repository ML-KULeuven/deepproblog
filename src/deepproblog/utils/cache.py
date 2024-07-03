import pickle
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Callable, Generic, Hashable, TypeVar, Union

from deepproblog.utils import check_path

K = TypeVar("K")
T = TypeVar("T")


class Cache(ABC, Generic[K, T]):
    def __init__(
        self,
        func: Callable[[K], T],
        key_func: Callable[[K], Hashable] = lambda x: x,  # hash(str(x.query))
    ):
        """Create cache

        :param func: The evaluation function to cache
        :param key_func: Convert the key into a hashable object if needed
        """
        self.func = func
        self.key_func = key_func
        self.hits = 0
        self.misses = 0

    def __call__(self, item: K) -> T:
        result = self.retrieve(self.key_func(item))
        if result is None:
            self.misses += 1
            result = self.func(item)
            self.store(self.key_func(item), result)
        else:
            self.hits += 1
        return result

    @abstractmethod
    def retrieve(self, key):
        pass

    @abstractmethod
    def store(self, key, item):
        pass

    def invalidate(self):
        if isinstance(self.func, Cache):
            self.func.invalidate()

    def reset_count(self):
        self.hits = 0
        self.misses = 0

    @abstractmethod
    def __len__(self):
        pass

    def get_hit_rate(self):
        return self.hits / (self.hits + self.misses)

    def __str__(self):
        return "Cache size: {}\tHit rate: {}".format(len(self), self.get_hit_rate())


class DictCache(Cache):
    def __init__(
        self, func: Callable[[K], T], key_func: Callable[[K], Hashable] = lambda x: x
    ):
        super().__init__(func, key_func)
        self._cache = dict()
        print("Caching to memory.")

    def retrieve(self, key):
        return self._cache.get(key)

    def store(self, key, item):
        self._cache[key] = item

    def invalidate(self):
        super().invalidate()
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


class FileCache(Cache):
    def __init__(
        self,
        func: Callable[[K], T],
        root: Union[str, PathLike],
        key_func: Callable[[K], Hashable] = lambda x: x,
    ):
        super().__init__(func, key_func)
        self.root = Path(root)
        print("Caching in {}.".format(self.root.resolve()))

    def get_filepath(self, key):
        return self.root / (str(key) + ".cache")

    def retrieve(self, key):
        """Implements file based cache."""
        filepath = self.get_filepath(key)

        if filepath.exists():
            with open(filepath, "rb") as f:
                result = pickle.load(f)
                return result
        return None

    def store(self, key, item):
        filepath = self.get_filepath(key)
        check_path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(item, f)

    def invalidate(self):
        super().invalidate()
        for p in self.root.glob("*.cache"):
            p.unlink()

    def __len__(self):
        return len([p for p in self.root.glob("*.cache")])


# class NoCache(Cache):
#
#     def __init__(self, func: Callable[[K], T]):
#         super().__init__(func, lambda x: None)
#
#     def retrieve(self, key):
#         return None
#
#     def store(self, key, item):
#         pass
#
#     def invalidate(self):
#         pass
#
#     def __len__(self):
#         return 0
#
#     def __str__(self):
#         return ''

# class CascadeCache(Cache):
#
#
#     def __init__(self, func: Callable[[K], T], key_func: Callable[[K], Hashable] = lambda x: x):
#         super().__init__(func, key_func)
#
#     def retrieve(self, key):
#         pass
#
#     def store(self, key, item):
#         pass
#
#     def invalidate(self):
#         pass
#
#     def __len__(self):
#         pass
