import pickle
from os import PathLike
from pathlib import Path
from typing import Callable, Generic, Hashable, TypeVar, Union

from deepproblog.utils import check_path

K = TypeVar("K")
T = TypeVar("T")


class Cache(Generic[K, T]):
    def __init__(
        self,
        func: Callable[[K], T],
        cache: bool,
        cache_root: Union[None, str, PathLike],
        key_func: Callable[[K], Hashable] = lambda x: x,
    ):
        """Create cache

        :param func: Function this cache evaluates
        :param cache: If true, do in memory caching.
        :param cache_root: If not None, cache to files at the provided path.
        :param key_func: Convert the key into a hashable object if needed
        """
        self.func = func
        self.key_func = key_func
        self.cache = cache
        if cache is True:
            print("Caching ACs")
        self.cache_file = cache_root is not None
        self.cache_dict = dict()
        self.first = func
        if self.cache_file:
            self.cache_root = Path(cache_root)
            self.first = self.get_from_file
        if self.cache:
            self.first = self.get_from_dict

    def __getitem__(self, item: K) -> T:
        return self.first(item)

    def invalidate(self):
        """Invalidate entire cache."""
        self.cache_dict.clear()
        if self.cache_file:
            for p in self.cache_root.iterdir():
                p.unlink()

    def get(self, item: K) -> T:
        return self.first(item)

    def get_from_dict(self, item: K) -> T:
        """Implements dict based cache."""
        cache_key = self.key_func(item)
        result = self.cache_dict.get(cache_key)
        if result is None:
            if self.cache_file:
                result = self.get_from_file(item, cache_key)
            else:
                result = self.func(item)
                result.from_cache = False
            self.cache_dict[cache_key] = result
        else:
            result.from_cache = True
        return result

    def get_from_file(self, item: K, cache_key=None) -> T:
        """Implements file based cache."""
        if cache_key is None:
            cache_key = self.key_func(item)
        filepath = self.cache_root / str(hash(cache_key))
        result = None
        if filepath.exists():
            with open(filepath, "rb") as f:
                (key, result) = pickle.load(f)
            if key == cache_key:
                return result
            else:
                # Hash collision! Handle this by overwriting the cache with the new query
                result = None
        if result is None:
            result = self.func(item)
            check_path(filepath)
            with open(filepath, "wb") as f:
                pickle.dump((cache_key, result), f)
        return result
