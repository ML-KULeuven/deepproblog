import dataclasses
from typing import Callable, TypeVar, List

import pytest

from deepproblog.utils.cache import Cache

K = TypeVar("K")
T = TypeVar("T")


@dataclasses.dataclass
class _ResultType:
    value: str


@pytest.fixture(
    params=[
        {"cache_dict": False, "cache_file": False},
        {"cache_dict": True, "cache_file": False},
        {"cache_dict": False, "cache_file": True},
        {"cache_dict": True, "cache_file": True},
    ]
)
def cache_factory(request, tmpdir):
    kwargs = {
        "cache": request.param["cache_dict"],
        "cache_root": tmpdir.join("cache_dir") if request.param["cache_file"] else None,
    }

    def create_cache(func: Callable[[K], T], key_func=lambda x: x) -> Cache[K, T]:
        return Cache[K, T](func=func, key_func=key_func, **kwargs)

    # Certain tests are only valid if a cache is used.
    create_cache.no_cache = not (
        request.param["cache_dict"] or request.param["cache_file"]
    )
    return create_cache


def test_basic(cache_factory):
    cache = cache_factory(lambda x: _ResultType(str(x)))
    assert cache[32] == _ResultType("32")
    assert cache[1] == _ResultType("1")
    # This should retrieve it from cache
    assert cache[32] == _ResultType("32")


class _TerribleHash:
    """Test class to test hash collisions"""

    def __init__(self, v: int):
        self.v = v

    def __eq__(self, other):
        return isinstance(other, _TerribleHash) and other.v == self.v

    def __hash__(self):
        """Intentionally terrible hash to test hash collisions"""
        return 5 if self.v > 5 else 1


def test_hash_collision(cache_factory):
    cache = cache_factory(lambda x: _ResultType(str(x.v)))
    assert cache[_TerribleHash(1)] == _ResultType("1")
    assert cache[_TerribleHash(6)] == _ResultType("6")
    assert cache[_TerribleHash(2)] == _ResultType("2")


def test_invalidate(cache_factory):
    """Test that cache is properly invalidated on a call to invalidate()"""
    hash_breaker = 3
    cache = cache_factory(lambda x: _ResultType(str(x + hash_breaker)))
    assert cache[1] == _ResultType("4")
    hash_breaker = 2
    if not cache_factory.no_cache:
        # If caching is used, this should not yet return the new value.
        assert cache[1] == _ResultType("4")
    cache.invalidate()
    assert cache[1] == _ResultType("3")


@dataclasses.dataclass
class UnhashableData:
    cache_key: int
    unhashable_data: List[int]


def test_unhashable(cache_factory):
    if cache_factory.no_cache:
        pytest.skip("Unsupported when cache is not used.")
    cache = cache_factory(lambda x: _ResultType(str(x.cache_key)))
    with pytest.raises(TypeError):
        assert cache[UnhashableData(1, [])] == _ResultType("1")


def test_key_func(cache_factory):
    cache = cache_factory(
        lambda x: _ResultType(str(x.cache_key)), key_func=lambda x: x.cache_key
    )
    assert cache[UnhashableData(1, [])] == _ResultType("1")
    assert cache[UnhashableData(1, [1, 2, 3])] == _ResultType("1")
