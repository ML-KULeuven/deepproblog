import copy

from deepproblog.dataset import (
    DataLoader,
    QueryDataset,
    MutatingDataset,
    NumericIncorrectOutputMutator,
    NoiseMutatorDecorator,
)
from deepproblog.query import Query
from deepproblog.utils import parse
from problog.logic import Term, Constant


def test_query_dataset(tmpdir):
    with tmpdir.join("queries.txt").open(mode="wt") as tmpfile:
        tmpfile.write(
            "a(2,3,5).\nsubs(a(3,3,b),[[b,123]]).\n0.5 :: subs(a(3,3,c),[[c,123]]).\n0.7 :: b(1,2,3)."
        )
    dataset_file = QueryDataset(tmpdir.join("queries.txt"))
    dataset_list = QueryDataset(
        [
            Query(parse("a(2,3,5).")),
            Query(parse("a(3,3,b)."), substitution={Term("b"): Constant(123)}),
            Query(parse("a(3,3,c)."), substitution={Term("c"): Constant(123)}, p=0.5),
            Query(parse("b(1,2,3)."), p=0.7),
        ]
    )
    assert len(dataset_file) == 4
    assert len(dataset_list) == 4
    assert dataset_file.queries == dataset_list.queries
    assert dataset_list.to_queries() == dataset_list.queries


def test_dataset_write_to_file(tmpdir):
    dataset_list = QueryDataset(
        [
            Query(parse("a(2,3,5).")),
            Query(parse("a(3,3,b)."), substitution={Term("b"): Constant(123)}),
        ]
    )
    with tmpdir.join("queries_out.txt").open(mode="wt") as tmpfile:
        dataset_list.write_to_file(tmpfile)
    # Test that we can reload it.
    dataset_reloaded = QueryDataset(tmpdir.join("queries_out.txt"))
    assert dataset_reloaded.queries == dataset_list.queries


def test_subset():
    dataset = QueryDataset(
        [
            Query(parse("a(2,3,5).")),
            Query(parse("a(3,3,b)."), substitution={Term("b"): Constant(123)}),
        ]
    )
    assert dataset.subset(1).to_queries() == [dataset.queries[0]]


def test_extension():
    dataset1 = QueryDataset(
        [
            Query(parse("a(2,3,5).")),
            Query(parse("a(3,3,b)."), substitution={Term("b"): Constant(123)}),
        ]
    )
    dataset2 = QueryDataset([Query(parse("a(1,2,3)."))])
    assert (dataset1 + dataset2).to_queries() == dataset1.queries + dataset2.queries


def test_mutating_dataset():
    dataset1 = QueryDataset(
        [
            Query(parse("a(2,3,5).")),
            Query(parse("a(3,3,b)."), substitution={Term("b"): Constant(123)}),
        ]
    )

    def mutator(i: int, q: Query):
        q2 = copy.copy(q)
        q2.visited = True
        return q2

    mutated = MutatingDataset(dataset1, mutator)
    assert all(hasattr(e, "visited") for e in mutated.to_queries())


def test_noise_mutator():
    """Test that the noise is approximately correct"""
    hit_count = 0

    def inner_mutator(i: int, q: Query):
        nonlocal hit_count
        hit_count += 1
        return q

    mutator = NoiseMutatorDecorator(p=0.75, inner_mutator=inner_mutator, seed=123)

    total_count = 2000
    for i in range(total_count):
        mutator(i, Query(Term("dummy")))

    assert 0.7 < hit_count / total_count < 0.8

    # Check we get the same result twice
    hit_count1 = hit_count
    hit_count = 0
    for i in range(total_count):
        mutator(i, Query(Term("dummy")))
    assert hit_count == hit_count1


def test_numeric_incorrect_output_mutator():
    mutator = NumericIncorrectOutputMutator(domain=list(range(10)), seed=123)
    r1 = mutator(1, Query(parse("a(1,2,3).")))
    r2 = mutator(1, Query(parse("a(1,2,3).")))
    r3 = mutator(2, Query(parse("a(1,2,3).")))
    r4 = mutator(2, Query(parse("a(1,2,3).")))
    assert r1 == r2
    assert r3 == r4
    assert r1 != r3
    assert r1.query.args[-1].value != 3
    assert r3.query.args[-1].value != 3


def test_dataloader():
    dataset = QueryDataset(
        [
            Query(parse("a(2,3,5).")),
            Query(parse("a(1,2,3).")),
            Query(parse("a(3,3,b)."), substitution={Term("b"): Constant(123)}),
        ]
    )
    loader = DataLoader(dataset, 2, False)
    one_epoch = list(loader)
    assert one_epoch[0] + one_epoch[1] == dataset.queries
