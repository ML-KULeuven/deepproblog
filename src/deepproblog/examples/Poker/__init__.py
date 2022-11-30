from collections import defaultdict
from typing import Tuple, List

import torchvision.transforms as transforms
from problog.logic import Term, Constant, list2term

from deepproblog.dataset import ImageDataset
from deepproblog.query import Query

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

Card = List[str]
Hand = Tuple[str, str]


class PokerSeparate(ImageDataset):
    def __getitem__(self, i):
        return super().__getitem__("{}_{}".format(*i))

    def __len__(self):
        return len(self.data)

    def to_query(self, i):
        cards = self.data[i]
        cards, h1, h2, outcome, p = cards[:5], cards[5], cards[6], cards[7], cards[8]
        sub_images = [Term("card_{}".format(x)) for x in range(4)]
        images = [
            Term("tensor", Term(self.dataset, Constant(i), Constant(x)))
            for x in range(4)
        ]
        sub = {sub_images[i]: images[i] for i in range(4)}
        if self.extra_supervision and i % 5 == 0:
            return Query(
                Term(
                    "game",
                    list2term(sub_images),
                    Term("card", Term(cards[-1][0])),
                    Term(outcome),
                ),
                sub,
                p,
            )
        else:
            return Query(Term("game", list2term(sub_images), Term(outcome)), sub, p)

    def __init__(
        self, dataset, extra_supervision=False, probs=(0.25, 0.25, 0.25, 0.25)
    ):
        super().__init__("data/images/{}".format(dataset), transform=transform)
        self.data: List[
            Tuple[Card, Card, Card, Card, Card, Hand, Hand, str, float]
        ] = []
        self.suits = False
        self.extra_supervision = extra_supervision
        self.dataset = dataset
        count = defaultdict(int)
        with open("data/labels/{}.csv".format(dataset)) as f:
            for line in f:
                cards: List[Card] = [c.split(" of ") for c in line.strip().split(",")]
                count[cards[-1][0]] += 1
                h1, h2 = (
                    best_hand(cards[:2] + [cards[-1]], self.suits),
                    best_hand(cards[2:], self.suits),
                )
                result = get_outcome(h1, h2)
                result_probabilities = get_probabilities(
                    cards[:2], cards[2:4], self.suits, probs
                )
                p = result_probabilities[result]
                self.data.append(tuple(cards + [h1, h2, result, p]))
        print(dataset)
        for c in count:
            print("\t", c, count[c] / len(self.data))


values = ["jack", "queen", "king", "ace"]
suits = ["hearts", "clubs", "diamonds", "spades"]
types = ["high", "pair", "threeofakind", "straight", "flush"]
subranks = {v: values.index(v) for v in values}
subranks.update({"low": 0, "high": 1, None: 0})


def is_flush(hand):
    suit = hand[0][1]
    for _, s in hand:
        if s != suit:
            return False
    return "flush", None


def is_straight(hand):
    sorted_hand = sorted(hand, key=lambda x: values.index(x[0]))
    sorted_hand = list(x[0] for x in sorted_hand)
    if sorted_hand == ["jack", "queen", "king"]:
        return "straight", "low"
    if sorted_hand == ["queen", "king", "ace"]:
        return "straight", "high"
    return False


def three_of_a_kind(hand):
    value = hand[0][0]
    for v, _ in hand:
        if v != value:
            return False
    return "threeofakind", value


def two_of_a_kind(hand):
    sorted_hand = sorted(hand, key=lambda x: values.index(x[0]))
    sorted_hand = list(x[0] for x in sorted_hand)
    value = sorted_hand[1]
    if sorted_hand[0] == value or sorted_hand[2] == value:
        return "pair", value
    return False


def high_card(hand):
    sorted_hand = sorted(hand, key=lambda x: values.index(x[0]))
    sorted_hand = list(x[0] for x in sorted_hand)
    return "high", sorted_hand[-1]


def best_hand(hand, with_suits):
    if with_suits:
        possibilities = [
            is_flush,
            is_straight,
            three_of_a_kind,
            two_of_a_kind,
            high_card,
        ]
    else:
        possibilities = [is_straight, three_of_a_kind, two_of_a_kind, high_card]
    for t in possibilities:
        result = t(hand)
        if result:
            return result
    return None


def get_outcome(t1, t2):
    i1, i2 = types.index(t1[0]), types.index(t2[0])
    if i1 > i2:
        return "win"
    if i1 < i2:
        return "loss"
    i1, i2 = subranks[t1[1]], subranks[t2[1]]
    if i1 > i2:
        return "win"
    if i1 < i2:
        return "loss"
    return "draw"


def hand_to_term(hand):
    if hand[1] is None:
        return Term(hand[0])
    else:
        return Term(hand[0], Term(hand[1]))


def get_probabilities(c1, c2, use_suits, p):
    outcomes = {"win": 0.0, "loss": 0.0, "draw": 0.0}
    for c in [(v, s) for v in values for s in suits]:
        h1, h2 = best_hand(c1 + [c], use_suits), best_hand(c2 + [c], use_suits)
        result = get_outcome(h1, h2)
        outcomes[result] += p[values.index(c[0])] * 0.25
    return outcomes
