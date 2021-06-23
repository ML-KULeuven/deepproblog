from random import sample, random, shuffle

values = ["jack", "queen", "king", "ace"]
suits = ["hearts", "clubs", "diamonds", "spades"]

deck = [(value, suit) for value in values for suit in suits]


def draw_cards(n_cards=4, n_decks=2):
    return list(sample(deck * n_decks, n_cards))


def weighted_sample(distribution):
    sample = random()
    for w, v in distribution:
        if sample < w:
            return v
        sample -= w
    return distribution[-1][1]


def next_game():
    cards = draw_cards()
    return cards[0:2], cards[2:4]


fair = [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]

unfair = [0.2, 0.4, 0.15, 0.25], [0.25, 0.25, 0.25, 0.25]


def card_to_string(v, s):
    return "{} of {}".format(v, s)


def get_N(N, distribution):
    out = []
    p_tot = 1.0
    for p in distribution:
        n = int(round(p / p_tot * N))
        out.append(n)
        N -= n
        p_tot -= p
    # out.append(N - total)
    return out


def write_examples(distribution, N, location):
    lines = []
    house_cards = []
    for i, n1 in enumerate(get_N(N, distribution[0])):
        for j, n2 in enumerate(get_N(n1, distribution[1])):
            house_cards += [(values[i], suits[j])] * n2

    shuffle(house_cards)
    print(house_cards[:10])
    for house in house_cards:
        h1, h2 = next_game()
        h1, h2 = [card_to_string(*x) for x in h1], [card_to_string(*x) for x in h2]
        house = card_to_string(*house)
        lines.append(",".join(h1 + h2 + [house]))
        with open(location, "w") as f:
            f.write("\n".join(lines))


write_examples(fair, 500, "labels/fair.csv")
write_examples(unfair, 500, "labels/unfair.csv")

write_examples(fair, 100, "labels/fair_test.csv")
write_examples(unfair, 100, "labels/unfair_test.csv")
