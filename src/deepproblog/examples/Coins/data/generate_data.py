import random

coin1 = [("heads", 0.5), ("tails", 0.5)]
coin2 = [("heads", 0.5), ("tails", 0.5)]


def sample(dist):
    r = random.random()
    for v, p in dist:
        r -= p
        if r < 0:
            return v
    return dist[-1][1]


def next_example():
    c1 = sample(coin1)
    c2 = sample(coin1)
    return c1, c2


def generate_examples(N, fname):
    lines = list()
    for i in range(N):
        c1, c2 = next_example()
        lines.append(", ".join([c1, c2]))
    with open(fname + ".csv", "w") as f:
        f.write("\n".join(lines))


random.seed(1234)

generate_examples(100, "label_data/train")

generate_examples(20, "label_data/test")
