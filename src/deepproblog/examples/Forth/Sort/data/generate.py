import random


def random_list(length):
    l = []
    for i in range(length):
        l.append(random.randint(0, 9))

    return l


def next_example(length):
    l1 = random_list(length)
    l2 = sorted(l1)
    return l1, l2


def generate_examples(N, length, out_file):
    with open(out_file, "w") as f:
        for i in range(N):
            l1, l2 = next_example(length)
            f.write(
                "forth_sort([{}],[{}]).\n".format(
                    ",".join([str(l) for l in l1]), ",".join([str(l) for l in l2])
                )
            )


for i in [5, 6, 7, 8]:
    for j in [8, 64]:
        generate_examples(256, i, "train{}_test{}_train.txt".format(i, j))
        generate_examples(32, j, "train{}_test{}_test.txt".format(i, j))
        generate_examples(32, j, "train{}_test{}_dev.txt".format(i, j))
