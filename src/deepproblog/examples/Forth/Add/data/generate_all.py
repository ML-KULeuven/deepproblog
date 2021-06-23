carries = list()
results = list()

for n1 in range(0, 10):
    for n2 in range(0, 10):
        for carry in [0, 1]:
            ncarry = (n1 + n2 + carry) // 10
            res = (n1 + n2 + carry) % 10
            carries.append((n1, n2, carry, ncarry))
            results.append((n1, n2, carry, res))

carries = ["carry({},{},{},{}).".format(*c) for c in carries]
results = ["result({},{},{},{}).".format(*r) for r in results]

with open("tests.pl", "w") as f:
    f.write("\n".join(carries + results))
