import re

for train in [2, 3, 4]:
    for test in [8, 64]:
        for mode in ["train", "test", "dev"]:
            with open("train{}_test{}/{}.txt".format(train, test, mode)) as f:
                text = f.read()
            matches = re.findall("\[([0-9 ]*)\]\t\[([0-9 ]*) (\d*)\]", text)
            text = list()
            for match in matches:
                l_sorted = [int(i) for i in match[0].split(" ")]
                l = [int(i) for i in match[1].split(" ")]
                l.reverse()
                l_sorted.reverse()
                text.append("forth_sort({},{}).".format(l, l_sorted))

            with open("train{}_test{}_{}.txt".format(train, test, mode), "w") as f:
                f.write("\n".join(text))
