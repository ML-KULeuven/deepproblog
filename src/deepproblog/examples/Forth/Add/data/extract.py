import re

for train in [2, 4, 8]:
    for test in [8, 64]:
        for mode in ["train", "test", "dev"]:
            with open("train{}_test{}/{}.txt".format(train, test, mode)) as f:
                text = f.read()
            matches = re.findall("\[([0-9 ]*)\]\t\[([0-9 ]*) (\d) (\d*)\]", text)
            text = list()
            for match in matches:
                res = match[0].strip().split(" ")
                digits = match[1].strip().split(" ")
                carry = [match[2]]
                length = int(match[3])
                digit1 = list()
                digit2 = list()
                for i in range(0, len(digits), 2):
                    digit1.append(digits[i])
                    digit2.append(digits[i + 1])
                text.append(
                    "add([{}],[{}],{},[{}]).".format(
                        *[",".join(l) for l in [digit1, digit2, carry, res]]
                    )
                )

            with open("train{}_test{}_{}.txt".format(train, test, mode), "w") as f:
                f.write("\n".join(text))
