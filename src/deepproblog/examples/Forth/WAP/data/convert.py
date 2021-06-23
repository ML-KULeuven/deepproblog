import re

for filename in ["train", "test", "dev"]:
    out_lines = []
    with open(filename + ".txt") as f:
        for line in f:
            solution, question = line.rstrip().split("\t")
            solution = int(float(solution))
            numbers = [int(i) for i in re.findall(" (\d+)", question)]
            out_lines.append(
                'wap("{}",{},{},{},{}).'.format(question, *numbers, solution)
            )
    with open("{}.pl".format(filename), "w") as f:
        f.write("\n".join(out_lines))
