from deepproblog.utils import check_path

template = """
[Default]
batch_size = {0}
infoloss = {1}
name = poker_batch_{0}_infoloss_{1}
"""
i = 0

check_path("parameter_cfg/0.cfg")

for batch_size in [10, 25, 50, 100]:
    for infoloss in [0, 0.5, 1.0, 2.0, 4.0]:
        with open("parameter_cfg/{}.cfg".format(i), "w") as f:
            f.write(template.format(batch_size, infoloss))
        i += 1

template = """
[Default]
batch_size = 10
infoloss = 0.5
labeled = {0}
name = poker_batch_labeled_{0}_{1}
"""
i = 0

check_path("experiment/0.cfg")

for labeled in [300]:
    for rep in range(10):
        with open("experiment/{}.cfg".format(i), "w") as f:
            f.write(template.format(labeled, rep))
        i += 1
