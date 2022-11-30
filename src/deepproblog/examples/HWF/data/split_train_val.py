import json

train = list()
val = list()

with open("expr_train.json") as f:
    data = json.load(f)

for i, d in enumerate(data):
    if i % 10 == 0:
        val.append(d)
    else:
        train.append(d)

print("Train size", len(train))
print("Val size", len(val))

with open("expr_train2.json", "w") as f:
    json.dump(train, f)
with open("expr_val.json", "w") as f:
    json.dump(val, f)
