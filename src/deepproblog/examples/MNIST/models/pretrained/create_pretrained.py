import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from deepproblog.examples.MNIST.data import datasets
from deepproblog.examples.MNIST.network import MNIST_Net


def get_accuracy(model, dataloader: DataLoader):
    total = 0
    correct = 0
    for input_data, gt_labels in dataloader:
        _, predicted = torch.max(model(input_data), 1)
        total += len(gt_labels)
        correct_labels = torch.eq(predicted, gt_labels)
        correct += correct_labels.sum().item()
    return correct / total


accuracies = dict()


def get_subset(dataset, n, allowed=None):
    print(dataset, n, allowed)
    indices = []
    for i, d in enumerate(dataset):
        if len(indices) > n:
            break
        if allowed is None or d[1] in allowed:
            indices.append(i)
    return Subset(dataset, indices)


# allowed_digits = {'even': [0, 2, 4, 6, 8], 'odd': [1, 3, 5, 7, 9], 'all': list(range(10))}
allowed_digits = {"all": list(range(10))}
for nr_examples in [8, 16, 32, 64, 128, 256]:
    for k in allowed_digits:
        net = MNIST_Net(with_softmax=False)
        subset = get_subset(datasets["train"], nr_examples, allowed_digits[k])
        dataloader = DataLoader(subset, 4, shuffle=True)
        dataloader_test = DataLoader(datasets["test"], 4)
        optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = CrossEntropyLoss()

        cumulative_loss = 0
        i = 0

        for _ in range(4):
            for epoch in range(10):
                for batch in dataloader:
                    i += 1
                    data, labels = batch

                    optimizer.zero_grad()

                    data = Variable(data)
                    out = net(data)

                    loss = criterion(out, labels)
                    cumulative_loss += float(loss)
                    loss.backward()
                    optimizer.step()

                    if i % 50 == 0:
                        print("Loss: ", cumulative_loss / 100.0)
                        cumulative_loss = 0
            print("Accuracy", get_accuracy(net, dataloader_test))
        accuracies[(k, nr_examples)] = get_accuracy(net, dataloader_test)
        torch.save(net.state_dict(), "{}_{}.pth".format(k, nr_examples))

with open("accuracies.txt", "w") as f:
    for k in accuracies:
        f.write("{}\t{}\n".format(k, accuracies[k]))
